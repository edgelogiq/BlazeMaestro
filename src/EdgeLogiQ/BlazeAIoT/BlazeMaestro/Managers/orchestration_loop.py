"""
Orchestration Loop Module

Handles the main orchestration loop with threading support.
"""

from BlazeAIoT.Core.Utils import BlazeLogger
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.orchestrator_utils import OrchestratorUtils
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
import time
import signal
import threading
from datetime import datetime


class OrchestrationLoop:
    """Manages the orchestration loop execution."""
    
    def __init__(self, config: dict, orchestrator, metrics_provider, services, nodes, layers, strategy=None):
        """
        Initialize Orchestration Loop.
        
        Args:
            config: Configuration dictionary
            orchestrator: Orchestrator instance (OrchestrationContext)
            metrics_provider: Metrics facade
            services: List of services
            nodes: List of nodes
            layers: List of layers
            strategy: Orchestration strategy to use
        """
        self.config = config
        self.orchestrator = orchestrator
        self.metrics_provider = metrics_provider
        self.services = services
        self.nodes = nodes
        self.layers = layers
        self.strategy = strategy
        
        self.running = False
        self.paused = False
        self._thread = None
        self._lock = threading.RLock()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused by default
        
        self.current_allocation = {}
        self.config_updater = None
        self.service_discovery_manager = None
        self.allocation_facade = None
        self.allocation_emitter = None
        self.simpy_env = None  # SimPy environment for simulator mode
        self.metrics_populated_once = False
        self.services_update_xml_path = None
        self.compare_strategies = {}
        self.compare_log_changes_only = self.config.get('orchestrator', {}).get('compare', {}).get('log_changes_only', True)
        self.compare_show_no_differences = self.config.get('orchestrator', {}).get('compare', {}).get('show_no_differences', False)
        
        # CSV logging
        self.csv_logger = None
        
        # Orchestration timing
        orchestration_rate = self.config.get('orchestrator', {}).get('rate_hz', 0.1)
        self.orchestration_interval_s = 1.0 / orchestration_rate
        
        # Track orchestrator results for CSV logging
        self.orchestrator_results = {}  # {orchestrator_name: {'allocation': ..., 'violations': ..., 'execution_time_ms': ...}}
        
        # Get active orchestrator type
        self.active_orchestrator_type = self.config.get('orchestrator', {}).get('active', 
                                                        self.config.get('orchestrator', {}).get('type', 'nsga2'))
        
        # Track previous allocations for change detection
        self.previous_allocations = {}
        
        # WiFi mobility controller (optional)
        self.wifi_controller = None
        self.wifi_enabled = False
        
        # Async metrics fetcher (runs independently of orchestration)
        self.async_metrics_fetcher = None
        
        # Initialize logger first (needed by other init methods)
        self._logger = BlazeLogger.get_instance()
        
        # Battery state tracking for battery-powered nodes
        self.battery_state = {}  # {node_id: remaining_wh}
        self._latest_battery_node_metrics = {}  # {node_id: {voltage_v, current_a, power_w}}
        self._initialize_battery_state()

        # Non-blocking allocation execution (one worker thread per allocation)
        allocation_workers = int(self.config.get('orchestrator', {}).get('allocation_workers', 16))
        self._allocation_executor = ThreadPoolExecutor(
            max_workers=max(1, allocation_workers),
            thread_name_prefix="AllocationWorker"
        )
        self._pending_allocation_futures: Dict[str, Future] = {}
        self._allocation_futures_lock = threading.RLock()
        
        self._logger.debug("OrchestrationLoop initialized")
    
    def _initialize_battery_state(self):
        """Initialize battery state for all battery-powered nodes."""
        if not hasattr(self, 'nodes') or not self.nodes:
            return
        
        for node in self.nodes:
            battery_powered = getattr(node, 'battery_powered', False)
            if battery_powered:
                node_id = getattr(node, 'node_id', None)
                battery_voltage = getattr(node, 'battery_voltage', 0.0)
                battery_capacity_ah = getattr(node, 'battery_capacity_ah', 0.0)
                
                if node_id and battery_voltage > 0 and battery_capacity_ah > 0:
                    # Calculate total energy in Wh (Watt-hours)
                    total_energy_wh = battery_voltage * battery_capacity_ah
                    self.battery_state[node_id] = total_energy_wh
                    self._logger.debug(f"Initialized battery for {node_id}: {total_energy_wh:.2f}Wh")

    def _collect_battery_node_metrics(self, iteration: int) -> Dict[str, Dict[str, Optional[float]]]:
        """Collect live battery metrics for all battery-powered nodes.

        Returns dict in normalized format:
        {node_id: {"voltage_v": float|None, "current_a": float|None, "power_w": float|None}}
        """
        metrics_by_node: Dict[str, Dict[str, Optional[float]]] = {}

        if not self.nodes or not self.metrics_provider:
            return metrics_by_node
        
        # Debug: Log battery nodes detection
        battery_nodes = [getattr(n, 'node_id', '?') for n in self.nodes if getattr(n, 'battery_powered', False)]
        if battery_nodes:
            self._logger.info(f"[{iteration}] Collecting battery metrics for nodes: {battery_nodes}")
        else:
            self._logger.debug(f"[{iteration}] No battery-powered nodes found in {len(self.nodes)} nodes")

        metrics_facade = getattr(self.metrics_provider, 'metrics_facade', None) or self.metrics_provider

        def _extract_scalar(results):
            if not results:
                return None
            try:
                value = results[0].get("value", [None, None])[1]
                return float(value) if value is not None else None
            except Exception:
                return None

        def _query_with_fallbacks(promql: str, preferred_layer_id: Optional[str]):
            if hasattr(metrics_facade, 'query'):
                try:
                    res = metrics_facade.query(promql) or []
                    if res:
                        return res
                except Exception:
                    pass

            if hasattr(metrics_facade, '_query_at_url'):
                cluster_urls = getattr(metrics_facade, 'cluster_urls', {}) or {}
                base_url = getattr(metrics_facade, 'base_url', None)
                ordered_urls = []

                if preferred_layer_id and cluster_urls.get(preferred_layer_id):
                    ordered_urls.append(cluster_urls.get(preferred_layer_id))
                if base_url:
                    ordered_urls.append(base_url)
                ordered_urls.extend(cluster_urls.values())

                seen = set()
                unique_urls = []
                for url in ordered_urls:
                    if not url:
                        continue
                    key = str(url).rstrip('/')
                    if key in seen:
                        continue
                    seen.add(key)
                    unique_urls.append(key)

                for url in unique_urls:
                    try:
                        res = metrics_facade._query_at_url(promql, url) or []
                        if res:
                            return res
                    except Exception:
                        continue

            return []

        def _resolve_metric_for_node(metric_name: str, target_node_id: str, preferred_layer_id: Optional[str]):
            label_selectors = [
                f'nodeName="{target_node_id}"',
                f'node="{target_node_id}"',
                f'nodename="{target_node_id}"',
                f'instance=~"^{target_node_id}(:.*)?$"',
            ]

            for selector in label_selectors:
                result = _query_with_fallbacks(f'max({metric_name}{{{selector}}})', preferred_layer_id)
                scalar = _extract_scalar(result)
                if scalar is not None:
                    return scalar

            vector = _query_with_fallbacks(metric_name, preferred_layer_id)
            for sample in vector:
                labels = sample.get("metric", {}) or {}
                if any(str(v) == target_node_id or target_node_id in str(v) for v in labels.values()):
                    scalar = _extract_scalar([sample])
                    if scalar is not None:
                        return scalar
            return None

        for node in self.nodes:
            node_id = getattr(node, "node_id", None)
            battery_powered = getattr(node, "battery_powered", False)
            if not node_id or not battery_powered:
                continue

            layer_id = getattr(node, "layer_id", None)
            try:
                voltage_v = _resolve_metric_for_node("blazeaiot_node_battery_voltage", node_id, layer_id)
                current_a = _resolve_metric_for_node("blazeaiot_node_battery_current", node_id, layer_id)
                power_w = (voltage_v * current_a) if voltage_v is not None and current_a is not None else None
                metrics_by_node[node_id] = {
                    "voltage_v": voltage_v,
                    "current_a": current_a,
                    "power_w": power_w,
                }
            except Exception as e:
                self._logger.warning(f"[{iteration}] Failed to fetch battery metrics for {node_id}: {e}")
                metrics_by_node[node_id] = {
                    "voltage_v": None,
                    "current_a": None,
                    "power_w": None,
                }

        return metrics_by_node

    def _apply_live_battery_inputs(self, battery_node_metrics: Dict[str, Dict[str, Optional[float]]], iteration: int) -> None:
        """Convert live battery telemetry to orchestrator input format.

        Orchestrators expect node-level fields like power_active/power_base and battery properties.
        This method updates those fields per battery-powered node using live V/I-derived power.
        """
        if not battery_node_metrics or not self.nodes:
            return

        try:
            orchestration_rate = self.config.get('orchestrator', {}).get('rate_hz', 0.1)
            interval_s = 1.0 / orchestration_rate if orchestration_rate > 0 else 10.0
        except Exception:
            interval_s = 10.0

        for node in self.nodes:
            node_id = getattr(node, "node_id", None)
            if not node_id:
                continue
            if not getattr(node, "battery_powered", False):
                continue

            node_metrics = battery_node_metrics.get(node_id, {})
            voltage_v = node_metrics.get("voltage_v")
            current_a = node_metrics.get("current_a")
            power_w = node_metrics.get("power_w")

            if voltage_v is not None:
                setattr(node, "battery_voltage", float(voltage_v))
            if current_a is not None:
                setattr(node, "battery_current_a", float(current_a))

            if power_w is not None and power_w > 0:
                setattr(node, "power_active", float(power_w))
                existing_base = getattr(node, "power_base", None)
                if existing_base is None:
                    setattr(node, "power_base", max(0.5, float(power_w) * 0.3))
                else:
                    try:
                        base_val = float(existing_base)
                        if base_val > float(power_w):
                            setattr(node, "power_base", max(0.5, float(power_w) * 0.3))
                    except Exception:
                        setattr(node, "power_base", max(0.5, float(power_w) * 0.3))

                # Keep internal remaining energy state in Wh aligned with live power draw
                cap_ah = getattr(node, "battery_capacity_ah", None)
                bat_v = getattr(node, "battery_voltage", None)
                if cap_ah and bat_v:
                    try:
                        total_wh = float(cap_ah) * float(bat_v)
                        prev_wh = self.battery_state.get(node_id, total_wh)
                        consumed_wh = float(power_w) * (interval_s / 3600.0)
                        remaining_wh = max(0.0, prev_wh - consumed_wh)
                        remaining_pct = (remaining_wh / total_wh * 100.0) if total_wh > 0 else 0.0
                        self.battery_state[node_id] = remaining_wh
                        
                        self._logger.info(
                            f"[{iteration}] Battery {node_id}: "
                            f"V={voltage_v:.2f}V, I={current_a:.2f}A, P={power_w:.2f}W, "
                            f"Remaining={remaining_wh:.2f}Wh ({remaining_pct:.1f}%), "
                            f"Consumed={consumed_wh*1000:.2f}mWh"
                        )
                    except Exception:
                        pass
                else:
                    self._logger.info(
                        f"[{iteration}] Battery inputs {node_id}: "
                        f"V={voltage_v}V I={current_a}A P={power_w}W "
                        f"power_active={getattr(node, 'power_active', None)}"
                    )
            else:
                # No live power metrics available, use existing power_active value
                self._logger.info(
                    f"[{iteration}] Battery inputs {node_id}: "
                    f"No live metrics (V={voltage_v}, I={current_a}, P={power_w}), "
                    f"using power_active={getattr(node, 'power_active', None)}W"
                )
    
    def run(self):
        """Main orchestration loop."""
        self.running = True
        orchestration_rate = self.config.get('orchestrator', {}).get('rate_hz', 0.1)
        interval = 1.0 / orchestration_rate
        
        env_mode = self.config.get('environment', {}).get('mode', 'kubernetes')
        max_iterations = self.config.get('orchestrator', {}).get('max_iterations')
        
        # Initialize WiFi if enabled
        self._initialize_wifi()
        
        # Initialize async metrics fetcher
        self._initialize_async_metrics_fetcher()
        
        print("\n" + "=" * 80)
        print(f"STARTING ORCHESTRATION LOOP (mode: {env_mode}, rate: {orchestration_rate} Hz)")
        if self.wifi_enabled:
            print("WiFi mobility simulation: ENABLED")
        if self.async_metrics_fetcher:
            print("Async metrics fetching: ENABLED")
        print("=" * 80)
        print("Press Ctrl+C to stop\n")
        
        iteration = 0
        
        try:
            while self.running:
                iteration += 1
                start_time = time.time()
                self._logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] Iteration {iteration}")
                
                # Check iteration limit
                if max_iterations and iteration > max_iterations:
                    self._logger.info(f"✓ Reached max iterations ({max_iterations})")
                    break
                
                # Run orchestration iteration (mode-agnostic)
                self._run_iteration(iteration)
                
                # Sleep until next iteration
                elapsed = time.time() - start_time
                sleep_time = interval - elapsed
                if sleep_time < 0:
                    self._logger.warning(f"Iteration took {elapsed:.2f}s (target: {interval:.2f}s)")
                    sleep_time = 0
                
                # Check pause state before sleeping
                self._pause_event.wait()  # Block if paused
                
                if self.running:  # Check if still running after pause
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\n\nReceived interrupt signal, shutting down...")
            self.running = False
        finally:
            # Clean up async fetcher
            if self.async_metrics_fetcher:
                self.async_metrics_fetcher.stop()

    
    def _run_iteration(self, iteration: int):
        """Execute one iteration of the orchestration loop.
        
        Updates metrics (simulator or Prometheus), runs orchestration,
        and executes allocations. Mode-agnostic via config_updater interface.
        """
        try:
            # Refresh current allocation from Kubernetes before orchestrating
            env_mode = self.config.get('environment', {}).get('mode', 'kubernetes')
            # No training-specific randomization in orchestration loop
            if env_mode == 'kubernetes':
                if self.service_discovery_manager:
                    try:
                        k8s_alloc = self.service_discovery_manager.discover_current_allocation()
                        refreshed = {}
                        if isinstance(k8s_alloc, dict):
                            for sid, pod_list in k8s_alloc.items():
                                if pod_list and isinstance(pod_list, list):
                                    # Choose the best pod: Ready list is already filtered; pick most recent start_time
                                    best = None
                                    best_start = None
                                    for entry in pod_list:
                                        start_ts = entry.get('start_time')
                                        # Prefer entries with a start_time; otherwise fall back to first encountered
                                        if start_ts is not None:
                                            if best_start is None or start_ts > best_start:
                                                best_start = start_ts
                                                best = entry
                                        elif best is None:
                                            best = entry
                                    target_entry = best or pod_list[0]
                                    node = target_entry.get('node')
                                    if node:
                                        refreshed[sid] = node
                        self.current_allocation = refreshed
                        self._logger.debug(
                            f"[{iteration}] Refreshed allocation from K8s (Ready pods): {self.current_allocation if self.current_allocation else 'none'}"
                        )
                    except Exception as e:
                        self._logger.debug(f"[{iteration}] Allocation refresh skipped: {e}")
                else:
                    self._logger.debug(f"[{iteration}] No service_discovery_manager; skipping K8s refresh")
            else:
                self._logger.debug(f"[{iteration}] Non-K8s env_mode={env_mode}; skipping K8s refresh")

            # Update services_update.xml placement nodes from current allocation (flexible/layer_lock only)
            self._sync_current_allocation_to_services_xml()

            # Step 1: Metrics are fetched asynchronously and passed into orchestrators
            metrics_health = None
            if self.async_metrics_fetcher:
                metrics_health = self.async_metrics_fetcher.get_health()
                snapshot_facade = self.async_metrics_fetcher.get_snapshot_facade()
                if snapshot_facade:
                    self._apply_metrics_snapshot(snapshot_facade)
                    self._logger.debug(
                        f"[{iteration}] Using async metrics (age: {metrics_health['last_fetch_age_seconds']:.2f}s)"
                    )
                else:
                    self._logger.warning(
                        f"[{iteration}] Async metrics not ready yet (error: {metrics_health['last_fetch_error']})"
                    )

            # Convert live battery telemetry (V/I) into orchestrator input fields for all battery nodes
            self._latest_battery_node_metrics = self._collect_battery_node_metrics(iteration)
            self._apply_live_battery_inputs(self._latest_battery_node_metrics, iteration)
            
            # Step 2: Run orchestration algorithm
            if self.orchestrator and self.strategy:
                try:
                    import time
                    self._logger.debug(f"[{iteration}] Running orchestration algorithm...")
                    self._logger.info(f"[{iteration}] Active orchestrator type: {type(self.strategy).__name__}")
                    self._logger.debug(f"[{iteration}] Services count: {len(self.services) if self.services else 0}")
                    self._logger.debug(f"[{iteration}] Current allocation: {self.current_allocation}")
                    
                    # Time the active orchestrator execution
                    active_start = time.time()
                    active_hysteresis = self._get_hysteresis_threshold(self.active_orchestrator_type)
                    # Pass only optimizable services (excludes pure-daemonset)
                    optimizable_services = self._get_optimizable_services()
                    proposed_allocation = self.orchestrator.optimize(
                        strategy=self.strategy,
                        current_assignment=self.current_allocation,
                        hysteresis_threshold=active_hysteresis,
                        services=optimizable_services if optimizable_services else self.services
                    )
                    active_execution_time = time.time() - active_start
                    # Store execution time for CSV logging
                    self._last_execution_time_ms = active_execution_time * 1000  # Convert to milliseconds
                    self._logger.debug(f"[{iteration}] Proposed allocation: {proposed_allocation}")
                    
                    # Restore current allocations for non-optimizable services (e.g., pure-daemonset)
                    proposed_allocation = self._restore_non_optimizable_allocations(proposed_allocation, self.current_allocation)
                    
                    # Handle empty allocation (e.g., first iteration with no metrics)
                    if not proposed_allocation:
                        self._logger.warning(f"[{iteration}] Orchestrator returned empty allocation (metrics may not be ready)")
                        # Use current allocation as fallback to ensure CSV logging works
                        proposed_allocation = self.current_allocation.copy() if self.current_allocation else {}
                    
                    # Calculate violations for active orchestrator
                    active_violations = []
                    if proposed_allocation:
                        try:
                            active_violations, _ = self.orchestrator._assignment_score(proposed_allocation, orchestrator_name=self.active_orchestrator_type)
                        except Exception as e:
                            self._logger.debug(f"[{iteration}] Failed to score active allocation: {e}")
                    
                    # Store active orchestrator results for CSV logging
                    self.orchestrator_results = {
                        self.active_orchestrator_type: {
                            'allocation': proposed_allocation,
                            'violations': active_violations,
                            'execution_time_ms': self._last_execution_time_ms
                        }
                    }
                    self._logger.debug(f"[{iteration}] Stored orchestrator results for {self.active_orchestrator_type}")

                    # Run comparison strategies (side-by-side) without applying changes
                    if self.compare_strategies:
                        self._run_comparison_strategies(
                            iteration=iteration,
                            main_allocation=proposed_allocation,
                            active_execution_time=active_execution_time
                        )
                    
                    if proposed_allocation:
                        self._logger.info(f"[{iteration}] Orchestrator proposed {len(proposed_allocation)} placements")
                        
                        # Step 3: Execute allocation changes
                        if self.allocation_facade:
                            try:
                                changes = self._execute_allocation_changes(proposed_allocation, iteration)
                                
                                # Update current allocation
                                self.current_allocation = proposed_allocation.copy()
                                
                                self._logger.info(
                                    f"[{iteration}] Allocation dispatch: {changes.get('scheduled', 0)} scheduled, "
                                    f"{changes.get('in_flight', 0)} in-flight, "
                                    f"{changes.get('succeeded', 0)} completed, {changes.get('failed', 0)} failed"
                                )
                            except Exception as e:
                                self._logger.error(f"Allocation execution failed: {e}", exc_info=True)
                    else:
                        self._logger.debug(f"[{iteration}] No allocation changes needed")
                        
                except Exception as e:
                    self._logger.error(f"Orchestration algorithm failed: {e}", exc_info=True)
            elif not self.strategy:
                self._logger.warning(f"[{iteration}] No orchestration strategy configured")
            else:
                self._logger.warning(f"[{iteration}] Orchestrator not initialized")
            
            # Step 0.5: Run services through SimPy simulator if available (before metrics update)
            if env_mode == 'simulator' and self.simpy_env:
                self._run_services_in_simulator(iteration)
            
            # Step 0.75: Update WiFi quality if enabled
            if self.wifi_enabled and self.wifi_controller:
                self._update_wifi_quality(iteration)
            
            # Step 0.8: Log mobile/battery node power status
            if env_mode == 'simulator':
                self._log_mobile_power_status(iteration)
            
            # Step 0.9: Log metrics to CSV
            self._logger.info(f"[{iteration}] Starting CSV logging...")
            self._log_iteration_to_csv(iteration)
            self._logger.info(f"[{iteration}] CSV logging complete")
                
        except Exception as e:
            self._logger.error(f"Iteration {iteration} failed: {e}", exc_info=True)

    def _run_services_in_simulator(self, iteration: int) -> None:
        """Generate realistic metrics for services based on historical ranges."""
        import random
        import xml.etree.ElementTree as ET
        
        try:
            if not self.services_update_xml_path:
                return
            
            # Parse the services_update.xml to generate new metrics
            try:
                tree = ET.parse(self.services_update_xml_path)
                root = tree.getroot()
            except Exception:
                return
            
            # Generate new metrics for all containers
            updated = False
            for service_elem in root.findall('service'):
                instances = service_elem.find('instances')
                if instances is None:
                    continue
                
                for container_elem in instances.findall('container'):
                    # Read historical min/max
                    cpu_min_str = container_elem.get('cpu_usage_min')
                    cpu_max_str = container_elem.get('cpu_usage_max')
                    ram_min_str = container_elem.get('ram_usage_min_gb')
                    ram_max_str = container_elem.get('ram_usage_max_gb')
                    
                    try:
                        if cpu_min_str and cpu_max_str:
                            cpu_min = float(cpu_min_str)
                            cpu_max = float(cpu_max_str)
                            # Generate new value within range with ±10% jitter
                            new_cpu = random.uniform(cpu_min, cpu_max)
                            container_elem.set('cpu_usage_last', f"{new_cpu:.3f}")
                            updated = True
                        
                        if ram_min_str and ram_max_str:
                            ram_min = float(ram_min_str)
                            ram_max = float(ram_max_str)
                            # Generate new value within range with ±10% jitter
                            new_ram = random.uniform(ram_min, ram_max)
                            container_elem.set('ram_usage_last_gb', f"{new_ram:.3f}")
                            updated = True
                    except (ValueError, TypeError):
                        continue
            
            # Write updated metrics back
            if updated:
                self._indent_xml(root)
                tree.write(self.services_update_xml_path, encoding='utf-8', xml_declaration=True)
        
        except Exception as e:
            self._logger.debug(f"[{iteration}] Failed to generate service metrics: {e}")

    def _run_comparison_strategies(self, iteration: int, main_allocation: Optional[Dict[str, str]], 
                                   active_execution_time: float = 0.0) -> None:
        """Run comparison strategies and log allocations in service-centric table format."""
        import time
        
        if not main_allocation:
            return
        
        # Get active orchestrator type
        active_type = self.config.get('orchestrator', {}).get('active', self.config.get('orchestrator', {}).get('type', 'nsga2'))
        
        # Collect allocations and execution times: {strategy_name: {service_id: node_id}}
        allocations = {'current': self.current_allocation}
        execution_times = {}  # Track execution times for each orchestrator
        
        # Record the active orchestrator's allocation and execution time
        allocations[active_type] = main_allocation
        execution_times[active_type] = active_execution_time
        self._logger.debug(f"[{iteration}] Active orchestrator ({active_type}) proposed: {main_allocation}")
        
        # Calculate violations for active orchestrator
        active_violations, active_energy = self.orchestrator._assignment_score(main_allocation, orchestrator_name=active_type)
        violations_by_algo = {active_type: active_violations}
        
        for compare_type, strategy in self.compare_strategies.items():
            # Skip if this is the active orchestrator (already have its result)
            if compare_type == active_type:
                self._logger.debug(f"[{iteration}] Skipping {compare_type} (is active orchestrator)")
                continue
                
            try:
                self._logger.debug(f"[{iteration}] Running comparison orchestrator: {compare_type}")
                compare_start = time.time()
                context = getattr(strategy, 'context', None) or self.orchestrator
                # Get per-orchestrator hysteresis threshold for comparison orchestrator
                compare_hysteresis = self._get_hysteresis_threshold(compare_type)
                # Pass only optimizable services (excludes pure-daemonset)
                optimizable_services = self._get_optimizable_services()
                compare_allocation = context.optimize(
                    strategy=strategy,
                    current_assignment=self.current_allocation,
                    hysteresis_threshold=compare_hysteresis,
                    services=optimizable_services if optimizable_services else self.services
                )
                execution_times[compare_type] = time.time() - compare_start
                self._logger.debug(f"[{iteration}] {compare_type} proposed: {compare_allocation}")
                
                # Restore current allocations for non-optimizable services
                compare_allocation = self._restore_non_optimizable_allocations(compare_allocation, self.current_allocation)

                if compare_allocation:
                    allocations[compare_type] = compare_allocation
                    # Calculate violations for this comparison orchestrator
                    comp_violations, comp_energy = self.orchestrator._assignment_score(compare_allocation, orchestrator_name=compare_type)
                    violations_by_algo[compare_type] = comp_violations
                    self._logger.debug(f"[{iteration}] {compare_type} violations: {comp_violations}, energy: {comp_energy:.2f}")
                    
                    # Store in orchestrator_results for CSV logging
                    self.orchestrator_results[compare_type] = {
                        'allocation': compare_allocation,
                        'violations': comp_violations,
                        'execution_time_ms': execution_times[compare_type] * 1000
                    }
                else:
                    allocations[compare_type] = {}
            except Exception as e:
                execution_times[compare_type] = time.time() - compare_start
                error_msg = str(e)
                self._logger.error(f"[{iteration}] Orchestrator {compare_type} failed: {error_msg}")
                # Count orchestrator failures as violations
                violations_by_algo[compare_type] = 999  # Mark as failed
                allocations[compare_type] = {'_error': error_msg}

        # Build service-centric table
        all_services = set(main_allocation.keys())
        service_by_id = {
            str(getattr(svc, 'service_id', '')): svc
            for svc in (self.services or [])
        }
        table_rows = []
        
        # Debug: log which orchestrators are being compared
        self._logger.debug(f"[{iteration}] Active orchestrator: {active_type}")
        self._logger.debug(f"[{iteration}] Comparison strategies: {list(self.compare_strategies.keys())}")
        self._logger.debug(f"[{iteration}] Allocations collected: {list(allocations.keys())}")

        def should_show_in_comparison(svc) -> bool:
            """Exclude daemonset or locked services from comparison table."""
            if svc is None:
                return True

            # Exclude daemonset services (including multi-layer daemonsets)
            if getattr(svc, 'is_daemonset', False):
                return False

            # Exclude node-locked or layer-locked services
            if getattr(svc, 'locked_node_id', None) is not None:
                return False
            if getattr(svc, 'locked_layer_id', None) is not None:
                return False

            return True

        for service_id in sorted(all_services):
            svc = service_by_id.get(service_id)
            if not should_show_in_comparison(svc):
                continue
            row_data = {'service': service_id}
            row_data['current'] = self.current_allocation.get(service_id, '-')
            
            # Build columns dynamically based on what orchestrators are available
            for strategy_name in sorted(allocations.keys()):
                if strategy_name == 'current':
                    continue  # Already added
                    
                alloc = allocations.get(strategy_name, {})
                if '_error' in alloc:
                    node_name = f"ERROR: {alloc['_error']}"
                else:
                    node_name = alloc.get(service_id, '-')
                    
                    # Mark changed nodes with "-" prefix/suffix
                    if strategy_name in self.previous_allocations:
                        prev_node = self.previous_allocations[strategy_name].get(service_id, '-')
                        if node_name != '-' and node_name != prev_node:
                            node_name = f"-{node_name}-"
                
                row_data[strategy_name] = node_name
            
            table_rows.append(row_data)
        
        # Store current allocations for next iteration
        self.previous_allocations = allocations.copy()

        # Format and log table
        col_names = ['service', 'current'] + sorted([k for k in allocations.keys() if k != 'current'])
        col_widths = {}
        
        for col in col_names:
            col_widths[col] = max(
                len(col),
                max((len(str(row.get(col, '-'))) for row in table_rows), default=0)
            )

        def fmt_row(row_dict):
            return " | ".join(
                str(row_dict.get(col, '-')).ljust(col_widths[col]) for col in col_names
            )

        divider = "-+-".join("-" * col_widths[col] for col in col_names)

        self._logger.info(f"[{iteration}] Orchestrator comparison table:")
        self._logger.info(fmt_row({col: col for col in col_names}))
        self._logger.info(divider)
        for row in table_rows:
            self._logger.info(fmt_row(row))
        
        # Log execution times
        if execution_times:
            self._logger.info(f"[{iteration}] Execution times:")
            for algo_name in sorted(execution_times.keys()):
                elapsed = execution_times[algo_name]
                marker = " (active)" if algo_name == active_type else ""
                self._logger.info(f"  {algo_name:15s}: {elapsed*1000:7.2f} ms{marker}")
        
        # Log violations for each orchestrator
        if violations_by_algo:
            self._logger.info(f"[{iteration}] Constraint violations:")
            for algo_name in sorted(violations_by_algo.keys()):
                num_violations = violations_by_algo[algo_name]
                marker = " (active)" if algo_name == active_type else ""
                violation_status = "✓ PASS" if num_violations == 0 else f"✗ FAIL ({num_violations})"
                self._logger.info(f"  {algo_name:15s}: {violation_status}{marker}")

            # Log concise violation details only for algorithms that failed
            latency_matrix = getattr(self.orchestrator, "latency_matrix", None)
            expected_service_ids = [
                str(getattr(svc, 'service_id', ''))
                for svc in (self.services or [])
                if not getattr(svc, 'exclude_from_orchestration', False)
                and not getattr(svc, 'is_daemonset', False)
            ]
            for algo_name in sorted(violations_by_algo.keys()):
                if violations_by_algo[algo_name] == 0:
                    continue
                placement = allocations.get(algo_name) or {}
                if not placement:
                    self._logger.info(f"[{iteration}] {algo_name} produced no placement for comparison.")
                    if expected_service_ids:
                        self._logger.info(
                            f"[{iteration}] {algo_name} missing {len(expected_service_ids)} services: "
                            f"{', '.join(expected_service_ids[:5])}"
                        )
                    continue
                utils = OrchestratorUtils(self.services, self.nodes, latency_matrix)
                details = utils.validate_placement(placement, log_warnings=False)
                if details:
                    self._logger.info(f"[{iteration}] {algo_name} violation details:")
                    for item in details[:5]:
                        self._logger.info(f"  - {item}")
                else:
                    missing = [sid for sid in expected_service_ids if sid not in placement]
                    if missing:
                        self._logger.info(
                            f"[{iteration}] {algo_name} missing {len(missing)} services: "
                            f"{', '.join(missing[:5])}"
                        )

    def _initialize_wifi(self) -> None:
        """Initialize WiFi mobility controller if enabled in config."""
        env_config = self.config.get('environment', {})
        if env_config.get('mode') != 'simulator':
            return
        
        simulator_config = env_config.get('simulator', {})
        wifi_config = simulator_config.get('wifi', {})
        
        if not wifi_config.get('enabled'):
            return
        
        try:
            from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Utils.wifi_mobility_controller import WiFiMobilityController
            
            self.wifi_controller = WiFiMobilityController(logger=self._logger)
            self.wifi_enabled = True
            
            # Register access points
            for ap_config in wifi_config.get('access_points', []):
                host_id = ap_config['host_id']
                pos = ap_config['position']
                radius = ap_config['coverage_radius']
                self.wifi_controller.register_access_point(host_id, pos[0], pos[1], radius)
                self._logger.info(f"WiFi AP registered: {host_id} at ({pos[0]}, {pos[1]}) radius={radius}m")
            
            # Register robot
            robot_config = wifi_config.get('robot', {})
            if robot_config:
                robot_id = robot_config['host_id']
                pos = robot_config['initial_position']
                vel = robot_config.get('velocity', [0.0, 0.0])
                self.wifi_controller.register_robot(robot_id, pos[0], pos[1], vel[0], vel[1])
                self._logger.info(f"WiFi robot registered: {robot_id} at ({pos[0]}, {pos[1]}) vel=({vel[0]}, {vel[1]}) m/s")
                
                # Register WiFi link for robot
                self.wifi_controller.register_wifi_link(
                    link_id=f"wifi-{robot_id}",
                    robot_id=robot_id
                )
            
            # Set update interval
            update_interval = wifi_config.get('update_interval', 1.0)
            self.wifi_controller.set_update_interval(update_interval)
            
            # Pass WiFi controller to metrics provider so latencies are WiFi-aware
            if hasattr(self.metrics_provider, '_wifi_controller'):
                self.metrics_provider._wifi_controller = self.wifi_controller
            
            # Force initial WiFi status update
            initial_updates = self.wifi_controller.update(0.0)
            if initial_updates:
                for link_id, (snr_level, ap_id, signal_strength) in initial_updates.items():
                    snr_names = ["Good", "Medium", "Poor"]
                    self._logger.info(
                        f"WiFi {link_id} initial: SNR={snr_names[snr_level]} "
                        f"AP={ap_id} signal={signal_strength:.1%}"
                    )
            
            self._logger.info("WiFi mobility simulation initialized")
            
        except Exception as e:
            self._logger.warning(f"Failed to initialize WiFi simulation: {e}")
            self.wifi_enabled = False
            self.wifi_controller = None

    def _initialize_async_metrics_fetcher(self) -> None:
        """Initialize async metrics fetcher that runs independently of orchestration."""
        if not self.config_updater:
            self._logger.debug("No config_updater; skipping async metrics fetcher")
            return
        
        try:
            from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Metrics.async_metrics_fetcher import AsyncMetricsFetcher
            
            metrics_rate_hz = self.config.get('metrics', {}).get('fetch_rate_hz', 1.0)
            
            self.async_metrics_fetcher = AsyncMetricsFetcher(
                config_updater=self.config_updater,
                metrics_provider=self.metrics_provider,
                nodes=self.nodes,
                services=self.services,
                services_xml_path=self.services_update_xml_path,
                namespace=self.config.get('services', {}).get('namespace', 'default')
            )
            
            # Start fetcher in background thread
            self.async_metrics_fetcher.start(fetch_interval_hz=metrics_rate_hz)
            
            # Wait for first fetch to ensure metrics are available
            wait_timeout = self.config.get('metrics', {}).get('initial_fetch_timeout_s', 30.0)
            if self.async_metrics_fetcher.wait_for_first_fetch(timeout=wait_timeout):
                self._logger.info(f"✓ Async metrics fetcher started (rate: {metrics_rate_hz:.1f} Hz)")
            else:
                self._logger.warning(
                    f"First metrics fetch timed out after {wait_timeout}s; "
                    f"orchestrators will use fallback node properties"
                )
            
        except Exception as e:
            self._logger.warning(f"Failed to initialize async metrics fetcher: {e}")
            self.async_metrics_fetcher = None

    def _apply_metrics_snapshot(self, snapshot_facade: Any) -> None:
        """Apply metrics snapshot to orchestration contexts and strategies."""
        if not snapshot_facade:
            return

        if self.orchestrator:
            self.orchestrator.metrics = snapshot_facade

        if self.strategy and hasattr(self.strategy, 'context'):
            self.strategy.context.metrics = snapshot_facade

        for strategy in self.compare_strategies.values():
            context = getattr(strategy, 'context', None)
            if context:
                context.metrics = snapshot_facade
    
    def _get_hysteresis_threshold(self, orchestrator_type: str) -> float:
        """Get hysteresis threshold for a specific orchestrator type."""
        strategies_config = self.config.get('orchestrator', {}).get('strategies', {})
        if orchestrator_type in strategies_config:
            threshold = strategies_config[orchestrator_type].get('hysteresis_threshold')
            if threshold is not None:
                return threshold
        return self.config.get('orchestrator', {}).get('hysteresis_threshold', 0.0)
    
    def _get_optimizable_services(self) -> List[Any]:
        """Get list of services that should be included in orchestration optimization.
        
        Excludes:
        - Services with multi-layer placement where ALL layers are daemonset
        - Services marked with exclude_from_orchestration flag
        
        Includes:
        - Services with flexible placement
        - Services with multi-layer where at least one layer is not daemonset
        - Services with node_lock (locked in place)
        """
        optimizable = []
        for svc in self.services:
            # Check exclude_from_orchestration flag
            if getattr(svc, 'exclude_from_orchestration', False):
                continue
            
            placement = getattr(svc, 'placement', None)
            if placement is None:
                optimizable.append(svc)
                continue
            
            placement_type = getattr(placement, 'type', None)
            
            # Skip if it's a multi-layer with ALL layers as daemonset
            if placement_type == 'multi-layer':
                layer_rules = getattr(placement, 'layer_rules', {})
                if layer_rules:
                    # Check if all layers are daemonset
                    all_daemonset = all(
                        getattr(layer, 'type', None) == 'daemonset' 
                        for layer in layer_rules.values()
                    )
                    if all_daemonset:
                        continue  # Skip this service
            
            optimizable.append(svc)
        
        return optimizable
    
    def _restore_non_optimizable_allocations(self, proposal: Dict[str, str], 
                                            current: Dict[str, str]) -> Dict[str, str]:
        """Restore allocations for non-optimizable services to their current values.
        
        Services with pure-daemonset placement shouldn't be reallocated.
        """
        result = proposal.copy() if proposal else {}
        
        # Get service IDs that are pure-daemonset
        pure_daemonset_ids = self._get_pure_daemonset_service_ids()
        
        # Restore current allocation for non-optimizable services
        for service_id, node_id in current.items():
            if service_id in pure_daemonset_ids:
                result[service_id] = node_id
        
        return result
    
    def _get_pure_daemonset_service_ids(self) -> set:
        """Get IDs of services that shouldn't be optimized.
        
        This includes:
        - Services with multi-layer where ALL layers are daemonset
        - Services marked with exclude_from_orchestration flag
        
        These services shouldn't be reallocated as they either run on all nodes in their layers
        (daemonset) or are manually managed (exclude_from_orchestration).
        """
        non_optimizable_ids = set()
        
        # Try to detect from services list
        for svc in self.services:
            service_id = str(getattr(svc, 'service_id', ''))
            if not service_id:
                continue
            
            # Check exclude_from_orchestration flag
            if getattr(svc, 'exclude_from_orchestration', False):
                non_optimizable_ids.add(service_id)
                continue
            
            placement = getattr(svc, 'placement', None)
            if placement is None:
                continue
            
            placement_type = getattr(placement, 'type', None)
            if placement_type != 'multi-layer':
                continue
            
            layer_rules = getattr(placement, 'layer_rules', {})
            if not layer_rules:
                continue
            
            # Check if all layers are daemonset
            all_daemonset = all(
                getattr(layer, 'type', None) == 'daemonset' 
                for layer in layer_rules.values()
            )
            
            if all_daemonset:
                non_optimizable_ids.add(service_id)
        
        # Fallback: hardcode known daemonset services if detection failed
        if not non_optimizable_ids:
            non_optimizable_ids = {'blazemonitoring'}
        
        return non_optimizable_ids
    
    def _update_wifi_quality(self, iteration: int) -> None:
        """Update WiFi link quality based on robot movement."""
        if not self.wifi_controller:
            return
        
        try:
            # Simulation time based on iteration and rate
            orchestration_rate = self.config.get('orchestrator', {}).get('rate_hz', 0.1)
            interval = 1.0 / orchestration_rate
            simulation_time = iteration * interval
            
            self._logger.debug(f"[{iteration}] WiFi update: sim_time={simulation_time:.1f}s")
            
            # Update WiFi controller
            updates = self.wifi_controller.update(simulation_time)
            
            if updates:
                # Get robot positions for logging
                robot_config = self.config.get('environment', {}).get('simulator', {}).get('wifi', {}).get('robot', {})
                robot_id = robot_config.get('host_id')
                robot_status = self.wifi_controller.get_robot_status(robot_id) if robot_id else None
                
                for link_id, (snr_level, ap_id, signal_strength) in updates.items():
                    snr_names = ["Good", "Medium", "Poor"]
                    pos_str = ""
                    if robot_status:
                        pos = robot_status['position']
                        pos_str = f" pos=({pos['x']:.1f}, {pos['y']:.1f})"
                    
                    self._logger.info(
                        f"[{iteration}] WiFi {link_id}: SNR={snr_names[snr_level]} "
                        f"AP={ap_id} signal={signal_strength:.1%}{pos_str}"
                    )
            else:
                self._logger.debug(f"[{iteration}] WiFi: No updates this iteration")
        
        except Exception as e:
            self._logger.debug(f"[{iteration}] WiFi update failed: {e}")

    def _log_mobile_power_status(self, iteration: int) -> None:
        """Log power consumption and battery status for mobile/battery-powered nodes.
        
        Power model for Jetson Orin AGX + Motors:
        - Jetson base (idle): 5W
        - Jetson active (max CPU): 50W  
        - Motors: 0-24W depending on movement (WiFi quality proxy)
        - Variation: ±10% random noise on each component
        """
        import random
        import os
        import xml.etree.ElementTree as ET
        import statistics
        
        if not hasattr(self, 'nodes') or not self.nodes:
            return
        
        try:
            def _load_idle_baseline() -> dict:
                if hasattr(self, '_idle_baseline_cache') and self._idle_baseline_cache is not None:
                    return self._idle_baseline_cache

                cpu_by_host = {}
                ram_by_host = {}
                cpu_vals = []
                ram_vals = []

                candidate_paths = []
                metrics_path = getattr(self.metrics_provider, 'platform_xml_path', None) if self.metrics_provider else None
                if metrics_path:
                    candidate_paths.append(metrics_path)
                candidate_paths.append(
                    os.path.abspath(
                        os.path.join(os.path.dirname(__file__), '../../../../config/platform.xml')
                    )
                )

                platform_xml = next((p for p in candidate_paths if p and os.path.exists(p)), None)
                if platform_xml:
                    try:
                        tree = ET.parse(platform_xml)
                        root = tree.getroot()
                        for host in root.findall('.//host'):
                            host_id = host.get('id') or ''
                            cpu_prop = host.find(".//prop[@id='debug.cpu_used']")
                            ram_prop = host.find(".//prop[@id='debug.ram_used_gb']")
                            if cpu_prop is not None:
                                try:
                                    cpu_val = float(cpu_prop.get('value', '0') or 0)
                                    cpu_by_host[host_id] = cpu_val
                                    cpu_vals.append(cpu_val)
                                except Exception:
                                    pass
                            if ram_prop is not None:
                                try:
                                    ram_val = float(ram_prop.get('value', '0') or 0)
                                    ram_by_host[host_id] = ram_val
                                    ram_vals.append(ram_val)
                                except Exception:
                                    pass
                    except Exception:
                        pass

                cpu_avg = statistics.mean(cpu_vals) if cpu_vals else 0.1
                ram_avg = statistics.mean(ram_vals) if ram_vals else 0.2
                self._idle_baseline_cache = {
                    'cpu_by_host': cpu_by_host,
                    'ram_by_host': ram_by_host,
                    'cpu_avg': cpu_avg,
                    'ram_avg': ram_avg,
                }
                return self._idle_baseline_cache

            # Get mobile/battery nodes
            mobile_nodes = [
                n for n in self.nodes
                if getattr(n, 'battery_powered', False)
                or str(getattr(n, 'layer_id', '')).lower() == 'mobile'
                or str(getattr(n, 'node_id', '')).startswith('frank-')
            ]
            
            if not mobile_nodes:
                return
            
            # Calculate time since last iteration (for energy consumption)
            orchestration_rate = self.config.get('orchestrator', {}).get('rate_hz', 0.1)
            interval = 1.0 / orchestration_rate
            
            # Log power status for each mobile node
            for node in mobile_nodes:
                node_id = getattr(node, 'node_id', 'unknown')
                battery_powered = getattr(node, 'battery_powered', False)
                
                if not battery_powered:
                    continue  # Skip non-battery nodes
                
                # Get power specs for Jetson Orin AGX + Motors
                jetson_base = float(getattr(node, 'power.jetson.base', 5.0))
                jetson_active = float(getattr(node, 'power.jetson.active', 50.0))
                motor_max = float(getattr(node, 'power.motor.max', 24.0))
                battery_voltage = getattr(node, 'battery_voltage', None)
                battery_capacity_ah = getattr(node, 'battery_capacity_ah', None)
                
                if battery_voltage and battery_capacity_ah:
                    total_energy_wh = battery_voltage * battery_capacity_ah
                    remaining_wh = self.battery_state.get(node_id, total_energy_wh)
                    remaining_pct = (remaining_wh / total_energy_wh) * 100.0 if total_energy_wh > 0 else 0.0
                    
                    # === JETSON POWER CALCULATION ===
                    # Calculate actual power consumption based on services running on this node
                    services_on_node = []
                    total_cpu_demand = 0.0
                    services_cpu = 0.0

                    # Add idle baseline (from platform.xml) with slight variation
                    baseline = _load_idle_baseline()
                    idle_jitter = random.uniform(0.95, 1.05)
                    idle_cpu = float(baseline['cpu_by_host'].get(node_id, baseline['cpu_avg'])) * idle_jitter
                    total_cpu_demand += idle_cpu
                    
                    for svc in self.services:
                        svc_id = str(getattr(svc, 'service_id', ''))
                        
                        # Check if service is allocated to this node
                        allocated_node = self.current_allocation.get(svc_id)
                        
                        # Also check for locked services (node-lock, daemonset, layer-lock)
                        locked_node_id = getattr(svc, 'locked_node_id', None)
                        locked_layer_id = getattr(svc, 'locked_layer_id', None)
                        is_daemonset = getattr(svc, 'is_daemonset', False)
                        node_layer = getattr(node, 'layer_id', None)
                        
                        is_on_node = False
                        
                        # Service is on this node if:
                        # 1. Explicitly allocated to this node
                        if allocated_node == node_id:
                            is_on_node = True
                        # 2. Node-locked to this specific node
                        elif locked_node_id and str(locked_node_id) == node_id:
                            is_on_node = True
                        # 3. Daemonset running on this node's layer
                        elif is_daemonset and locked_layer_id and str(locked_layer_id) == node_layer:
                            is_on_node = True
                        # 4. Layer-locked to this node's layer (but not daemonset)
                        elif not is_daemonset and locked_layer_id and str(locked_layer_id) == node_layer and allocated_node == node_id:
                            is_on_node = True
                        
                        if is_on_node:
                            services_on_node.append(svc_id)

                            # Prefer measured usage (matches platform_update.xml debug.cpu_used)
                            measured_cpu = None
                            if hasattr(svc, 'get_measured_usage'):
                                try:
                                    usage = svc.get_measured_usage() or {}
                                    measured_cpu = 0.0
                                    for _, container_usage in usage.items():
                                        if isinstance(container_usage, dict):
                                            measured_cpu += float(container_usage.get('cpu_usage_last', 0.0))
                                except Exception:
                                    measured_cpu = None

                            if measured_cpu is not None:
                                services_cpu += float(measured_cpu)
                            else:
                                # Fallback to container specs / observed maxima
                                containers = getattr(svc, 'containers', [])
                                for container in containers:
                                    cpu_usage_last = getattr(container, 'cpu_usage_last', None)
                                    cpu_usage_max = getattr(container, 'cpu_usage_max', None)
                                    if cpu_usage_last is not None:
                                        services_cpu += float(cpu_usage_last)
                                    elif cpu_usage_max is not None:
                                        services_cpu += float(cpu_usage_max)
                                    else:
                                        services_cpu += float(getattr(container, 'cpu', 0.0))

                    total_cpu_demand += services_cpu
                    
                    # Calculate Jetson power based on CPU utilization
                    node_capacity = float(getattr(node, 'cpu', 1.0))
                    if hasattr(node, 'metrics') and node.metrics:
                        try:
                            capacity = node.metrics.get_capacity()
                            node_capacity = float(capacity.get('cpu', node_capacity))
                        except Exception:
                            pass
                    cpu_utilization = min(1.0, total_cpu_demand / node_capacity) if node_capacity > 0 else 0.0
                    
                    # Linear scaling: jetson_power = base + (active - base) * utilization
                    jetson_power = jetson_base + (jetson_active - jetson_base) * cpu_utilization
                    
                    # === MOTOR POWER CALCULATION ===
                    # Motor power depends on movement/WiFi quality
                    # Use WiFi signal quality as proxy for mobility intensity
                    motor_power = 0.0
                    wifi_quality = "Unknown"
                    
                    if hasattr(self, 'wifi_controller') and self.wifi_controller:
                        try:
                            # Get robot status to determine movement
                            robot_config = self.config.get('environment', {}).get('simulator', {}).get('wifi', {}).get('robot', {})
                            robot_id = robot_config.get('host_id')
                            if robot_id:
                                robot_status = self.wifi_controller.get_robot_status(robot_id)
                                if robot_status:
                                    # SNR level: 0=Good, 1=Medium, 2=Poor
                                    # Map to motor usage: Good=High movement (need motors), Medium=Medium, Poor=Low
                                    snr_level = robot_status.get('snr_level', 0)
                                    if snr_level == 0:  # Good signal = high movement
                                        motor_power = motor_max * 0.9
                                        wifi_quality = "Good"
                                    elif snr_level == 1:  # Medium signal = medium movement
                                        motor_power = motor_max * 0.5
                                        wifi_quality = "Medium"
                                    else:  # Poor signal = low movement
                                        motor_power = motor_max * 0.1
                                        wifi_quality = "Poor"
                        except Exception:
                            motor_power = motor_max * 0.3  # Default to medium
                    else:
                        # No WiFi controller, use random motor activity
                        motor_power = motor_max * random.uniform(0.1, 0.8)
                    
                    # === ADD VARIATION ===
                    # Add ±10% random noise to both Jetson and Motor power
                    jetson_variation = jetson_power * random.uniform(-0.1, 0.1)
                    motor_variation = motor_power * random.uniform(-0.1, 0.1)
                    
                    jetson_power_actual = jetson_power + jetson_variation
                    motor_power_actual = motor_power + motor_variation
                    
                    # Total power consumption
                    total_power = jetson_power_actual + motor_power_actual
                    
                    # Calculate energy consumed in this interval (Wh = W * hours)
                    energy_consumed_wh = total_power * (interval / 3600.0)
                    
                    # Update battery state
                    remaining_wh = max(0.0, remaining_wh - energy_consumed_wh)
                    self.battery_state[node_id] = remaining_wh
                    
                    self._logger.info(
                        f"[{iteration}] Battery {node_id}: "
                        f"Jetson CPU={cpu_utilization*100:.0f}% ({jetson_power_actual:.1f}W), "
                        f"Motor={wifi_quality} ({motor_power_actual:.1f}W), "
                        f"Total={total_power:.1f}W, "
                        f"Remaining={remaining_wh:.2f}Wh ({remaining_pct:.1f}%), "
                        f"Consumed={energy_consumed_wh*1000:.2f}mWh"
                    )
                else:
                    self._logger.info(
                        f"[{iteration}] Battery {node_id}: No battery specs available"
                    )
                
        except Exception as e:
            self._logger.debug(f"[{iteration}] Mobile power logging failed: {e}")

    
    def _is_service_orchestrable(self, service_id: str) -> bool:
        """
        Check if a service can be orchestrated (is not purely daemonset).
        
        Args:
            service_id: Service ID to check
            
        Returns:
            True if service can be orchestrated, False if it's a pure daemonset
        """
        # Check if service has daemonset-only placement
        # Services with ONLY daemonset type layers cannot be moved
        
        # Find service in XML if available
        if self.services_update_xml_path:
            try:
                import xml.etree.ElementTree as ET
                # ALWAYS read from base services.xml (source of truth), never from _update.xml
                base_path = self.services_update_xml_path.replace('_update.xml', '.xml')
                tree = ET.parse(base_path)
                root = tree.getroot()
                
                for svc_elem in root.findall('.//service'):
                    if svc_elem.get('id') != service_id:
                        continue
                    
                    placement = svc_elem.find('placement')
                    if placement is None:
                        # No placement = flexible, can be orchestrated
                        return True
                    
                    # Check if multi-layer with layer-rules
                    layer_rules = placement.find('layer-rules')
                    if layer_rules is not None:
                        # Check if ALL layers are daemonset or node_lock
                        all_fixed = True
                        has_layers = False
                        
                        for layer_rule in layer_rules.findall('layer'):
                            has_layers = True
                            rule_type = layer_rule.get('type', 'flexible')
                            # Both daemonset and node_lock mean fixed placement (not orchestratable)
                            if rule_type not in ('daemonset', 'node_lock', 'node-lock'):
                                all_fixed = False
                                break
                        
                        if has_layers and all_fixed:
                            # All layers have fixed placement (daemonset or node_lock), cannot orchestrate
                            self._logger.debug(f"[{service_id}] Skipping - all layers have fixed placement")
                            return False
                    
                    # Has placement but no multi-layer rules, likely orchestrable
                    return True
                    
            except Exception as e:
                self._logger.debug(f"Error checking service type: {e}")
        
        # Default: assume orchestrable
        return True
    
    def _execute_allocation_changes(self, proposed_allocation: dict, iteration: int) -> dict:
        """
        Execute the proposed allocation changes.
        
        Args:
            proposed_allocation: Dict of {service_id: target_node_id}
            iteration: Current iteration number
            
        Returns:
            Dict with 'succeeded' and 'failed' counts
        """
        completed = self._collect_completed_allocation_futures(iteration)
        changes = {
            'succeeded': completed.get('succeeded', 0),
            'failed': completed.get('failed', 0),
            'scheduled': 0,
            'in_flight': 0
        }

        if not self.config.get('orchestrator', {}).get('apply_changes', True):
            self._logger.info(f"[{iteration}] apply_changes=false; skipping allocation execution")
            return changes
        
        if not self.allocation_facade:
            self._logger.warning("Allocation facade not available")
            return changes
        
        for service_id, target_node_id in proposed_allocation.items():
            try:
                # Skip if service is a pure daemonset (cannot be orchestrated)
                if not self._is_service_orchestrable(service_id):
                    self._logger.debug(f"[{iteration}] Skipping {service_id} - pure daemonset service")
                    continue
                
                # Find current location
                source_node_id = self.current_allocation.get(service_id)
                
                # Find node objects for layer information
                # Support both .id and .node_id attributes
                def get_node_id(node):
                    return getattr(node, 'id', None) or getattr(node, 'node_id', None)

                def get_node_layer(node):
                    if not node:
                        return 'default'
                    return (
                        getattr(node, 'layer_id', None)
                        or getattr(node, 'layer', None)
                        or 'default'
                    )
                
                target_node = next((n for n in self.nodes if get_node_id(n) == target_node_id), None) if self.nodes else None
                source_node = next((n for n in self.nodes if get_node_id(n) == source_node_id), None) if self.nodes and source_node_id else None
                
                target_layer = get_node_layer(target_node)
                source_layer = get_node_layer(source_node)
                
                self._logger.debug(f"[{iteration}] {service_id}: source={source_node_id} ({source_layer}), target={target_node_id} ({target_layer})")
                
                # Emit allocation summary (even if no move needed)
                if self.allocation_emitter and hasattr(self.allocation_emitter, 'emit_allocation_summary'):
                    eligible_nodes = [get_node_id(n) for n in self.nodes] if self.nodes else []
                    reason = "no change" if source_node_id == target_node_id else "orchestrator decision"
                    self.allocation_emitter.emit_allocation_summary(
                        service_id=service_id,
                        original_node=source_node_id,
                        recommended_node=target_node_id,
                        eligible_nodes=eligible_nodes,
                        reason=reason
                    )
                
                # Skip if already on target node
                if source_node_id == target_node_id:
                    self._logger.debug(f"[{iteration}] {service_id} already on {target_node_id}, skipping")
                    continue
                
                # Skip if an allocation for this service is already in progress
                with self._allocation_futures_lock:
                    pending_future = self._pending_allocation_futures.get(service_id)
                    if pending_future and not pending_future.done():
                        self._logger.debug(f"[{iteration}] {service_id} allocation already in-flight, skipping resubmit")
                        continue

                # Find the service object to check for multiple deployments
                service = next((s for s in self.services if s.service_id == service_id), None) if self.services else None
                deployment_names = []
                
                # If service has multiple deployments (e.g., wifi, ethernet variants)
                if service and hasattr(service, 'get_deployment_names'):
                    deployment_names = service.get_deployment_names()
                
                if not deployment_names:
                    # Check for layer-specific deployment variants (edge uses -eth0, fog/cloud uses base)
                    # Services with split deployments: robot-slam, robot-localization, emergency-stop-service
                    split_deployment_services = {'robot-slam', 'robot-localization', 'emergency-stop-service'}
                    
                    if service_id in split_deployment_services:
                        if target_layer == 'edge':
                            # Edge layer uses -eth0 variant
                            deployment_names = [f"{service_id}-eth0"]
                        else:
                            # Fog/cloud layers use base name (wifi variant)
                            deployment_names = [service_id]
                    else:
                        # Single deployment (default case)
                        deployment_names = [service_id]
                
                # Also determine source deployment name for proper scale-down
                source_deployment_names = []
                if source_layer:
                    split_deployment_services = {'robot-slam', 'robot-localization', 'emergency-stop-service'}
                    if service_id in split_deployment_services:
                        if source_layer == 'edge':
                            source_deployment_names = [f"{service_id}-eth0"]
                        else:
                            source_deployment_names = [service_id]
                    else:
                        source_deployment_names = [service_id]
                
                namespace = self.config.get('services', {}).get('namespace', 'default')

                # Dispatch asynchronously (non-blocking): one thread per allocation
                future = self._allocation_executor.submit(
                    self._run_service_allocation,
                    service_id,
                    target_node_id,
                    target_layer,
                    source_node_id,
                    source_layer,
                    namespace,
                    deployment_names,
                    source_deployment_names,
                    iteration
                )
                with self._allocation_futures_lock:
                    self._pending_allocation_futures[service_id] = future
                changes['scheduled'] += 1
                self._logger.debug(f"[{iteration}] Scheduled async allocation for {service_id} → {target_node_id}")
                    
            except Exception as e:
                changes['failed'] += 1
                self._logger.error(f"Error allocating {service_id}: {e}", exc_info=True)

        with self._allocation_futures_lock:
            changes['in_flight'] = sum(1 for fut in self._pending_allocation_futures.values() if not fut.done())
        
        return changes

    def _run_service_allocation(
        self,
        service_id: str,
        target_node_id: str,
        target_layer: str,
        source_node_id: Optional[str],
        source_layer: Optional[str],
        namespace: str,
        deployment_names: List[str],
        source_deployment_names: List[str],
        iteration: int
    ) -> Tuple[bool, Optional[str], str]:
        """Worker that executes one service allocation (possibly multiple deployments)."""
        all_succeeded = True
        last_reason = None

        for i, deployment_name in enumerate(deployment_names):
            try:
                # Get corresponding source deployment name (same index or last if fewer)
                source_deployment_name = (
                    source_deployment_names[min(i, len(source_deployment_names) - 1)]
                    if source_deployment_names else None
                )
                
                result = self.allocation_facade.allocate(
                    service_id=service_id,
                    target_node=target_node_id,
                    target_layer=target_layer,
                    source_node=source_node_id,
                    source_layer=source_layer,
                    namespace=namespace,
                    deployment_name=deployment_name,
                    source_deployment_name=source_deployment_name
                )

                if isinstance(result, tuple):
                    success, reason = result
                else:
                    success, reason = result, None

                if success:
                    self._logger.info(
                        f"[{iteration}] ✓ Allocated {service_id} deployment '{deployment_name}' → {target_node_id}"
                    )
                else:
                    all_succeeded = False
                    last_reason = reason
                    failure_msg = (
                        f"[{iteration}] ✗ Failed to allocate {service_id} deployment "
                        f"'{deployment_name}' to {target_node_id}"
                    )
                    if reason:
                        failure_msg += f" - Reason: {reason}"
                    self._logger.warning(failure_msg)

            except Exception as e:
                all_succeeded = False
                last_reason = str(e)
                self._logger.error(
                    f"Error allocating {service_id} deployment '{deployment_name}': {e}",
                    exc_info=True
                )

        return all_succeeded, last_reason, target_node_id

    def _collect_completed_allocation_futures(self, iteration: int) -> Dict[str, int]:
        """Collect completed async allocations without blocking."""
        summary = {'succeeded': 0, 'failed': 0}

        with self._allocation_futures_lock:
            completed_items = [
                (service_id, fut)
                for service_id, fut in self._pending_allocation_futures.items()
                if fut.done()
            ]

        for service_id, fut in completed_items:
            try:
                success, reason, target_node_id = fut.result()
                if success:
                    summary['succeeded'] += 1
                    self.current_allocation[service_id] = target_node_id
                    self._sync_current_allocation_to_services_xml()
                else:
                    summary['failed'] += 1
                    if reason:
                        self._logger.warning(f"[{iteration}] Async allocation failed for {service_id}: {reason}")
            except Exception as e:
                summary['failed'] += 1
                self._logger.error(f"[{iteration}] Async allocation future failed for {service_id}: {e}", exc_info=True)
            finally:
                with self._allocation_futures_lock:
                    existing = self._pending_allocation_futures.get(service_id)
                    if existing is fut:
                        self._pending_allocation_futures.pop(service_id, None)

        return summary

    def _sync_current_allocation_to_services_xml(self) -> None:
        """Persist current allocation into services_update.xml for flexible/layer_lock placements."""
        if not self.services_update_xml_path or not self.current_allocation:
            return

        try:
            import os
            import shutil
            import xml.etree.ElementTree as ET

            update_path = self.services_update_xml_path
            if not os.path.exists(update_path):
                base_path = update_path.replace('_update.xml', '.xml')
                if os.path.exists(base_path):
                    shutil.copy2(base_path, update_path)
                else:
                    return

            tree = ET.parse(update_path)
            root = tree.getroot()
            updated = 0

            for svc in root.findall('.//service'):
                sid = svc.get('id') or svc.get('service_id')
                if not sid:
                    continue
                node = self.current_allocation.get(sid)
                if not node:
                    continue

                placement = svc.find('placement')
                if placement is None:
                    continue

                ptype = placement.get('type', '').lower()
                if ptype in ('flexible', 'layer_lock', 'layer-lock'):
                    if placement.get('node') != node:
                        placement.set('node', node)
                        updated += 1

            if updated > 0:
                tree.write(update_path, encoding='utf-8', xml_declaration=True)
        except Exception as e:
            self._logger.debug(f"Failed to sync current allocation to services XML: {e}")
    
    def start_threaded(self) -> bool:
        """
        Start the orchestration loop in a background thread.
        
        Returns:
            bool: True if thread started successfully, False if already running
        """
        with self._lock:
            if self.running or (self._thread and self._thread.is_alive()):
                self._logger.warning("Orchestration loop already running")
                return False
            
            self.running = True
            self.paused = False
            self._pause_event.set()
            self._thread = threading.Thread(target=self.run, name="BlazeMaestro", daemon=True)
            self._thread.start()
            self._logger.info("Orchestration loop started in background thread")
            return True
    
    def stop_threaded(self, timeout: float = 10.0) -> bool:
        """
        Stop the orchestration loop and wait for thread to complete.
        
        Args:
            timeout: Maximum seconds to wait for thread to stop
            
        Returns:
            bool: True if stopped successfully, False if timeout occurred
        """
        with self._lock:
            if not self.running:
                self._logger.info("Orchestration loop not running")
                return True
            
            self.running = False
            # Unpause if paused to allow thread to exit
            if self.paused:
                self._pause_event.set()
            
            thread = self._thread
        
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                self._logger.warning(f"Thread did not stop within {timeout}s timeout")
                return False

        # Stop allocation worker pool
        try:
            self._allocation_executor.shutdown(wait=False, cancel_futures=False)
        except Exception:
            pass
        
        self._logger.info("Orchestration loop stopped")
        return True
    
    def pause(self) -> bool:
        """
        Pause the orchestration loop.
        
        Returns:
            bool: True if paused successfully, False if not running
        """
        with self._lock:
            if not self.running:
                self._logger.warning("Cannot pause: orchestration loop not running")
                return False
            
            if self.paused:
                self._logger.info("Orchestration loop already paused")
                return True
            
            self.paused = True
            self._pause_event.clear()
            self._logger.info("Orchestration loop paused")
            return True
    
    def resume(self) -> bool:
        """
        Resume the orchestration loop after pausing.
        
        Returns:
            bool: True if resumed successfully, False if not paused or not running
        """
        with self._lock:
            if not self.running:
                self._logger.warning("Cannot resume: orchestration loop not running")
                return False
            
            if not self.paused:
                self._logger.info("Orchestration loop not paused")
                return True
            
            self.paused = False
            self._pause_event.set()
            self._logger.info("Orchestration loop resumed")
            return True
    
    def is_running(self) -> bool:
        """Check if orchestration loop is running."""
        with self._lock:
            return self.running and (self._thread is None or self._thread.is_alive())
    
    def is_paused(self) -> bool:
        """Check if orchestration loop is paused."""
        with self._lock:
            return self.paused
    
    def _indent_xml(self, elem, level=0):
        """Add pretty-print indentation to XML element tree."""
        indent = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for child in elem:
                self._indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent
    
    def _log_iteration_to_csv(self, iteration: int) -> None:
        """Log current iteration metrics to CSV for all orchestrators.
        
        Captures allocation, constraint status, execution time, and latencies
        for the active orchestrator and any comparison orchestrators.
        """
        if not self.csv_logger:
            self._logger.warning(f"[{iteration}] No CSV logger available")
            return
        
        self._logger.info(f"[{iteration}] CSV logging: checking orchestrator results")
        
        try:
            # Get latency matrix (convert to seconds if needed, logger expects seconds)
            latency_matrix = []
            if hasattr(self.metrics_provider, 'get_latency_matrix'):
                try:
                    latency_matrix = self.metrics_provider.get_latency_matrix(self.nodes or [])
                except Exception as e:
                    self._logger.debug(f"[{iteration}] Failed to get latency matrix: {e}")
                    latency_matrix = []
            elif hasattr(self.metrics_provider, 'latency_matrix'):
                latency_matrix = self.metrics_provider.latency_matrix
            
            # Export latency matrix on first iteration
            if iteration == 1 and latency_matrix is not None and len(latency_matrix) > 0 and self.nodes:
                self.csv_logger.export_latency_matrix(
                    nodes=self.nodes,
                    latency_matrix=latency_matrix,
                    frank2_node_id="frank-2"
                )
                self._logger.debug(f"[{iteration}] Exported latency matrix to CSV")

            # Reuse battery metrics already collected before orchestration.
            battery_node_metrics = self._latest_battery_node_metrics or self._collect_battery_node_metrics(iteration)
            
            # Debug logging for orchestrator results
            if not self.orchestrator_results:
                self._logger.warning(f"[{iteration}] No orchestrator results to log to CSV")
                return
            
            self._logger.info(f"[{iteration}] Logging {len(self.orchestrator_results)} orchestrator(s) to CSV: {list(self.orchestrator_results.keys())}")
            
            # Log each orchestrator's results
            for orchestrator_name, result in self.orchestrator_results.items():
                allocation = result.get('allocation', {})
                violations = result.get('violations', [])
                execution_time_ms = result.get('execution_time_ms', 0.0)
                
                # Get previous allocation for this specific orchestrator
                prev_key = f"{orchestrator_name}_{iteration - 1}"
                previous_allocation = self.previous_allocations.get(prev_key) if iteration > 1 else None
                
                # Handle violations being either int (count) or list (details)
                violations_count = violations if isinstance(violations, int) else len(violations)
                self._logger.info(f"[{iteration}] Writing CSV row for {orchestrator_name} (allocation_size={len(allocation)}, violations={violations_count})")
                
                # Log to CSV
                is_active_orchestrator = orchestrator_name == self.active_orchestrator_type
                self._logger.debug(f"[{iteration}] {orchestrator_name}: is_active={is_active_orchestrator} (active_type={self.active_orchestrator_type})")
                self.csv_logger.log_iteration(
                    orchestrator_name=orchestrator_name,
                    iteration=iteration,
                    assignments=allocation,
                    previous_assignments=previous_allocation,
                    services=self.services or [],
                    nodes=self.nodes or [],
                    latency_matrix=latency_matrix,
                    violations=violations,
                    execution_time_ms=execution_time_ms,
                    frank2_node_id="frank-2",
                    include_battery_metrics=is_active_orchestrator,
                    battery_metrics=battery_node_metrics if is_active_orchestrator else None,
                    battery_state=self.battery_state if is_active_orchestrator else None,
                    battery_interval_s=self.orchestration_interval_s
                )
                
                self._logger.info(f"[{iteration}] Successfully wrote CSV row for {orchestrator_name}")
                
                # Store current allocation for next iteration's "previous"
                curr_key = f"{orchestrator_name}_{iteration}"
                self.previous_allocations[curr_key] = allocation.copy() if allocation else {}
            
        except Exception as e:
            self._logger.error(f"[{iteration}] CSV logging failed: {e}", exc_info=True)
    
    def stop(self):
        """Legacy stop method for compatibility."""
        self.running = False
        try:
            self._allocation_executor.shutdown(wait=False, cancel_futures=False)
        except Exception:
            pass
