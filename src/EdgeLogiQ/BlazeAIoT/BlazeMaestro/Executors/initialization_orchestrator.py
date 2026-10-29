"""
Initialization Orchestrator Module

Implements Single Responsibility Principle (SRP) for initialization coordination.
Coordinates the startup sequence of all BlazeMaestro components.
"""

import os
from typing import Dict, Any
from BlazeAIoT.Core.Utils import BlazeLogger


class InitializationOrchestrator:
    """
    Orchestrates the initialization sequence of BlazeMaestro components.
    
    Responsibilities:
    - Coordinate initialization order
    - Initialize platform (Kubernetes/Simulator)
    - Initialize services discovery
    - Initialize metrics providers
    - Initialize orchestrator
    - Handle initialization errors gracefully
    
    Follows Template Method pattern for initialization sequence.
    """
    
    def __init__(
        self,
        config_loader,
        platform_manager,
        service_discovery_manager,
        xml_manager,
        csv_output_dir=None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            config_loader: ConfigLoader instance
            platform_manager: PlatformManager instance
            service_discovery_manager: ServiceDiscoveryManager instance
            xml_manager: XMLManager instance
            csv_output_dir: Optional output directory for CSV files
        """
        self._logger = BlazeLogger.get_instance()
        self.config_loader = config_loader
        self.platform_manager = platform_manager
        self.service_discovery_manager = service_discovery_manager
        self.xml_manager = xml_manager
        self.csv_output_dir = csv_output_dir
        
        # State containers
        self.layers = None
        self.nodes = None
        self.services = None
        self.topology_manager = None
        self.service_manager = None
        self.metrics_provider = None
        self.orchestrator = None
        self.allocation_facade = None
        self.config_updater = None
        self.compare_strategies = {}
        self.compare_log_changes_only = True
        
        # Tracking
        self.current_allocation = {}
        self.current_pod_refs = {}
        self.locked_services = set()
        self.metrics_populated_once = False
    
    def initialize_all(self) -> bool:
        """
        Execute complete initialization sequence.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("=" * 80)
            print("BLAZE MAESTRO INITIALIZATION")
            print("=" * 80)
            
            # Step 1: Initialize platform
            if not self._init_platform():
                return False
            
            # Step 2: Discover services
            if not self._init_services():
                return False
            
            # Step 3: Get initial allocation
            if not self._init_allocation():
                return False
            
            # Step 4: Initialize metrics
            if not self._init_metrics():
                return False
            
            # Step 5: Initialize orchestrator
            if not self._init_orchestrator():
                return False
            
            print("\n" + "=" * 80)
            print("✓ INITIALIZATION COMPLETE")
            print("=" * 80)
            
            return True
        
        except Exception as e:
            self._logger.error(f"Initialization failed: {e}", exc_info=True)
            return False
    
    def _init_platform(self) -> bool:
        """
        Initialize platform (Kubernetes or Simulator).
        
        Returns:
            True if successful
        """
        print("\n[1/5] Initializing platform...")
        
        try:
            self.layers, self.nodes = self.platform_manager.initialize_platform()
            self.topology_manager = self.platform_manager.topology_manager
            
            # Update service discovery manager with discovered topology
            self.service_discovery_manager.set_topology(self.nodes, self.layers)
            
            # Copy simpy_env for simulator mode if available
            if hasattr(self.platform_manager, 'simpy_env'):
                self.simpy_env = self.platform_manager.simpy_env
            if hasattr(self.platform_manager, 'platform_xml_path'):
                self.platform_xml_path = self.platform_manager.platform_xml_path
            
            print(f"  ✓ Platform initialized: {len(self.layers)} layers, {len(self.nodes)} nodes")
            return True
        
        except Exception as e:
            self._logger.error(f"Platform initialization failed: {e}", exc_info=True)
            return False
    
    def _init_services(self) -> bool:
        """
        Discover and load services.
        
        Returns:
            True if successful
        """
        print("\n[2/5] Discovering services...")
        
        try:
            simpy_env = getattr(self, 'simpy_env', None)
            self.services = self.service_discovery_manager.discover_services(simpy_env)
            
            # Sync state from manager
            self.service_manager = self.service_discovery_manager.service_manager
            self.current_allocation = self.service_discovery_manager.current_allocation
            self.current_pod_refs = self.service_discovery_manager.current_pod_refs
            self.locked_services = self.service_discovery_manager.locked_services
            
            print(f"  ✓ Discovered {len(self.services)} services")
            return True
        
        except Exception as e:
            self._logger.error(f"Service discovery failed: {e}", exc_info=True)
            return False
    
    def _init_allocation(self) -> bool:
        """
        Get initial service allocation (Kubernetes mode only).
        
        Returns:
            True if successful
        """
        print("\n[3/5] Getting initial service allocation...")
        
        env_mode = self.config_loader.get_environment_mode()
        
        if env_mode == 'simulator':
            print("  Skipping K8s allocation discovery (simulator mode)")
            return True
        
        try:
            # Get allocation from Kubernetes
            k8s_alloc = self.service_discovery_manager.discover_current_allocation()
            
            # Map to orchestrator-friendly format
            if isinstance(k8s_alloc, dict) and k8s_alloc:
                for sid, pod_list in k8s_alloc.items():
                    if pod_list and isinstance(pod_list, list):
                        first_pod_info = pod_list[0]
                        if first_pod_info.get('node'):
                            self.current_allocation[sid] = first_pod_info['node']
                        if first_pod_info.get('pod'):
                            self.current_pod_refs[sid] = first_pod_info['pod']
            
            print(f"  ✓ Found {len(k8s_alloc)} services currently allocated")
            
            # Update services XML with K8s allocation
            services_xml_path = self.service_discovery_manager.get_services_xml_path()
            self.xml_manager.update_with_allocation(services_xml_path, k8s_alloc)
            
            return True
        
        except Exception as e:
            self._logger.warning(f"Could not get initial allocation: {e}")
            return True  # Non-fatal
    
    def _init_metrics(self) -> bool:
        """
        Initialize metrics provider (Prometheus or Simulator).
        
        Returns:
            True if successful
        """
        print("\n[4/5] Initializing metrics provider...")
        
        try:
            from ..Metrics.simulator_facade import SimulatorMetricsFacade
            
            env_mode = self.config_loader.get_environment_mode()
            metrics_source = self.config_loader.get_metrics_config().get('source')
            
            # Default based on environment mode if not explicitly set
            if not metrics_source:
                metrics_source = 'simulator' if env_mode == 'simulator' else 'prometheus'
            
            if metrics_source == 'simulator':
                print(f"  Using Simulator metrics (environment mode: {env_mode})")
                # Build latency matrix from platform.xml when available (with jitter)
                latency_matrix = None
                simulator_cfg = self.config_loader.get_environment_config().get('simulator', {})
                platform_xml = getattr(self, 'platform_xml_path', None) or simulator_cfg.get('platform_xml')
                jitter_cfg = simulator_cfg.get('network_jitter', {})
                enable_jitter = jitter_cfg.get('enabled', True)
                jitter_percent = float(jitter_cfg.get('jitter_percent', 10.0))
                enable_packet_loss = bool(jitter_cfg.get('enable_packet_loss', False))
                packet_loss_rate = float(jitter_cfg.get('packet_loss_rate', 0.01))
                seed = jitter_cfg.get('seed')

                if platform_xml:
                    try:
                        from ..Simulator.util.network_util import NetworkUtil
                        latency_matrix = NetworkUtil.build_latency_matrix_with_jitter(
                            platform_xml,
                            self.nodes,
                            enable_jitter=enable_jitter,
                            jitter_percent=jitter_percent,
                            enable_packet_loss=enable_packet_loss,
                            packet_loss_rate=packet_loss_rate,
                            seed=seed,
                        )
                    except Exception as e:
                        self._logger.warning(f"Failed to parse latency matrix from {platform_xml}: {e}")

                self.metrics_provider = SimulatorMetricsFacade(self.nodes, latency_matrix=latency_matrix)
                for node in self.nodes:
                    node.metrics.set_facade(self.metrics_provider)
                from ..ConfigUpdater.simulator_xml_updater import SimulatorXMLUpdater
                self.config_updater = SimulatorXMLUpdater(
                    metrics_facade=self.metrics_provider,
                    nodes=self.nodes,
                    services=self.services,
                    current_allocation=self.current_allocation,
                    output_dir=self.csv_output_dir
                )
                # Ensure platform _update.xml exists in simulator mode
                platform_xml = getattr(self, 'platform_xml_path', None)
                if not platform_xml:
                    simulator_cfg = self.config_loader.get_environment_config().get('simulator', {})
                    platform_xml = simulator_cfg.get('platform_xml')
                if platform_xml:
                    update_path = platform_xml.replace('.xml', '_update.xml')
                    if os.path.exists(platform_xml) and not os.path.exists(update_path):
                        import shutil
                        shutil.copy2(platform_xml, update_path)
                        print(f"  ✓ Created {os.path.basename(update_path)} from source")
                print("  ✓ Config updater ready (simulator)")
            else:
                print(f"  Using Prometheus metrics (environment mode: {env_mode})")
                self.metrics_provider = self._init_prometheus_metrics()
            
            print("  ✓ Metrics provider initialized")
            return True
        
        except Exception as e:
            self._logger.error(f"Metrics initialization failed: {e}", exc_info=True)
            return False
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics provider."""
        from ..Metrics.prometheus_facade import PrometheusMetricsFacade
        from ..ConfigUpdater.prometheus_xml_updater import PrometheusXMLUpdater
        
        cluster_configs = self.config_loader.get_clusters_config()
        
        if not cluster_configs:
            raise ValueError("No cluster configurations found for Prometheus")
        
        # Build node selectors
        node_selectors = {}
        for node in self.nodes:
            if hasattr(node, 'k8s_node_name'):
                node_selectors[node.node_id] = f'node="{node.k8s_node_name}"'
            else:
                node_selectors[node.node_id] = f'node="{node.node_id}"'

        # Build optional static node properties (battery characteristics)
        node_properties = {}

        # Load platform.xml capacities as fallback for Prometheus (cpu/ram/gpu)
        try:
            import xml.etree.ElementTree as ET
            env_cfg = self.config_loader.get_environment_config()
            platform_xml = env_cfg.get('simulator', {}).get('platform_xml')
            if platform_xml and os.path.exists(platform_xml):
                tree = ET.parse(platform_xml)
                root = tree.getroot()
                for host in root.findall('.//host'):
                    host_id = host.get('id')
                    if not host_id:
                        continue
                    props = node_properties.setdefault(host_id, {})
                    # CPU capacity from core count
                    core_str = host.get('core')
                    if core_str:
                        try:
                            props['cpu'] = float(core_str)
                        except Exception:
                            pass
                    # RAM capacity from ram.GB
                    ram_prop = host.find("./prop[@id='ram.GB']")
                    if ram_prop is not None:
                        ram_val = ram_prop.get('value')
                        if ram_val is not None:
                            try:
                                props['ram_gb'] = float(ram_val)
                            except Exception:
                                pass
                    # GPU capacity from gpu.memGB (optionally times gpu.count)
                    gpu_mem = None
                    gpu_mem_prop = host.find("./prop[@id='gpu.memGB']")
                    if gpu_mem_prop is not None:
                        gpu_mem_val = gpu_mem_prop.get('value')
                        if gpu_mem_val is not None:
                            try:
                                gpu_mem = float(gpu_mem_val)
                            except Exception:
                                gpu_mem = None
                    gpu_count = None
                    gpu_count_prop = host.find("./prop[@id='gpu.count']")
                    if gpu_count_prop is not None:
                        gpu_count_val = gpu_count_prop.get('value')
                        if gpu_count_val is not None:
                            try:
                                gpu_count = float(gpu_count_val)
                            except Exception:
                                gpu_count = None
                    if gpu_mem is not None:
                        props['gpu_gb'] = gpu_mem * (gpu_count if gpu_count else 1.0)
        except Exception:
            # Best-effort fallback; keep Prometheus as primary
            pass
        for node in self.nodes:
            props = {}
            # Static capacity from topology (used if Prometheus returns 0)
            cpu = getattr(node, 'cpu', None)
            ram = getattr(node, 'ram', None)
            gpu = getattr(node, 'gpu', None)
            if cpu is not None:
                try:
                    props['cpu'] = float(getattr(cpu, 'capacity', cpu))
                except Exception:
                    pass
            if ram is not None:
                try:
                    props['ram_gb'] = float(getattr(ram, 'capacity', ram))
                except Exception:
                    pass
            if gpu is not None:
                try:
                    props['gpu_gb'] = float(getattr(gpu, 'capacity', gpu))
                except Exception:
                    pass
            for key in ("battery_capacity_ah", "battery_voltage", "battery_powered"):
                val = getattr(node, key, None)
                if val is not None:
                    props[key] = val
            if props:
                node_properties.setdefault(node.node_id, {}).update(props)
        
        # Get Prometheus URL from first cluster
        configured_url = cluster_configs[0].get('prometheus_url', 'http://localhost:9090')
        verify_ssl = configured_url.startswith('https') and not any(
            x in configured_url for x in ['localhost', '127.0.0.1', '10.', '172.', '192.']
        )
        
        # Create facade
        metrics_provider = PrometheusMetricsFacade(
            base_url=configured_url,
            token='',
            node_selectors=node_selectors,
            verify_ssl=verify_ssl,
            cluster_configs=cluster_configs,
            node_properties=node_properties
        )
        
        print("  Setting up Prometheus access for all layers...")
        
        # Check if running inside a Kubernetes pod (in-cluster)
        in_cluster = os.path.exists('/var/run/secrets/kubernetes.io/serviceaccount/token')
        print(f"  Running in-cluster: {in_cluster}")
        
        # Set up access for each cluster
        cluster_urls = {}
        for i, cluster in enumerate(cluster_configs):
            cluster_name = cluster.get('name', cluster['layer_id'])
            layer_id = cluster.get('layer_id', f'layer-{i}')
            print(f"    [{layer_id.upper()}] Setting up {cluster_name}...")
            
            # Check if this cluster uses external kubeconfig (multi-cluster setup)
            has_external_kubeconfig = cluster.get('kubeconfig_path') and cluster.get('kubeconfig_path') != 'null'
            
            if in_cluster and not has_external_kubeconfig:
                # Same cluster - use Kubernetes service URL for in-cluster access
                cluster_namespace = cluster.get('prometheus_namespace', 'cattle-monitoring-system')
                url = f"http://rancher-monitoring-prometheus.{cluster_namespace}:9090"
                print(f"      Using in-cluster service URL: {url}")
                cluster_urls[layer_id] = url
            else:
                # External cluster or running outside - need port-forward
                configured_url = cluster.get('prometheus_url')
                if configured_url:
                    cluster_urls[layer_id] = configured_url
                    print(f"      Will set up port-forward to: {configured_url}")
                    # Trigger port-forward setup
                    actual_url = metrics_provider.setup_prometheus_access(cluster_name)
                    if actual_url:
                        cluster_urls[layer_id] = actual_url
                        print(f"      ✓ Port-forward established: {actual_url}")
                    else:
                        print(f"      ⚠ Port-forward failed, using configured URL")
                else:
                    print(f"      ✗ ERROR: No prometheus_url configured for {cluster_name}")
                    raise ValueError(f"Missing prometheus_url for cluster {cluster_name}")
        
        print(f"  Primary URL: {metrics_provider.base_url}")
        if len(cluster_urls) > 1:
            print(f"  Layer URLs: {cluster_urls}")
        print(f"  Mapped {len(node_selectors)} nodes")
        
        # Initialize config updater if enabled
        if self.config_loader.get('update_configs_from_prometheus', False):
            print("\n  Initializing config updater...")
            first_cluster = cluster_configs[0]
            self.config_updater = PrometheusXMLUpdater(
                metrics_facade=metrics_provider,
                kube_context=first_cluster.get('context'),
                kubeconfig_path=first_cluster.get('kubeconfig_path'),
                cluster_urls=cluster_urls,
                output_dir=self.csv_output_dir
            )
            print("  ✓ Config updater ready")
            if len(cluster_urls) > 1:
                print(f"  Multi-layer routing enabled: {list(cluster_urls.keys())}")
            
            # Update configuration files from Prometheus (write to *_update.xml only)
            print("\n  Updating configuration files from Prometheus...")
            try:
                # Get paths - XMLManager uses methods, not attributes
                platform_xml = str(self.xml_manager.base_path / 'platform.xml')
                services_xml = str(self.xml_manager.base_path / 'services.xml')
                # Use configured Kubernetes namespace for services
                services_cfg = self.config_loader.get_services_config()
                namespace = services_cfg.get('namespace', 'default')
                
                # Update configs (creates platform_update.xml with all discovered nodes/layers)
                self.config_updater.update_all_configs(
                    platform_xml=platform_xml,
                    services_xml=services_xml,
                    namespace=namespace
                )
                print("  ✓ Configuration update files generated (_update.xml)")
            except Exception as e:
                import traceback
                self._logger.warning(f"Failed to update configs from Prometheus: {e}")
                self._logger.warning(f"Traceback: {traceback.format_exc()}")
        return metrics_provider
    
    def _init_orchestrator(self) -> bool:
        """
        Initialize orchestrator.
        
        Returns:
            True if successful
        """
        print("\n[5/5] Initializing orchestrator...")
        
        try:
            from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.strategy_adapters import create_strategy
            from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Orchestrator.orchestration_strategy import OrchestrationContext
            
            orchestrator_config = self.config_loader.get_orchestrator_config()
            
            # Get active orchestrator type (new config: orchestrator.active)
            orchestrator_type = orchestrator_config.get('active', orchestrator_config.get('type', 'nsga2'))
            
            # Get per-orchestrator configurations
            strategies_config = orchestrator_config.get('strategies', {})
            active_strategy_config = strategies_config.get(orchestrator_type, {})
            
            # Merge active orchestrator config with base config
            merged_config = {**orchestrator_config, **active_strategy_config}
            
            # Create orchestration context
            self.orchestration_context = OrchestrationContext(
                services=self.services,
                nodes=self.nodes,
                metrics=self.metrics_provider,
                config=merged_config
            )
            
            # Create strategy with merged config
            self.strategy = create_strategy(
                orchestrator_type,
                self.orchestration_context,
                merged_config
            )

            # Optional comparison strategies
            compare_cfg = orchestrator_config.get('compare', {})
            self.compare_log_changes_only = compare_cfg.get('log_changes_only', True)
            self.compare_strategies = {}
            if compare_cfg.get('enabled'):
                compare_types = compare_cfg.get('types', [])
                self._logger.info(f"Initializing {len(compare_types)} comparison orchestrators: {compare_types}")
                for compare_type in compare_types:
                    try:
                        # Get strategy-specific config
                        compare_strategy_config = strategies_config.get(compare_type, {})
                        merged_compare_config = {**orchestrator_config, **compare_strategy_config}
                        
                        self._logger.debug(f"  {compare_type}: config keys = {list(merged_compare_config.keys())}")
                        
                        compare_context = OrchestrationContext(
                            services=self.services,
                            nodes=self.nodes,
                            metrics=self.metrics_provider,
                            config=merged_compare_config
                        )
                        created_strategy = create_strategy(
                            compare_type,
                            compare_context,
                            merged_compare_config
                        )
                        self.compare_strategies[compare_type] = created_strategy
                        self._logger.info(f"  ✓ Created {compare_type} strategy: {type(created_strategy).__name__}")
                    except Exception as e:
                        self._logger.warning(
                            f"Comparison orchestrator init failed for '{compare_type}': {e}"
                        )
            
            # Log orchestrator creation
            from BlazeAIoT.Core.Utils import BlazeLogger
            logger = BlazeLogger.get_instance()
            
            # Build detailed log message with key config parameters
            config_details = []
            if orchestrator_type == 'test':
                config_details.append(f"service_id={orchestrator_config.get('service_id')}")
                if orchestrator_config.get('target_node_id'):
                    config_details.append(f"target_node_id={orchestrator_config.get('target_node_id')}")
                if orchestrator_config.get('target_layer_id'):
                    config_details.append(f"target_layer_id={orchestrator_config.get('target_layer_id')}")
            elif orchestrator_type == 'nsga2':
                config_details.append(f"population_size={orchestrator_config.get('population_size', 'default')}")
                config_details.append(f"generations={orchestrator_config.get('generations', 'default')}")
            elif orchestrator_type == 'dqn':
                config_details.append(f"checkpoint_path={orchestrator_config.get('checkpoint_path', 'training')}")
            
            details_str = ", ".join(config_details) if config_details else "default config"
            logger.info(f"Orchestrator initialized: type={orchestrator_type} ({details_str})")
            
            print(f"  ✓ Orchestration strategy created: {orchestrator_type}")
            if self.compare_strategies:
                compare_list = ", ".join(self.compare_strategies.keys())
                logger.info(f"Comparison orchestrators enabled: {compare_list}")
                print(f"  ✓ Comparison orchestrators enabled: {compare_list}")
            
            # Initialize service allocation facade
            self._init_service_allocation()
            
            return True
        
        except Exception as e:
            self._logger.error(f"Orchestrator initialization failed: {e}", exc_info=True)
            return False
    
    def _init_service_allocation(self) -> bool:
        """Initialize service allocation facade."""
        try:
            from EdgeLogiQ.BlazeAIoT.BlazeMaestro.ServiceAllocation import (
                ServiceAllocationFacade
            )
            
            env_mode = self.config_loader.get_environment_mode()
            
            if env_mode == 'simulator':
                self.allocation_facade = ServiceAllocationFacade.create_for_simulator(verbose=False)
                print("  ✓ Service Allocation (Simulator mode) initialized")
            else:
                cluster_configs = self.config_loader.get_clusters_config()
                if cluster_configs:
                    self.allocation_facade = ServiceAllocationFacade.create_for_kubernetes(
                        cluster_configs=cluster_configs,
                        nodes=self.nodes,
                        verbose=True
                    )
                    print(f"  ✓ Service Allocation (Kubernetes mode) initialized")
                    print(f"    Multi-cluster support: {len(cluster_configs)} cluster(s)")
                else:
                    print("  ⚠ No cluster configs available for Kubernetes allocation")
                    self.allocation_facade = None
            
            return True
        
        except Exception as e:
            self._logger.warning(f"Service allocation initialization failed: {e}")
            return True  # Non-fatal
    
    def get_initialized_components(self) -> Dict[str, Any]:
        """
        Get all initialized components.
        
        Returns:
            Dict containing all initialized components
        """
        return {
            'layers': self.layers,
            'nodes': self.nodes,
            'services': self.services,
            'topology_manager': self.topology_manager,
            'service_manager': self.service_manager,
            'metrics_provider': self.metrics_provider,
            'orchestrator': self.orchestration_context,
            'strategy': self.strategy,
            'compare_strategies': self.compare_strategies,
            'compare_log_changes_only': self.compare_log_changes_only,
            'allocation_facade': self.allocation_facade,
            'config_updater': self.config_updater,
            'current_allocation': self.current_allocation,
            'current_pod_refs': self.current_pod_refs,
            'locked_services': self.locked_services,
            'simpy_env': getattr(self, 'simpy_env', None),
            'platform_xml_path': getattr(self, 'platform_xml_path', None)
        }
