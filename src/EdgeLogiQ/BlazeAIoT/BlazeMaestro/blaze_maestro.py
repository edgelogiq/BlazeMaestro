"""
Blaze Maestro - AI-Powered Edge Orchestration System

This is the refactored main class that delegates to specialized components.
Original 2,123-line monolithic file has been decomposed following SOLID principles.

Architecture:
- ConfigLoader: Configuration management
- PlatformManager: Platform initialization (K8s/Simulator)
- ServiceDiscoveryManager: Service discovery coordination  
- KubernetesServiceDiscovery: K8s API interactions
- XMLManager: XML file operations
- AllocationExecutor: Service allocation execution
- InitializationOrchestrator: Startup coordination
- OrchestrationLoop: Main orchestration loop

See REFACTORING.md for detailed documentation.
"""

from BlazeAIoT.Core.Utils import BlazeLogger
from typing import Dict, List, Optional
import os

# Import refactored components
from .Config.config_loader import ConfigLoader
from .Managers.platform_manager import PlatformManager
from .Managers.service_discovery_manager import ServiceDiscoveryManager
from .Executors.xml_manager import XMLManager
from .Executors.allocation_executor import AllocationExecutor
from .Executors.initialization_orchestrator import InitializationOrchestrator
from .Managers.orchestration_loop import OrchestrationLoop
from .Utils.orchestrator_csv_logger import OrchestratorCSVLogger


class BlazeMaestro:
    """
    Blaze Maestro - Main Facade Class
    
    Provides a simplified interface to the edge orchestration system.
    Coordinates initialization, orchestration, and service allocation.
    
    Usage:
        maestro = BlazeMaestro('config.yaml')
        maestro.initialize()
        maestro.run()
    """
    
    def __init__(self, config_path: str):
        """
        Initialize Blaze Maestro.
        
        Args:
            config_path: Path to YAML configuration file
        """
        
        # Load configuration
        self._logger = BlazeLogger.get_instance()
        self._logger.info(f"Loading YAML config from: {config_path}")
        self.config_loader = ConfigLoader(config_path)
        
        # Initialize platform manager
        self.platform_manager = PlatformManager(self.config_loader.config)
        
        # Determine XML base path
        xml_base_path = self._determine_xml_base_path()
        self.xml_manager = XMLManager(xml_base_path)
        
        # Service discovery manager (nodes/layers set after platform init)
        self.service_discovery_manager = ServiceDiscoveryManager(
            config=self.config_loader.config,
            nodes=None,
            layers=None
        )
        
        # Components initialized during startup
        self.initialization_orchestrator = None
        self.allocation_executor = None
        self.orchestration_loop = None
        
        # CSV logging for metrics collection
        self.csv_logger = OrchestratorCSVLogger(output_base_dir="out")
        
        # State (populated during initialization)
        self.layers = None
        self.nodes = None
        self.services = None
        self.metrics_provider = None
        self.orchestrator = None
        self.orchestration_context = None
        self.strategy = None
        self.current_allocation = {}
        self.current_pod_refs = {}
        self.locked_services = set()
        
        # Additional state
        self.topology_manager = None
        self.service_manager = None
        self.config_updater = None
        self.allocation_facade = None
        self.allocation_emitter = None
        self.metrics_populated_once = False
        
        # Metrics-only mode (disable orchestration)
        self.metrics_only_mode = self.config_loader.config.get('orchestrator', {}).get('metrics_only', False)
        
        self._logger.info("BlazeMaestro instance created")
    
    def _determine_xml_base_path(self) -> str:
        """Determine base path for XML files from configuration."""
        services_config = self.config_loader.get_services_config()
        environment_config = self.config_loader.get_environment_config()
        
        # Check for explicit services_xml path (try both old and new key names)
        if 'services_xml' in services_config:
            return os.path.dirname(services_config['services_xml'])
        if 'xml_path' in services_config:
            return os.path.dirname(services_config['xml_path'])
        
        # Check for config_dir
        if 'config_dir' in services_config:
            return services_config['config_dir']
        
        # Try to get path from simulator or environment config
        simulator_config = environment_config.get('simulator', {})
        if 'platform_xml' in simulator_config:
            return os.path.dirname(simulator_config['platform_xml'])
        
        # Default to config directory relative to config file location
        config_dir = os.path.dirname(os.path.abspath(str(self.config_loader.config_path)))
        return config_dir
    
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Executes the complete initialization sequence:
        1. Platform initialization (Kubernetes/Simulator)
        2. Service discovery
        3. Initial allocation discovery
        4. Metrics provider setup
        5. Orchestrator initialization
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create initialization orchestrator
            self.initialization_orchestrator = InitializationOrchestrator(
                config_loader=self.config_loader,
                platform_manager=self.platform_manager,
                service_discovery_manager=self.service_discovery_manager,
                xml_manager=self.xml_manager,
                csv_output_dir=self.csv_logger.session_dir
            )
            
            # Execute initialization
            success = self.initialization_orchestrator.initialize_all()
            
            if not success:
                self._logger.error("Initialization failed")
                return False
            
            # Extract components
            components = self.initialization_orchestrator.get_initialized_components()
            self._populate_from_components(components)
            
            # Create allocation executor
            self.allocation_executor = AllocationExecutor(
                allocation_facade=components['allocation_facade'],
                nodes=self.nodes,
                config=self.config_loader.config
            )
            
            # Create allocation emitter
            from .Metrics.allocation_emitter import PerNodeAllocationEmitter
            self.allocation_emitter = PerNodeAllocationEmitter(
                nodes=self.nodes,
                use_logging=True
            )
            
            # Create orchestration loop
            self.orchestration_loop = OrchestrationLoop(
                config=self.config_loader.config,
                orchestrator=self.orchestrator,
                metrics_provider=self.metrics_provider,
                services=self.services,
                nodes=self.nodes,
                layers=self.layers,
                strategy=self.strategy
            )
            
            # Share state with orchestration loop
            self.orchestration_loop.current_allocation = self.current_allocation
            self.orchestration_loop.service_discovery_manager = self.service_discovery_manager
            self.orchestration_loop.allocation_facade = components['allocation_facade']
            self.orchestration_loop.config_updater = components['config_updater']
            self.orchestration_loop.allocation_emitter = self.allocation_emitter
            self.orchestration_loop.compare_strategies = components.get('compare_strategies', {})
            self.orchestration_loop.compare_log_changes_only = components.get('compare_log_changes_only', True)
            
            # Set CSV logger
            self.orchestration_loop.csv_logger = self.csv_logger
            # Pass SimPy environment if available (for simulator mode)
            if hasattr(self.initialization_orchestrator, 'simpy_env'):
                self.orchestration_loop.simpy_env = self.initialization_orchestrator.simpy_env
            
            # Pass services to config_updater for multi-deployment aggregation support
            if components['config_updater'] and hasattr(components['config_updater'], '_services_for_aggregation'):
                components['config_updater']._services_for_aggregation = self.services
            
            # Ensure runtime metrics updates target the correct services_update.xml path
            services_update_path = None
            # Use service_discovery_manager to get the source path, then convert to update path
            if self.service_discovery_manager:
                services_source_path = self.service_discovery_manager.get_services_xml_path()
                services_update_path = services_source_path.replace('.xml', '_update.xml')
            if not services_update_path:
                try:
                    services_update_path = str(self.xml_manager.get_services_update_path('services.xml'))
                except Exception:
                    services_update_path = str(self.xml_manager.base_path / 'services_update.xml')
            # Attach for use by the orchestration loop on each iteration
            setattr(self.orchestration_loop, 'services_update_xml_path', services_update_path)
            
            return True
        
        except Exception as e:
            self._logger.error(f"Initialization error: {e}", exc_info=True)
            return False
    
    def _populate_from_components(self, components: Dict) -> None:
        """Populate instance variables from initialized components."""
        self.layers = components['layers']
        self.nodes = components['nodes']
        self.services = components['services']
        self.metrics_provider = components['metrics_provider']
        self.orchestrator = components['orchestrator']
        self.orchestration_context = components['orchestrator']
        self.strategy = components.get('strategy')
        self.current_allocation = components['current_allocation']
        self.current_pod_refs = components.get('current_pod_refs', {})
        self.locked_services = components.get('locked_services', set())
        
        # Additional components
        self.topology_manager = components.get('topology_manager')
        self.service_manager = components.get('service_manager')
        self.config_updater = components.get('config_updater')
        self.allocation_facade = components.get('allocation_facade')
        
        # Simulator-specific
        if hasattr(self.initialization_orchestrator, 'simpy_env'):
            self.simpy_env = self.initialization_orchestrator.simpy_env
        if hasattr(self.initialization_orchestrator, 'platform_xml_path'):
            self.platform_xml_path = self.initialization_orchestrator.platform_xml_path
    
    async def start(self) -> None:
        """Asynchronously start the orchestration (for compatibility with async code)."""
        import asyncio
        # Respect metrics-only mode even when started asynchronously
        if self.metrics_only_mode:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._run_metrics_only)
            return

        if not self.orchestration_loop:
            self._logger.error("Orchestration loop not initialized. Call initialize() first.")
            return
        
        # Run orchestration in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.orchestration_loop.run)
    
    def run(self) -> None:
        """Run the orchestration loop (blocking) or metrics collection if metrics_only mode."""
        if self.metrics_only_mode:
            self._run_metrics_only()
            return
        
        if not self.orchestration_loop:
            self._logger.error("Orchestration loop not initialized. Call initialize() first.")
            return
        
        self.orchestration_loop.run()
    
    def _run_metrics_only(self) -> None:
        """Run metrics collection without orchestration."""
        if not self.config_updater:
            self._logger.error("Config updater not initialized")
            return
        
        orchestration_rate = self.config.get('orchestrator', {}).get('rate_hz', 0.1)
        interval = 1.0 / orchestration_rate
        max_iterations = self.config.get('orchestrator', {}).get('max_iterations')
        
        print("\n" + "=" * 80)
        print("STARTING METRICS COLLECTION (orchestration disabled)")
        print("=" * 80)
        print("Press Ctrl+C to stop\n")
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                
                # Check iteration limit
                if max_iterations and iteration > max_iterations:
                    self._logger.info(f"✓ Reached max iterations ({max_iterations})")
                    break
                
                self._logger.info(f"[Iteration {iteration}] Collecting metrics...")
                
                try:
                    # Update metrics only
                    self.config_updater.update_instance_metrics(
                        metrics_provider=self.metrics_provider,
                        nodes=self.nodes,
                        services=self.services,
                        current_allocation=self.current_allocation,
                        services_xml_path=self.service_discovery_manager.get_services_xml_path(),
                        namespace=self.config.get('services', {}).get('namespace', 'default')
                    )
                    self._logger.info(f"[Iteration {iteration}] ✓ Metrics updated")
                    
                except Exception as e:
                    self._logger.error(f"[Iteration {iteration}] Metrics update failed: {e}", exc_info=True)
                
                # Sleep before next iteration
                import time
                time.sleep(interval)
        
        except KeyboardInterrupt:
            print("\n" + "=" * 80)
            print("METRICS COLLECTION STOPPED")
            print("=" * 80)
    
    def start_threaded(self) -> bool:
        """Start orchestration loop in background thread."""
        if not self.orchestration_loop:
            self._logger.error("Orchestration loop not initialized. Call initialize() first.")
            return False
        
        return self.orchestration_loop.start_threaded()
    
    def stop_threaded(self, timeout: float = 10.0) -> bool:
        """Stop orchestration loop and wait for thread."""
        if self.orchestration_loop:
            return self.orchestration_loop.stop_threaded(timeout)
        return True
    
    def pause(self) -> bool:
        """Pause orchestration loop."""
        if self.orchestration_loop:
            return self.orchestration_loop.pause()
        self._logger.warning("Orchestration loop not initialized")
        return False
    
    def resume(self) -> bool:
        """Resume orchestration loop after pausing."""
        if self.orchestration_loop:
            return self.orchestration_loop.resume()
        self._logger.warning("Orchestration loop not initialized")
        return False
    
    def stop(self) -> None:
        """Stop orchestration loop."""
        if self.orchestration_loop:
            self.orchestration_loop.stop()
        # Close CSV logger
        if self.csv_logger:
            self.csv_logger.close()
            self._logger.info(f"CSV metrics saved to: {self.csv_logger.get_session_dir()}")
    
    def is_running(self) -> bool:
        """Check if orchestration loop is running."""
        if self.orchestration_loop:
            return self.orchestration_loop.is_running()
        return False
    
    def is_paused(self) -> bool:
        """Check if orchestration loop is paused."""
        if self.orchestration_loop:
            return self.orchestration_loop.is_paused()
        return False
    
    def allocate_service(
        self,
        service_id: str,
        target_node: str,
        target_layer: str,
        source_node: Optional[str] = None,
        source_layer: Optional[str] = None,
        namespace: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Allocate (move) a service to target node/layer."""
        if not self.allocation_executor:
            self._logger.error("Allocation executor not initialized. Call initialize() first.")
            return False
        
        success = self.allocation_executor.allocate_service(
            service_id=service_id,
            target_node=target_node,
            target_layer=target_layer,
            source_node=source_node,
            source_layer=source_layer,
            namespace=namespace,
            **kwargs
        )
        
        if success:
            self.current_allocation[service_id] = target_node
        
        return success
    
    def deallocate_service(
        self,
        service_id: str,
        node: str,
        layer: str,
        namespace: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Deallocate (remove) service from node/layer."""
        if not self.allocation_executor:
            self._logger.error("Allocation executor not initialized. Call initialize() first.")
            return False
        
        return self.allocation_executor.deallocate_service(
            service_id=service_id,
            node=node,
            layer=layer,
            namespace=namespace,
            **kwargs
        )
    
    def optimize(
        self,
        services: Optional[List] = None,
        current_assignment: Optional[Dict[str, str]] = None,
        hysteresis_threshold: float = 0.0
    ) -> Dict[str, str]:
        """Run orchestration optimization manually."""
        if not self.orchestrator:
            self._logger.error("Orchestrator not initialized. Call initialize() first.")
            return {}
        
        # Check if using strategy pattern
        if self.orchestration_context and self.strategy:
            return self.orchestration_context.optimize(
                strategy=self.strategy,
                current_assignment=current_assignment or self.current_allocation,
                hysteresis_threshold=hysteresis_threshold,
                services=services or self.services
            )
        else:
            # Direct orchestrator call (backward compatibility)
            return self.orchestrator.optimize(
                services=services or self.services,
                current_assignment=current_assignment or self.current_allocation,
                hysteresis_threshold=hysteresis_threshold
            )
    
    def shutdown(self) -> None:
        """Graceful shutdown."""
        print("\n" + "=" * 80)
        print("SHUTTING DOWN BLAZE MAESTRO")
        print("=" * 80)
        
        if self.orchestration_loop:
            self.orchestration_loop.stop()
        
        print("✓ Shutdown complete")
    
    # Backward compatibility properties
    @property
    def config(self) -> Dict:
        """Get configuration dictionary (backward compatibility)."""
        return self.config_loader.config
    
    @property
    def running(self) -> bool:
        """Check if running (backward compatibility)."""
        return self.is_running()
    
    @running.setter
    def running(self, value: bool) -> None:
        """Set running state (backward compatibility)."""
        if not value:
            self.stop()
    
    # Legacy method stubs for backward compatibility
    def _load_config(self, config_path: str) -> Dict:
        """Legacy method - delegates to ConfigLoader."""
        return ConfigLoader(config_path).config
    
