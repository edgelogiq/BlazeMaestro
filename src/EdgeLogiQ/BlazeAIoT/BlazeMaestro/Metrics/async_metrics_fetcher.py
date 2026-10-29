"""
Async metrics fetcher module.

Fetches metrics independently from orchestration loop to avoid blocking.
Metrics are pre-computed and provided to orchestrators as input.
"""

import threading
import time
from typing import Dict, List, Any, Optional
from BlazeAIoT.Core.Utils import BlazeLogger

from .snapshot_metrics_facade import MetricsSnapshot, SnapshotMetricsFacade


class AsyncMetricsFetcher:
    """
    Asynchronously fetches metrics from Prometheus/Simulator.
    
    Runs in a background thread and updates a cached snapshot that
    orchestrators can read synchronously without blocking.
    """
    
    def __init__(self, config_updater, metrics_provider, nodes: List, 
                 services: List, services_xml_path: str, namespace: str = 'default'):
        """
        Initialize async metrics fetcher.
        
        Args:
            config_updater: ConfigUpdater instance (PrometheusXMLUpdater or SimulatorXMLUpdater)
            metrics_provider: Metrics facade (PrometheusMetricsFacade or SimulatorMetricsFacade)
            nodes: List of Node objects
            services: List of Service objects
            services_xml_path: Path to services_update.xml
            namespace: Kubernetes namespace (default: 'default')
        """
        self.config_updater = config_updater
        self.metrics_provider = metrics_provider
        self.nodes = nodes
        self.services = services
        self.services_xml_path = services_xml_path
        self.namespace = namespace
        
        self._logger = BlazeLogger.get_instance()
        
        # Synchronization
        self._thread = None
        self._running = False
        self._lock = threading.RLock()
        self._ready_event = threading.Event()  # Signals when first fetch is done
        
        # Cached metrics snapshot (safe to read from orchestration thread)
        self._metrics_snapshot = {
            'timestamp': None,
            'success': False,
            'error': None,
            'snapshot': None,
            'facade': None,
        }
    
    def start(self, fetch_interval_hz: float = 1.0):
        """
        Start background fetching thread.
        
        Args:
            fetch_interval_hz: Fetch rate in Hz (e.g., 1.0 Hz = every 1 second)
        """
        if self._running:
            self._logger.warning("AsyncMetricsFetcher already running")
            return
        
        self._running = True
        interval_seconds = 1.0 / fetch_interval_hz
        
        def _fetch_loop():
            """Background thread that fetches metrics periodically."""
            self._logger.info(f"AsyncMetricsFetcher started (interval: {interval_seconds:.2f}s)")
            
            first_iteration = True
            while self._running:
                try:
                    start = time.time()
                    
                    # Fetch metrics via config_updater
                    # This updates the underlying services_update.xml in memory
                    success = self.config_updater.update_instance_metrics(
                        metrics_provider=self.metrics_provider,
                        nodes=self.nodes,
                        services=self.services,
                        current_allocation={},
                        services_xml_path=self.services_xml_path,
                        namespace=self.namespace
                    )
                    
                    snapshot = self._build_snapshot() if success else None
                    facade = SnapshotMetricsFacade(snapshot) if snapshot else None

                    # Update snapshot (thread-safe with lock)
                    with self._lock:
                        self._metrics_snapshot = {
                            'timestamp': time.time(),
                            'success': success,
                            'error': None,
                            'snapshot': snapshot,
                            'facade': facade,
                        }
                        
                        if first_iteration and success:
                            self._ready_event.set()  # Signal that first fetch is done
                            first_iteration = False
                            self._logger.info("AsyncMetricsFetcher: First metrics fetch complete")
                    
                    elapsed = time.time() - start
                    
                    # Sleep for remaining time
                    sleep_time = interval_seconds - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        self._logger.debug(
                            f"Metrics fetch took {elapsed:.2f}s (target: {interval_seconds:.2f}s)"
                        )
                
                except Exception as e:
                    self._logger.error(f"Metrics fetch failed: {e}", exc_info=True)
                    
                    with self._lock:
                        self._metrics_snapshot = {
                            'timestamp': time.time(),
                            'success': False,
                            'error': str(e),
                            'snapshot': None,
                            'facade': None,
                        }
                    
                    time.sleep(interval_seconds)
        
        self._thread = threading.Thread(target=_fetch_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop background fetching thread."""
        if not self._running:
            return
        
        self._logger.info("Stopping AsyncMetricsFetcher...")
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                self._logger.warning("AsyncMetricsFetcher thread did not stop cleanly")
    
    def wait_for_first_fetch(self, timeout: float = 30.0) -> bool:
        """
        Block until first metrics fetch completes.
        
        Useful for initialization to ensure orchestrators have valid metrics
        before running.
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            True if first fetch succeeded, False if timeout or error
        """
        return self._ready_event.wait(timeout=timeout)
    
    def get_snapshot(self) -> Dict[str, Any]:
        """
        Get current metrics snapshot (thread-safe).
        
        Returns:
            Dictionary with:
            - timestamp: Unix timestamp of last fetch
            - success: bool indicating if fetch succeeded
            - error: Error message if fetch failed (None if successful)
        """
        with self._lock:
            return self._metrics_snapshot.copy()

    def get_snapshot_facade(self) -> Optional[SnapshotMetricsFacade]:
        """Get a read-only facade for the latest metrics snapshot."""
        with self._lock:
            return self._metrics_snapshot.get('facade')
    
    def get_health(self) -> Dict[str, Any]:
        """
        Get fetcher health status.
        
        Returns:
            Dictionary with health info
        """
        snapshot = self.get_snapshot()
        age = None
        
        if snapshot['timestamp']:
            age = time.time() - snapshot['timestamp']
        
        return {
            'running': self._running,
            'thread_alive': self._thread.is_alive() if self._thread else False,
            'last_fetch_timestamp': snapshot['timestamp'],
            'last_fetch_age_seconds': age,
            'last_fetch_success': snapshot['success'],
            'last_fetch_error': snapshot['error']
        }

    def _build_snapshot(self) -> MetricsSnapshot:
        node_ids: List[str] = [str(getattr(n, 'node_id', '')) for n in self.nodes]

        capacities: Dict[str, Dict[str, float]] = {}
        available: Dict[str, Dict[str, float]] = {}
        properties: Dict[str, Dict[str, Any]] = {}

        for node_id in node_ids:
            capacities[node_id] = self.metrics_provider.get_node_capacity(node_id)
            available[node_id] = self.metrics_provider.get_node_available(node_id)
            properties[node_id] = {
                'gpu_shared': self.metrics_provider.get_node_property(node_id, 'gpu_shared')
            }

        latency_matrix = None
        if hasattr(self.metrics_provider, 'get_latency_matrix'):
            try:
                latency_matrix = self.metrics_provider.get_latency_matrix(self.nodes)
            except TypeError:
                latency_matrix = self.metrics_provider.get_latency_matrix()

        return MetricsSnapshot(
            timestamp=time.time(),
            node_ids=node_ids,
            node_capacity=capacities,
            node_available=available,
            node_properties=properties,
            latency_matrix=latency_matrix,
        )
