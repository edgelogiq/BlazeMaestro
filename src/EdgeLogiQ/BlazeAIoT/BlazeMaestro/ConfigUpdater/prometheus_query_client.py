"""
Prometheus query client for executing PromQL queries.

Handles direct queries to Prometheus instances and multi-layer query routing.
Includes port-forward validation and automatic restart on connection failures.
"""

from typing import Dict, List, Optional, Any
import subprocess
import time
import socket
from urllib.parse import urlparse


class PrometheusQueryClient:
    """Client for querying Prometheus metrics."""

    def __init__(self, metrics_facade=None, prometheus_url: str = None, verify_ssl: bool = False):
        """
        Initialize the Prometheus query client.

        Args:
            metrics_facade: PrometheusMetricsFacade instance (preferred)
            prometheus_url: Direct Prometheus URL (legacy fallback)
            verify_ssl: Whether to verify SSL certificates
        """
        from BlazeAIoT.Core.Utils import BlazeLogger
        
        self.metrics_facade = metrics_facade
        self.verify_ssl = verify_ssl
        
        # Get cluster_urls and base_url from metrics_facade if available
        if self.metrics_facade and hasattr(self.metrics_facade, 'cluster_urls'):
            self.cluster_urls = self.metrics_facade.cluster_urls
            self.prometheus_url = getattr(self.metrics_facade, 'base_url', prometheus_url or 'http://localhost:9090').rstrip('/')
        else:
            self.cluster_urls = {}
            self.prometheus_url = (prometheus_url or 'http://localhost:9090').rstrip('/')
        
        # Legacy support: if using legacy mode (no facade), create session
        if not self.metrics_facade:
            #if prometheus_url:
            #    BlazeLogger.get_instance().warning("PrometheusQueryClient: Using legacy direct query mode. Consider passing metrics_facade instead.")
            import requests
            from urllib3.util.retry import Retry
            from requests.adapters import HTTPAdapter
            
            self.session = requests.Session()
            
            # Disable retries on connection errors to fail fast
            retry_strategy = Retry(
                total=0,
                connect=0,
                read=0,
                status_forcelist=[],
                backoff_factor=0
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
            
            if not verify_ssl:
                import warnings
                from urllib3.exceptions import InsecureRequestWarning
                warnings.filterwarnings('ignore', category=InsecureRequestWarning)
        
        self._port_forward_attempts = {}  # Track retry attempts per URL

    def _is_localhost_url(self, url: str) -> bool:
        """Check if URL is a localhost port-forward (localhost, 127.0.0.1, etc.)."""
        parsed = urlparse(url)
        return parsed.hostname in ['localhost', '127.0.0.1', '::1']

    def _validate_port_forward(self, url: str) -> bool:
        """
        Validate if port-forward is accessible.
        
        Args:
            url: Prometheus URL to validate
            
        Returns:
            True if accessible, False otherwise
        """
        from BlazeAIoT.Core.Utils import BlazeLogger
        
        try:
            parsed = urlparse(url)
            host = parsed.hostname or 'localhost'
            port = parsed.port or 9090
            
            # Try to connect to the port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                BlazeLogger.get_instance().debug(f"[PORT_FORWARD] Port {port} is accessible")
                return True
            else:
                BlazeLogger.get_instance().warning(f"[PORT_FORWARD] Port {port} is NOT accessible (connection refused)")
                return False
        except Exception as e:
            BlazeLogger.get_instance().warning(f"[PORT_FORWARD] Failed to validate {url}: {e}")
            return False

    def _restart_port_forward(self, url: str, layer_name: str = None) -> bool:
        """
        Attempt to restart the port-forward for the given URL.
        
        Args:
            url: Prometheus URL (e.g., http://localhost:9091)
            layer_name: Optional layer name for logging (e.g., 'fog', 'edge')
            
        Returns:
            True if restart successful, False otherwise
        """
        from BlazeAIoT.Core.Utils import BlazeLogger
        
        if not self._is_localhost_url(url):
            return False  # Only restart localhost port-forwards
        
        parsed = urlparse(url)
        port = parsed.port or 9090
        
        # Track attempts to avoid infinite restart loops
        attempt_key = f"{url}_{time.time()}"
        if attempt_key in self._port_forward_attempts:
            if self._port_forward_attempts[attempt_key] > 2:
                BlazeLogger.get_instance().warning(f"[PORT_FORWARD] Max restart attempts reached for {layer_name or url}")
                return False
            self._port_forward_attempts[attempt_key] += 1
        else:
            self._port_forward_attempts[attempt_key] = 1
        
        layer_label = f"{layer_name} " if layer_name else ""
        BlazeLogger.get_instance().info(f"[PORT_FORWARD] Attempting to restart port-forward for {layer_label}({url})...")
        
        try:
            # Kill any existing port-forward on this port
            kill_cmd = f"pkill -f 'port-forward.*{port}' || true"
            subprocess.run(kill_cmd, shell=True, timeout=5)
            time.sleep(1)
            
            # Determine kubeconfig and namespace based on port
            # Port 9091 = fog, 9090 = edge
            if port == 9091:
                kubeconfig = "~/.kube/config.lab.fog"
                context = "lab-fog"
            else:
                kubeconfig = "~/.kube/config.lab.edge"
                context = "lab-edge"
            
            # Start new port-forward
            pf_cmd = f"KUBECONFIG={kubeconfig} kubectl port-forward -n cattle-monitoring-system svc/rancher-monitoring-prometheus {port}:9090 > /tmp/pf_{port}.log 2>&1 &"
            subprocess.run(pf_cmd, shell=True, timeout=5)
            
            # Wait for port-forward to establish
            for attempt in range(5):
                time.sleep(1)
                if self._validate_port_forward(url):
                    BlazeLogger.get_instance().info(f"[PORT_FORWARD] Successfully restarted port-forward for {layer_label}({port})")
                    return True
            
            BlazeLogger.get_instance().warning(f"[PORT_FORWARD] Port-forward restart timed out for {layer_label}({port})")
            return False
            
        except subprocess.TimeoutExpired:
            BlazeLogger.get_instance().error(f"[PORT_FORWARD] Timeout restarting port-forward for {layer_label}({port})")
            return False
        except Exception as e:
            BlazeLogger.get_instance().error(f"[PORT_FORWARD] Failed to restart port-forward for {layer_label}({url}): {e}")
            return False

    def _query_with_port_forward_recovery(self, query: str, url: str, layer_name: str = None) -> Optional[Dict]:
        """
        Execute query with automatic port-forward validation and restart.
        
        Args:
            query: PromQL query string
            url: Prometheus URL
            layer_name: Optional layer name for logging
            
        Returns:
            Response object or None if failed
        """
        from BlazeAIoT.Core.Utils import BlazeLogger
        
        if not hasattr(self, 'session'):
            import requests
            from urllib3.util.retry import Retry
            from requests.adapters import HTTPAdapter
            
            self.session = requests.Session()
            
            # Disable retries on connection errors to fail fast
            retry_strategy = Retry(
                total=0,
                connect=0,
                read=0,
                status_forcelist=[],
                backoff_factor=0
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
        
        # Validate port-forward for localhost URLs
        if self._is_localhost_url(url):
            if not self._validate_port_forward(url):
                BlazeLogger.get_instance().warning(f"[PORT_FORWARD] Port-forward not accessible for {layer_name or url}, attempting restart...")
                if not self._restart_port_forward(url, layer_name):
                    BlazeLogger.get_instance().error(f"[PORT_FORWARD] Failed to restore port-forward for {layer_name or url}")
                    return None
        
        # Execute query
        try:
            response = self.session.get(
                f"{url.rstrip('/')}/api/v1/query",
                params={'query': query},
                timeout=10,
                verify=self.verify_ssl
            )
            return response
        except Exception as e:
            BlazeLogger.get_instance().error(f"[PORT_FORWARD] Query failed for {layer_name or url}: {e}")
            return None

    def query(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a PromQL query using the metrics facade if available, otherwise legacy mode.

        Args:
            query: PromQL query string

        Returns:
            Query result list or None if failed
        """
        from BlazeAIoT.Core.Utils import BlazeLogger
        
        # Prefer metrics_facade if available
        if self.metrics_facade:
            try:
                return self.metrics_facade.query(query)
            except Exception as e:
                BlazeLogger.get_instance().error(f"Failed to query via metrics_facade: {e}")
                return None
        
        # Legacy mode - direct query
        if hasattr(self, 'session') and hasattr(self, 'prometheus_url'):
            try:
                response = self.session.get(
                    f"{self.prometheus_url}/api/v1/query",
                    params={'query': query},
                    timeout=10,
                    verify=self.verify_ssl
                )

                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'success':
                        return data.get('data', {}).get('result', [])

                BlazeLogger.get_instance().warning(f"Prometheus query failed: {query}")
                return None
            except Exception as e:
                #BlazeLogger.get_instance().error(f"Failed to query Prometheus: {e}")
                return None
        
        BlazeLogger.get_instance().error("No query method available (no facade or legacy session)")
        return None

    def query_at_url(self, query: str, url: str, layer_name: str = None) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a PromQL query at a specific Prometheus URL (for layer-specific queries).
        
        Includes automatic port-forward validation and restart on connection failures.

        Args:
            query: PromQL query string
            url: Prometheus URL to query directly
            layer_name: Optional layer name for logging (e.g., 'fog', 'edge')

        Returns:
            Query result list or None if failed
        """
        from BlazeAIoT.Core.Utils import BlazeLogger
        
        # Try query with port-forward recovery
        response = self._query_with_port_forward_recovery(query, url, layer_name)
        
        if response is None:
            return None
        
        try:
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    return data.get('data', {}).get('result', [])

            BlazeLogger.get_instance().debug(f"Prometheus query failed at {url}: {query}")
            return None
        except Exception as e:
            BlazeLogger.get_instance().debug(f"Failed to parse Prometheus response at {url}: {e}")
            return None
