from __future__ import annotations
from BlazeAIoT.Core.Utils import BlazeLogger

import subprocess
import warnings
import socket
import atexit
import time
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Any, List

import requests
from urllib3.exceptions import InsecureRequestWarning

from .base import BaseMetricsFacade



def find_kubectl() -> Optional[str]:
    """
    Find kubectl executable even if not in PATH.

    Searches common locations and uses heuristics to locate kubectl.

    Returns:
        Full path to kubectl, or None if not found
    """
    # First, try the normal way (if in PATH)
    kubectl_path = shutil.which('kubectl')
    if kubectl_path:
        return kubectl_path

    # Common installation locations
    common_paths = [
        '/usr/local/bin/kubectl',
        '/usr/bin/kubectl',
        '/snap/bin/kubectl',
        '/opt/kubectl/bin/kubectl',
        os.path.expanduser('~/.local/bin/kubectl'),
        os.path.expanduser('~/bin/kubectl'),
        # Windows paths (in case of WSL)
        '/mnt/c/Program Files/kubectl/kubectl.exe',
        '/mnt/c/ProgramData/chocolatey/bin/kubectl.exe',
    ]

    for path in common_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            BlazeLogger.get_instance().info(f"Found kubectl at: {path}")
            return path

    # Try to find via locate-like search in common directories
    search_dirs = [
        '/usr/local',
        '/usr',
        '/opt',
        '/snap',
        os.path.expanduser('~/.local'),
        os.path.expanduser('~/bin'),
    ]

    for search_dir in search_dirs:
        if not os.path.isdir(search_dir):
            continue

        try:
            for root, dirs, files in os.walk(search_dir, followlinks=False):
                # Limit depth to avoid excessive searching
                depth = root[len(search_dir):].count(os.sep)
                if depth > 3:
                    dirs[:] = []  # Don't recurse deeper
                    continue

                if 'kubectl' in files:
                    kubectl_candidate = os.path.join(root, 'kubectl')
                    if os.access(kubectl_candidate, os.X_OK):
                        BlazeLogger.get_instance().info(f"Found kubectl at: {kubectl_candidate}")
                        return kubectl_candidate
        except (PermissionError, OSError):
            continue

    BlazeLogger.get_instance().warning("kubectl not found in common locations")
    return None


class PortForwardManager:
    """Manages kubectl port-forward processes with intelligent kubectl detection."""

    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.urls: Dict[str, str] = {}
        self.kubectl_path: Optional[str] = None
        self._kubectl_checked = False
        atexit.register(self.cleanup)

    def _find_kubectl(self):
        """Find kubectl executable and cache the path."""
        if self._kubectl_checked:
            return
        self._kubectl_checked = True
        self.kubectl_path = find_kubectl()
        try:
            if self.kubectl_path:
                BlazeLogger.get_instance().info(f"Using kubectl: {self.kubectl_path}")
            else:
                BlazeLogger.get_instance().warning("kubectl not found - port-forward will not be available")
        except:
            # Logger not initialized yet, skip logging
            pass

    def cleanup(self):
        """Cleanup all port-forward processes."""
        for cluster_name, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception as e:
                BlazeLogger.get_instance().debug(f"Failed to terminate port-forward for {cluster_name}: {e}")

    def setup_port_forward(
        self,
        cluster_name: str,
        context: str,
        kubeconfig_path: Optional[str],
        local_port: int,
        remote_service: str = "svc/prometheus",
        remote_port: int = 9090,
        namespace: str = "cattle-monitoring-system"
    ) -> Optional[str]:
        """
        Set up a port-forward for Prometheus.

        Args:
            cluster_name: Name of the cluster
            context: Kubernetes context name
            kubeconfig_path: Path to kubeconfig (optional)
            local_port: Local port to forward to
            remote_service: Kubernetes service to forward (default: svc/prometheus)
            remote_port: Remote port to forward (default: 9090)
            namespace: Kubernetes namespace (default: cattle-monitoring-system)

        Returns:
            URL to access Prometheus, or None if setup failed
        """
        if cluster_name in self.processes:
            # Port-forward already established
            return self.urls.get(cluster_name)

        # Check if kubectl is available
        if not self.kubectl_path:
            BlazeLogger.get_instance().warning(f"Cannot set up port-forward for {cluster_name}: kubectl not found")
            return None

        try:
            # Build kubectl command using found path
            cmd = [self.kubectl_path, '--context', context]

            if kubeconfig_path and kubeconfig_path != 'null':
                cmd.extend(['--kubeconfig', kubeconfig_path])

            cmd.extend([
                '-n', namespace,
                'port-forward', remote_service,
                f'{local_port}:{remote_port}'
            ])

            BlazeLogger.get_instance().info(f"Starting port-forward: {' '.join(cmd)}")
            print(f"    ℹ Running: {' '.join(cmd[:3])} ... port-forward {remote_service} {local_port}:{remote_port}")

            # Start port-forward process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait a moment for the port-forward to establish
            time.sleep(2)

            # Check if process is still running
            if process.poll() is not None:
                stderr = process.stderr.read() if process.stderr else "Unknown error"
                BlazeLogger.get_instance().error(f"Port-forward failed: {stderr}")
                print(f"    ✗ Port-forward failed: {stderr.strip()}")
                return None

            # Check if port is actually accessible
            url = f"http://localhost:{local_port}"
            if self._is_url_accessible(url):
                self.processes[cluster_name] = process
                self.urls[cluster_name] = url
                BlazeLogger.get_instance().info(f"Port-forward established for {cluster_name} at {url}")
                print(f"    ✓ Port-forward established: {url}")
                return url
            else:
                process.terminate()
                BlazeLogger.get_instance().warning(f"Port-forward started but port {local_port} not accessible")
                print(f"    ✗ Port {local_port} not accessible after port-forward start")
                return None

        except FileNotFoundError:
            BlazeLogger.get_instance().error(f"kubectl not found at {self.kubectl_path}")
            print(f"    ✗ kubectl not found at {self.kubectl_path}")
            return None
        except Exception as e:
            BlazeLogger.get_instance().error(f"Failed to set up port-forward: {e}")
            print(f"    ✗ Port-forward setup failed: {e}")
            return None

    def _is_url_accessible(self, url: str, timeout: float = 2.0) -> bool:
        """Check if a URL is accessible."""
        try:
            response = requests.get(f"{url}/api/v1/query?query=up", timeout=timeout, verify=False)
            return response.status_code == 200
        except Exception:
            return False


def find_free_port(start_port: int = 9090) -> int:
    """Find a free port starting from start_port."""
    port = start_port
    while port < 65535:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            port += 1
    raise RuntimeError("Could not find a free port")


class PrometheusMetricsFacade(BaseMetricsFacade):
    """Prometheus-backed metrics facade with automatic port-forward setup.

    Provide your Prometheus URL, a bearer token for authentication, and a mapping of
    node_id -> PromQL label selector.

    Automatically sets up port-forwards to Prometheus services in Kubernetes clusters
    if they are not directly accessible.
    """

    # Class-level port-forward manager (shared across instances, lazy-initialized)
    _port_forward_manager = None

    @classmethod
    def _get_port_forward_manager(cls):
        """Lazy initialization of port forward manager."""
        if cls._port_forward_manager is None:
            cls._port_forward_manager = PortForwardManager()
        return cls._port_forward_manager

    def __init__(
        self,
        base_url: str,
        token: str,
        node_selectors: Dict[str, str],
        *,
        verify_ssl: bool = True,
        cluster_configs: Optional[List[Dict]] = None,
        node_properties: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.node_selectors = dict(node_selectors)
        self.session = requests.Session()
        
        # Disable retries on connection errors to fail fast
        # urllib3 defaults to retrying 3 times with backoff, causing long delays when Prometheus is down
        from urllib3.util.retry import Retry
        from requests.adapters import HTTPAdapter
        
        retry_strategy = Retry(
            total=0,  # No retries - fail immediately
            connect=0,
            read=0,
            status_forcelist=[],  # Don't retry on any status code
            backoff_factor=0
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.session.headers.update({"Authorization": f"Bearer {token}"})
        self.session.verify = verify_ssl

        # Optional static node properties (e.g., battery capacity from config)
        self.node_properties: Dict[str, Dict[str, Any]] = node_properties or {}
        self._battery_state: Dict[str, Dict[str, float]] = {}

        # Suppress SSL warnings if verification is disabled
        if not verify_ssl:
            warnings.filterwarnings('ignore', category=InsecureRequestWarning)

        # Store cluster configs for auto-setup
        self.cluster_configs = cluster_configs or []
        self.cluster_contexts = {}
        self.cluster_urls = {}  # Per-layer URL mapping (layer_id -> url)
        self.cluster_name_to_layer = {}  # cluster_name -> layer_id mapping
        self.next_port = 9090

        for cluster_config in self.cluster_configs:
            cluster_name = cluster_config.get('name', cluster_config.get('layer_id'))
            layer_id = cluster_config.get('layer_id')
            # Store cluster-specific Prometheus URL by layer_id
            prom_url = cluster_config.get('prometheus_url', base_url)
            self.cluster_urls[layer_id] = prom_url.rstrip("/")
            self.cluster_name_to_layer[cluster_name] = layer_id

            self.cluster_contexts[cluster_name] = {
                'context': cluster_config.get('context'),
                'kubeconfig_path': cluster_config.get('kubeconfig_path'),
                'layer_id': layer_id
            }


    def setup_prometheus_access(self, cluster_name: str) -> Optional[str]:
        """
        Attempt to set up Prometheus access for a cluster.

        Tries the following methods in order:
        1. Check if cluster-specific URL is accessible
        2. Set up kubectl port-forward (auto-finds kubectl)
        3. Set up NodePort service
        4. Fall back to cluster URL

        Args:
            cluster_name: Name of the cluster

        Returns:
            Accessible Prometheus URL, or None if all methods fail
        """
        # Get layer_id for this cluster
        layer_id = self.cluster_contexts.get(cluster_name, {}).get('layer_id')
        # Get cluster-specific URL from config using layer_id
        cluster_url = self.cluster_urls.get(layer_id, self.base_url) if layer_id else self.base_url

        # Method 1: Check if already accessible at cluster-specific URL
        if self._is_url_accessible(cluster_url):
            print(f"    ✓ Prometheus accessible at {cluster_url}")
            return cluster_url

        print(f"    ✗ Prometheus not accessible at {cluster_url}")

        if cluster_name not in self.cluster_contexts:
            print(f"    ✗ No cluster context found for {cluster_name}")
            return self.base_url

        context_info = self.cluster_contexts[cluster_name]
        context = context_info.get('context')
        kubeconfig_path = context_info.get('kubeconfig_path')
        layer_id = context_info.get('layer_id')

        # Method 2: Try port-forward (PortForwardManager finds kubectl automatically)
        port_mgr = self._get_port_forward_manager()
        port_mgr._find_kubectl()  # Ensure kubectl is found
        if port_mgr.kubectl_path:
            print(f"    ℹ Attempting kubectl port-forward for {cluster_name}...")
            print(f"    ℹ Using kubectl: {port_mgr.kubectl_path}")

            # Extract port from cluster_url (from config's prometheus_url)
            # e.g., http://localhost:9091 -> port 9091
            try:
                from urllib.parse import urlparse
                parsed = urlparse(cluster_url)
                local_port = parsed.port if parsed.port else 9090
            except:
                local_port = find_free_port(self.next_port)
                self.next_port = local_port + 1
            
            print(f"    ℹ Using port {local_port} from config")

            # Try multiple common Prometheus service names in Rancher
            service_names = [
                "svc/rancher-monitoring-prometheus",  # Rancher Monitoring v2
                "svc/prometheus",  # Generic
                "svc/prometheus-operated",  # Prometheus Operator
                "svc/kube-prometheus-stack-prometheus",  # kube-prometheus-stack
                "svc/prometheus-server",  # Helm chart default
            ]

            for service_name in service_names:
                print(f"    ℹ Trying service: {service_name}")
                url = port_mgr.setup_port_forward(
                    cluster_name=cluster_name,
                    context=context,
                    kubeconfig_path=kubeconfig_path,
                    local_port=local_port,
                    remote_service=service_name,
                    remote_port=9090,
                    namespace="cattle-monitoring-system"
                )

                if url:
                    print(f"    ✓ Found working service: {service_name}")
                    self.base_url = url
                    # Update cluster_urls with the new port-forwarded URL
                    layer_id = self.cluster_name_to_layer.get(cluster_name)
                    if layer_id:
                        self.cluster_urls[layer_id] = url
                    return url
                else:
                    # Service not found or port-forward failed, try next one
                    continue

            print(f"    ✗ None of the common Prometheus service names worked")
        else:
            print("    ℹ kubectl not found - searching common locations...")
            print(f"    ✗ kubectl not available - cannot set up port-forward automatically")
            print("    Tip: Install kubectl or use manual port-forward")

        # Method 3: Try NodePort (legacy fallback)
        if port_mgr.kubectl_path:
            print(f"    ℹ Attempting NodePort setup for {cluster_name}...")
            url = self._setup_nodeport(cluster_name, context, kubeconfig_path)

            if url:
                print(f"    ✓ NodePort established: {url}")
                self.base_url = url
                return url
            return url

        print(f"    ⚠ Could not establish Prometheus access")
        print(f"    Tip: Manually set up port-forward with:")
        print(f"      kubectl --context {context} -n cattle-monitoring-system port-forward svc/prometheus 9090:9090")
        return self.base_url

    def _is_url_accessible(self, url: str, timeout: float = 2.0) -> bool:
        """Check if a URL is accessible."""
        try:
            response = self.session.get(f"{url}/api/v1/query?query=up", timeout=timeout, verify=False)
            return response.status_code == 200
        except Exception:
            return False

    def _setup_nodeport(self, cluster_name: str, context: str, kubeconfig_path: Optional[str]) -> Optional[str]:
        """
        Set up a NodePort service for Prometheus (fallback method).

        Args:
            cluster_name: Name of the cluster
            context: Kubernetes context name
            kubeconfig_path: Path to kubeconfig

        Returns:
            URL to access Prometheus via NodePort, or None
        """
        try:
            kubectl_env = {}
            if kubeconfig_path and kubeconfig_path != 'null':
                kubectl_env['KUBECONFIG'] = kubeconfig_path

            # Check if service already exists
            check_cmd = [
                'kubectl', '--context', context, '-n', 'cattle-monitoring-system',
                'get', 'svc', 'prometheus-nodeport', '-o', 'jsonpath={.spec.ports[0].nodePort}'
            ]

            result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=5, env=kubectl_env)
            if result.returncode == 0 and result.stdout:
                nodeport = result.stdout.strip()
                BlazeLogger.get_instance().info(f"NodePort service already exists on port {nodeport}")
            else:
                # Create NodePort service
                BlazeLogger.get_instance().info("Creating NodePort service...")

                create_cmd = [
                    'kubectl', '--context', context, '-n', 'cattle-monitoring-system',
                    'create', 'service', 'nodeport', 'prometheus-nodeport',
                    '--tcp=9090:9090', '--node-port=30090'
                ]

                subprocess.run(create_cmd, capture_output=True, timeout=10, env=kubectl_env)

                # Patch selector
                patch_cmd = [
                    'kubectl', '--context', context, '-n', 'cattle-monitoring-system',
                    'patch', 'svc', 'prometheus-nodeport',
                    '-p', '{"spec":{"selector":{"app":"prometheus"}}}'
                ]

                subprocess.run(patch_cmd, capture_output=True, timeout=10, env=kubectl_env)

                BlazeLogger.get_instance().info("NodePort service created on port 30090")
                nodeport = "30090"

            # Get node IP
            get_nodes_cmd = [
                'kubectl', '--context', context,
                'get', 'nodes', '-o', 'jsonpath={.items[*].status.addresses[?(@.type=="ExternalIP")].address}'
            ]

            result = subprocess.run(get_nodes_cmd, capture_output=True, text=True, timeout=5, env=kubectl_env)
            node_ips = result.stdout.strip().split() if result.stdout.strip() else []

            if not node_ips:
                # Try InternalIP
                get_nodes_cmd = [
                    'kubectl', '--context', context,
                    'get', 'nodes', '-o', 'jsonpath={.items[*].status.addresses[?(@.type=="InternalIP")].address}'
                ]
                result = subprocess.run(get_nodes_cmd, capture_output=True, text=True, timeout=5, env=kubectl_env)
                node_ips = result.stdout.strip().split() if result.stdout.strip() else []

            if node_ips:
                node_ip = node_ips[0]
                new_url = f"http://{node_ip}:{nodeport}"
                BlazeLogger.get_instance().info(f"Prometheus available at {new_url}")
                return new_url
            else:
                BlazeLogger.get_instance().warning("Could not determine node IP for NodePort setup")
                return None

        except subprocess.TimeoutExpired:
            BlazeLogger.get_instance().warning("kubectl command timed out")
            return None
        except Exception as e:
            BlazeLogger.get_instance().debug(f"Failed to set up NodePort: {type(e).__name__}: {e}")
            return None

    def _selector(self, node_id: str) -> str:
        """Get PromQL label selector for a node.

        For Rancher Monitoring, uses the 'node' label which contains the Kubernetes node name.
        If custom mappings are provided, uses those instead.

        Handles both:
        - Simple node IDs: "fog-cpu1" → node="fog-cpu1"
        - Composite IDs: "Fog Cluster-fog-cpu1" → node="fog-cpu1" (extracts k8s name)
        """
        # If explicit selector is provided, use it
        if node_id in self.node_selectors:
            return self.node_selectors[node_id]

        # Handle composite IDs only when they actually start with a known cluster prefix.
        # Do NOT split plain hyphenated node names like "frank-2" or "fog-cpu3".
        # Example composite: "Fog Cluster-fog-cpu1" -> node="fog-cpu1"
        if '-' in node_id and self.cluster_name_to_layer:
            for cluster_name in self.cluster_name_to_layer.keys():
                prefix = f"{cluster_name}-"
                if node_id.startswith(prefix):
                    k8s_node_name = node_id[len(prefix):]
                    if k8s_node_name:
                        return f'node="{k8s_node_name}"'

        # Fallback: use node_id as-is (for simple node names without cluster prefix)
        return f'node="{node_id}"'

    def _query(self, promql: str) -> float:
        """Run a PromQL query and return the scalar result.
        
        Uses shorter timeout (2s) to fail fast if Prometheus is unavailable,
        falling back to static node properties or zero.
        """
        api_url = f"{self.base_url}/api/v1/query"
        try:
            # Shorter timeout to fail fast if Prometheus is down
            response = self.session.get(api_url, params={"query": promql}, timeout=2)
            response.raise_for_status()
            data = response.json()

            if data["status"] != "success":
                BlazeLogger.get_instance().debug(f"Prometheus query returned non-success: {data.get('error', 'Unknown error')}")
                return 0.0

            result = data.get("data", {}).get("result", [])
            if not result:
                BlazeLogger.get_instance().debug(f"Query returned no results: {promql}")
                return 0.0

            # Expecting a scalar or a vector with one element
            value = result[0]["value"][1]
            return float(value)

        except requests.exceptions.Timeout:
            # Prometheus is slow/unavailable - fail fast and use fallback
            BlazeLogger.get_instance().debug(f"Prometheus query timeout (2s) - using fallback: {promql}")
            return 0.0
        except requests.exceptions.RequestException as e:
            BlazeLogger.get_instance().debug(f"Prometheus connection unavailable (using fallback): {e}")
            return 0.0
        except (KeyError, IndexError, ValueError) as e:
            BlazeLogger.get_instance().debug(f"Failed to parse Prometheus response: {e}")
            return 0.0

    def _query_first_scalar(self, promql_candidates: List[str]) -> Optional[float]:
        """Run a list of PromQL queries and return the first scalar result found.

        Uses the full query API so we can detect empty results vs. numeric 0.
        """
        for promql in promql_candidates:
            try:
                results = self.query(promql)
                if not results:
                    continue
                value = results[0].get("value", [None, None])[1]
                if value is None:
                    continue
                return float(value)
            except Exception:
                continue
        return self._get_node_property_static(node_id, key)

    def _battery_selectors(self, node_id: str) -> List[str]:
        """Build a list of label selectors to match battery metrics for a node."""
        selectors = []
        sel = self._selector(node_id)
        if sel:
            selectors.append(sel)

        node_label = None
        if sel.startswith('node="') and sel.endswith('"'):
            node_label = sel[len('node="'):-1]
        if node_label:
            selectors.append(f'instance=~"^{node_label}(:.*)?$"')
            selectors.append(f'nodename="{node_label}"')
            selectors.append(f'nodeName="{node_label}"')
        return selectors

    def _get_node_property_static(self, node_id: str, key: str) -> Optional[Any]:
        props = self.node_properties.get(node_id, {}) or {}
        if key in props:
            return props.get(key)
        # Accept a few alternate key names
        aliases = {
            "battery_capacity_ah": ["battery.capacity.ah", "capacity_ah"],
            "battery_voltage": ["battery.voltage", "voltage_v"],
            "battery_powered": ["battery.powered"],
        }
        for alias in aliases.get(key, []):
            if alias in props:
                return props.get(alias)
        return None

    def _get_battery_capacity_j(self, node_id: str, voltage_v: Optional[float]) -> Optional[float]:
        cap_ah = self._get_node_property_static(node_id, "battery_capacity_ah")
        if cap_ah is None:
            return None
        try:
            cap_ah = float(cap_ah)
        except Exception:
            return None
        if cap_ah <= 0:
            return None
        if voltage_v is None:
            voltage_v = self._get_node_property_static(node_id, "battery_voltage")
        try:
            voltage_v = float(voltage_v) if voltage_v is not None else None
        except Exception:
            voltage_v = None
        if not voltage_v or voltage_v <= 0:
            return None
        return cap_ah * voltage_v * 3600.0

    def _get_battery_voltage_v(self, node_id: str) -> Optional[float]:
        selectors = self._battery_selectors(node_id)
        power_supply_filter = 'power_supply=~"BAT.*|battery|CMB.*"'

        def _with_sel(metric: str, sel: str) -> str:
            if sel:
                return f'{metric}{{{sel}, {power_supply_filter}}}'
            return f'{metric}{{{power_supply_filter}}}'

        volt_queries = []
        for sel in selectors:
            volt_queries.append(f'max({"blazeaiot_node_battery_voltage"}{{{sel}}})')
            volt_queries.append(f'max({_with_sel("node_power_supply_voltage_now", sel)})')
            volt_queries.append(f'max({"upower_voltage"}{{{sel}}})')
        voltage_uv = self._query_first_scalar(volt_queries)
        if voltage_uv is None:
            static_v = self._get_node_property_static(node_id, "battery_voltage")
            try:
                return float(static_v) if static_v is not None else None
            except Exception:
                return None
        # blazeaiot_node_battery_voltage is expected in volts; node_exporter uses microvolts
        if voltage_uv > 1e3:
            return float(voltage_uv) * 1e-6
        return float(voltage_uv)

    def _get_battery_current_a(self, node_id: str) -> Optional[float]:
        selectors = self._battery_selectors(node_id)
        power_supply_filter = 'power_supply=~"BAT.*|battery|CMB.*"'

        def _with_sel(metric: str, sel: str) -> str:
            if sel:
                return f'{metric}{{{sel}, {power_supply_filter}}}'
            return f'{metric}{{{power_supply_filter}}}'

        current_queries = []
        for sel in selectors:
            current_queries.append(f'max({"blazeaiot_node_battery_current"}{{{sel}}})')
            current_queries.append(f'max({_with_sel("node_power_supply_current_now", sel)})')
            current_queries.append(f'max({"upower_current"}{{{sel}}})')
        current_ua = self._query_first_scalar(current_queries)
        if current_ua is None:
            return None
        # blazeaiot_node_battery_current is expected in amps; node_exporter uses microamps
        if current_ua > 1e3:
            return float(current_ua) * 1e-6
        return float(current_ua)

    def _get_battery_power_w(self, node_id: str) -> Optional[float]:
        selectors = self._battery_selectors(node_id)
        power_supply_filter = 'power_supply=~"BAT.*|battery|CMB.*"'

        def _with_sel(metric: str, sel: str) -> str:
            if sel:
                return f'{metric}{{{sel}, {power_supply_filter}}}'
            return f'{metric}{{{power_supply_filter}}}'

        power_queries = []
        for sel in selectors:
            power_queries.append(f'max({_with_sel("node_power_supply_power_now", sel)})')
            power_queries.append(f'max({"upower_power"}{{{sel}}})')
        power_uw = self._query_first_scalar(power_queries)
        if power_uw is not None:
            return float(power_uw) * 1e-6

        voltage_v = self._get_battery_voltage_v(node_id)
        current_a = self._get_battery_current_a(node_id)
        if voltage_v is None or current_a is None:
            return None
        return float(voltage_v) * float(current_a)

    def _get_battery_charge_ah_raw(self, node_id: str) -> Optional[float]:
        selectors = self._battery_selectors(node_id)
        power_supply_filter = 'power_supply=~"BAT.*|battery|CMB.*"'

        def _with_sel(metric: str, sel: str) -> str:
            if sel:
                return f'{metric}{{{sel}, {power_supply_filter}}}'
            return f'{metric}{{{power_supply_filter}}}'

        charge_queries = []
        for sel in selectors:
            charge_queries.append(f'max({_with_sel("node_power_supply_charge_now", sel)})')
            charge_queries.append(f'max({"upower_charge"}{{{sel}}})')
        charge_uah = self._query_first_scalar(charge_queries)
        if charge_uah is None:
            return None
        return float(charge_uah) * 1e-6

    def query(self, promql: str) -> list:
        """Run a PromQL query and return the full result list.

        Args:
            promql: Prometheus query string

        Returns:
            List of result dictionaries from Prometheus
        """
        api_url = f"{self.base_url}/api/v1/query"
        try:
            # Use shorter timeout (2s) to prevent hanging
            response = self.session.get(api_url, params={"query": promql}, timeout=2.0)
            response.raise_for_status()
            data = response.json()

            if data["status"] != "success":
                BlazeLogger.get_instance().debug(f"Prometheus query failed: {data.get('error', 'Unknown error')}")
                return []

            result = data.get("data", {}).get("result", [])
            return result

        except requests.exceptions.Timeout as e:
            BlazeLogger.get_instance().debug(f"Prometheus query timeout (2s): {promql[:100]}")
            return []
        except requests.exceptions.RequestException as e:
            BlazeLogger.get_instance().debug(f"Failed to connect to Prometheus: {e}. Attempting to re-establish access...")
            self._reestablish_access()
            # Retry once
            try:
                response = self.session.get(api_url, params={"query": promql}, timeout=2.0)
                response.raise_for_status()
                data = response.json()
                if data.get("status") == "success":
                    return data.get("data", {}).get("result", [])
            except Exception:
                pass
            return []
        except (KeyError, ValueError) as e:
            BlazeLogger.get_instance().debug(f"Failed to parse Prometheus response: {e}")
            return []

    def _query_at_url(self, promql: str, base_url: str) -> list:
        """Run a PromQL query against a specific Prometheus base URL."""
        api_url = f"{base_url.rstrip('/')}/api/v1/query"
        try:
            response = self.session.get(api_url, params={"query": promql}, timeout=2.0)
            response.raise_for_status()
            data = response.json()
            if data.get("status") != "success":
                return []
            return data.get("data", {}).get("result", [])
        except Exception:
            return []

    def get_node_capacity(self, node_id: str) -> Dict[str, float]:
        sel = self._selector(node_id)
        # Number of CPU cores
        cpu = self._query(f'count(node_cpu_seconds_total{{{sel}, mode="idle"}})')
        # Total memory in GiB
        ram = self._query(f"node_memory_MemTotal_bytes{{{sel}}}") / (1024**3)
        # Total GPU memory in GiB (assumes NVIDIA DCGM exporter)
        gpu = self._query(f'sum(node_nvsmi_memory_total_bytes{{{sel}}}) by (instance)') / (1024**3)
        # Fallback to static capacities if Prometheus returns 0
        props = self.node_properties.get(node_id, {}) or {}
        if float(cpu) <= 0.0 and props.get("cpu") is not None:
            try:
                cpu = float(props.get("cpu"))
            except Exception:
                pass
        if float(ram) <= 0.0 and props.get("ram_gb") is not None:
            try:
                ram = float(props.get("ram_gb"))
            except Exception:
                pass
        if float(gpu) <= 0.0 and props.get("gpu_gb") is not None:
            try:
                gpu = float(props.get("gpu_gb"))
            except Exception:
                pass
        return {"cpu": float(cpu), "ram_gb": float(ram), "gpu_gb": float(gpu)}

    def get_node_available(self, node_id: str) -> Dict[str, float]:
        """Get available node resources.
        
        Queries Prometheus with short timeout (2s) to fail fast if unavailable,
        then falls back to static properties from config.
        """
        sel = self._selector(node_id)
        # Available CPU cores (1 - usage) * capacity
        cpu_usage = self._query(f'1 - avg(rate(node_cpu_seconds_total{{{sel}, mode="idle"}}[5m]))')
        cpu_capacity = self.get_node_capacity(node_id)["cpu"]
        cpu_avail = (1 - cpu_usage) * cpu_capacity
        # Available memory in GiB
        ram_avail = self._query(f"node_memory_MemAvailable_bytes{{{sel}}}") / (1024**3)
        # Available GPU memory in GiB (assumes NVIDIA DCGM exporter)
        gpu_free = self._query(f'sum(node_nvsmi_memory_free_bytes{{{sel}}}) by (instance)') / (1024**3)
        # Fallback to static capacities if Prometheus returns 0
        props = self.node_properties.get(node_id, {}) or {}
        if float(cpu_avail) <= 0.0 and props.get("cpu") is not None:
            try:
                cpu_avail = float(props.get("cpu"))
            except Exception:
                pass
        if float(ram_avail) <= 0.0 and props.get("ram_gb") is not None:
            try:
                ram_avail = float(props.get("ram_gb"))
            except Exception:
                pass
        if float(gpu_free) <= 0.0 and props.get("gpu_gb") is not None:
            try:
                gpu_free = float(props.get("gpu_gb"))
            except Exception:
                pass
        return {"cpu": float(cpu_avail), "ram_gb": float(ram_avail), "gpu_gb": float(gpu_free)}

    def _reestablish_access(self):
        """Attempt to re-establish Prometheus access by re-running setup for clusters.

        This handles cases where kubectl port-forward dropped or URLs changed.
        Updates `base_url` and per-layer `cluster_urls` when successful.
        """
        try:
            # Try each known cluster context
            for cluster_name in list(self.cluster_contexts.keys()):
                url = self.setup_prometheus_access(cluster_name)
                if url:
                    # Update base_url to a working endpoint, prefer first success
                    self.base_url = url
                    # Also update per-layer mapping
                    layer_id = self.cluster_name_to_layer.get(cluster_name)
                    if layer_id:
                        self.cluster_urls[layer_id] = url
                    # If we found any working URL, we can stop early
                    break
        except Exception as e:
            BlazeLogger.get_instance().debug(f"Re-establishing Prometheus access failed: {e}")

    def get_node_property(self, node_id: str, key: str) -> Optional[Any]:
        # Properties could be provided via labels or a side-channel config.
        # Example: query for a specific label on a node
        # promql = f'kube_node_labels{{{self._selector(node_id)}, label_{key}=~".+"}}'
        # You would need to parse the label value from the result.

        # Battery metrics via Prometheus (node_exporter / upower-exporter style)
        battery_keys = {
            "battery_percent",
            "battery_level",
            "battery_capacity_percent",
            "battery_energy_j",
            "battery_remaining_j",
            "energy_level",
            "battery_charge_ah",
            "battery_voltage_v",
            "battery_current_a",
            "battery_power_w",
            "battery_present",
        }

        if key in battery_keys:
            selectors = self._battery_selectors(node_id)
            power_supply_filter = 'power_supply=~"BAT.*|battery|CMB.*"'

            def _with_sel(metric: str, sel: str) -> str:
                if sel:
                    return f'{metric}{{{sel}, {power_supply_filter}}}'
                return f'{metric}{{{power_supply_filter}}}'

            # Battery percentage
            if key in {"battery_percent", "battery_level", "battery_capacity_percent"}:
                percent_queries = []
                for sel in selectors:
                    percent_queries.append(f'max({_with_sel("node_power_supply_capacity", sel)})')
                    percent_queries.append(f'max({_with_sel("node_power_supply_charge_percent", sel)})')
                    percent_queries.append(f'max({"upower_percentage"}{{{sel}}})')
                percent = self._query_first_scalar(percent_queries)
                if percent is not None:
                    return percent

            # Battery present (0/1)
            if key == "battery_present":
                present_queries = []
                for sel in selectors:
                    present_queries.append(f'max({_with_sel("node_power_supply_present", sel)})')
                    present_queries.append(f'max({"upower_is_present"}{{{sel}}})')
                present = self._query_first_scalar(present_queries)
                return present if present is not None else None

            # Battery voltage in V
            if key == "battery_voltage_v":
                return self._get_battery_voltage_v(node_id)

            # Battery current in A
            if key == "battery_current_a":
                return self._get_battery_current_a(node_id)

            # Battery power in W
            if key == "battery_power_w":
                return self._get_battery_power_w(node_id)

            # Battery charge in Ah
            if key == "battery_charge_ah":
                charge_ah = self._get_battery_charge_ah_raw(node_id)
                if charge_ah is not None:
                    return charge_ah

            # Battery energy in J (remaining)
            if key in {"battery_energy_j", "battery_remaining_j", "energy_level"}:
                energy_queries = []
                for sel in selectors:
                    energy_queries.append(f'max({_with_sel("node_power_supply_energy_now", sel)})')
                    energy_queries.append(f'max({"upower_energy"}{{{sel}}})')
                energy_uwh = self._query_first_scalar(energy_queries)
                if energy_uwh is not None:
                    # node_exporter uses micro-watt-hours (uWh) => J
                    return float(energy_uwh) * 1e-6 * 3600.0

                # Fallback: compute from charge_now * voltage_now
                charge_ah = self._get_battery_charge_ah_raw(node_id)
                voltage_v = self.get_node_property(node_id, "battery_voltage_v")
                if charge_ah is not None and voltage_v is not None:
                    return float(charge_ah) * float(voltage_v) * 3600.0

                # Fallback: estimate from power draw over time using capacity
                capacity_j = self._get_battery_capacity_j(node_id, voltage_v)
                if capacity_j is None:
                    return None

                now = time.time()
                state = self._battery_state.get(node_id)
                if not state:
                    state = {
                        "capacity_j": capacity_j,
                        "remaining_j": capacity_j,
                        "last_ts": now,
                    }
                    self._battery_state[node_id] = state
                    return state["remaining_j"]

                # Update capacity if changed
                if capacity_j != state.get("capacity_j") and capacity_j > 0:
                    state["capacity_j"] = capacity_j
                    state["remaining_j"] = min(state.get("remaining_j", capacity_j), capacity_j)

                power_w = self._get_battery_power_w(node_id)
                if power_w is None:
                    state["last_ts"] = now
                    return state.get("remaining_j")

                dt = max(0.0, now - float(state.get("last_ts", now)))
                delta_j = float(power_w) * dt
                remaining = float(state.get("remaining_j", capacity_j)) - delta_j
                remaining = min(max(0.0, remaining), float(state.get("capacity_j", capacity_j)))
                state["remaining_j"] = remaining
                state["last_ts"] = now
                return remaining

            # Battery percent from estimated remaining energy
            if key in {"battery_percent", "battery_level", "battery_capacity_percent"}:
                voltage_v = self.get_node_property(node_id, "battery_voltage_v")
                capacity_j = self._get_battery_capacity_j(node_id, voltage_v)
                if capacity_j is None:
                    return None
                remaining_j = self.get_node_property(node_id, "battery_remaining_j")
                if remaining_j is None:
                    return None
                try:
                    return float(remaining_j) / float(capacity_j) * 100.0
                except Exception:
                    return None

            # Battery charge from remaining energy
            if key == "battery_charge_ah":
                remaining_j = self.get_node_property(node_id, "battery_remaining_j")
                voltage_v = self.get_node_property(node_id, "battery_voltage_v")
                if remaining_j is None or voltage_v is None or float(voltage_v) <= 0:
                    return None
                return float(remaining_j) / (float(voltage_v) * 3600.0)

        return None

    def get_latency_matrix(self, nodes: list) -> Any:
        """Query Prometheus for network latency between nodes with caching.

        Queries ping_latency metric with labels:
          - nodeName: origin node of the ping
          - sourceId: destination in format "<layer_name:node_name>"

        Uses caching (5 second TTL) to avoid N×N queries every call.

        Args:
            nodes: List of Node objects

        Returns:
            2D numpy array where latency_matrix[i][j] is latency from node i to node j in seconds
        """
        import numpy as np
        import time

        # Check cache (5 second TTL)
        cache_key = tuple(sorted([n.node_id for n in nodes]))
        current_time = time.time()

        if not hasattr(self, '_latency_cache'):
            self._latency_cache = {}
            self._latency_cache_time = {}

        if cache_key in self._latency_cache:
            cache_age = current_time - self._latency_cache_time.get(cache_key, 0)
            if cache_age < 5.0:  # 5 second TTL
                BlazeLogger.get_instance().debug(f"Using cached latency matrix (age: {cache_age:.1f}s)")
                return self._latency_cache[cache_key]

        n_nodes = len(nodes)
        latency_matrix = np.ones((n_nodes, n_nodes)) * 0.01  # Default 10ms

        # Build node to layer mapping
        node_to_layer = {}
        for node in nodes:
            layer_id = getattr(node, 'layer_id', None)
            if not layer_id and hasattr(node, 'node_id') and '-' in node.node_id:
                layer_id = node.node_id.split('-')[0]
            node_to_layer[node.node_id] = layer_id or 'unknown'

        # Try to query Prometheus for ping latency metrics
        # Use timeout to prevent hanging on slow queries
        try:
            BlazeLogger.get_instance().debug(f"Querying Prometheus for latency matrix ({n_nodes}×{n_nodes} = {n_nodes*n_nodes} metrics)")

            # Try a single batch query first (much faster than N×N individual queries)
            try:
                batch_query = 'ping_latency'

                # Query all cluster Prometheus endpoints if configured
                results = []
                if self.cluster_urls:
                    for layer_id, url in self.cluster_urls.items():
                        layer_results = self._query_at_url(batch_query, url)
                        layer_id_norm = str(layer_id).lower() if layer_id else None
                        for result in layer_results:
                            metric = result.get('metric', {})
                            node_name = metric.get('nodeName')
                            if not node_name:
                                continue
                            node_layer = node_to_layer.get(node_name, '').lower()
                            # Only accept results from the matching layer cluster
                            if not layer_id_norm or node_layer == layer_id_norm:
                                results.append(result)
                else:
                    results = self.query(batch_query)

                if results:
                    BlazeLogger.get_instance().debug(f"Batch query returned {len(results)} latency metrics")
                    # Parse batch results
                    for result in results:
                        metric = result.get('metric', {})
                        node_name = metric.get('nodeName')
                        source_id = metric.get('sourceId')

                        if node_name and source_id:
                            # Find node indices
                            i = next((idx for idx, n in enumerate(nodes) if n.node_id == node_name), None)
                            if i is not None and ':' in source_id:
                                source_node = source_id.split(':', 1)[1]
                                j = next((idx for idx, n in enumerate(nodes) if n.node_id == source_node), None)
                                if j is not None:
                                    raw_value = float(result.get('value', [0, 0.01])[1])
                                    # ping_latency is in milliseconds -> convert to seconds
                                    latency_value = raw_value / 1_000.0
                                    latency_matrix[i][j] = latency_value
            except Exception as e:
                BlazeLogger.get_instance().debug(f"Batch query failed, using defaults: {e}")

        except Exception as e:
            BlazeLogger.get_instance().warning(f"Latency query failed, using layer-based defaults: {e}")

        # Fill missing values with 0.0 (no hardcoded defaults)
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i == j:
                    latency_matrix[i][j] = 0.0
                elif latency_matrix[i][j] == 0.01:  # Still at default
                    latency_matrix[i][j] = 0.0

        # Cache the result
        self._latency_cache[cache_key] = latency_matrix
        self._latency_cache_time[cache_key] = current_time

        BlazeLogger.get_instance().debug(f"Latency matrix cached for {n_nodes} nodes")
        return latency_matrix

    def export_latency_matrix_to_xml(self, nodes: list, output_path: str, metadata: Optional[dict] = None) -> str:
        """
        Export current latency matrix to XML file for comparison analysis.

        Args:
            nodes: List of Node objects
            output_path: Path where XML file will be written
            metadata: Optional dict with additional metadata

        Returns:
            Full path to written XML file
        """
        from .latency_matrix_exporter import LatencyMatrixExporter

        latency_matrix = self.get_latency_matrix(nodes)
        meta = metadata or {}
        meta["environment"] = "prometheus"
        meta["prometheus_url"] = self.base_url

        return LatencyMatrixExporter.export_to_xml(
            latency_matrix, nodes, output_path, source="prometheus", metadata=meta
        )
