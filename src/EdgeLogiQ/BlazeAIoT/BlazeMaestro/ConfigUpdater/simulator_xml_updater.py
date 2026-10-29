"""
Simulator XML updater: writes per-instance CPU/RAM usage metrics into services_update.xml
based on current template container specs and any observed in-memory usage.

This mimics Prometheus updater behavior for simulator runs.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional, Dict
import os
import statistics
from dataclasses import dataclass

from .base_xml_updater import BaseXMLUpdater


@dataclass
class _Baseline:
    cpu_avg: Optional[float]
    ram_avg: Optional[float]
    cpu_by_host: Dict[str, float]
    ram_by_host: Dict[str, float]


class SimulatorXMLUpdater(BaseXMLUpdater):
    def __init__(self, metrics_facade=None, nodes=None, services=None, current_allocation: Optional[dict] = None, output_dir: str = None):
        """
        Initialize simulator XML updater.
        
        Args:
            metrics_facade: SimulatorMetricsFacade instance for accessing node metrics
            nodes: List of SimPy nodes from simulation
            services: Dict of service_id -> ServiceSimulator for accessing service metrics
            output_dir: Optional output directory for CSV files
        """
        super().__init__()
        self.metrics_facade = metrics_facade
        self.nodes = nodes or []
        self.services = services or []
        self.current_allocation = current_allocation or {}
        self.output_dir = output_dir or 'out'
        
        # Track iteration for latency matrix export
        self._iteration_counter = 0

    def update_instance_metrics(self, xml_path: str = None, namespace: str = 'default', 
                               overrides: dict = None, metrics_provider=None, nodes=None, 
                               services=None, current_allocation: dict = None, **kwargs) -> bool:
        """
        Update simulator metrics in a mode-agnostic way.
        
        Can be called with either:
        1. Old API: update_instance_metrics(xml_path, overrides={...})
        2. New API: update_instance_metrics(metrics_provider=..., nodes=..., services=..., current_allocation=...)
        
        Args:
            xml_path: Path to services XML file (old API)
            namespace: Kubernetes namespace (ignored in simulator)
            overrides: Service metrics overrides (old API)
            metrics_provider: Metrics facade (new API, updates self.metrics_facade)
            nodes: List of nodes (new API, updates self.nodes)
            services: List/dict of services (new API, updates self.services)
            current_allocation: Service-to-node mapping (new API, updates self.current_allocation)
            **kwargs: Additional arguments
            
        Returns:
            True if successful, False otherwise
        """
        
        # Support both old and new API
        if metrics_provider is not None:
            self.metrics_facade = metrics_provider
        if nodes is not None:
            self.nodes = nodes
        if services is not None:
            self.services = services
        if current_allocation is not None:
            self.current_allocation = current_allocation
        
        # If called with new API, need to determine XML path
        if xml_path is None:
            # Try to infer from services list or use default
            xml_path = kwargs.get('services_xml_path', 'services_update.xml')
        
        # Ensure we write to _update.xml, not the source file
        source_path = xml_path
        if xml_path.endswith('_update.xml'):
            source_path = xml_path.replace('_update.xml', '.xml')
        else:
            xml_path = xml_path.replace('.xml', '_update.xml')
        
        # Create update file from source if it doesn't exist
        import os
        import shutil
        if not os.path.exists(xml_path) and os.path.exists(source_path):
            shutil.copy2(source_path, xml_path)
        
        # Now call original implementation
        result = self._update_instance_metrics_impl(xml_path, overrides=overrides)
        
        # Export latency matrix after metrics update (if metrics_provider available)
        if result and self.metrics_facade and self.nodes:
            self._iteration_counter += 1
            self._export_latency_matrix_if_enabled(self._iteration_counter)
        
        return result
    
    def _update_instance_metrics_impl(self, xml_path: str, namespace: str = 'default', overrides: dict | None = None) -> bool:
        """
        Populate <instances>/<container> entries with cpu_usage_* and ram_usage_* attributes.
        Strategy:
        - Apply measured overrides from services (current iteration demand)
        - Initialize min/max/last from template if not present
        - Min/max track historical range, last tracks current measured usage
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            updated = 0

            for service in root.findall('service'):
                instances = service.find('instances')
                containers_elem = service.find('containers')
                if instances is None or containers_elem is None:
                    continue

                # Build template map for defaults
                tmpl_by_id = {}
                for tmpl in containers_elem.findall('container'):
                    cid = tmpl.get('id')
                    if cid:
                        tmpl_by_id[cid] = {
                            'cpu': float(tmpl.get('cpu', '0.0')),
                            'ram_gb': float(tmpl.get('ram_gb', '0.0'))
                        }

                # Update each instance container
                for c in instances.findall('container'):
                    cid = c.get('id')
                    if not cid:
                        continue
                    tmpl = tmpl_by_id.get(cid, {'cpu': 0.0, 'ram_gb': 0.0})
                    # Use overrides when present for measured last values: {service_id: {container_id: {..}}}
                    svc_id = service.get('id') or service.get('service_id') or ''
                    ov = overrides.get(svc_id, {}).get(cid, {}) if isinstance(overrides, dict) else {}

                    # Read existing metrics
                    cpu_min = c.get('cpu_usage_min')
                    cpu_max = c.get('cpu_usage_max')
                    cpu_last = c.get('cpu_usage_last')
                    ram_min = c.get('ram_usage_min_gb')
                    ram_max = c.get('ram_usage_max_gb')
                    ram_last = c.get('ram_usage_last_gb')
                    gpu_min = c.get('gpu_usage_min_gb')
                    gpu_max = c.get('gpu_usage_max_gb')
                    gpu_last = c.get('gpu_usage_last_gb')

                    # Initialize if missing
                    def _f(v: Optional[str]) -> Optional[float]:
                        try:
                            return float(v) if v is not None else None
                        except Exception:
                            return None

                    cpu_last_val = _f(cpu_last)
                    ram_last_val = _f(ram_last)
                    gpu_last_val = _f(gpu_last)
                    
                    # Apply measured overrides (from service simulator this iteration)
                    if ov.get('cpu_usage_last') is not None:
                        cpu_last_val = float(ov['cpu_usage_last'])
                    if ov.get('ram_usage_last_gb') is not None:
                        ram_last_val = float(ov['ram_usage_last_gb'])
                    if ov.get('gpu_usage_last_gb') is not None:
                        gpu_last_val = float(ov['gpu_usage_last_gb'])
                    
                    # If still no measured value, use template as baseline
                    if cpu_last_val is None:
                        cpu_last_val = float(tmpl['cpu'])
                    if ram_last_val is None:
                        ram_last_val = float(tmpl['ram_gb'])
                    if gpu_last_val is None:
                        gpu_last_val = 0.0

                    # Parse existing min/max or initialize from current last
                    cpu_min_val = _f(cpu_min)
                    cpu_max_val = _f(cpu_max)
                    ram_min_val = _f(ram_min)
                    ram_max_val = _f(ram_max)
                    gpu_min_val = _f(gpu_min)
                    gpu_max_val = _f(gpu_max)

                    if cpu_min_val is None:
                        cpu_min_val = cpu_last_val
                    else:
                        cpu_min_val = min(cpu_min_val, cpu_last_val)
                    if cpu_max_val is None:
                        cpu_max_val = cpu_last_val
                    else:
                        cpu_max_val = max(cpu_max_val, cpu_last_val)
                        
                    if ram_min_val is None:
                        ram_min_val = ram_last_val
                    else:
                        ram_min_val = min(ram_min_val, ram_last_val)
                    if ram_max_val is None:
                        ram_max_val = ram_last_val
                    else:
                        ram_max_val = max(ram_max_val, ram_last_val)
                        
                    if gpu_min_val is None:
                        gpu_min_val = gpu_last_val
                    else:
                        gpu_min_val = min(gpu_min_val, gpu_last_val)
                    if gpu_max_val is None:
                        gpu_max_val = gpu_last_val
                    else:
                        gpu_max_val = max(gpu_max_val, gpu_last_val)

                    # Write back updated metrics
                    c.set('cpu_usage_min', f"{cpu_min_val:.3f}")
                    c.set('cpu_usage_max', f"{cpu_max_val:.3f}")
                    c.set('cpu_usage_last', f"{cpu_last_val:.3f}")
                    c.set('ram_usage_min_gb', f"{ram_min_val:.3f}")
                    c.set('ram_usage_max_gb', f"{ram_max_val:.3f}")
                    c.set('ram_usage_last_gb', f"{ram_last_val:.3f}")
                    c.set('gpu_usage_min_gb', f"{gpu_min_val:.3f}")
                    c.set('gpu_usage_max_gb', f"{gpu_max_val:.3f}")
                    c.set('gpu_usage_last_gb', f"{gpu_last_val:.3f}")
                    updated += 1

            if updated > 0:
                self._remove_empty_instances_blocks(root)
                self._indent_xml(root)
                tree.write(xml_path, encoding='utf-8', xml_declaration=True)
                return True
            return False
        except Exception:
            return False

    # Abstract method implementations - collect from simulation
    def get_node_metrics(self) -> Dict[str, Dict[str, float]]:
        """Collect node metrics from SimPy simulation via metrics facade.
        
        Computes per-node CPU/RAM used as:
          used = measured_service_usage + idle_overhead
        where:
          measured_service_usage = sum of all service containers on that node (from service.get_measured_usage())
          idle_overhead = per-node background overhead from real platform baseline
        """
        if not self.metrics_facade or not self.nodes:
            return {}

        # Load per-host idle overhead from real platform.xml (base file)
        baseline = self._load_real_baseline(self.metrics_facade.platform_xml_path if hasattr(self.metrics_facade, 'platform_xml_path') else None)
        
        # Aggregate measured service usage per node from all allocated services
        service_usage_cpu: Dict[str, float] = {}
        service_usage_ram: Dict[str, float] = {}
        service_usage_gpu: Dict[str, float] = {}
        try:
            for svc in (self.services or []):
                sid = getattr(svc, 'service_id', None) or getattr(svc, 'id', None)
                if not sid:
                    continue
                assigned_node = self.current_allocation.get(str(sid)) or getattr(svc, 'locked_node_id', None)
                if not assigned_node:
                    continue

                # Get measured usage from service if available
                if hasattr(svc, 'get_measured_usage'):
                    try:
                        usage = svc.get_measured_usage() or {}
                        cpu_used = 0.0
                        ram_used = 0.0
                        gpu_used = 0.0
                        # Sum across all containers in this service
                        for cid, container_usage in usage.items():
                            if isinstance(container_usage, dict):
                                cpu_used += float(container_usage.get('cpu_usage_last', 0.0))
                                ram_used += float(container_usage.get('ram_usage_last_gb', 0.0))
                                gpu_used += float(container_usage.get('gpu_usage_last_gb', 0.0))
                        service_usage_cpu[assigned_node] = service_usage_cpu.get(assigned_node, 0.0) + cpu_used
                        service_usage_ram[assigned_node] = service_usage_ram.get(assigned_node, 0.0) + ram_used
                        service_usage_gpu[assigned_node] = service_usage_gpu.get(assigned_node, 0.0) + gpu_used
                    except Exception:
                        pass
        except Exception:
            pass

        metrics = {}
        for node in self.nodes:
            node_id = getattr(node, 'node_id', str(node))
            try:
                capacity = self.metrics_facade.get_node_capacity(node_id)
                
                # Total used = service usage + idle overhead (real baseline)
                cpu_used = service_usage_cpu.get(node_id, 0.0)
                ram_used = service_usage_ram.get(node_id, 0.0)
                gpu_used = service_usage_gpu.get(node_id, 0.0)
                
                # Add per-node idle baseline with realistic per-iteration variation (±5% jitter)
                import random
                idle_jitter = random.uniform(0.95, 1.05)
                idle_cpu = (baseline.cpu_by_host.get(node_id, baseline.cpu_avg or 0.1)) * idle_jitter
                idle_ram = (baseline.ram_by_host.get(node_id, baseline.ram_avg or 0.2)) * idle_jitter
                
                cpu_used = cpu_used + idle_cpu
                ram_used = ram_used + idle_ram

                metrics[node_id] = {
                    'cpu_cores': capacity.get('cpu', 0.0),
                    'ram_gb': capacity.get('ram_gb', 0.0),
                    'cpu_used': max(0.0, cpu_used),
                    'ram_used_gb': max(0.0, ram_used),
                }
                
                # Add GPU if present
                if capacity.get('gpu_gb', 0.0) > 0:
                    metrics[node_id]['gpu_count'] = capacity['gpu_gb']
                    metrics[node_id]['gpu_used_gb'] = max(0.0, gpu_used)
                
                # Add layer info if available
                layer = self.metrics_facade.get_node_property(node_id, 'layer')
                if layer:
                    metrics[node_id]['layer'] = layer
                
            except Exception:
                continue
        
        return metrics

    def get_service_metrics(self, namespace: str = 'default') -> Dict[str, Dict[str, float]]:
        """Collect service metrics from ServiceSimulator instances."""
        if not self.services:
            return {}
        
        metrics = {}
        for service_id, service in self.services.items():
            try:
                # Get measured usage from the service simulator
                measured = getattr(service, 'get_measured_usage', lambda: {})()
                
                if not measured:
                    # Fall back to template specs if no measured data
                    containers = getattr(service, 'containers', [])
                    if containers:
                        # Average across all containers
                        total_cpu = sum(float(getattr(c, 'cpu', 0.0)) for c in containers)
                        total_ram = sum(float(getattr(c, 'ram_gb', 0.0)) for c in containers)
                        metrics[service_id] = {
                            'cpu_usage': total_cpu / len(containers),
                            'ram_usage_gb': total_ram / len(containers),
                        }
                else:
                    # Use measured values from simulation
                    cpu_values = [v for k, v in measured.items() if 'cpu' in k.lower()]
                    ram_values = [v for k, v in measured.items() if 'ram' in k.lower()]
                    
                    if cpu_values:
                        metrics[service_id] = {
                            'cpu_usage': sum(cpu_values) / len(cpu_values),
                        }
                    if ram_values:
                        metrics[service_id]['ram_usage_gb'] = sum(ram_values) / len(ram_values)
                
            except Exception:
                continue
        
        return metrics

    def update_platform_xml(self, xml_path: str, backup: bool = True) -> bool:
        """
        Update platform.xml with node metrics from simulation.
        Primarily uses update_instance_metrics for simulator workflows.
        """
        # Could be implemented if needed, but typically simulator doesn't modify platform.xml
        return False

    def update_services_xml(self, xml_path: str, namespace: str = 'default', backup: bool = True) -> bool:
        """
        Update services.xml with service metrics from simulation.
        Simulator primarily uses update_instance_metrics for per-instance tracking.
        """
        # Could be implemented if needed, but typically simulator uses update_instance_metrics
        return False

    # _indent_xml inherited from BaseXMLUpdater

    def _load_real_baseline(self, services_xml_path: Optional[str]) -> _Baseline:
        """Load per-host and average cpu_used/ram_used_gb from platform.xml (base file, not _update).

        Falls back to src/config/platform.xml when no path is provided or file missing.
        """
        try:
            candidate_paths = []
            if services_xml_path:
                # Always use base platform.xml, not platform_update.xml
                candidate_paths.append(os.path.join(os.path.dirname(services_xml_path), 'platform.xml'))
            # Fallback to repo config default (../../../../config from ConfigUpdater dir)
            candidate_paths.append(
                os.path.abspath(
                    os.path.join(os.path.dirname(__file__), '../../../../config/platform.xml')
                )
            )

            platform_xml = next((p for p in candidate_paths if os.path.exists(p)), None)
            if not platform_xml:
                return _Baseline(None, None, {}, {})

            tree = ET.parse(platform_xml)
            root = tree.getroot()
            cpu_vals = []
            ram_vals = []
            cpu_by_host: Dict[str, float] = {}
            ram_by_host: Dict[str, float] = {}
            for host in root.findall('.//host'):
                host_id = host.get('id') or ''
                cpu_prop = host.find(".//prop[@id='debug.cpu_used']")
                ram_prop = host.find(".//prop[@id='debug.ram_used_gb']")
                if cpu_prop is not None:
                    try:
                        cpu_val = float(cpu_prop.get('value', '0') or 0)
                        cpu_vals.append(cpu_val)
                        cpu_by_host[host_id] = cpu_val
                    except Exception:
                        pass
                if ram_prop is not None:
                    try:
                        ram_val = float(ram_prop.get('value', '0') or 0)
                        ram_vals.append(ram_val)
                        ram_by_host[host_id] = ram_val
                    except Exception:
                        pass

            cpu_avg = statistics.mean(cpu_vals) if cpu_vals else None
            ram_avg = statistics.mean(ram_vals) if ram_vals else None
            return _Baseline(cpu_avg, ram_avg, cpu_by_host, ram_by_host)
        except Exception:
            return _Baseline(None, None, {}, {})
    
    def _export_latency_matrix_if_enabled(self, iteration=None) -> None:
        """Export latency matrix to CSV after metrics update."""
        try:
            from ..Metrics.latency_matrix_exporter import LatencyMatrixExporter
            import os
            
            if not self.metrics_facade or not self.nodes:
                return
            
            # Create output directory (use timestamped dir from csv_logger if available)
            output_dir = self.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Use fixed filename (append mode) instead of timestamp
            csv_path = f"{output_dir}/latency_matrix_simulator.csv"
            
            try:
                latency_matrix = self.metrics_facade.get_latency_matrix(self.nodes)
                LatencyMatrixExporter.export_to_csv(
                    latency_matrix=latency_matrix,
                    nodes=self.nodes,
                    output_path=csv_path,
                    source='simulator',
                    iteration=iteration,
                    metadata={'mode': 'simulator'}
                )
            except Exception as e:
                from BlazeAIoT.Core.Utils import BlazeLogger
                BlazeLogger.get_instance().debug(f"Latency matrix export failed: {e}")
        except Exception:
            pass
