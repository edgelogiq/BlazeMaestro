"""Applies metrics to XML elements."""
from typing import Dict, Optional
import xml.etree.ElementTree as ET

from BlazeAIoT.Core.Utils import BlazeLogger


class MetricsApplier:
    """Applies metrics to XML elements."""
    
    def apply_to_container(self, container: ET.Element, metrics: Dict) -> bool:
        """Apply CPU/RAM/exec metrics to container."""
        changed = False
        for attr, key, fmt in [
            ('cpu', 'cpu_usage', '.2f'),
            ('ram_gb', 'ram_usage_gb', '.2f'),
            ('exec_time', 'exec_time', '.4f'),
            ('execution_rate_hz', 'execution_rate_hz', '.2f')
        ]:
            if key in metrics:
                val = float(metrics[key]) * (1.2 if 'usage' in key else 1.0)  # 20% headroom
                new = f"{val:{fmt}}"
                old = container.get(attr, '')
                if new != old:
                    container.set(attr, new)
                    changed = True
        return changed
    
    def apply_to_instance(self, instance: ET.Element, pod_node_metrics: Dict, service_id: str, 
                         pod_name: str, node_name: str, fallback: Optional[Dict] = None) -> bool:
        """Apply metrics from lookup to instance."""
        logger = BlazeLogger.get_instance()
        changed = False
        metrics = None
        if pod_name and node_name:
            metrics = pod_node_metrics.get((pod_name, node_name))
        if not metrics and fallback is not None:
            metrics = fallback
        
        if not metrics:
            return False
        
        # CPU
        cpu = metrics.get('cpu_usage', 0.0)
        if cpu > 0:
            new_cpu = f"{cpu:.4f}"
            if instance.get('cpu_usage_last') != new_cpu:
                self._update_min_max_last(instance, 'cpu_usage', cpu)
                changed = True
        
        # RAM
        ram = metrics.get('ram_usage_gb', 0.0)
        if ram > 0:
            new_ram = f"{ram:.4f}"
            if instance.get('ram_usage_last_gb') != new_ram:
                self._update_min_max_last(instance, 'ram_usage', ram, suffix='_gb')
                changed = True
        
        # Exec metrics
        for key, attr, fmt in [
            ('exec_time', 'exec_time', '.6f'),
            ('execution_rate_hz', 'execution_rate_hz', '.2f')
        ]:
            val = metrics.get(key) or metrics.get(f'workload_level_{key}')
            if val and float(val) > 0:
                new_val = f"{float(val):{fmt}}"
                if instance.get(attr) != new_val:
                    instance.set(attr, new_val)
                    changed = True
        
        return changed
    
    def _update_min_max_last(self, elem: ET.Element, base: str, val: float, suffix: str = ''):
        """Update min/max/last metrics."""
        min_key, max_key, last_key = f'{base}_min{suffix}', f'{base}_max{suffix}', f'{base}_last{suffix}'
        is_uninit = elem.get(min_key) is None or (float(elem.get(min_key, 0)) == 0 and float(elem.get(max_key, 0)) == 0)
        
        if is_uninit:
            elem.set(min_key, f'{val:.3f}')
            elem.set(max_key, f'{val:.3f}')
        else:
            elem.set(min_key, f'{min(float(elem.get(min_key, 0)), val):.3f}')
            elem.set(max_key, f'{max(float(elem.get(max_key, 0)), val):.3f}')
        elem.set(last_key, f'{val:.3f}')
