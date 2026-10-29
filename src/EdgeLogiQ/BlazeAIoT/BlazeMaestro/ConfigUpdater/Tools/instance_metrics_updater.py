"""Handles instance metric updates from Prometheus."""
import xml.etree.ElementTree as ET
from typing import Dict

from BlazeAIoT.Core.Utils import BlazeLogger


class InstanceMetricsUpdater:
    """Handles updating instance metrics from Prometheus data."""
    
    def __init__(self, base_updater):
        self.base_updater = base_updater
    
    def update_cpu_metric(self, container: ET.Element, cpu_current: float, service_id: str, pod_name: str) -> bool:
        """Update CPU metrics. Returns True if changed."""
        logger = BlazeLogger.get_instance()
        
        cpu_min = container.get('cpu_usage_min')
        cpu_max = container.get('cpu_usage_max')
        is_uninitialized = cpu_min is None or (float(cpu_min or 0) == 0.0 and float(cpu_max or 0) == 0.0)
        
        if is_uninitialized:
            container.set('cpu_usage_min', f'{cpu_current:.3f}')
            container.set('cpu_usage_max', f'{cpu_current:.3f}')
            container.set('cpu_usage_last', f'{cpu_current:.3f}')
            return True
        else:
            cpu_min_val = min(float(cpu_min), cpu_current)
            cpu_max_val = max(float(cpu_max), cpu_current)
            old_last = container.get('cpu_usage_last')
            container.set('cpu_usage_min', f'{cpu_min_val:.3f}')
            container.set('cpu_usage_max', f'{cpu_max_val:.3f}')
            container.set('cpu_usage_last', f'{cpu_current:.3f}')
            return True
    
    def update_ram_metric(self, container: ET.Element, ram_current: float, service_id: str, pod_name: str) -> bool:
        """Update RAM metrics. Returns True if changed."""
        logger = BlazeLogger.get_instance()
        
        ram_min = container.get('ram_usage_min_gb')
        ram_max = container.get('ram_usage_max_gb')
        is_uninitialized = ram_min is None or (float(ram_min or 0) == 0.0 and float(ram_max or 0) == 0.0)
        
        if is_uninitialized:
            container.set('ram_usage_min_gb', f'{ram_current:.3f}')
            container.set('ram_usage_max_gb', f'{ram_current:.3f}')
            container.set('ram_usage_last_gb', f'{ram_current:.3f}')
            return True
        else:
            ram_min_val = min(float(ram_min), ram_current)
            ram_max_val = max(float(ram_max), ram_current)
            container.set('ram_usage_min_gb', f'{ram_min_val:.3f}')
            container.set('ram_usage_max_gb', f'{ram_max_val:.3f}')
            container.set('ram_usage_last_gb', f'{ram_current:.3f}')
            return True
    
    def update_instance_metrics(self, container: ET.Element, metrics_entry: Dict,
                                service_id: str, pod_name: str, node_name: str) -> bool:
        """Update all metrics for an instance (CPU, RAM, exec). Returns True if any changed."""
        changed = False
        
        # Update CPU
        cpu_usage = metrics_entry.get('cpu_usage')
        if cpu_usage is not None and float(cpu_usage) > 0:
            if self.update_cpu_metric(container, float(cpu_usage), service_id, pod_name):
                changed = True
        
        # Update RAM
        ram_usage_gb = metrics_entry.get('ram_usage_gb')
        if ram_usage_gb is not None and float(ram_usage_gb) > 0:
            if self.update_ram_metric(container, float(ram_usage_gb), service_id, pod_name):
                changed = True
        
        # Update execution metrics
        if self.update_exec_metrics(container, metrics_entry, service_id, pod_name, node_name):
            changed = True
        
        return changed
    
    def update_exec_metrics(self, container: ET.Element, metrics_entry: Dict, 
                           service_id: str, pod_name: str, node_name: str) -> bool:
        """Update execution metrics. Returns True if changed."""
        logger = BlazeLogger.get_instance()
        changed = False
        
        # Apply exec_time
        exec_time_val = metrics_entry.get('exec_time')
        if not exec_time_val or float(exec_time_val) == 0:
            exec_time_val = metrics_entry.get('workload_level_exec_time')
        
        if exec_time_val is not None:
            try:
                exec_time_float = float(exec_time_val)
                if exec_time_float > 0:
                    new_exec = f"{exec_time_float:.6f}"
                    old_exec = container.get('exec_time')
                    if old_exec != new_exec:
                        container.set('exec_time', new_exec)
                        logger.debug(f"[PROM_EXEC] {service_id}/{pod_name}@{node_name}: exec_time={new_exec}")
                        changed = True
            except (TypeError, ValueError):
                logger.warning(f"[PROM_EXEC] Invalid exec_time value for {service_id}/{pod_name}: {exec_time_val}")
        
        # Apply execution_rate_hz
        exec_rate_val = metrics_entry.get('execution_rate_hz')
        if not exec_rate_val or float(exec_rate_val) == 0:
            exec_rate_val = metrics_entry.get('workload_level_exec_rate')
        
        if exec_rate_val is not None:
            try:
                exec_rate_float = float(exec_rate_val)
                if exec_rate_float > 0:
                    new_rate = f"{exec_rate_float:.2f}"
                    old_rate = container.get('execution_rate_hz')
                    if old_rate != new_rate:
                        container.set('execution_rate_hz', new_rate)
                        logger.debug(f"[PROM_EXEC] {service_id}/{pod_name}@{node_name}: execution_rate_hz={new_rate}")
                        changed = True
            except (TypeError, ValueError):
                logger.warning(f"[PROM_EXEC] Invalid execution_rate_hz value for {service_id}/{pod_name}: {exec_rate_val}")
        
        return changed
    
