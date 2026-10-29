from __future__ import annotations
from BlazeAIoT.Core.Utils import BlazeLogger

from typing import Dict, Optional, Sequence, Any



class PerNodeAllocationEmitter:
    """
    Emits per-node allocation status for services.

    This is intentionally decoupled from BaseMetricsFacade so we don't
    break existing consumers. For now, it logs and keeps an in-memory view,
    and optionally can integrate with Prometheus (future hook).
    """

    def __init__(self, nodes: Sequence[Any], use_logging: bool = True):
        # Build node -> layer map once for easy labeling
        self.node_to_layer: Dict[str, str] = {}
        for n in nodes:
            node_id = getattr(n, 'node_id', None)
            layer_id = getattr(n, 'layer_id', None)
            if not layer_id and node_id and '-' in node_id:
                # Infer layer from node_id prefix (e.g., edge-cpu1, but NOT frank-*)
                prefix = node_id.split('-')[0]
                # Map frank prefix to edge
                layer_id = 'edge' if prefix == 'frank' else prefix
            if node_id:
                self.node_to_layer[node_id] = layer_id or 'unknown'
        self.use_logging = use_logging
        # Simple store for last emitted values: (service_id, node_id) -> {eligible, selected}
        self.last_status: Dict[tuple, Dict[str, int]] = {}

        # Optional: Prometheus client integration (lazy)
        self._prom_gauge = None
        try:
            import importlib
            prom_mod = importlib.import_module('prometheus_client')
            Gauge = getattr(prom_mod, 'Gauge', None)
            if Gauge is not None:
                # Note: eligible/selected are encoded as values (0/1); labels carry service/node/layer
                self._prom_gauge = Gauge(
                    'maestro_allocation_status',
                    'Per-node allocation status (eligible/selected) for services',
                    labelnames=['service_id', 'node_id', 'layer_id', 'flag']
                )
                BlazeLogger.get_instance().info('Prometheus Gauge initialized: maestro_allocation_status')
        except Exception:
            # Prometheus client not available; continue with logging-only
            self._prom_gauge = None

    def emit(self, service_id: str, node_id: str, eligible: bool, selected: bool):
        layer_id = self.node_to_layer.get(node_id, 'unknown')
        status = {'eligible': int(bool(eligible)), 'selected': int(bool(selected))}
        self.last_status[(service_id, node_id)] = status

        if self.use_logging:
            BlazeLogger.get_instance().info("[alloc] service=%s node=%s layer=%s eligible=%s selected=%s",
                     service_id, node_id, layer_id, bool(eligible), bool(selected))

        if self._prom_gauge is not None:
            # Emit two labeled metrics: flag=eligible and flag=selected
            try:
                self._prom_gauge.labels(service_id=service_id, node_id=node_id, layer_id=layer_id, flag='eligible').set(status['eligible'])
                self._prom_gauge.labels(service_id=service_id, node_id=node_id, layer_id=layer_id, flag='selected').set(status['selected'])
            except Exception as e:
                BlazeLogger.get_instance().debug('Prometheus gauge emit failed: %s', e)

    def emit_allocation_summary(self, service_id: str, original_node: Optional[str], 
                               recommended_node: Optional[str], eligible_nodes: Sequence[str],
                               reason: str = ''):
        """
        Emit a summary of allocation decision for a service.
        
        Args:
            service_id: Service ID
            original_node: Where the service was previously running (or None if new)
            recommended_node: Where the orchestrator recommends placing it (or None if removing)
            eligible_nodes: List of nodes where service can run
            reason: Optional explanation for the change
        """
        