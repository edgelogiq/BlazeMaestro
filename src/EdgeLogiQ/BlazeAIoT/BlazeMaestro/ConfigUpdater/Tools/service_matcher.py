"""Service name matching logic with fuzzy matching support."""
from typing import Dict, Optional
import xml.etree.ElementTree as ET


class ServiceMatcher:
    """Service name matching logic with fuzzy matching support."""
    
    def find_match(self, service_id: str, service_metrics: Dict, service_elem: Optional[ET.Element] = None) -> Optional[str]:
        """Find matching metric key for service_id with optional XML hint."""
        override = service_elem.get('metrics_service_name') or service_elem.get('metrics_id') if service_elem else None
        
        for name in (override, service_id):
            if name is None:
                continue
            if name in service_metrics:
                return name
            # Substring match
            for svc in service_metrics:
                if (name in svc or svc in name) and '-daemonset' not in svc.lower():
                    return svc
            # Fuzzy match
            for svc in service_metrics:
                if self._fuzzy_match(name, svc):
                    return svc
        
        # DaemonSet variants
        for svc in service_metrics:
            if '-daemonset' in svc.lower() and service_id == svc.replace('-daemonset', '').replace('_daemonset', ''):
                return svc
        return None
    
    def _fuzzy_match(self, s1: str, s2: str) -> bool:
        """Fuzzy match by removing hyphens/underscores."""
        n1 = s1.replace('-', '').replace('_', '').lower()
        n2 = s2.replace('-', '').replace('_', '').lower()
        if n1 == n2 or n1 in n2 or n2 in n1:
            return True
        if len(n1) >= 5 and len(n2) >= 5:
            c1, c2 = n1.rstrip('0123456789'), n2.rstrip('0123456789')
            if c1.startswith(c2) or c2.startswith(c1):
                return True
        return False
