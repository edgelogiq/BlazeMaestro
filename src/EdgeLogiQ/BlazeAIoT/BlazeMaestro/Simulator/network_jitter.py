"""
Network latency variability and jitter simulation for realistic network modeling.

Provides configurable network noise to simulate real-world conditions:
- Gaussian jitter for general network variability
- WiFi-specific jitter (higher variability than wired)
- Congestion modeling (latency increases with the load)
- Packet loss simulation (optional, requires retransmissions)
"""

import numpy as np
from typing import Dict, Optional


class NetworkJitterModel:
    """Model network latency variability with configurable jitter and packet loss."""
    def __init__(
        self,
        base_jitter_percent: float = 10.0,
        wifi_jitter_multiplier: float = 2.0,
        congestion_enabled: bool = True,
        packet_loss_enabled: bool = False,
        packet_loss_rate: float = 0.01,
        seed: Optional[int] = None
    ):
        """
        Initialize the network jitter model.
        
        Args:
            base_jitter_percent: Base jitter as percentage of nominal latency (default: 10%)
            wifi_jitter_multiplier: WiFi jitter multiplier vs. wired (default: 2x)
            congestion_enabled: Enable congestion modeling (default: True)
            packet_loss_enabled: Enable packet loss simulation (default: False)
            packet_loss_rate: Probability of packet loss requiring retransmission (default: 1%)
            seed: Random seed for reproducibility
        """
        self.base_jitter_percent = base_jitter_percent
        self.wifi_jitter_multiplier = wifi_jitter_multiplier
        self.congestion_enabled = congestion_enabled
        self.packet_loss_enabled = packet_loss_enabled
        self.packet_loss_rate = packet_loss_rate
        
        # Track network load for congestion modeling
        self.link_load: Dict[str, float] = {}
        
        # Random number generator
        self.rng = np.random.RandomState(seed)
    
    def apply_jitter(
        self,
        base_latency: float,
        link_id: str,
        is_wifi: bool = False
    ) -> float:
        """
        Apply jitter to base latency value.
        
        Args:
            base_latency: Nominal latency from platform.xml (seconds)
            link_id: Identifier for the link (e.g., "frank-1-wifi6")
            is_wifi: Whether this is a Wi-Fi link (higher jitter)
        
        Returns:
            Actual latency with jitter applied (seconds)
        """
        if base_latency <= 0:
            return base_latency
        
        # Calculate jitter percentage
        jitter_percent = self.base_jitter_percent
        if is_wifi:
            jitter_percent *= self.wifi_jitter_multiplier
        
        # Apply Gaussian noise (±jitter_percent of base latency)
        # Using 3-sigma rule: 99.7% of values within ±3σ
        sigma = (jitter_percent / 100.0) * base_latency / 3.0
        jitter = self.rng.normal(0, sigma)
        actual_latency = base_latency + jitter
        
        # Apply congestion if enabled
        if self.congestion_enabled:
            congestion_factor = self._get_congestion_factor(link_id)
            actual_latency *= congestion_factor
        
        # Apply packet loss if enabled
        if self.packet_loss_enabled:
            if self.rng.random() < self.packet_loss_rate:
                # Packet lost, needs retransmission (adds ~1 RTT)
                actual_latency += base_latency
        
        # Ensure latency is non-negative
        return max(0.0, actual_latency)
    
    def _get_congestion_factor(self, link_id: str) -> float:
        """
        Calculate the congestion factor based on the link load.
        
        Light congestion increases latency by up to 50%.
        Real networks show exponential increase under a heavy load.
        
        Args:
            link_id: Link identifier
        
        Returns:
            Congestion multiplier (1.0 = no congestion, 1.5 = moderate)
        """
        load = self.link_load.get(link_id, 0.0)
        
        # Simple congestion model: latency increases as the load approaches capacity
        # load = 0.0 -> factor = 1.0
        # load = 0.5 -> factor = 1.1
        # load = 0.8 -> factor = 1.3
        # load = 1.0 -> factor = 2.0 (heavy congestion)
        if load < 0.8:
            return 1.0 + (load * 0.375)  # Linear up to 80%
        else:
            # Exponential growth beyond 80%
            return 1.3 + 3.5 * ((load - 0.8) / 0.2) ** 2
    
