from typing import Dict, Optional
import numpy as np
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.network_jitter import NetworkJitterModel


class LatencyMatrixWithJitter:
    """Wrapper for latency matrix that applies jitter on-demand."""

    def __init__(
            self,
            base_matrix: np.ndarray,
            link_mapping: Dict[tuple, str],
            wifi_links: set,
            jitter_model: Optional[NetworkJitterModel] = None
    ):
        """
        Initialize latency matrix with jitter.

        Args:
            base_matrix: Base latency matrix from platform.xml
            link_mapping: Mapping from (src_idx, dst_idx) to link_id
            wifi_links: Set of WiFi link IDs
            jitter_model: Network jitter model (None = no jitter)
        """
        self.base_matrix = base_matrix
        self.link_mapping = link_mapping
        self.wifi_links = wifi_links
        self.jitter_model = jitter_model

    def get_latency(self, src_idx: int, dst_idx: int) -> float:
        """
        Get latency with jitter applied.

        Args:
            src_idx: Source node index
            dst_idx: Destination node index

        Returns:
            Actual latency with jitter (seconds)
        """
        base_latency = float(self.base_matrix[src_idx, dst_idx])

        if self.jitter_model is None:
            return base_latency

        # Determine if this is a WiFi link
        link_id = self.link_mapping.get((src_idx, dst_idx), "unknown")
        is_wifi = link_id in self.wifi_links

        return self.jitter_model.apply_jitter(base_latency, link_id, is_wifi)

    def __getitem__(self, key):
        """Support array-style indexing: matrix[i, j] or matrix[i][j]."""
        if isinstance(key, tuple) and len(key) == 2:
            return self.get_latency(key[0], key[1])
        else:
            # Return a row accessor that supports further indexing
            return _LatencyMatrixRow(self, key)

    @property
    def shape(self):
        """Return shape of underlying matrix."""
        return self.base_matrix.shape

    def flatten(self):
        """Return flattened view of base latency matrix (without jitter)."""
        return self.base_matrix.flatten()


class _LatencyMatrixRow:
    """Helper class for row-based indexing of LatencyMatrixWithJitter."""

    def __init__(self, parent, row_idx):
        self.parent = parent
        self.row_idx = row_idx

    def __getitem__(self, col_idx):
        """Support matrix[row][col] indexing."""
        return self.parent.get_latency(self.row_idx, col_idx)