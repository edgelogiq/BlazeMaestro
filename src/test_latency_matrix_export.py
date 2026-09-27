#!/usr/bin/env python3
"""
Integration test for latency matrix export functionality.

Tests that latency matrices can be successfully exported from both
Prometheus and Simulator facades.
"""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Metrics.latency_matrix_exporter import LatencyMatrixExporter
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Metrics.simulator_facade import SimulatorMetricsFacade
import numpy as np


class MockNode:
    """Mock Node for testing."""
    def __init__(self, node_id: str, layer_id: str = None):
        self.node_id = node_id
        self.layer_id = layer_id or node_id.split("-")[0]


def test_latency_matrix_export():
    """Test basic latency matrix export to XML."""
    print("Testing Latency Matrix Export...")

    # Create test nodes
    nodes = [
        MockNode("edge-cpu1", "edge"),
        MockNode("edge-cpu2", "edge"),
        MockNode("fog-cpu1", "fog"),
        MockNode("cloud-cpu1", "cloud"),
    ]

    # Create test matrix
    latency_matrix = np.array([
        [0.000, 0.001, 0.020, 0.040],
        [0.001, 0.000, 0.021, 0.041],
        [0.020, 0.021, 0.000, 0.025],
        [0.040, 0.041, 0.025, 0.000],
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = f"{tmpdir}/latency_test.xml"
        
        # Export matrix
        result_path = LatencyMatrixExporter.export_to_xml(
            latency_matrix,
            nodes,
            output_path,
            source="test",
            metadata={"test": "true", "nodes": "4"}
        )

        # Verify file exists
        assert Path(result_path).exists(), "XML file was not created"
        print(f"✓ XML file created: {result_path}")

        # Read and verify content
        with open(result_path, "r") as f:
            content = f.read()
            assert "<latency_matrix" in content, "Missing root element"
            assert 'source="test"' in content, "Missing source attribute"
            assert "<latencies>" in content, "Missing latencies section"
            assert "edge-cpu1" in content, "Missing node reference"
        
        print("✓ XML structure validated")


def test_simulator_facade_export():
    """Test simulator facade export functionality."""
    print("\nTesting Simulator Facade Export...")

    # Create test nodes
    nodes = [
        MockNode("edge-cpu1", "edge"),
        MockNode("edge-cpu2", "edge"),
        MockNode("fog-cpu1", "fog"),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = f"{tmpdir}/simulator_latency.xml"
        
        # Create facade and export
        facade = SimulatorMetricsFacade(nodes)
        result_path = facade.export_latency_matrix_to_xml(
            nodes,
            output_path,
            metadata={"test": "simulator"}
        )

        # Verify file exists
        assert Path(result_path).exists(), "Simulator XML file was not created"
        print(f"✓ Simulator XML file created: {result_path}")

        # Verify it's valid
        with open(result_path, "r") as f:
            content = f.read()
            assert 'source="simulator"' in content, "Missing simulator source"

        print("✓ Simulator export validated")


def test_comparison_export():
    """Test comparison matrix export."""
    print("\nTesting Comparison Export...")

    # Create test nodes
    nodes = [
        MockNode("edge-cpu1", "edge"),
        MockNode("fog-cpu1", "fog"),
        MockNode("cloud-cpu1", "cloud"),
    ]

    # Create test matrices
    prometheus_matrix = np.array([
        [0.000, 0.018, 0.042],
        [0.019, 0.000, 0.025],
        [0.038, 0.026, 0.000],
    ])

    simulator_matrix = np.array([
        [0.000, 0.020, 0.040],
        [0.020, 0.000, 0.020],
        [0.040, 0.020, 0.000],
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        # Export comparison
        prom_xml, sim_xml, stats_xml = LatencyMatrixExporter.export_comparison(
            prometheus_matrix,
            simulator_matrix,
            nodes,
            tmpdir,
            metadata={"test": "comparison"}
        )

        # Verify all files exist
        assert Path(prom_xml).exists(), "Prometheus XML not created"
        assert Path(sim_xml).exists(), "Simulator XML not created"
        assert Path(stats_xml).exists(), "Stats XML not created"
        
        print(f"✓ Prometheus XML: {prom_xml}")
        print(f"✓ Simulator XML: {sim_xml}")
        print(f"✓ Stats XML: {stats_xml}")

        # Verify stats content
        with open(stats_xml, "r") as f:
            content = f.read()
            assert "<latency_comparison" in content, "Missing comparison root"
            assert "<overall_stats>" in content, "Missing overall stats"
            assert "<mean_difference_ms>" in content, "Missing mean difference"
            assert "<per_node_stats>" in content, "Missing per-node stats"

        print("✓ Comparison statistics validated")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Latency Matrix Export Integration Tests")
    print("=" * 60)

    try:
        test_latency_matrix_export()
        test_simulator_facade_export()
        test_comparison_export()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
