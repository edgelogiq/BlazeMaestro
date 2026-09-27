#!/usr/bin/env python3
"""
Latency Matrix Comparison Tool

Exports latency matrices from both Prometheus (real environment) and Simulator
to XML format and generates detailed comparison statistics.

Usage:
    python compare_latency_matrices.py [--prometheus-url URL] [--output-dir DIR]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Metrics.prometheus_facade import PrometheusMetricsFacade
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Metrics.simulator_facade import SimulatorMetricsFacade
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Simulator.util.network_util import NetworkUtil
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.Metrics.latency_matrix_exporter import LatencyMatrixExporter
from EdgeLogiQ.BlazeAIoT.BlazeMaestro.ConfigUpdater.prometheus_xml_updater import PrometheusXMLUpdater
from BlazeAIoT.Core.Utils import BlazeLogger


class Node:
    """Simple Node class for testing."""

    def __init__(self, node_id: str, layer_id: str = None):
        self.node_id = node_id
        self.layer_id = layer_id or node_id.split("-")[0] if "-" in node_id else "unknown"


def get_prometheus_nodes(prom_url: str) -> list:
    """Fetch nodes from Prometheus metrics."""
    try:
        BlazeLogger.get_instance().info(f"Fetching nodes from Prometheus: {prom_url}")

        # Use PrometheusXMLUpdater to discover nodes
        updater = PrometheusXMLUpdater(prom_url=prom_url)
        nodes = []

        # Try to get nodes from node metadata metric
        try:
            nodes_query = 'node_uname_info'
            facade = PrometheusMetricsFacade(prom_url)
            results = facade.query(nodes_query)

            if results:
                for result in results:
                    metric = result.get('metric', {})
                    node_name = metric.get('nodename') or metric.get('node') or metric.get('instance', '').split(':')[0]
                    if node_name:
                        nodes.append(Node(node_name))

            if nodes:
                BlazeLogger.get_instance().info(f"Discovered {len(nodes)} nodes from Prometheus")
                return nodes
        except Exception as e:
            BlazeLogger.get_instance().warning(f"Could not discover nodes from metrics: {e}")

        # Fallback: create sample nodes based on typical topology
        BlazeLogger.get_instance().info("Using default node topology")
        nodes = [
            Node("edge-cpu1", "edge"),
            Node("edge-cpu2", "edge"),
            Node("edge-gpu1", "edge"),
            Node("fog-cpu1", "fog"),
            Node("fog-cpu2", "fog"),
            Node("fog-gpu1", "fog"),
            Node("cloud-cpu1", "cloud"),
            Node("cloud-cpu2", "cloud"),
        ]
        return nodes

    except Exception as e:
        BlazeLogger.get_instance().error(f"Failed to get Prometheus nodes: {e}")
        raise


def compare_latency_matrices(prometheus_url: str, output_dir: str, platform_xml: str | None, enable_jitter: bool,
                             jitter_percent: float, enable_packet_loss: bool, packet_loss_rate: float, seed: int | None):
    """Compare Prometheus and Simulator latency matrices."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    BlazeLogger.get_instance().info(f"Latency comparison starting")
    BlazeLogger.get_instance().info(f"Output directory: {output_dir}")

    # Get nodes from Prometheus
    nodes = get_prometheus_nodes(prometheus_url)
    BlazeLogger.get_instance().info(f"Using {len(nodes)} nodes for comparison")

    # Get Prometheus latency matrix
    try:
        BlazeLogger.get_instance().info("Fetching latency matrix from Prometheus...")
        prometheus_facade = PrometheusMetricsFacade(prometheus_url)
        prometheus_matrix = prometheus_facade.get_latency_matrix(nodes)
        BlazeLogger.get_instance().info("✓ Prometheus latency matrix retrieved")

        # Export Prometheus matrix
        prom_xml = output_dir / "latency_matrix_prometheus.xml"
        prometheus_facade.export_latency_matrix_to_xml(
            nodes,
            str(prom_xml),
            metadata={"source": "prometheus", "url": prometheus_url},
        )
        BlazeLogger.get_instance().info(f"✓ Prometheus matrix exported: {prom_xml}")

    except Exception as e:
        BlazeLogger.get_instance().error(f"Failed to fetch Prometheus matrix: {e}")
        prometheus_matrix = None

    # Get Simulator latency matrix
    try:
        BlazeLogger.get_instance().info("Generating latency matrix from Simulator...")
        latency_matrix = None
        if platform_xml:
            latency_matrix = NetworkUtil.build_latency_matrix_with_jitter(
                platform_xml,
                nodes,
                enable_jitter=enable_jitter,
                jitter_percent=jitter_percent,
                enable_packet_loss=enable_packet_loss,
                packet_loss_rate=packet_loss_rate,
                seed=seed,
            )
        simulator_facade = SimulatorMetricsFacade(nodes, latency_matrix=latency_matrix)
        simulator_matrix = simulator_facade.get_latency_matrix(nodes)
        BlazeLogger.get_instance().info("✓ Simulator latency matrix generated")

        # Export Simulator matrix
        sim_xml = output_dir / "latency_matrix_simulator.xml"
        simulator_facade.export_latency_matrix_to_xml(
            nodes,
            str(sim_xml),
            metadata={"source": "simulator", "model": "simgrid", "platform_xml": platform_xml or "none"},
        )
        BlazeLogger.get_instance().info(f"✓ Simulator matrix exported: {sim_xml}")

    except Exception as e:
        BlazeLogger.get_instance().error(f"Failed to generate Simulator matrix: {e}")
        simulator_matrix = None

    # Generate comparison statistics
    if prometheus_matrix is not None and simulator_matrix is not None:
        try:
            BlazeLogger.get_instance().info("Generating comparison statistics...")
            prom_path, sim_path, stats_path = LatencyMatrixExporter.export_comparison(
                prometheus_matrix,
                simulator_matrix,
                nodes,
                str(output_dir),
                metadata={"prometheus_url": prometheus_url},
            )
            BlazeLogger.get_instance().info(f"✓ Comparison statistics: {stats_path}")

            # Export CSV for evolution tracking
            BlazeLogger.get_instance().info("Exporting CSV for evolution tracking...")
            csv_path = LatencyMatrixExporter.export_comparison_to_csv(
                prometheus_matrix,
                simulator_matrix,
                nodes,
                str(output_dir / "latency_comparison.csv"),
                run_id="latest",
            )
            BlazeLogger.get_instance().info(f"✓ Comparison CSV: {csv_path}")

            # Print summary
            import numpy as np

            prom_arr = np.array(prometheus_matrix)
            sim_arr = np.array(simulator_matrix)
            diff = np.abs(prom_arr - sim_arr) * 1000  # Convert to ms

            print("\n" + "=" * 60)
            print("LATENCY MATRIX COMPARISON SUMMARY")
            print("=" * 60)
            print(f"Nodes compared: {len(nodes)}")
            print(f"Mean difference: {np.mean(diff[np.nonzero(diff)]):.4f} ms")
            print(f"Max difference: {np.max(diff):.4f} ms")
            print(f"Std dev: {np.std(diff):.4f} ms")
            print("=" * 60 + "\n")

        except Exception as e:
            BlazeLogger.get_instance().error(f"Failed to generate comparison stats: {e}")

    BlazeLogger.get_instance().info("Comparison complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Prometheus and Simulator latency matrices"
    )
    parser.add_argument(
        "--prometheus-url",
        type=str,
        default="http://localhost:9090",
        help="Prometheus URL (default: http://localhost:9090)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out/latency_comparison",
        help="Output directory (default: out/latency_comparison)",
    )
    parser.add_argument(
        "--platform",
        type=str,
        default=None,
        help="Platform XML for SimGrid latency (optional)",
    )
    parser.add_argument("--enable-jitter", action="store_true", help="Enable network jitter")
    parser.add_argument("--jitter-percent", type=float, default=10.0, help="Base jitter percentage")
    parser.add_argument("--enable-packet-loss", action="store_true", help="Enable packet loss")
    parser.add_argument("--packet-loss-rate", type=float, default=0.01, help="Packet loss rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    try:
        compare_latency_matrices(
            args.prometheus_url,
            args.output_dir,
            args.platform,
            args.enable_jitter,
            args.jitter_percent,
            args.enable_packet_loss,
            args.packet_loss_rate,
            args.seed,
        )
    except Exception as e:
        BlazeLogger.get_instance().error(f"Comparison failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
