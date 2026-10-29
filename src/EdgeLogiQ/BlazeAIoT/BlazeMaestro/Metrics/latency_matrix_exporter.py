"""
Latency Matrix Exporter - Export latency matrices to XML and CSV for comparison analysis.

This module provides utilities to serialize latency matrices from both Prometheus
(real environment) and Simulator (simulated environment) to XML and CSV formats for
detailed analysis and comparison, including time-series tracking of latency evolution.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom
import csv
from pathlib import Path
from typing import List, Optional, Any, Union
import numpy as np
from datetime import datetime


class LatencyMatrixExporter:
    """Export latency matrices to XML format for comparison analysis."""

    @staticmethod
    def export_to_xml(
        latency_matrix: Union[List[List[float]], np.ndarray],
        nodes: List[Any],
        output_path: str,
        source: str = "unknown",
        timestamp: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Export latency matrix to XML file.

        Args:
            latency_matrix: 2D array of latencies (seconds)
            nodes: List of Node objects with node_id attribute
            output_path: Path where XML file will be written
            source: Source label ("prometheus" or "simulator")
            timestamp: Optional timestamp (default: current time)
            metadata: Optional dict with additional metadata (hyperparameters, etc.)

        Returns:
            Full path to written XML file
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        # Create root element
        root = ET.Element("latency_matrix")
        root.set("version", "1.0")
        root.set("source", source)
        root.set("timestamp", timestamp)

        # Add metadata
        metadata_elem = ET.SubElement(root, "metadata")
        metadata_elem.set("nodes", str(len(nodes)))
        metadata_elem.set("unit", "seconds")

        if metadata:
            for key, value in metadata.items():
                meta_item = ET.SubElement(metadata_elem, "meta")
                meta_item.set("key", str(key))
                meta_item.set("value", str(value))

        # Build node ID mapping
        node_ids = [getattr(n, "node_id", f"node_{i}") for i, n in enumerate(nodes)]

        # Add nodes section
        nodes_elem = ET.SubElement(root, "nodes")
        for i, node_id in enumerate(node_ids):
            node_elem = ET.SubElement(nodes_elem, "node")
            node_elem.set("id", node_id)
            node_elem.set("index", str(i))
            if nodes and i < len(nodes):
                node_obj = nodes[i]
                if hasattr(node_obj, "layer_id"):
                    node_elem.set("layer", str(node_obj.layer_id))
                elif hasattr(node_obj, "node_id") and "-" in node_obj.node_id:
                    layer = node_obj.node_id.split("-")[0]
                    node_elem.set("layer", layer)

        # Add latency matrix as entries
        matrix_elem = ET.SubElement(root, "latencies")
        if isinstance(latency_matrix, np.ndarray):
            latency_matrix = latency_matrix.tolist()

        for i, row in enumerate(latency_matrix):
            for j, latency in enumerate(row):
                if i != j:  # Skip diagonal (self-latencies)
                    latency_elem = ET.SubElement(matrix_elem, "latency")
                    latency_elem.set("from", node_ids[i])
                    latency_elem.set("to", node_ids[j])
                    latency_elem.set("from_idx", str(i))
                    latency_elem.set("to_idx", str(j))
                    latency_elem.set("value", f"{float(latency):.6f}")

        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Pretty-print XML
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        # Remove extra blank lines
        xml_str = "\n".join([line for line in xml_str.split("\n") if line.strip()])

        # Write to file
        with open(output_path, "w") as f:
            f.write(xml_str)

        return output_path

    @staticmethod
    def export_comparison(
        prometheus_matrix: Union[List[List[float]], np.ndarray],
        simulator_matrix: Union[List[List[float]], np.ndarray],
        nodes: List[Any],
        output_dir: str,
        metadata: Optional[dict] = None,
    ) -> tuple:
        """
        Export both matrices and generate comparison statistics.

        Args:
            prometheus_matrix: Real latency matrix from Prometheus
            simulator_matrix: Simulated latency matrix
            nodes: List of Node objects
            output_dir: Directory where comparison files will be written
            metadata: Optional metadata dict

        Returns:
            Tuple of (prometheus_xml_path, simulator_xml_path, stats_path)
        """
        timestamp = datetime.now().isoformat()

        # Export Prometheus matrix
        prom_path = f"{output_dir}/latency_matrix_prometheus_{timestamp.replace(':', '-')}.xml"
        LatencyMatrixExporter.export_to_xml(
            prometheus_matrix,
            nodes,
            prom_path,
            source="prometheus",
            timestamp=timestamp,
            metadata={**(metadata or {}), "environment": "real"},
        )

        # Export Simulator matrix
        sim_path = f"{output_dir}/latency_matrix_simulator_{timestamp.replace(':', '-')}.xml"
        LatencyMatrixExporter.export_to_xml(
            simulator_matrix,
            nodes,
            sim_path,
            source="simulator",
            timestamp=timestamp,
            metadata={**(metadata or {}), "environment": "simulated"},
        )

        # Generate comparison statistics
        stats_path = LatencyMatrixExporter._generate_comparison_stats(
            prometheus_matrix, simulator_matrix, nodes, output_dir, timestamp
        )

        return prom_path, sim_path, stats_path

    @staticmethod
    def _generate_comparison_stats(
        prometheus_matrix: Union[List[List[float]], np.ndarray],
        simulator_matrix: Union[List[List[float]], np.ndarray],
        nodes: List[Any],
        output_dir: str,
        timestamp: str,
    ) -> str:
        """Generate comparison statistics between real and simulated latencies."""
        if isinstance(prometheus_matrix, list):
            prometheus_matrix = np.array(prometheus_matrix)
        if isinstance(simulator_matrix, list):
            simulator_matrix = np.array(simulator_matrix)

        # Convert to milliseconds for readability
        prom_ms = prometheus_matrix * 1000
        sim_ms = simulator_matrix * 1000

        # Calculate statistics
        diff = np.abs(prom_ms - sim_ms)
        diff_percent = (diff / (np.abs(prom_ms) + 1e-6)) * 100

        # Create stats XML
        root = ET.Element("latency_comparison")
        root.set("timestamp", timestamp)

        # Overall statistics
        stats_elem = ET.SubElement(root, "overall_stats")

        # Mean absolute difference
        mean_diff = np.mean(diff[np.nonzero(diff)])
        mean_diff_elem = ET.SubElement(stats_elem, "mean_difference_ms")
        mean_diff_elem.text = f"{mean_diff:.4f}"

        # Max difference
        max_diff = np.max(diff)
        max_diff_elem = ET.SubElement(stats_elem, "max_difference_ms")
        max_diff_elem.text = f"{max_diff:.4f}"

        # Std dev
        std_diff = np.std(diff)
        std_diff_elem = ET.SubElement(stats_elem, "std_difference_ms")
        std_diff_elem.text = f"{std_diff:.4f}"

        # Mean percent error
        nonzero_diff = diff_percent[np.nonzero(diff_percent)]
        if len(nonzero_diff) > 0:
            mean_percent = np.mean(nonzero_diff)
            mean_percent_elem = ET.SubElement(stats_elem, "mean_percent_error")
            mean_percent_elem.text = f"{mean_percent:.2f}%"

        # Per-node statistics
        node_ids = [getattr(n, "node_id", f"node_{i}") for i, n in enumerate(nodes)]
        nodes_stats = ET.SubElement(root, "per_node_stats")

        for i, from_node in enumerate(node_ids):
            from_elem = ET.SubElement(nodes_stats, "from_node")
            from_elem.set("id", from_node)

            for j, to_node in enumerate(node_ids):
                if i != j:
                    link_elem = ET.SubElement(from_elem, "link")
                    link_elem.set("to", to_node)
                    link_elem.set("prom_ms", f"{prom_ms[i, j]:.4f}")
                    link_elem.set("sim_ms", f"{sim_ms[i, j]:.4f}")
                    link_elem.set("diff_ms", f"{diff[i, j]:.4f}")
                    if prom_ms[i, j] > 0:
                        link_elem.set("error_percent", f"{diff_percent[i, j]:.2f}%")

        # Pretty-print XML
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        xml_str = "\n".join([line for line in xml_str.split("\n") if line.strip()])

        # Write to file
        stats_path = f"{output_dir}/latency_comparison_stats_{timestamp.replace(':', '-')}.xml"
        Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as f:
            f.write(xml_str)

        return stats_path

    @staticmethod
    def export_to_csv(
        latency_matrix: Union[List[List[float]], np.ndarray],
        nodes: List[Any],
        output_path: str,
        source: str = "unknown",
        timestamp: Optional[str] = None,
        iteration: Optional[int] = None,
        run_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Export latency matrix as CSV in matrix format with node columns.

        Each row represents an iteration with latency values between all node pairs.
        Columns: iteration, node1->node1, node1->node2, ..., node_i->node_j, ...
        Diagonal values (node->node) are 0.0

        Args:
            latency_matrix: 2D array of latencies (seconds)
            nodes: List of Node objects with node_id attribute
            output_path: Path where CSV file will be written
            source: Source label ("prometheus" or "simulator")
            timestamp: Optional timestamp (deprecated - use iteration instead)
            iteration: Optional iteration number (default: current timestamp if None)
            run_id: Optional run identifier for tracking evolution
            metadata: Optional dict with additional metadata

        Returns:
            Full path to written CSV file
        """
        # Use iteration if provided, otherwise fall back to timestamp
        if iteration is not None:
            row_identifier = str(iteration)
        elif timestamp is not None:
            row_identifier = timestamp
        else:
            row_identifier = datetime.now().isoformat()

        # Build node ID mapping
        node_ids = [getattr(n, "node_id", f"node_{i}") for i, n in enumerate(nodes)]

        # Convert to milliseconds
        if isinstance(latency_matrix, np.ndarray):
            latency_matrix = latency_matrix.tolist()
        latency_ms = [[float(val) * 1000 for val in row] for row in latency_matrix]

        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Determine if file exists (to write header or not)
        file_exists = Path(output_path).exists() and Path(output_path).stat().st_size > 0

        # Build column headers for all node pairs
        column_headers = ["iteration"]
        for i, from_node in enumerate(node_ids):
            for j, to_node in enumerate(node_ids):
                column_headers.append(f"{from_node}->{to_node}")

        # Prepare data row
        data_row = [row_identifier]
        for i in range(len(node_ids)):
            for j in range(len(node_ids)):
                if i == j:
                    # Diagonal: node to itself is 0.0
                    data_row.append("0.0")
                else:
                    data_row.append(f"{latency_ms[i][j]:.4f}")

        # Write or append to CSV
        with open(output_path, "a", newline="") as f:
            writer = csv.writer(f)

            # Write header only if file doesn't exist or is empty
            if not file_exists:
                writer.writerow(column_headers)

            # Write data row
            writer.writerow(data_row)

        return output_path

    @staticmethod
    def export_comparison_to_csv(
        prometheus_matrix: Union[List[List[float]], np.ndarray],
        simulator_matrix: Union[List[List[float]], np.ndarray],
        nodes: List[Any],
        output_path: str,
        timestamp: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Export comparison as flat CSV with detailed statistics per link.

        One row per latency pair with Prometheus, Simulator, and delta values.

        Args:
            prometheus_matrix: Real latency matrix (seconds)
            simulator_matrix: Simulated latency matrix (seconds)
            nodes: List of Node objects
            output_path: Path where CSV file will be written
            timestamp: Optional timestamp (default: current time)
            run_id: Optional run identifier
            metadata: Optional metadata dict

        Returns:
            Full path to written CSV file
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        if isinstance(prometheus_matrix, list):
            prometheus_matrix = np.array(prometheus_matrix)
        if isinstance(simulator_matrix, list):
            simulator_matrix = np.array(simulator_matrix)

        # Convert to milliseconds
        prom_ms = prometheus_matrix * 1000
        sim_ms = simulator_matrix * 1000
        diff_ms = np.abs(prom_ms - sim_ms)
        error_percent = (diff_ms / (np.abs(prom_ms) + 1e-6)) * 100

        # Build node ID mapping
        node_ids = [getattr(n, "node_id", f"node_{i}") for i, n in enumerate(nodes)]

        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Write CSV
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "timestamp",
                    "run_id",
                    "from_node",
                    "to_node",
                    "from_layer",
                    "to_layer",
                    "prometheus_latency_ms",
                    "simulator_latency_ms",
                    "diff_ms",
                    "error_percent",
                ]
            )

            # Data rows
            for i, from_node in enumerate(node_ids):
                from_layer = from_node.split("-")[0] if "-" in from_node else "unknown"
                for j, to_node in enumerate(node_ids):
                    if i != j:  # Skip diagonal
                        to_layer = to_node.split("-")[0] if "-" in to_node else "unknown"
                        writer.writerow(
                            [
                                timestamp,
                                run_id or "default",
                                from_node,
                                to_node,
                                from_layer,
                                to_layer,
                                f"{prom_ms[i, j]:.4f}",
                                f"{sim_ms[i, j]:.4f}",
                                f"{diff_ms[i, j]:.4f}",
                                f"{error_percent[i, j]:.2f}",
                            ]
                        )

        return output_path
