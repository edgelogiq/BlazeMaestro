"""
Orchestrator CSV Logger for BlazeMaestro - metrics tracking across iterations.

Logs orchestrator performance metrics to timestamped CSV files during orchestration.
Supports multiple orchestrators (one file per orchestrator).
"""

import os
import csv
from datetime import datetime
from typing import Dict, List, Optional, Any


class OrchestratorCSVLogger:
    """Manages CSV logging for orchestrator iterations in BlazeMaestro."""
    
    def __init__(self, output_base_dir: str = "out"):
        """Initialize the CSV logger.
        
        Args:
            output_base_dir: Base directory for CSV outputs (default: 'out')
        """
        self.output_base_dir = output_base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(output_base_dir, self.timestamp)
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Support multiple orchestrators
        self.csv_handles = {}  # orchestrator_name -> file handle
        self.csv_writers = {}  # orchestrator_name -> csv.DictWriter
        self.field_names = {}  # orchestrator_name -> list of field names
        self.initialized_orchestrators = set()
        
    def initialize_orchestrator(self, 
                                orchestrator_name: str,
                                services: List[Any],
                                nodes: List[Any],
                                frank2_node_id: str = "frank-2",
                                include_battery_metrics: bool = False):
        """Initialize CSV logging for a specific orchestrator.
        
        Args:
            orchestrator_name: Name of the orchestrator (e.g., "nsga2", "dqn")
            services: List of Service objects
            nodes: List of Node objects  
            frank2_node_id: Node ID to use as latency reference (default: "frank-2")
        """
        if orchestrator_name in self.initialized_orchestrators:
            return
        
        # Build field names based on services and nodes
        fields = ["iteration"]
        
        # Service allocations - current and resulting
        for svc in services:
            service_id = getattr(svc, "service_id", f"service_{services.index(svc)}")
            fields.append(f"{service_id}_current_node")
            fields.append(f"{service_id}_allocation")
        
        # Constraint violations per service
        for svc in services:
            service_id = getattr(svc, "service_id", f"service_{services.index(svc)}")
            fields.append(f"{service_id}_constraint_status")
        
        # Orchestrator execution time
        fields.append("orchestrator_execution_time_ms")

        # Optional battery metric fields for battery-powered nodes (active orchestrator only)
        if include_battery_metrics:
            battery_nodes = []
            for node in nodes:
                battery_powered = getattr(node, "battery_powered", False)
                if not battery_powered:
                    continue
                # Only include schedulable battery nodes
                schedulable = getattr(node, "schedulable", True)
                if not schedulable:
                    continue
                node_id = getattr(node, "node_id", f"node_{nodes.index(node)}")
                battery_nodes.append(node_id)
                fields.append(f"{node_id}_battery_voltage_v")
                fields.append(f"{node_id}_battery_current_a")
                fields.append(f"{node_id}_battery_power_w")
                fields.append(f"{node_id}_battery_remaining_wh")
                fields.append(f"{node_id}_battery_remaining_pct")
                fields.append(f"{node_id}_battery_consumed_mwh")
            print(f"[CSV Logger] Added battery metric columns for nodes: {battery_nodes}")
        else:
            print(f"[CSV Logger] Skipping battery metrics (include_battery_metrics={include_battery_metrics})")
        
        # Latency from each node to frank-2
        for node in nodes:
            node_id = getattr(node, "node_id", f"node_{nodes.index(node)}")
            fields.append(f"{node_id}_latency_to_{frank2_node_id}_ms")
        
        self.field_names[orchestrator_name] = fields
        
        # Create CSV file for this orchestrator
        safe_name = orchestrator_name.replace(' ', '_').replace('-', '_').lower()
        csv_path = os.path.join(self.session_dir, f"{safe_name}.csv")
        
        # Check if file exists - if so, append; otherwise create new
        file_exists = os.path.exists(csv_path)
        mode = 'a' if file_exists else 'w'
        file_handle = open(csv_path, mode, newline='')
        writer = csv.DictWriter(file_handle, fieldnames=fields)
        
        # Only write header if creating new file
        if not file_exists:
            writer.writeheader()
        
        self.csv_handles[orchestrator_name] = file_handle
        self.csv_writers[orchestrator_name] = writer
        self.initialized_orchestrators.add(orchestrator_name)
        
        action = "Reopened" if file_exists else "Initialized"
        print(f"[CSV Logger] {action} CSV file for {orchestrator_name}: {csv_path}")
    
    def export_latency_matrix(self, nodes: List[Any], latency_matrix: List[List[float]], 
                              frank2_node_id: str = "frank-2"):
        """Export the latency matrix to CSV.
        
        Args:
            nodes: List of Node objects
            latency_matrix: Network latency matrix (in seconds, will convert to ms)
            frank2_node_id: Reference node for documentation
        """
        csv_path = os.path.join(self.session_dir, "latency_matrix.csv")
        
        try:
            with open(csv_path, 'w', newline='') as f:
                # Create header: source node, then each destination node
                node_ids = [getattr(node, "node_id", f"node_{i}") for i, node in enumerate(nodes)]
                writer = csv.writer(f)
                
                # Header row
                writer.writerow(["source_node"] + node_ids)
                
                # Data rows
                for i, source_node in enumerate(nodes):
                    source_id = getattr(source_node, "node_id", f"node_{i}")
                    row = [source_id]
                    
                    for j in range(len(nodes)):
                        try:
                            # Convert from seconds to milliseconds
                            latency_ms = float(latency_matrix[i][j]) * 1000
                            row.append(f"{latency_ms:.4f}")
                        except (IndexError, TypeError, ValueError):
                            row.append("N/A")
                    
                    writer.writerow(row)
                    
        except Exception as e:
            # Silently fail - latency matrix export is optional
            pass
    
    def log_iteration(self,
                     orchestrator_name: str,
                     iteration: int,
                     assignments: Dict[str, str],
                     previous_assignments: Optional[Dict[str, str]],
                     services: List[Any],
                     nodes: List[Any],
                     latency_matrix: List[List[float]],
                     violations: List[str],
                     execution_time_ms: float,
                     frank2_node_id: str = "frank-2",
                     include_battery_metrics: bool = False,
                     battery_metrics: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
                     battery_state: Optional[Dict[str, float]] = None,
                     battery_interval_s: float = 10.0):
        """Log a single iteration for a specific orchestrator.
        
        Args:
            orchestrator_name: Name of the orchestrator (e.g., "nsga2", "dqn")
            iteration: Current iteration number
            assignments: Current service -> node assignments
            previous_assignments: Previous iteration assignments (for "current" column)
            services: List of Service objects
            nodes: List of Node objects
            latency_matrix: Network latency matrix (in seconds, will convert to ms)
            violations: List of constraint violation descriptions
            execution_time_ms: Execution time of orchestrator in milliseconds
            frank2_node_id: Node ID to use as latency reference
            include_battery_metrics: Whether to log battery metrics for battery-powered nodes
            battery_metrics: Per-battery-node metrics dict: {node_id: {"voltage_v":..., "current_a":..., "power_w":...}}
            battery_state: Per-battery-node remaining energy tracking dict: {node_id: remaining_wh}
            battery_interval_s: Time interval between orchestration iterations in seconds (default 10.0)
        """
        if orchestrator_name not in self.initialized_orchestrators:
            self.initialize_orchestrator(
                orchestrator_name,
                services,
                nodes,
                frank2_node_id,
                include_battery_metrics=include_battery_metrics
            )
        
        if orchestrator_name not in self.csv_writers:
            return
        
        row = {"iteration": iteration}
        
        # Service allocations and current allocations
        for svc in services:
            service_id = getattr(svc, "service_id", f"service_{services.index(svc)}")
            
            # Current allocation (from previous_assignments)
            current_node = previous_assignments.get(service_id, "unassigned") if previous_assignments else "initial"
            row[f"{service_id}_current_node"] = current_node
            
            # Resulting allocation
            new_node = assignments.get(service_id, "unassigned")
            row[f"{service_id}_allocation"] = new_node
        
        # Constraint status for each service
        for svc in services:
            service_id = getattr(svc, "service_id", f"service_{services.index(svc)}")
            # Handle violations being either int (count) or list (details)
            if isinstance(violations, int):
                # If violations is a count, we can't determine per-service status, mark all as PASS if count is 0
                has_violation = violations > 0
            elif violations:
                has_violation = any(service_id in v for v in violations)
            else:
                has_violation = False
            row[f"{service_id}_constraint_status"] = "FAIL" if has_violation else "PASS"
        
        # Orchestrator execution time
        row["orchestrator_execution_time_ms"] = f"{execution_time_ms:.2f}"

        # Optional battery metrics for all battery-powered nodes (active orchestrator only)
        if include_battery_metrics:
            battery_metrics = battery_metrics or {}
            battery_state = battery_state or {}
            for node in nodes:
                battery_powered = getattr(node, "battery_powered", False)
                if not battery_powered:
                    continue
                # Only include schedulable battery nodes
                schedulable = getattr(node, "schedulable", True)
                if not schedulable:
                    continue
                node_id = getattr(node, "node_id", f"node_{nodes.index(node)}")
                node_metrics = battery_metrics.get(node_id, {})
                row[f"{node_id}_battery_voltage_v"] = (
                    f"{node_metrics.get('voltage_v'):.2f}" if node_metrics.get('voltage_v') is not None else "N/A"
                )
                row[f"{node_id}_battery_current_a"] = (
                    f"{node_metrics.get('current_a'):.2f}" if node_metrics.get('current_a') is not None else "N/A"
                )
                row[f"{node_id}_battery_power_w"] = (
                    f"{node_metrics.get('power_w'):.2f}" if node_metrics.get('power_w') is not None else "N/A"
                )
                
                # Battery energy state (remaining and consumed)
                remaining_wh = battery_state.get(node_id, None)
                row[f"{node_id}_battery_remaining_wh"] = (
                    f"{remaining_wh:.2f}" if remaining_wh is not None else "N/A"
                )
                
                # Calculate remaining percentage and consumed energy from power and interval
                power_w = node_metrics.get('power_w')
                cap_ah = getattr(node, "battery_capacity_ah", None)
                bat_v = getattr(node, "battery_voltage", None)
                
                if remaining_wh is not None and cap_ah and bat_v:
                    total_wh = float(cap_ah) * float(bat_v)
                    remaining_pct = (remaining_wh / total_wh * 100.0) if total_wh > 0 else 0.0
                    row[f"{node_id}_battery_remaining_pct"] = f"{remaining_pct:.1f}"
                else:
                    row[f"{node_id}_battery_remaining_pct"] = "N/A"
                
                if power_w is not None:
                    consumed_wh = float(power_w) * (battery_interval_s / 3600.0)
                    consumed_mwh = consumed_wh * 1000.0
                    row[f"{node_id}_battery_consumed_mwh"] = f"{consumed_mwh:.2f}"
                else:
                    row[f"{node_id}_battery_consumed_mwh"] = "N/A"
        
        # Latencies from each node to frank-2
        node_to_frank2_idx = None
        for i, node in enumerate(nodes):
            node_id = getattr(node, "node_id", f"node_{i}")
            if node_id == frank2_node_id:
                node_to_frank2_idx = i
                break
        
        if node_to_frank2_idx is not None:
            for i, node in enumerate(nodes):
                node_id = getattr(node, "node_id", f"node_{i}")
                try:
                    # latency_matrix is in seconds, convert to milliseconds
                    latency_ms = float(latency_matrix[i][node_to_frank2_idx]) * 1000
                    row[f"{node_id}_latency_to_{frank2_node_id}_ms"] = f"{latency_ms:.4f}"
                except (IndexError, TypeError, ValueError):
                    row[f"{node_id}_latency_to_{frank2_node_id}_ms"] = "N/A"
        
        # Write row
        self.csv_writers[orchestrator_name].writerow(row)
        self.csv_handles[orchestrator_name].flush()
        print(f"[CSV Logger] Wrote iteration {iteration} for {orchestrator_name}")
    
    def close(self):
        """Close all CSV files."""
        for handle in self.csv_handles.values():
            if handle:
                handle.close()
        self.csv_handles.clear()
        self.csv_writers.clear()
        self.initialized_orchestrators.clear()
    
    def get_session_dir(self) -> str:
        """Get the session directory where CSV files are stored."""
        return self.session_dir
    
    def __del__(self):
        """Ensure file is closed on deletion."""
        self.close()
