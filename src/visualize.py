"""
Visualize the results from dynamic orchestrator comparison.
Creates plots showing:
- Latency over time for each orchestrator
- Energy consumption over time
- Reassignment events
- Battery depletion over time
"""
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_results(json_path: str):
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_metrics_over_time(results_data, output_dir):
    """Plot latency and energy metrics over time."""
    config = results_data["config"]
    results = results_data["results"]
    
    # Prepare data
    orchestrators = list(results.keys())
    
    # Get time_step_minutes from config, or calculate from data, or use default
    if "time_step_minutes" in config:
        time_step_minutes = config["time_step_minutes"]
    else:
        # Try to infer from the data or use a default
        # If we have results, check if there's a pattern in the time field
        time_step_minutes = 5  # Default to 5 minutes if not specified
        print(f"Warning: time_step_minutes not found in config, using default: {time_step_minutes} minutes")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Latency over time
    ax1 = axes[0]
    for orch_name in orchestrators:
        orch_results = results[orch_name]
        if not orch_results:
            continue
        times = [r["step"] * time_step_minutes for r in orch_results]
        latencies = [r["latency"] for r in orch_results]
        ax1.plot(times, latencies, marker='o', label=orch_name, linewidth=2)
    
    ax1.set_xlabel("Time (minutes)", fontsize=12)
    ax1.set_ylabel("Total Latency (seconds)", fontsize=12)
    ax1.set_title("Service Pipeline Latency Over Time", fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Add scenario phase backgrounds
    ax1.axvspan(0, 10, alpha=0.1, color='green', label='Normal')
    ax1.axvspan(10, 15, alpha=0.1, color='orange', label='Peak Traffic')
    ax1.axvspan(15, 20, alpha=0.1, color='red', label='Node Degraded')
    ax1.axvspan(20, 25, alpha=0.1, color='yellow', label='Network Congestion')
    ax1.axvspan(25, 30, alpha=0.1, color='blue', label='Low Traffic')
    
    # Plot 2: Energy consumption over time
    ax2 = axes[1]
    for orch_name in orchestrators:
        orch_results = results[orch_name]
        if not orch_results:
            continue
        times = [r["step"] * time_step_minutes for r in orch_results]
        energies = [r["energy"] for r in orch_results]
        ax2.plot(times, energies, marker='s', label=orch_name, linewidth=2)
    
    ax2.set_xlabel("Time (minutes)", fontsize=12)
    ax2.set_ylabel("Energy Consumption (Joules)", fontsize=12)
    ax2.set_title("Battery Energy Consumption Over Time", fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Add scenario phase backgrounds
    ax2.axvspan(0, 10, alpha=0.1, color='green')
    ax2.axvspan(10, 15, alpha=0.1, color='orange')
    ax2.axvspan(15, 20, alpha=0.1, color='red')
    ax2.axvspan(20, 25, alpha=0.1, color='yellow')
    ax2.axvspan(25, 30, alpha=0.1, color='blue')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "metrics_over_time.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_reassignments(results_data, output_dir, time_step_minutes=5):
    """Plot reassignment events over time."""
    config = results_data["config"]
    results = results_data["results"]
    reassignment_counts = results_data["reassignment_counts"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    orchestrators = list(results.keys())
    try:
        # Try modern matplotlib API (3.7+)
        cmap = plt.colormaps.get_cmap('tab10')
    except AttributeError:
        # Fall back to older API
        cmap = plt.cm.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(orchestrators)))
    
    for idx, orch_name in enumerate(orchestrators):
        orch_results = results[orch_name]
        if not orch_results:
            continue
        
        # Mark reassignment events
        reassign_times = []
        for r in orch_results:
            if r.get("reassigned", False):
                reassign_times.append(r["step"] * time_step_minutes)
        
        # Plot as vertical lines
        for t in reassign_times:
            ax.axvline(x=t, color=colors[idx], alpha=0.6, linewidth=2, linestyle='--')
        
        # Add dummy line for legend
        ax.plot([], [], color=colors[idx], linewidth=2, linestyle='--', 
                label=f"{orch_name} ({len(reassign_times)} reassignments)")
    
    # Add scenario phase backgrounds
    ax.axvspan(0, 10, alpha=0.1, color='green', label='Normal')
    ax.axvspan(10, 15, alpha=0.1, color='orange', label='Peak Traffic')
    ax.axvspan(15, 20, alpha=0.1, color='red', label='Node Degraded')
    ax.axvspan(20, 25, alpha=0.1, color='yellow', label='Network Congestion')
    ax.axvspan(25, 30, alpha=0.1, color='blue', label='Low Traffic')
    
    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Reassignment Events", fontsize=12)
    ax.set_title("Service Reassignment Events Over Time", fontsize=14, fontweight='bold')
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "reassignments.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_battery_depletion(results_data, output_dir, time_step_minutes=5):
    """Plot cumulative battery depletion over time."""
    config = results_data["config"]
    results = results_data["results"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    orchestrators = list(results.keys())
    
    for orch_name in orchestrators:
        orch_results = results[orch_name]
        if not orch_results:
            continue
        
        times = [r["step"] * time_step_minutes for r in orch_results]
        
        # Calculate cumulative battery usage for each node
        # Sum across all battery nodes
        cumulative_battery = []
        cumulative_sum = 0.0
        
        for r in orch_results:
            battery_usage = r.get("battery_usage", {})
            step_total = sum(usage.get("energy_ah", 0.0) for usage in battery_usage.values())
            cumulative_sum += step_total
            cumulative_battery.append(cumulative_sum)
        
        ax.plot(times, cumulative_battery, marker='o', label=orch_name, linewidth=2)
    
    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Cumulative Battery Drain (Ah)", fontsize=12)
    ax.set_title("Cumulative Battery Energy Consumption", fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add scenario phase backgrounds
    ax.axvspan(0, 10, alpha=0.1, color='green')
    ax.axvspan(10, 15, alpha=0.1, color='orange')
    ax.axvspan(15, 20, alpha=0.1, color='red')
    ax.axvspan(20, 25, alpha=0.1, color='yellow')
    ax.axvspan(25, 30, alpha=0.1, color='blue')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "battery_depletion.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_violations_over_time(results_data, output_dir, time_step_minutes=5):
    """Plot constraint violations over time."""
    config = results_data["config"]
    results = results_data["results"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    orchestrators = list(results.keys())
    
    for orch_name in orchestrators:
        orch_results = results[orch_name]
        if not orch_results:
            continue
        
        times = [r["step"] * time_step_minutes for r in orch_results]
        violations = [r.get("violations", 0) for r in orch_results]
        ax.plot(times, violations, marker='x', label=orch_name, linewidth=2, markersize=8)
    
    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Number of Violations", fontsize=12)
    ax.set_title("Constraint Violations Over Time", fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add scenario phase backgrounds
    ax.axvspan(0, 10, alpha=0.1, color='green')
    ax.axvspan(10, 15, alpha=0.1, color='orange')
    ax.axvspan(15, 20, alpha=0.1, color='red')
    ax.axvspan(20, 25, alpha=0.1, color='yellow')
    ax.axvspan(25, 30, alpha=0.1, color='blue')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "violations_over_time.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_resource_utilization(results_data, output_dir, time_step_minutes=5):
    """Plot resource utilization (CPU, RAM, GPU) over time for each orchestrator."""
    results = results_data["results"]
    orchestrators = list(results.keys())
    
    # Create a figure with 3 subplots (CPU, RAM, GPU)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    for orch_name in orchestrators:
        orch_results = results[orch_name]
        if not orch_results:
            continue
        
        # Check if node_usage data is available
        if not orch_results[0].get("node_usage"):
            print(f"⚠️  Warning: No resource utilization data for {orch_name}. Rerun with updated script.")
            continue
        
        times = [r["step"] * time_step_minutes for r in orch_results]
        
        # Aggregate resource usage across all nodes
        cpu_utils = []
        ram_utils = []
        gpu_utils = []
        
        for r in orch_results:
            node_usage = r.get("node_usage", {})
            if node_usage:
                # Calculate average utilization across all nodes (weighted by capacity)
                total_cpu_used = sum(u["cpu_used"] for u in node_usage.values())
                total_cpu_cap = sum(u["cpu_capacity"] for u in node_usage.values())
                cpu_utils.append((total_cpu_used / total_cpu_cap * 100) if total_cpu_cap > 0 else 0)
                
                total_ram_used = sum(u["ram_used"] for u in node_usage.values())
                total_ram_cap = sum(u["ram_capacity"] for u in node_usage.values())
                ram_utils.append((total_ram_used / total_ram_cap * 100) if total_ram_cap > 0 else 0)
                
                total_gpu_used = sum(u["gpu_used"] for u in node_usage.values())
                total_gpu_cap = sum(u["gpu_capacity"] for u in node_usage.values())
                gpu_utils.append((total_gpu_used / total_gpu_cap * 100) if total_gpu_cap > 0 else 0)
            else:
                cpu_utils.append(0)
                ram_utils.append(0)
                gpu_utils.append(0)
        
        # Plot CPU utilization
        axes[0].plot(times, cpu_utils, marker='o', label=orch_name, linewidth=2)
        
        # Plot RAM utilization
        axes[1].plot(times, ram_utils, marker='s', label=orch_name, linewidth=2)
        
        # Plot GPU utilization
        axes[2].plot(times, gpu_utils, marker='^', label=orch_name, linewidth=2)
    
    # Configure CPU plot
    axes[0].set_ylabel("CPU Utilization (%)", fontsize=12)
    axes[0].set_title("Resource Utilization Over Time", fontsize=14, fontweight='bold')
    axes[0].axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100% capacity')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, max(110, axes[0].get_ylim()[1]))
    
    # Configure RAM plot
    axes[1].set_ylabel("RAM Utilization (%)", fontsize=12)
    axes[1].axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100% capacity')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, max(110, axes[1].get_ylim()[1]))
    
    # Configure GPU plot
    axes[2].set_xlabel("Time (minutes)", fontsize=12)
    axes[2].set_ylabel("GPU Utilization (%)", fontsize=12)
    axes[2].axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100% capacity')
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, max(110, axes[2].get_ylim()[1]))
    
    plt.tight_layout()
    output_path = Path(output_dir) / "resource_utilization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_violations_with_resources(results_data, output_dir, time_step_minutes=5):
    """Plot violations alongside resource utilization for correlation analysis."""
    results = results_data["results"]
    orchestrators = list(results.keys())
    
    # Create a figure with 4 subplots stacked vertically
    fig, axes = plt.subplots(4, 1, figsize=(14, 14))
    
    for orch_name in orchestrators:
        orch_results = results[orch_name]
        if not orch_results:
            continue
        
        times = [r["step"] * time_step_minutes for r in orch_results]
        violations = [r.get("violations", 0) for r in orch_results]
        
        # Plot violations
        axes[0].plot(times, violations, marker='x', label=orch_name, linewidth=2, markersize=8)
        
        # Check if node_usage data is available
        if orch_results[0].get("node_usage"):
            # Aggregate resource usage
            cpu_utils = []
            ram_utils = []
            gpu_utils = []
            
            for r in orch_results:
                node_usage = r.get("node_usage", {})
                if node_usage:
                    total_cpu_used = sum(u["cpu_used"] for u in node_usage.values())
                    total_cpu_cap = sum(u["cpu_capacity"] for u in node_usage.values())
                    cpu_utils.append((total_cpu_used / total_cpu_cap * 100) if total_cpu_cap > 0 else 0)
                    
                    total_ram_used = sum(u["ram_used"] for u in node_usage.values())
                    total_ram_cap = sum(u["ram_capacity"] for u in node_usage.values())
                    ram_utils.append((total_ram_used / total_ram_cap * 100) if total_ram_cap > 0 else 0)
                    
                    total_gpu_used = sum(u["gpu_used"] for u in node_usage.values())
                    total_gpu_cap = sum(u["gpu_capacity"] for u in node_usage.values())
                    gpu_utils.append((total_gpu_used / total_gpu_cap * 100) if total_gpu_cap > 0 else 0)
                else:
                    cpu_utils.append(0)
                    ram_utils.append(0)
                    gpu_utils.append(0)
            
            # Plot resource utilization
            axes[1].plot(times, cpu_utils, marker='o', label=orch_name, linewidth=2)
            axes[2].plot(times, ram_utils, marker='s', label=orch_name, linewidth=2)
            axes[3].plot(times, gpu_utils, marker='^', label=orch_name, linewidth=2)
    
    # Configure plots
    axes[0].set_ylabel("Violations", fontsize=12)
    axes[0].set_title("Constraint Violations vs Resource Utilization", fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_ylabel("CPU Util (%)", fontsize=12)
    axes[1].axhline(y=100, color='r', linestyle='--', alpha=0.5, linewidth=1)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, max(110, axes[1].get_ylim()[1]) if axes[1].has_data() else 110)
    
    axes[2].set_ylabel("RAM Util (%)", fontsize=12)
    axes[2].axhline(y=100, color='r', linestyle='--', alpha=0.5, linewidth=1)
    axes[2].legend(loc='best')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, max(110, axes[2].get_ylim()[1]) if axes[2].has_data() else 110)
    
    axes[3].set_xlabel("Time (minutes)", fontsize=12)
    axes[3].set_ylabel("GPU Util (%)", fontsize=12)
    axes[3].axhline(y=100, color='r', linestyle='--', alpha=0.5, linewidth=1)
    axes[3].legend(loc='best')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_ylim(0, max(110, axes[3].get_ylim()[1]) if axes[3].has_data() else 110)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "violations_with_resources.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def plot_per_node_utilization(results_data, output_dir, time_step_minutes=5, orchestrator_name=None):
    """Plot detailed per-node resource utilization for a specific orchestrator."""
    results = results_data["results"]
    
    # Use first orchestrator if not specified
    if orchestrator_name is None:
        orchestrator_name = list(results.keys())[0]
    
    orch_results = results.get(orchestrator_name, [])
    if not orch_results or not orch_results[0].get("node_usage"):
        print(f"⚠️  Warning: No resource utilization data for {orchestrator_name}")
        return
    
    # Get all node IDs
    node_ids = list(orch_results[0]["node_usage"].keys())
    times = [r["step"] * time_step_minutes for r in orch_results]
    
    # Create subplots for each node
    num_nodes = len(node_ids)
    fig, axes = plt.subplots(num_nodes, 1, figsize=(14, 4 * num_nodes), squeeze=False)
    axes = axes.flatten()
    
    for idx, node_id in enumerate(node_ids):
        ax = axes[idx]
        
        # Extract data for this node
        cpu_utils = []
        ram_utils = []
        gpu_utils = []
        
        for r in orch_results:
            node_data = r["node_usage"].get(node_id, {})
            cpu_utils.append(node_data.get("cpu_util_pct", 0))
            ram_utils.append(node_data.get("ram_util_pct", 0))
            gpu_utils.append(node_data.get("gpu_util_pct", 0))
        
        # Plot all resources for this node
        ax.plot(times, cpu_utils, marker='o', label='CPU', linewidth=2)
        ax.plot(times, ram_utils, marker='s', label='RAM', linewidth=2)
        ax.plot(times, gpu_utils, marker='^', label='GPU', linewidth=2)
        
        ax.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100% capacity')
        ax.set_ylabel("Utilization (%)", fontsize=11)
        ax.set_title(f"Node: {node_id}", fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(110, ax.get_ylim()[1]))
    
    axes[-1].set_xlabel("Time (minutes)", fontsize=12)
    fig.suptitle(f"Per-Node Resource Utilization - {orchestrator_name}", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = Path(output_dir) / f"per_node_utilization_{orchestrator_name.replace(' ', '_').lower()}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize dynamic orchestrator comparison results")
    parser.add_argument("--results", type=str, default="out/dynamic_comparison/dynamic_results.json",
                        help="Path to JSON results file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots (default: same as results file)")
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results}")
    results_data = load_results(args.results)
    
    # Get time_step_minutes from config or use default
    config = results_data.get("config", {})
    if "time_step_minutes" in config:
        time_step_minutes = config["time_step_minutes"]
    else:
        time_step_minutes = 5  # Default to 5 minutes
        print(f"Warning: time_step_minutes not found in config, using default: {time_step_minutes} minutes")
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Path(args.results).parent
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating visualizations...")
    print(f"Output directory: {output_dir}\n")
    
    # Generate plots
    plot_metrics_over_time(results_data, output_dir)
    plot_reassignments(results_data, output_dir, time_step_minutes)
    plot_battery_depletion(results_data, output_dir, time_step_minutes)
    plot_violations_over_time(results_data, output_dir, time_step_minutes)
    
    # Generate resource utilization plots
    plot_resource_utilization(results_data, output_dir, time_step_minutes)
    plot_violations_with_resources(results_data, output_dir, time_step_minutes)
    
    # Generate per-node utilization for each orchestrator
    for orch_name in results_data["results"].keys():
        plot_per_node_utilization(results_data, output_dir, time_step_minutes, orch_name)
    
    print(f"\n✅ All visualizations generated successfully!")


if __name__ == "__main__":
    main()
