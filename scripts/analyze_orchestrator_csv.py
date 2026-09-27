#!/usr/bin/env python3
"""
Example script for analyzing orchestrator CSV logs.

This demonstrates how to load and analyze the CSV outputs from compare_orchestrators.py
"""

import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime


def find_latest_csv_session(base_dir="out"):
    """Find the most recent CSV session directory."""
    out_dir = Path(base_dir)
    if not out_dir.exists():
        return None
    
    # Find all timestamp directories
    dirs = [d for d in out_dir.iterdir() if d.is_dir() and len(d.name) == 15]  # YYYYMMDD_HHMMSS
    if not dirs:
        return None
    
    # Return the most recent one
    return max(dirs, key=lambda d: d.name)


def analyze_orchestrator_csv(csv_path):
    """Load and analyze a single orchestrator CSV file."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {Path(csv_path).name}")
    print(f"{'='*80}\n")
    
    df = pd.read_csv(csv_path)
    
    # Basic statistics
    print(f"Total iterations: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    
    # Execution time statistics
    if 'orchestrator_execution_time_ms' in df.columns:
        exec_times = df['orchestrator_execution_time_ms'].astype(float)
        print(f"\n⏱️  Execution Time Statistics (ms):")
        print(f"  Mean:    {exec_times.mean():.2f}")
        print(f"  Min:     {exec_times.min():.2f}")
        print(f"  Max:     {exec_times.max():.2f}")
        print(f"  Std Dev: {exec_times.std():.2f}")
    
    # Constraint violations
    constraint_cols = [col for col in df.columns if 'constraint_status' in col]
    if constraint_cols:
        print(f"\n🛡️  Constraint Violations:")
        for col in constraint_cols:
            failures = (df[col] == "FAIL").sum()
            total = len(df)
            pct = (failures / total * 100) if total > 0 else 0
            status = f"{failures}/{total} ({pct:.1f}%)"
            print(f"  {col.replace('_constraint_status', '')}: {status}")
        
        # Overall violation rate
        total_violations = 0
        for col in constraint_cols:
            total_violations += (df[col] == "FAIL").sum()
        overall_pct = (total_violations / (len(df) * len(constraint_cols)) * 100) if len(constraint_cols) > 0 else 0
        print(f"  Overall violation rate: {overall_pct:.1f}%")
    
    # Allocation changes
    allocation_pairs = []
    for col in df.columns:
        if col.endswith('_allocation'):
            service_name = col.replace('_allocation', '')
            current_col = f"{service_name}_current_node"
            if current_col in df.columns:
                allocation_pairs.append((service_name, col, current_col))
    
    if allocation_pairs:
        print(f"\n🔄 Allocation Changes:")
        for service_name, alloc_col, current_col in allocation_pairs:
            changes = (df[alloc_col] != df[current_col]).sum()
            pct = (changes / len(df) * 100) if len(df) > 0 else 0
            print(f"  {service_name}: {changes} changes ({pct:.1f}%)")
    
    # Latency statistics
    latency_cols = [col for col in df.columns if 'latency_to_frank-2_ms' in col]
    if latency_cols:
        print(f"\n🌐 Latency to frank-2 (ms):")
        for col in latency_cols:
            node_name = col.split('_latency')[0]
            latencies = pd.to_numeric(df[col], errors='coerce')
            if not latencies.isna().all():
                print(f"  {node_name}:")
                print(f"    Mean: {latencies.mean():.4f}")
                print(f"    Min:  {latencies.min():.4f}")
                print(f"    Max:  {latencies.max():.4f}")
    
    return df


def compare_orchestrators(session_dir):
    """Compare all orchestrators in a session."""
    csv_files = list(Path(session_dir).glob("*.csv"))
    # Exclude latency_matrix.csv from comparison
    csv_files = [f for f in csv_files if f.name != "latency_matrix.csv"]
    
    if not csv_files:
        print("❌ No orchestrator CSV files found in session")
        return
    
    print(f"\n{'='*80}")
    print(f"Orchestrator Comparison Summary")
    print(f"Session: {session_dir.name}")
    print(f"{'='*80}\n")
    
    results = {}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        orch_name = csv_file.stem
        
        exec_times = pd.to_numeric(df['orchestrator_execution_time_ms'], errors='coerce')
        constraint_cols = [col for col in df.columns if 'constraint_status' in col]
        
        violations = 0
        for col in constraint_cols:
            violations += (df[col] == "FAIL").sum()
        violation_pct = (violations / (len(df) * len(constraint_cols)) * 100) if len(constraint_cols) > 0 else 0
        
        results[orch_name] = {
            'avg_exec_time': exec_times.mean(),
            'min_exec_time': exec_times.min(),
            'max_exec_time': exec_times.max(),
            'violations': violations,
            'violation_pct': violation_pct,
            'iterations': len(df)
        }
    
    # Print comparison table
    print(f"{'Orchestrator':<20} {'Avg Time (ms)':<15} {'Violations (%)':<15} {'Iterations':<12}")
    print(f"{'-'*62}")
    for orch, metrics in sorted(results.items()):
        print(f"{orch:<20} {metrics['avg_exec_time']:>13.2f}  {metrics['violation_pct']:>13.1f}  {metrics['iterations']:>10}")
    
    if len(results) > 1:
        print(f"\n💡 Fastest orchestrator: {min(results, key=lambda x: results[x]['avg_exec_time'])}")
        print(f"✅ Most stable orchestrator: {min(results, key=lambda x: results[x]['violation_pct'])}")


def main():
    """Main analysis function."""
    # Find latest session or use provided path
    if len(sys.argv) > 1:
        session_path = Path(sys.argv[1])
    else:
        session_path = find_latest_csv_session()
        if not session_path:
            print("❌ No CSV session found. Run BlazeMaestro first.")
            return
    
    print(f"📊 CSV Analysis Tool")
    print(f"Session: {session_path}\n")
    
    # Check for latency matrix
    latency_file = session_path / "latency_matrix.csv"
    if latency_file.exists():
        print(f"🌐 Latency matrix found: {latency_file.name}")
        try:
            lat_df = pd.read_csv(latency_file, index_col=0)
            print(f"   Nodes: {len(lat_df)}")
            print(f"   Average latency: {lat_df.values.mean():.4f} ms")
            print(f"   Max latency: {lat_df.values.max():.4f} ms\n")
        except Exception:
            print(f"   (Unable to parse)\n")
    
    # Analyze CSV files
    csv_files = list(session_path.glob("*.csv"))
    # Exclude latency_matrix.csv from orchestrator analysis
    csv_files = [f for f in csv_files if f.name != "latency_matrix.csv"]
    
    if not csv_files:
        print("❌ No orchestrator CSV files found in session directory")
        return
    
    dataframes = {}
    for csv_file in csv_files:
        df = analyze_orchestrator_csv(csv_file)
        dataframes[csv_file.stem] = df
    
    # Compare orchestrators if multiple
    if len(csv_files) > 1:
        compare_orchestrators(session_path)
    
    print(f"\n{'='*80}")
    print(f"✅ Analysis complete!")
    print(f"Session directory: {session_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
