#!/usr/bin/env python3
"""
Latency Matrix Evolution Analyzer

Analyzes CSV files to track how latency differences evolve over time,
detecting trends and anomalies in simulator accuracy.

Usage:
    python analyze_latency_evolution.py [--csv-file FILE] [--output DIR]
"""

import argparse
import sys
from pathlib import Path
import csv
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def analyze_csv(csv_file: str) -> dict:
    """Analyze latency comparison CSV file."""
    stats = {
        "total_links": 0,
        "by_layer_pair": defaultdict(lambda: {
            "count": 0,
            "mean_diff": 0.0,
            "max_diff": 0.0,
            "mean_error": 0.0,
        }),
        "by_source_layer": defaultdict(lambda: {
            "count": 0,
            "mean_diff": 0.0,
            "max_diff": 0.0,
            "mean_error": 0.0,
        }),
        "timestamp": None,
        "run_id": None,
    }

    layer_pair_diffs = defaultdict(list)
    layer_pair_errors = defaultdict(list)
    source_diffs = defaultdict(list)
    source_errors = defaultdict(list)

    try:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stats["timestamp"] = row.get("timestamp")
                stats["run_id"] = row.get("run_id")
                
                diff = float(row.get("diff_ms", 0))
                error = float(row.get("error_percent", 0).rstrip("%"))
                
                from_layer = row.get("from_layer")
                to_layer = row.get("to_layer")
                layer_pair = f"{from_layer}→{to_layer}"
                
                layer_pair_diffs[layer_pair].append(diff)
                layer_pair_errors[layer_pair].append(error)
                source_diffs[from_layer].append(diff)
                source_errors[from_layer].append(error)
                
                stats["total_links"] += 1

        # Calculate aggregates
        for layer_pair, diffs in layer_pair_diffs.items():
            errors = layer_pair_errors[layer_pair]
            stats["by_layer_pair"][layer_pair] = {
                "count": len(diffs),
                "mean_diff": sum(diffs) / len(diffs),
                "max_diff": max(diffs),
                "mean_error": sum(errors) / len(errors),
            }

        for source, diffs in source_diffs.items():
            errors = source_errors[source]
            stats["by_source_layer"][source] = {
                "count": len(diffs),
                "mean_diff": sum(diffs) / len(diffs),
                "max_diff": max(diffs),
                "mean_error": sum(errors) / len(errors),
            }

    except Exception as e:
        print(f"✗ Error reading CSV: {e}")
        return None

    return stats


def print_analysis(stats: dict):
    """Print analysis results."""
    print("\n" + "=" * 70)
    print("LATENCY EVOLUTION ANALYSIS")
    print("=" * 70)
    print(f"Run ID: {stats['run_id']}")
    print(f"Timestamp: {stats['timestamp']}")
    print(f"Total links analyzed: {stats['total_links']}")
    print()

    print("By Layer Pair:")
    print("-" * 70)
    print(f"{'Pair':<15} {'Count':<8} {'Mean Δ (ms)':<15} {'Max Δ (ms)':<15} {'Err %':<10}")
    print("-" * 70)
    
    for pair in sorted(stats["by_layer_pair"].keys()):
        data = stats["by_layer_pair"][pair]
        print(
            f"{pair:<15} {data['count']:<8} "
            f"{data['mean_diff']:<15.4f} {data['max_diff']:<15.4f} "
            f"{data['mean_error']:<10.2f}%"
        )

    print()
    print("By Source Layer:")
    print("-" * 70)
    print(f"{'Layer':<15} {'Count':<8} {'Mean Δ (ms)':<15} {'Max Δ (ms)':<15} {'Err %':<10}")
    print("-" * 70)
    
    for layer in sorted(stats["by_source_layer"].keys()):
        data = stats["by_source_layer"][layer]
        print(
            f"{layer:<15} {data['count']:<8} "
            f"{data['mean_diff']:<15.4f} {data['max_diff']:<15.4f} "
            f"{data['mean_error']:<10.2f}%"
        )

    print("=" * 70 + "\n")


def compare_evolutions(csv_files: list):
    """Compare latency evolution across multiple CSV runs."""
    print("\n" + "=" * 70)
    print("LATENCY EVOLUTION COMPARISON")
    print("=" * 70)

    all_stats = []
    for csv_file in csv_files:
        stats = analyze_csv(csv_file)
        if stats:
            all_stats.append((csv_file, stats))

    # Print per-file analysis
    for csv_file, stats in all_stats:
        print(f"\nFile: {csv_file}")
        print(f"Run: {stats['run_id']} at {stats['timestamp']}")
        
        # Find overall statistics
        by_pair = stats["by_layer_pair"]
        if by_pair:
            mean_diffs = [d["mean_diff"] for d in by_pair.values()]
            max_diffs = [d["max_diff"] for d in by_pair.values()]
            mean_errors = [d["mean_error"] for d in by_pair.values()]
            
            print(f"  Mean difference (overall): {sum(mean_diffs)/len(mean_diffs):.4f} ms")
            print(f"  Max difference (overall): {max(max_diffs):.4f} ms")
            print(f"  Mean error (overall): {sum(mean_errors)/len(mean_errors):.2f}%")

    # Detect trends
    if len(all_stats) > 1:
        print("\n" + "-" * 70)
        print("TREND ANALYSIS")
        print("-" * 70)
        
        # Track same layer pairs across runs
        for layer_pair in ["edge→edge", "fog→fog", "cloud→cloud", "edge→fog", "fog→edge"]:
            diffs = []
            for _, stats in all_stats:
                if layer_pair in stats["by_layer_pair"]:
                    diffs.append(stats["by_layer_pair"][layer_pair]["mean_diff"])
            
            if diffs:
                trend = "↑ increasing" if diffs[-1] > diffs[0] else "↓ decreasing" if diffs[-1] < diffs[0] else "→ stable"
                print(f"{layer_pair:<15} {trend:<20} {diffs[0]:.4f}ms → {diffs[-1]:.4f}ms")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze latency matrix evolution from CSV files"
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        help="Single CSV file to analyze",
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        help="Directory containing multiple CSV files for evolution analysis",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="out/latency_analysis",
        help="Output directory for analysis results",
    )

    args = parser.parse_args()

    if args.csv_file:
        # Analyze single file
        stats = analyze_csv(args.csv_file)
        if stats:
            print_analysis(stats)
        else:
            sys.exit(1)

    elif args.csv_dir:
        # Analyze directory of CSVs
        csv_dir = Path(args.csv_dir)
        csv_files = sorted(csv_dir.glob("latency_comparison*.csv"))
        
        if not csv_files:
            print(f"✗ No CSV files found in {args.csv_dir}")
            sys.exit(1)
        
        print(f"Found {len(csv_files)} CSV files in {args.csv_dir}")
        compare_evolutions([str(f) for f in csv_files])

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
