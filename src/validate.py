#!/usr/bin/env python3
"""
Validates reassignment behavior to distinguish between:
1. Healthy reassignments: Adapting to changing conditions (good)
2. Thrashing: Constantly reassigning without clear benefit (bad)

Metrics:
- Reassignment rate: How often reassignments occur
- Flip-flop ratio: Services that swap between same 2-3 nodes repeatedly
- Stability periods: Duration between reassignments
- Benefit analysis: Energy/latency improvement per reassignment
- Churn score: Weighted metric combining frequency and flip-flops
"""

import argparse
import json
import os
import numpy as np
from collections import defaultdict
from tabulate import tabulate


def analyze_reassignment_patterns(results, orchestrator_name):
    """
    Analyze reassignment patterns for a single orchestrator.
    
    Returns:
        dict: Analysis results including metrics
    """
    if not results:
        return None
    
    analysis = {
        "orchestrator": orchestrator_name,
        "total_steps": len(results),
        "total_reassignments": 0,
        "service_reassignments": defaultdict(int),
        "service_flip_flops": defaultdict(int),
        "service_unique_nodes": defaultdict(set),
        "service_node_sequence": defaultdict(list),
        "stability_periods": [],
        "reassignment_benefits": [],
        "thrashing_services": [],
        "churn_score": 0.0
    }
    
    # Track per-service assignments over time
    for step_idx, result in enumerate(results):
        assignments = result.get("assignments", {})
        
        for service_id, node_id in assignments.items():
            analysis["service_node_sequence"][service_id].append(node_id)
            analysis["service_unique_nodes"][service_id].add(node_id)
    
    # Analyze each service's reassignment pattern
    for service_id, node_sequence in analysis["service_node_sequence"].items():
        if len(node_sequence) < 2:
            continue
        
        # Count reassignments for this service
        reassignments = 0
        flip_flops = 0
        stability_streak = 0
        
        for i in range(1, len(node_sequence)):
            if node_sequence[i] != node_sequence[i-1]:
                reassignments += 1
                analysis["service_reassignments"][service_id] += 1
                
                # Record stability period before this reassignment
                if stability_streak > 0:
                    analysis["stability_periods"].append(stability_streak)
                stability_streak = 0
                
                # Check for flip-flop (returns to a node seen recently)
                if i >= 2 and node_sequence[i] == node_sequence[i-2]:
                    flip_flops += 1
                    analysis["service_flip_flops"][service_id] += 1
            else:
                stability_streak += 1
        
        # Record final stability period
        if stability_streak > 0:
            analysis["stability_periods"].append(stability_streak)
    
    # Calculate aggregate metrics
    analysis["total_reassignments"] = sum(analysis["service_reassignments"].values())
    analysis["total_flip_flops"] = sum(analysis["service_flip_flops"].values())
    
    # Identify thrashing services (high flip-flop ratio)
    for service_id in analysis["service_reassignments"]:
        reassignments = analysis["service_reassignments"][service_id]
        flip_flops = analysis["service_flip_flops"][service_id]
        
        if reassignments > 0:
            flip_flop_ratio = flip_flops / reassignments
            # Consider thrashing if >40% of reassignments are flip-flops
            if flip_flop_ratio > 0.4 and reassignments >= 3:
                analysis["thrashing_services"].append({
                    "service": service_id,
                    "reassignments": reassignments,
                    "flip_flops": flip_flops,
                    "flip_flop_ratio": flip_flop_ratio,
                    "unique_nodes": len(analysis["service_unique_nodes"][service_id]),
                    "nodes": list(analysis["service_unique_nodes"][service_id])
                })
    
    # Calculate churn score (0-100, lower is better)
    # Factors: reassignment rate, flip-flop ratio, stability variance
    if len(results) > 1:
        reassignment_rate = analysis["total_reassignments"] / (len(results) - 1)
        flip_flop_rate = analysis["total_flip_flops"] / max(1, analysis["total_reassignments"])
        
        # Stability variance (high variance = unpredictable)
        stability_variance = np.var(analysis["stability_periods"]) if analysis["stability_periods"] else 0
        stability_std = np.sqrt(stability_variance)
        
        # Churn score components (normalized to 0-100)
        churn_from_rate = min(100, reassignment_rate * 20)  # 5+ reassignments/step = 100
        churn_from_flops = flip_flop_rate * 50  # 100% flip-flops = 50
        churn_from_variance = min(30, stability_std * 3)  # High variance adds up to 30
        
        analysis["churn_score"] = churn_from_rate + churn_from_flops + churn_from_variance
    
    # Analyze benefit of reassignments (if latency/energy data available)
    prev_latency = None
    prev_energy = None
    for step_idx, result in enumerate(results):
        if step_idx == 0:
            prev_latency = result.get("latency")
            prev_energy = result.get("energy")
            continue
        
        reassigned = result.get("reassigned", False)
        if reassigned and prev_latency is not None and prev_energy is not None:
            curr_latency = result.get("latency")
            curr_energy = result.get("energy")
            
            latency_improvement = prev_latency - curr_latency
            energy_improvement = prev_energy - curr_energy
            
            analysis["reassignment_benefits"].append({
                "step": step_idx,
                "latency_improvement": latency_improvement,
                "energy_improvement": energy_improvement,
                "beneficial": latency_improvement > 0 or energy_improvement > 0
            })
        
        prev_latency = result.get("latency")
        prev_energy = result.get("energy")
    
    return analysis


def print_stability_report(analyses):
    """Print a comprehensive stability report for all orchestrators."""
    
    print(f"\n{'='*100}")
    print(f"REASSIGNMENT STABILITY ANALYSIS")
    print(f"{'='*100}\n")
    
    # Overview table
    overview_table = []
    for name, analysis in analyses.items():
        if analysis is None:
            continue
        
        total_steps = analysis["total_steps"]
        total_reassignments = analysis["total_reassignments"]
        total_flip_flops = analysis["total_flip_flops"]
        churn_score = analysis["churn_score"]
        
        reassignment_rate = total_reassignments / max(1, total_steps - 1)
        flip_flop_ratio = total_flip_flops / max(1, total_reassignments)
        
        avg_stability = np.mean(analysis["stability_periods"]) if analysis["stability_periods"] else 0
        
        # Status emoji based on churn score
        if churn_score < 20:
            status = "✅ Stable"
        elif churn_score < 50:
            status = "⚠️  Moderate"
        else:
            status = "❌ Thrashing"
        
        overview_table.append([
            name,
            total_steps,
            total_reassignments,
            f"{reassignment_rate:.2f}",
            total_flip_flops,
            f"{flip_flop_ratio:.1%}",
            f"{avg_stability:.1f}",
            f"{churn_score:.1f}",
            status
        ])
    
    print(tabulate(
        overview_table,
        headers=["Orchestrator", "Steps", "Reassignments", "Rate", "Flip-Flops", "FF Ratio", "Avg Stability", "Churn Score", "Status"],
        tablefmt="grid"
    ))
    
    # Detailed per-orchestrator analysis
    for name, analysis in analyses.items():
        if analysis is None or analysis["total_reassignments"] == 0:
            continue
        
        print(f"\n{'─'*100}")
        print(f"📊 {name} - Detailed Analysis")
        print(f"{'─'*100}\n")
        
        # Service-level reassignment breakdown
        if analysis["service_reassignments"]:
            service_table = []
            for service_id in sorted(analysis["service_reassignments"].keys()):
                reassignments = analysis["service_reassignments"][service_id]
                flip_flops = analysis["service_flip_flops"][service_id]
                unique_nodes = len(analysis["service_unique_nodes"][service_id])
                
                # Get node sequence summary
                sequence = analysis["service_node_sequence"][service_id]
                if len(sequence) > 5:
                    sequence_str = f"{sequence[0]} → ... → {sequence[-1]}"
                else:
                    sequence_str = " → ".join(sequence)
                
                service_table.append([
                    service_id,
                    reassignments,
                    flip_flops,
                    unique_nodes,
                    sequence_str
                ])
            
            print("Per-Service Reassignments:")
            print(tabulate(
                service_table,
                headers=["Service", "Reassignments", "Flip-Flops", "Unique Nodes", "Node Sequence"],
                tablefmt="simple"
            ))
            print()
        
        # Thrashing services (if any)
        if analysis["thrashing_services"]:
            print(f"⚠️  WARNING: {len(analysis['thrashing_services'])} service(s) showing thrashing behavior:\n")
            for ts in analysis["thrashing_services"]:
                print(f"  • {ts['service']}:")
                print(f"    - Reassignments: {ts['reassignments']}")
                print(f"    - Flip-flops: {ts['flip_flops']} ({ts['flip_flop_ratio']:.1%})")
                print(f"    - Nodes: {', '.join(ts['nodes'])}")
                print()
        
        # Reassignment benefit analysis
        if analysis["reassignment_benefits"]:
            beneficial = sum(1 for b in analysis["reassignment_benefits"] if b["beneficial"])
            total_benefits = len(analysis["reassignment_benefits"])
            benefit_ratio = beneficial / total_benefits
            
            avg_latency_improvement = np.mean([b["latency_improvement"] for b in analysis["reassignment_benefits"]])
            avg_energy_improvement = np.mean([b["energy_improvement"] for b in analysis["reassignment_benefits"]])
            
            print(f"Reassignment Benefits:")
            print(f"  • Beneficial reassignments: {beneficial}/{total_benefits} ({benefit_ratio:.1%})")
            print(f"  • Avg latency improvement: {avg_latency_improvement*1000:.2f} ms")
            print(f"  • Avg energy improvement: {avg_energy_improvement:.2f} J")
            print()
        
        # Stability period distribution
        if analysis["stability_periods"]:
            print(f"Stability Periods:")
            print(f"  • Min: {min(analysis['stability_periods'])} steps")
            print(f"  • Avg: {np.mean(analysis['stability_periods']):.1f} steps")
            print(f"  • Max: {max(analysis['stability_periods'])} steps")
            print(f"  • Std Dev: {np.std(analysis['stability_periods']):.1f} steps")
            print()


def main():
    parser = argparse.ArgumentParser(description="Validate reassignment stability from comparison results.")
    parser.add_argument("--results", type=str, required=True, help="Path to dynamic_results.json")
    parser.add_argument("--output-dir", type=str, help="Output directory for stability report (optional)")
    parser.add_argument("--churn-threshold", type=float, default=50.0, help="Churn score threshold for warning (default: 50)")
    args = parser.parse_args()
    
    # Load results
    if not os.path.exists(args.results):
        print(f"❌ Error: Results file not found: {args.results}")
        return 1
    
    with open(args.results, 'r') as f:
        data = json.load(f)
    
    results = data.get("results", {})
    config = data.get("config", {})
    
    print(f"\n📁 Loaded results from: {args.results}")
    print(f"⚙️  Configuration:")
    print(f"   • Duration: {config.get('duration_minutes', 'N/A')} minutes")
    print(f"   • Time step: {config.get('time_step_minutes', 'N/A')} minutes")
    print(f"   • Hysteresis: {config.get('hysteresis_threshold', 'N/A')}")
    
    # Analyze each orchestrator
    analyses = {}
    for orchestrator_name, orchestrator_results in results.items():
        analyses[orchestrator_name] = analyze_reassignment_patterns(orchestrator_results, orchestrator_name)
    
    # Print comprehensive report
    print_stability_report(analyses)
    
    # Overall assessment
    print(f"\n{'='*100}")
    print(f"OVERALL ASSESSMENT")
    print(f"{'='*100}\n")
    
    stable_count = sum(1 for a in analyses.values() if a and a["churn_score"] < 20)
    moderate_count = sum(1 for a in analyses.values() if a and 20 <= a["churn_score"] < 50)
    thrashing_count = sum(1 for a in analyses.values() if a and a["churn_score"] >= 50)
    
    print(f"Orchestrator Stability Distribution:")
    print(f"  ✅ Stable (churn < 20): {stable_count}")
    print(f"  ⚠️  Moderate (20 ≤ churn < 50): {moderate_count}")
    print(f"  ❌ Thrashing (churn ≥ 50): {thrashing_count}")
    print()
    
    # Key recommendations
    print("Recommendations:")
    for name, analysis in analyses.items():
        if analysis is None:
            continue
        
        churn = analysis["churn_score"]
        if churn >= args.churn_threshold:
            print(f"  ⚠️  {name}: Consider increasing hysteresis threshold to reduce churn")
        
        if analysis["thrashing_services"]:
            print(f"  ⚠️  {name}: {len(analysis['thrashing_services'])} service(s) thrashing - review placement logic")
        
        if analysis["reassignment_benefits"]:
            beneficial_ratio = sum(1 for b in analysis["reassignment_benefits"] if b["beneficial"]) / len(analysis["reassignment_benefits"])
            if beneficial_ratio < 0.5:
                print(f"  ⚠️  {name}: Only {beneficial_ratio:.1%} of reassignments are beneficial - review reward function")
    
    # Save detailed report if output directory specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save JSON with full analysis
        output_file = os.path.join(args.output_dir, "stability_analysis.json")
        with open(output_file, 'w') as f:
            # Convert defaultdict and sets to regular dicts/lists for JSON
            serializable_analyses = {}
            for name, analysis in analyses.items():
                if analysis is None:
                    continue
                
                serializable = dict(analysis)
                serializable["service_reassignments"] = dict(serializable["service_reassignments"])
                serializable["service_flip_flops"] = dict(serializable["service_flip_flops"])
                serializable["service_unique_nodes"] = {k: list(v) for k, v in serializable["service_unique_nodes"].items()}
                serializable["service_node_sequence"] = dict(serializable["service_node_sequence"])
                serializable_analyses[name] = serializable
            
            json.dump({
                "config": config,
                "analyses": serializable_analyses
            }, f, indent=2)
        
        print(f"\n💾 Detailed analysis saved to: {output_file}")
    
    print()
    return 0


if __name__ == "__main__":
    exit(main())
