#!/usr/bin/env python3
"""
Generate publication-quality figures from test results and stability analysis.

This script creates 3 PDF figures for the paper:
1. Performance Comparison: Latency vs Energy trade-off
2. Dynamic Scenario Performance: Metrics over time across scenarios
3. Stability Analysis: Reassignments and churn metrics

Output: doc/img/*.pdf files ready for LaTeX inclusion
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path
import sys


class PaperFigureGenerator:
    """Generate publication-quality figures for IEEE paper."""
    
    def __init__(self, test_results_path: str, stability_analysis_path: str):
        """Initialize with paths to result JSON files."""
        self.test_results_path = Path(test_results_path)
        self.stability_analysis_path = Path(stability_analysis_path)
        self.output_dir = Path("doc/img")
        
        # Create output directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color palette for orchestrators (publication-ready)
        self.colors = {
            'NSGA-II': '#1f77b4',      # Blue
            'DQN': '#2ca02c',          # Green
            'Unsupervised': '#d62728', # Red
            'Autoencoder': '#9467bd',  # Purple
        }
        
        # Load data
        self.test_data = self._load_json(self.test_results_path)
        self.stability_data = self._load_json(self.stability_analysis_path)
    
    def _load_json(self, path: Path) -> dict:
        """Load JSON file safely."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            sys.exit(1)
    
    def generate_all(self):
        """Generate all figures."""
        print("Generating paper figures...")
        self.figure_1_performance_comparison()
        self.figure_2_dynamic_scenarios()
        self.figure_3_stability_metrics()
        print("✓ All figures generated successfully!")
    
    def figure_1_performance_comparison(self):
        """
        Figure 1: Performance Comparison (Latency vs Energy Trade-off).
        
        Shows latency and energy metrics for each orchestrator with dual axes.
        """
        print("  → Generating Figure 1: Performance Comparison...")
        
        results = self.test_data['results']
        orchestrators = sorted(results.keys())
        
        # Extract metrics
        latencies = []
        energies = []
        
        for orch in orchestrators:
            orch_results = results[orch]
            avg_latency = np.mean([r.get('latency', 0) for r in orch_results])
            avg_energy = np.mean([r.get('energy', 0) for r in orch_results])
            latencies.append(avg_latency)
            energies.append(avg_energy)
        
        # Create figure
        fig, ax1 = plt.subplots(figsize=(10, 6), dpi=150)
        fig.tight_layout(pad=3.0)
        
        x = np.arange(len(orchestrators))
        width = 0.35
        
        # Plot latency on primary y-axis
        bars1 = ax1.bar(x - width/2, latencies, width, label='Avg Latency', 
                        color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Orchestrator', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Latency (seconds)', fontsize=12, fontweight='bold', color='#1f77b4')
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        ax1.set_xticks(x)
        ax1.set_xticklabels(orchestrators, fontsize=11)
        
        # Plot energy on secondary y-axis
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width/2, energies, width, label='Avg Energy', 
                        color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Energy (Joules)', fontsize=12, fontweight='bold', color='#ff7f0e')
        ax2.tick_params(axis='y', labelcolor='#ff7f0e')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars1, latencies)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        for i, (bar, val) in enumerate(zip(bars2, energies)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        
        # Title and grid
        fig.suptitle('Orchestration Performance Comparison', fontsize=14, fontweight='bold', y=0.98)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_axisbelow(True)
        
        # Save
        output_path = self.output_dir / 'figure_01_performance_comparison.pdf'
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"    ✓ Saved: {output_path}")
    
    def figure_2_dynamic_scenarios(self):
        """
        Figure 2: Dynamic Scenario Performance Over Time.
        
        Shows how latency and energy evolve across 12 dynamic test scenarios.
        """
        print("  → Generating Figure 2: Dynamic Scenario Performance...")
        
        results = self.test_data['results']
        scenarios = self.test_data['config']['scenarios']
        orchestrators = sorted(results.keys())
        
        # Create subplots: latency and energy
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=150)
        fig.tight_layout(pad=2.5)
        
        steps = np.arange(len(scenarios))
        
        # Plot latency over scenarios
        for orch in orchestrators:
            orch_results = results[orch]
            latencies = [r.get('latency', 0) for r in orch_results]
            ax1.plot(steps, latencies, marker='o', linewidth=2.5, markersize=6,
                    label=orch, color=self.colors[orch], alpha=0.8)
        
        ax1.set_ylabel('Latency (seconds)', fontsize=11, fontweight='bold')
        ax1.set_title('Average End-to-End Latency Across Scenarios', fontsize=12, fontweight='bold', pad=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='best', fontsize=10, ncol=2)
        ax1.set_xticks(steps)
        ax1.set_xticklabels([f'S{i}\n{s}' for i, s in enumerate(scenarios)], 
                            fontsize=9, rotation=45, ha='right')
        
        # Plot energy over scenarios
        for orch in orchestrators:
            orch_results = results[orch]
            energies = [r.get('energy', 0) for r in orch_results]
            ax2.plot(steps, energies, marker='s', linewidth=2.5, markersize=6,
                    label=orch, color=self.colors[orch], alpha=0.8)
        
        ax2.set_xlabel('Scenario', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Energy (Joules)', fontsize=11, fontweight='bold')
        ax2.set_title('Total Energy Consumption Across Scenarios', fontsize=12, fontweight='bold', pad=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(loc='best', fontsize=10, ncol=2)
        ax2.set_xticks(steps)
        ax2.set_xticklabels([f'S{i}\n{s}' for i, s in enumerate(scenarios)], 
                            fontsize=9, rotation=45, ha='right')
        
        fig.suptitle('Dynamic Orchestration Performance Under Varying Scenarios', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        # Save
        output_path = self.output_dir / 'figure_02_dynamic_scenarios.pdf'
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"    ✓ Saved: {output_path}")
    
    def figure_3_stability_metrics(self):
        """
        Figure 3: Placement Stability and Churn Analysis.
        
        Shows reassignments, flip-flops, and churn scores with thresholds.
        """
        print("  → Generating Figure 3: Stability Analysis...")
        
        analyses = self.stability_data['analyses']
        orchestrators = sorted(analyses.keys())
        
        # Extract metrics
        reassignments = []
        flip_flops = []
        churn_scores = []
        
        for orch in orchestrators:
            analysis = analyses[orch]
            total_steps = analysis['total_steps']
            total_reassign = analysis['total_reassignments']
            total_ff = analysis['total_flip_flops']
            churn = analysis['churn_score']
            
            reassignments.append(total_reassign)
            flip_flops.append(total_ff)
            churn_scores.append(churn)
        
        # Create figure with GridSpec for better layout
        fig = plt.figure(figsize=(12, 7), dpi=150)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, :])  # Top: reassignments
        ax2 = fig.add_subplot(gs[1, 0])  # Bottom left: flip-flops
        ax3 = fig.add_subplot(gs[1, 1])  # Bottom right: churn score
        
        x = np.arange(len(orchestrators))
        width = 0.6
        
        # Subplot 1: Reassignments (stacked: reassignments vs flip-flops)
        bars1a = ax1.bar(x, reassignments, width, label='Reassignments', 
                        color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars1b = ax1.bar(x, flip_flops, width, bottom=reassignments, 
                        label='Flip-Flops (within reassignments)', 
                        color='#d62728', alpha=0.6, edgecolor='black', linewidth=1.5)
        
        ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax1.set_title('Service Reassignments and Flip-Flops', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(orchestrators, fontsize=10)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_axisbelow(True)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1a, reassignments)):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, val/2, f'{int(val)}',
                        ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        # Subplot 2: Flip-Flops percentage
        ff_pct = [ff/max(1, reassign)*100 for ff, reassign in zip(flip_flops, reassignments)]
        colors_ff = ['#2ca02c' if pct < 10 else '#ff7f0e' if pct < 40 else '#d62728' for pct in ff_pct]
        bars2 = ax2.bar(x, ff_pct, width, color=colors_ff, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax2.set_ylabel('Flip-Flop Ratio (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Flip-Flop Ratio\n(Stability Indicator)', fontsize=11, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(orchestrators, fontsize=10)
        ax2.axhline(y=40, color='red', linestyle='--', linewidth=2, label='Thrashing Threshold (40%)', alpha=0.7)
        ax2.legend(fontsize=9)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_axisbelow(True)
        
        for bar, val in zip(bars2, ff_pct):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Subplot 3: Churn Score
        colors_churn = ['#2ca02c' if score < 20 else '#ff7f0e' if score < 50 else '#d62728' 
                       for score in churn_scores]
        bars3 = ax3.bar(x, churn_scores, width, color=colors_churn, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax3.set_ylabel('Churn Score', fontsize=11, fontweight='bold')
        ax3.set_title('Orchestration Churn Score\n(Stability Metric)', fontsize=11, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(orchestrators, fontsize=10)
        ax3.axhline(y=20, color='orange', linestyle='--', linewidth=2, label='Moderate Threshold', alpha=0.7)
        ax3.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Thrashing Threshold', alpha=0.7)
        ax3.legend(fontsize=9)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        ax3.set_axisbelow(True)
        
        for bar, val in zip(bars3, churn_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add status annotations
        for i, (orch, score) in enumerate(zip(orchestrators, churn_scores)):
            if score < 20:
                status = '✓ Stable'
                color = '#2ca02c'
            elif score < 50:
                status = '⚠ Moderate'
                color = '#ff7f0e'
            else:
                status = '✗ Thrashing'
                color = '#d62728'
            ax3.text(i, -8, status, ha='center', va='top', fontsize=9, 
                    fontweight='bold', color=color)
        
        fig.suptitle('Placement Stability Analysis Over 30-Minute Dynamic Evaluation',
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Save
        output_path = self.output_dir / 'figure_03_stability_metrics.pdf'
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"    ✓ Saved: {output_path}")


def main():
    """Generate all figures."""
    test_results = 'out/test_results/dynamic_results.json'
    stability_analysis = 'out/train/stability_analysis.json'
    
    # Verify files exist
    for path in [test_results, stability_analysis]:
        if not Path(path).exists():
            print(f"❌ Error: {path} not found")
            sys.exit(1)
    
    # Generate figures
    generator = PaperFigureGenerator(test_results, stability_analysis)
    generator.generate_all()
    
    print("\n✅ All figures ready for LaTeX!")
    print("   Add to paper.tex:")
    print("   \\includegraphics[width=3.5in]{img/figure_01_performance_comparison.pdf}")
    print("   \\includegraphics[width=3.5in]{img/figure_02_dynamic_scenarios.pdf}")
    print("   \\includegraphics[width=3.5in]{img/figure_03_stability_metrics.pdf}")


if __name__ == '__main__':
    main()
