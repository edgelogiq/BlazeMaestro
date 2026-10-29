#!/usr/bin/env python3
"""Regenerate the real-deployment robot energy figure (Fig. 4 in the paper)
with annotated phase bands: stationary, path-following (motors active), and
the WiFi latency/packet-loss injection window.

Usage:
    python scripts/generate_fig_dqn_energy.py [run_dir] [output_path]

Defaults to the ten-minute real-deployment run used in the paper
(out/20260319_080017) and writes fig_dqn_energy.pdf in the current directory.

Phase boundaries (0-390s stationary, 390-520s motors active, 460-580s WiFi
injection) were determined empirically from the frank-2 battery power trace
and the per-iteration latency-to-frank-2 columns in dqn.csv; see the TPDS
review-response tracking document for the derivation.
"""
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "out/20260319_080017"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "fig_dqn_energy.pdf"

    with open(f"{run_dir}/dqn.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    power = np.array([float(r["frank-2_battery_power_w"]) for r in rows])
    t = np.arange(len(power)) * 10.0
    energy = power * 10.0  # J per 10s control cycle

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.axvspan(0, 390, color="tab:blue", alpha=0.08, label="Stationary (background load only)")
    ax.axvspan(390, 520, color="tab:orange", alpha=0.15, label="Path-following (motors active)")
    ax.axvspan(520, t[-1] + 10, color="tab:blue", alpha=0.08)
    ax.axvspan(460, 580, facecolor="none", edgecolor="tab:red", hatch="///", linewidth=0, alpha=0.5,
               label="WiFi latency/packet-loss injection")

    ax.plot(t, energy, color="tab:green", marker="o", markersize=3, linewidth=1.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (J)")
    ax.set_xlim(0, t[-1] + 10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
