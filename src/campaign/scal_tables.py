"""LaTeX rows of the supplementary scalability tables.

Main grid and variations: decision time (ms) / L_tot (ms) / % decisions in violation, plus robot
energy (kJ) for NSGA-II and Greedy Weighted; sub-grid: the four orchestrators. A trailing * marks
points where fewer than five runs completed within 30 min.
Usage: python src/campaign/scal_tables.py out/campaign/scal
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

D = sys.argv[1]


def load(dirs, pat):
    agg = defaultdict(lambda: defaultdict(list))
    for d in dirs:
        for f in glob.glob(os.path.join(d, "*.json")):
            tag = re.sub(pat, "", os.path.basename(f)[:-5])
            for p, rs in json.load(open(f))["results"].items():
                agg[tag][p].append((1000 * np.mean([r["decision_time_s"] for r in rs]),
                                    1000 * np.mean([r["latency"] for r in rs]),
                                    100 * np.mean([r["violations"] > 0 for r in rs]),
                                    sum(r["energy"] for r in rs) / 1000))
    return agg


def cell(v, energy=False):
    if not v:
        return "n/t"
    m = np.mean(v, axis=0)
    s = f"{m[0]:.0f} / {m[1]:.0f} / {m[2]:.0f}" + (f" / {m[3]:.2f}" if energy else "")
    return s + ("$^{*}$" if len(v) < 5 else "")


g = load([f"{D}/runs", f"{D}/baselines"], r"_seed\d+$")
rows = [(f"$|S|={s}$, $|N|={n}$", f"grid_n{n}_s{s}") for n in (5, 10, 25, 50) for s in (10, 25, 50, 100, 200)]
rows += [(f"Clusters {c}", f"clusters_c{c}") for c in (3, 6, 12)]
rows += [(f"Depth {d}", f"depth_d{d}") for d in (1, 2, 4, 8)] + [(f"Robots {r}", f"robots_r{r}") for r in (1, 2, 4, 8)]
for label, tag in rows:
    if label == "Clusters 3":
        print("\\midrule")
    p = g[tag]
    print(f"{label} & {cell(p.get('NSGA-II'), True)} & {cell(p.get('GreedyWeighted'), True)} & "
          f"{cell(p.get('Kinitos-inspired'))} & {cell(p.get('K8s-default'))} & {cell(p.get('All-fog'))} \\\\")
print("% sub-grid")
s = load([f"{D}/sub"], r"_run\d+$")
for n, sv in [(10, 10), (10, 25), (10, 50), (25, 10), (25, 25), (25, 50)]:
    p = s[f"grid_n{n}_s{sv}"]
    print(f"$|S|={sv}$, $|N|={n}$ & " + " & ".join(cell(p.get(k)) for k in
                                                ("NSGA-II", "GreedyWeighted", "Autoencoder", "DQN")) + " \\\\")
