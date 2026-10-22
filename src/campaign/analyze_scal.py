"""Scalability summary (R2-11): per instance family and policy, mean over seeds of the
decision time, L_tot, share of decisions with a violation and robot energy.
Usage: python src/campaign/analyze_scal.py out/campaign/scal/runs_v6 > summary.md"""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np

agg = defaultdict(lambda: defaultdict(list))
for f in glob.glob(os.path.join(sys.argv[1], "*.json")):
    tag = os.path.basename(f).rsplit("_seed", 1)[0]
    d = json.load(open(f))
    for p, rs in d["results"].items():
        agg[tag][p].append((1000 * np.mean([r["decision_time_s"] for r in rs]),
                            1000 * np.mean([r["latency"] for r in rs]),
                            100 * np.mean([r["violations"] > 0 for r in rs])))
pols = ["NSGA-II", "GreedyWeighted", "Kinitos-inspired", "OneShot-NSGA-II", "K8s-default", "All-fog"]
key = lambda t: (t.split("_")[0], [int(x) for x in re.findall(r"\d+", t)])
print("| Instance | " + " | ".join(f"{p} t(ms) / L_tot(ms) / viol%" for p in pols) + " |")
print("|---|" + "---|" * len(pols))
for tag in sorted(agg, key=key):
    cells = []
    for p in pols:
        v = agg[tag].get(p)
        cells.append("n/a (timeout)" if not v else
                     f"{np.mean([x[0] for x in v]):.0f} / {np.mean([x[1] for x in v]):.0f} / {np.mean([x[2] for x in v]):.0f}")
    print(f"| {tag} | " + " | ".join(cells) + " |")
