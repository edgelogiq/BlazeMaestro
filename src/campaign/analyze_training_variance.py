"""Training-variance summary for retrained DQN/AE (R2-10).

Usage: python src/campaign/analyze_training_variance.py out/campaign/v6_retrained
For each scenario, hysteresis and policy: grand mean over (train seed x run seed), the
standard deviation across training seeds (of per-model means) and the mean within-model
standard deviation across run seeds.
"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np

root = sys.argv[1]
vals = defaultdict(lambda: defaultdict(list))  # (sc,h,pol,metric) -> train -> [values]
for f in glob.glob(os.path.join(root, "train[0-9]*", "*", "seed*.json")):  # retrained only; trainorig excluded
    t = f.split(os.sep)[-3]; sc = f.split(os.sep)[-2]
    d = json.load(open(f)); h = d["config"]["hysteresis"]
    for pol, rs in d["results"].items():
        m = {"L_tot_ms": 1000 * np.mean([r["latency"] for r in rs]),
             "energy_kJ": sum(r["energy"] for r in rs) / 1000,
             "viol_steps": sum(r["violations"] > 0 for r in rs),
             "realloc": sum(r["reassigned"] for r in rs)}
        for k, v in m.items():
            vals[(sc, h, pol, k)][t].append(v)
lines = ["| Scénario | h | Politique | Métrique | Moyenne | σ entre entraînements | σ intra-modèle (seeds) |", "|---|---|---|---|---|---|---|"]
for key in sorted(vals):
    per = vals[key]
    means = [np.mean(v) for v in per.values()]
    within = np.mean([np.std(v, ddof=1) for v in per.values() if len(v) > 1])
    allv = [x for v in per.values() for x in v]
    lines.append(f"| {key[0]} | {key[1]:.0%} | {key[2]} | {key[3]} | {np.mean(allv):.2f} | {np.std(means, ddof=1):.2f} | {within:.2f} |")
out = "\n".join(lines) + "\n"
open(os.path.join(root, "training_variance.md"), "w").write(out)
print(out)
