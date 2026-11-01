"""Compare the SimGrid+SimPy scalability rerun with the reference sequential runs.

Usage: python src/campaign/compare_scal_sim.py out/campaign/scal out/campaign/scal_sim <output.md>
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

REF, SIM, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PAIRS = [("runs_idle", "runs"), ("sub_idle", "sub"), ("runs_idle_baselines", "baselines")]
key = lambda t: (t.split("_")[0], [int(x) for x in re.findall(r"\d+", t)])

backends, n_runs, n_diff, missing = set(), 0, [], []
agg = defaultdict(lambda: defaultdict(list))
for rdir, sdir in PAIRS:
    for f in sorted(glob.glob(f"{SIM}/{sdir}/*.json")):
        ref_path = f"{REF}/{rdir}/{os.path.basename(f)}"
        if not os.path.exists(ref_path):
            missing.append(f); continue
        s, r = json.load(open(f)), json.load(open(ref_path))
        backends.add(s["config"].get("latency_backend")); n_runs += 1
        tag = os.path.basename(f).rsplit("_seed", 1)[0].rsplit("_run", 1)[0] + ("" if sdir == "runs" else f" ({sdir})")
        for p, rs in s["results"].items():
            rr = r["results"].get(p)
            if rr is None:
                continue
            if any(x[k] != y[k] for x, y in zip(rs, rr) for k in ("assignments", "latency", "violations")):
                n_diff.append(f"{os.path.basename(f)} {p}")
            agg[tag][p].append((1000 * np.mean([x["decision_time_s"] for x in rr]), 1000 * np.mean([x["decision_time_s"] for x in rs]),
                                1000 * np.mean([x["latency"] for x in rr]), 1000 * np.mean([x["latency"] for x in rs]),
                                100 * np.mean([x["violations"] > 0 for x in rr]), 100 * np.mean([x["violations"] > 0 for x in rs]),
                                100 * np.mean([x.get("deadline_miss_services_sim", 0) > 0 for x in rs]),
                                1000 * max(x.get("wait_mean_sim", 0.0) for x in rs),
                                100 * (sum(x["energy_sim"] for x in rs) - sum(x["energy"] for x in rs)) / max(1e-9, sum(x["energy"] for x in rs))))

L = ["# Étude d'échelle : relance SimGrid + SimPy contre la mesure à vide de référence\n",
     f"- Runs comparés : {n_runs} ; moteur de latence : {', '.join(sorted(map(str, backends)))} ; runs sans référence : {len(missing)}.",
     f"- (run, politique) dont le placement, L_tot ou les violations diffèrent : {len(n_diff)}" + (f" ({', '.join(n_diff[:10])})" if n_diff else "") + ".\n",
     "Par instance et politique (moyenne sur les seeds) : temps de décision référence → relance (ms) | L_tot réf. → relance (ms) | % décisions en violation réf. → relance | % décisions avec échéance manquée (SimPy) | attente max (ms) | écart d'énergie SimPy (%)\n",
     "| Instance | Politique | t_dec (ms) | L_tot (ms) | Violations % | Échéances SimPy % | Attente max (ms) | ΔE SimPy % |", "|---|---|---|---|---|---|---|---|"]
ratios = defaultdict(list)
for tag in sorted(agg, key=key):
    for p, v in agg[tag].items():
        a = np.array(v); m = a.mean(0)
        ratios[p].append(m[1] / max(m[0], 1e-9))
        L.append(f"| {tag} | {p} | {m[0]:.1f} → {m[1]:.1f} | {m[2]:.1f} → {m[3]:.1f} | {m[4]:.0f} → {m[5]:.0f} | {m[6]:.0f} | {a[:, 7].max():.2f} | {m[8]:+.2f} |")
L.append("\nRapport médian des temps de décision (relance / référence) par politique : " +
         ", ".join(f"{p} {np.median(r):.2f} [{np.min(r):.2f}, {np.max(r):.2f}]" for p, r in sorted(ratios.items())))
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L[:3])); print(L[-1])
