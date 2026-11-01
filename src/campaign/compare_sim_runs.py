"""Check a SimGrid+SimPy campaign against a reference campaign and summarise the SimPy fields.

Usage: python src/campaign/compare_sim_runs.py out/campaign/final out/campaign/final_sim <output.md>
"""
import glob
import json
import os
import sys

import numpy as np

REF, SIM, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
KEYS = ("assignments", "latency", "latency_max_service", "energy", "violations", "reassigned")
CORE = ["NSGA-II", "GreedyWeighted", "Autoencoder", "DQN"]

files = sorted(glob.glob(f"{SIM}/**/seed*.json", recursive=True))
backends, n_diff, n_runs, rows = set(), 0, 0, {}
for f in files:
    sim = json.load(open(f))
    ref_path = f.replace(SIM, REF, 1)
    ref = json.load(open(ref_path))["results"] if os.path.exists(ref_path) else None
    backends.add(sim["config"].get("latency_backend"))
    n_runs += 1
    for p, rs in sim["results"].items():
        if ref is not None and any(x.get(k) != y.get(k) for x, y in zip(rs, ref[p]) for k in KEYS):
            n_diff += 1
        if "/main/" in f and p in CORE:
            sc = f.split("/main/")[1].split("/")[0]
            h = f.rsplit("_h", 1)[1][:-5]
            rows.setdefault((sc, h, p), []).append((
                sum(r["energy"] for r in rs), sum(r["energy_sim"] for r in rs),
                sum(r["violations"] > 0 for r in rs), sum(r["deadline_miss_services_sim"] > 0 for r in rs),
                sum(r["deadline_miss_jobs_sim"] for r in rs), sum(r["jobs_sim"] for r in rs),
                max(r["wait_mean_sim"] for r in rs),
                np.mean([r["latency"] for r in rs]), np.mean([r["latency_sim"] for r in rs])))

L = ["# Campagne SimGrid + SimPy (out/campaign/final_sim) contre la campagne finale\n",
     f"- Runs : {n_runs} ; moteur de latence : {', '.join(sorted(map(str, backends)))}.",
     f"- (run, politique) dont un champ analytique diffère de la campagne finale : {n_diff}.\n",
     "Ordonnanceurs principaux, modèle n° 1, moyennes par seed :\n",
     "| Scénario | Hyst. | Politique | L_tot analytique → SimPy (ms) | Énergie analytique → SimPy (kJ) | Décisions en violation / avec échéance manquée (/60) | Tâches manquées / lancées | Attente max (ms) |",
     "|---|---|---|---|---|---|---|---|"]
for (sc, h, p), v in sorted(rows.items()):
    a = np.array(v, float); m = a.mean(0)
    L.append(f"| {sc} | {float(h) * 100:.0f} % | {p} | {1000 * m[7]:.2f} → {1000 * m[8]:.2f} | {m[0] / 1000:.3f} → {m[1] / 1000:.3f} ({100 * (m[1] - m[0]) / m[0]:+.2f} %) | "
             f"{m[2]:.2f} / {m[3]:.2f} | {a[:, 4].sum():.0f} / {a[:, 5].sum():.0f} | {1000 * a[:, 6].max():.3f} |")
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L[:3]))
