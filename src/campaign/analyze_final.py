"""Summaries of the final campaign (out/campaign/final), written as Markdown/CSV.

Every policy of a given (scenario, seed, hysteresis) shares the same scenario realisation,
so paired tests across policies are valid. DQN and Autoencoder values per seed are the
mean over the 5 retrained models (model 1 from the main run, models 2-5 from ckptN runs).

Usage: python src/campaign/analyze_final.py out/campaign/final <output_dir>
"""
import csv
import glob
import itertools
import json
import os
import sys

import numpy as np
from scipy import stats

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)
LEARNED = ("DQN", "Autoencoder")
CORE = ["NSGA-II", "GreedyWeighted", "Autoencoder", "DQN"]
ALL = CORE + ["OneShot-NSGA-II", "Kinitos-inspired", "K8s-default", "Static-initial", "All-edge", "All-fog"]
LABEL = {"GreedyWeighted": "Greedy", "Autoencoder": "Autoencodeur", "OneShot-NSGA-II": "One-shot NSGA-II",
         "K8s-default": "K8s par défaut", "Static-initial": "Statique", "All-edge": "Tout-edge", "All-fog": "Tout-fog"}


def metrics(rs):
    return {
        "L_tot (ms)": 1000 * np.mean([r["latency"] for r in rs]),
        "max L_s (ms)": 1000 * np.mean([r.get("latency_max_service", 0.0) for r in rs]),
        "Énergie (kJ)": sum(r["energy"] for r in rs) / 1000,
        "Violations /60": float(sum(r["violations"] > 0 for r in rs)),
        "Réalloc. /60": float(sum(r["reassigned"] for r in rs)),
    }


def load(path):
    return json.load(open(path))["results"] if os.path.exists(path) else None


def per_seed(sc, h, n):
    """{policy: {seed: metrics}} and {policy: {model: [metrics per seed]}} for learned ones."""
    data, by_model = {}, {p: {} for p in LEARNED}
    for s in range(1, n + 1):
        main = load(f"{ROOT}/main/{sc}/seed{s}_h{h}.json")
        if main is None:
            continue
        for p in ALL:
            if p in main and p not in LEARNED:
                data.setdefault(p, {})[s] = metrics(main[p])
        for p in LEARNED:
            ms = [metrics(main[p])]
            by_model[p].setdefault(1, []).append(ms[0])
            for c in range(2, 6):
                r = load(f"{ROOT}/ckpt{c}/{sc}/seed{s}_h{h}.json")
                if r is not None:
                    m = metrics(r[p]); ms.append(m); by_model[p].setdefault(c, []).append(m)
            data.setdefault(p, {})[s] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
    return data, by_model


def ci(x):
    x = np.asarray(x, float)
    b = RNG.choice(x, (5000, len(x))).mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def holm(p):
    order, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


lines = ["# Campagne finale : statistiques complètes (10 politiques, appariement valide)\n",
         "Source : `out/campaign/final`, branche `rng-isolation` (e01da89).",
         "- Toutes les politiques d'une même (scénario, seed, hystérésis) partagent exactement la même réalisation du scénario, ce qui rend les tests appariés valides.",
         "- DQN et Autoencodeur : moyenne des 5 modèles réentraînés par seed ; σ entre entraînements = écart-type des 5 moyennes par modèle.",
         "- Valeurs : moyenne [IC 95 % bootstrap, 5 000 tirages].",
         "- Tests : Friedman sur les 4 ordonnanceurs, puis Wilcoxon apparié avec correction de Holm (6 paires).",
         "- Les temps de décision ne sont pas dans ce fichier : utiliser `decision_times_idle.md` et `decision_times_idle_replay.md` (mesures à vide).\n"]
for sc in ["table6", "disruption", "robot_pressure", "replay"]:
    n = 10 if sc == "replay" else 20
    for h in ["0.0", "0.05"]:
        data, by_model = per_seed(sc, h, n)
        seeds = sorted(set.intersection(*[set(data[p]) for p in CORE]))
        mets = list(next(iter(data["NSGA-II"].values())).keys())
        lines.append(f"\n## {sc}, hystérésis {float(h) * 100:.0f} % (n = {len(seeds)})\n")
        lines.append("| Politique | " + " | ".join(mets) + " | σ entre entraînements (L_tot) |")
        lines.append("|---|" + "---|" * (len(mets) + 1))
        for p in ALL:
            if p not in data:
                continue
            cells = ["%.2f [%.2f, %.2f]" % ci([data[p][s][m] for s in seeds]) for m in mets]
            sd = ""
            if p in LEARNED:
                sd = "%.2f" % np.std([np.mean([m["L_tot (ms)"] for m in v]) for v in by_model[p].values()], ddof=1)
            lines.append(f"| {LABEL.get(p, p)} | " + " | ".join(cells) + f" | {sd} |")
        for m in ["L_tot (ms)", "Énergie (kJ)", "Violations /60", "Réalloc. /60"]:
            mat = np.array([[data[p][s][m] for s in seeds] for p in CORE])
            try:
                fr = stats.friedmanchisquare(*mat)
                head = f"Friedman χ² = {fr.statistic:.2f}, p = {fr.pvalue:.2g}"
            except Exception:
                head = "Friedman non calculable (valeurs identiques)"
            lines.append(f"\n**{m}** : {head}")
            pairs = list(itertools.combinations(range(len(CORE)), 2))
            ps = [1.0 if np.allclose(mat[i], mat[j]) else stats.wilcoxon(mat[i], mat[j]).pvalue for i, j in pairs]
            for (i, j), p_, pa in zip(pairs, ps, holm(np.array(ps))):
                lines.append(f"- {LABEL.get(CORE[i], CORE[i])} vs {LABEL.get(CORE[j], CORE[j])} : Wilcoxon p = {p_:.2g}, Holm p = {pa:.2g}")
open(f"{OUT}/stats_final.md", "w").write("\n".join(lines) + "\n")

# Fig. 3 / Fig. 4 series (reference scenario)
rows = []
for h in ["0.0", "0.05"]:
    series = {}
    for s in range(1, 21):
        main = load(f"{ROOT}/main/table6/seed{s}_h{h}.json")
        for p in ALL:
            if p in LEARNED:
                runs = [main[p]] + [load(f"{ROOT}/ckpt{c}/table6/seed{s}_h{h}.json")[p] for c in range(2, 6)]
            else:
                runs = [main[p]]
            series.setdefault(p, []).extend(runs)
    for p, runs in series.items():
        L = np.array([[1000 * r["latency"] for r in rs] for rs in runs])
        E = np.array([[r["energy"] for r in rs] for rs in runs])
        for k in range(L.shape[1]):
            rows.append([h, p, k, L[:, k].mean(), L[:, k].std(), E[:, k].mean(), E[:, k].std(), len(runs)])
with open(f"{OUT}/fig3_fig4_timeseries_final.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["hysteresis", "policy", "step", "L_tot_ms_mean", "L_tot_ms_sd", "energy_J_mean", "energy_J_sd", "n_runs"])
    w.writerows(rows)

# Migration budget
M = ["# Budget de migration, campagne finale\n",
     "Hystérésis 0 %, 20 seeds, NSGA-II, Greedy et Autoencodeur (réentraînement n° 1) dans les mêmes runs. L'indisponibilité est modélisée comme services migrés × c_mig, à présenter comme une analyse de sensibilité.\n",
     "| Scénario | Budget k | Politique | L_tot (ms) | Décisions en violation /60 | Cycles avec migration /60 | Services migrés /run | Indisponibilité (s/run) à c_mig = 2 / 5 / 15 s |",
     "|---|---|---|---|---|---|---|---|"]
for sc in ["table6", "disruption", "robot_pressure"]:
    for k in ["-1", "2", "1", "0"]:
        R = {}
        for f in glob.glob(f"{ROOT}/migration/{sc}/k{k}/seed*.json"):
            for p, rs in json.load(open(f))["results"].items():
                R.setdefault(p, []).append((1000 * np.mean([r["latency"] for r in rs]), sum(r["violations"] > 0 for r in rs),
                                            sum(r["reassigned"] for r in rs), sum(r["services_moved"] for r in rs)))
        for p in ["NSGA-II", "GreedyWeighted", "Autoencoder"]:
            if p in R:
                a = np.array(R[p]).mean(0)
                M.append(f"| {sc} | {'∞' if k == '-1' else k} | {LABEL.get(p, p)} | {a[0]:.1f} | {a[1]:.1f} | {a[2]:.1f} | {a[3]:.1f} | "
                         f"{2 * a[3]:.0f} / {5 * a[3]:.0f} / {15 * a[3]:.0f} |")
open(f"{OUT}/migration_budget_final.md", "w").write("\n".join(M) + "\n")
print("written to", OUT)
