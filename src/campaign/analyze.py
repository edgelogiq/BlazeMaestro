"""Aggregate campaign runs (R2-10): mean and 95% bootstrap CI over seeds, plus paired tests.

Usage:
    python src/campaign/analyze.py out/campaign/r2-10_prelim
Writes summary.csv and summary.md next to the run files.
"""

import glob
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

METRICS = {
    "latency": ("L_tot avg (ms)", lambda rs: 1000.0 * float(np.mean([r["latency"] for r in rs]))),
    "lat_max": ("max L_s avg (ms)", lambda rs: 1000.0 * float(np.mean([r.get("latency_max_service", 0.0) for r in rs]))),
    "energy": ("Energy total (J)", lambda rs: float(sum(r["energy"] for r in rs))),
    "realloc": ("Realloc. (/60)", lambda rs: float(sum(1 for r in rs if r["reassigned"]))),
    "violations": ("Violations (sum)", lambda rs: float(sum(r["violations"] for r in rs))),
    "viol_steps": ("Steps w/ violation", lambda rs: float(sum(1 for r in rs if r["violations"] > 0))),
    "on_robot": ("Services on robot (avg)", lambda rs: float(np.mean([r.get("services_on_robot", 0) for r in rs]))),
    "t_dec_ms": ("Decision time (ms)", lambda rs: 1000.0 * float(np.mean([r["decision_time_s"] for r in rs]))),
}
CORE = ["NSGA-II", "DQN", "GreedyWeighted", "Autoencoder"]


def bootstrap_ci(x, n=5000, rng=np.random.default_rng(0)):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(x.mean()), float("nan"), float("nan")
    boots = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def main(run_dir):
    # values[h][policy][metric] -> {seed: value}
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for path in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        d = json.load(open(path))
        cfg = d["config"]
        if "seed" not in cfg or "hysteresis" not in cfg:
            continue
        for pol, rs in d["results"].items():
            for key, (_, fn) in METRICS.items():
                values[cfg["hysteresis"]][pol][key][cfg["seed"]] = fn(rs)

    rows, md = [], []
    for h in sorted(values):
        md.append(f"\n### Hysteresis {h * 100:.0f}%\n")
        md.append("| Policy | n | " + " | ".join(lbl for lbl, _ in METRICS.values()) + " |")
        md.append("|---|---|" + "---|" * len(METRICS))
        for pol, per_metric in values[h].items():
            cells = []
            n = len(per_metric["latency"])
            for key in METRICS:
                m, lo, hi = bootstrap_ci(list(per_metric[key].values()))
                rows.append([h, pol, key, n, m, lo, hi])
                fmt = "{:.2f}" if key in ("latency", "lat_max") else "{:.1f}"
                cells.append(f"{fmt.format(m)} [{fmt.format(lo)}, {fmt.format(hi)}]")
            md.append(f"| {pol} | {n} | " + " | ".join(cells) + " |")

        core = [p for p in CORE if p in values[h]]
        for key in ("latency", "energy", "realloc"):
            seeds = sorted(set.intersection(*[set(values[h][p][key]) for p in core]))
            if len(seeds) < 3 or len(core) < 3:
                continue
            mat = np.array([[values[h][p][key][s] for s in seeds] for p in core])
            try:
                fr = stats.friedmanchisquare(*mat)
                md.append(f"\nFriedman ({key}, n={len(seeds)}): chi2={fr.statistic:.2f}, p={fr.pvalue:.2g}")
            except ValueError as exc:
                md.append(f"\nFriedman ({key}): not computable ({exc})")
            pairs = list(itertools.combinations(range(len(core)), 2))
            ps = []
            for i, j in pairs:
                diff = mat[i] - mat[j]
                ps.append(1.0 if np.allclose(diff, 0) else stats.wilcoxon(mat[i], mat[j]).pvalue)
            for (i, j), p, pa in zip(pairs, ps, holm(np.array(ps))):
                md.append(f"- {core[i]} vs {core[j]}: Wilcoxon p={p:.2g}, Holm p={pa:.2g}")

    with open(os.path.join(run_dir, "summary.csv"), "w") as f:
        f.write("hysteresis,policy,metric,n,mean,ci_low,ci_high\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    with open(os.path.join(run_dir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main(sys.argv[1])
