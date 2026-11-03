#!/usr/bin/env bash
# Main simulation campaign of the paper (Tables VII-VIII, migration budget, Fig. 3): 940 runs.
# Scenarios table6 (reference), disruption, robot_pressure: 20 seeds x hysteresis {0, 0.05};
# replay of the real run: 10 seeds. DQN/Autoencoder: model 1 in the main runs, models 2-5 in ckptN.
# Migration budget k in {-1 (none), 2, 1, 0} with model 1. SimGrid routing and SimPy execution.
# Usage (from the repository root): scripts/campaign/main_campaign.sh [parallel jobs, default 14]
# Output: out/campaign/final; summary: python src/campaign/analyze_final.py out/campaign/final <dir>
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-.venv-linux/bin/python}
OUT=out/campaign/final
CK=out/campaign/train
export PYTHON OUT CK
jobs() {
  for sc in table6 disruption robot_pressure; do for s in $(seq 1 20); do for h in 0.0 0.05; do
    echo "main 1 $sc $s $h"; for c in 2 3 4 5; do echo "ckpt $c $sc $s $h"; done
  done; for k in -1 2 1 0; do echo "mig 1 $sc $s $k"; done; done; done
  for s in $(seq 1 10); do for h in 0.0 0.05; do
    echo "main 1 replay $s $h"; for c in 2 3 4 5; do echo "ckpt $c replay $s $h"; done
  done; done
}
run_one() {
  kind=$1; c=$2; sc=$3; s=$4; x=$5
  if [ "$sc" = replay ]; then extra="--trace-dir out/20260319_080017 --no-wifi --no-jitter"; else extra="--scenario $sc"; fi
  case $kind in
    main) o=$OUT/main/$sc; pol=NSGA-II,DQN,GreedyWeighted,Autoencoder,Static-initial,OneShot-NSGA-II,K8s-default,Kinitos-inspired,All-edge,All-fog; h=$x; mig="";;
    ckpt) o=$OUT/ckpt$c/$sc; pol=DQN,Autoencoder; h=$x; mig="";;
    mig)  o=$OUT/migration/$sc/k$x; pol=NSGA-II,GreedyWeighted,Autoencoder; h=0.0; mig="--migration-budget $x";;
  esac
  mkdir -p "$o/logs"; out=$o/seed${s}_h$h.json; [ "$kind" = mig ] && out=$o/seed$s.json
  [ -f "$out" ] && return 0
  OMP_NUM_THREADS=1 PYTHONPATH=src "$PYTHON" src/campaign/run_campaign.py $extra --seed "$s" --hysteresis "$h" $mig \
    --checkpoint-dir "$CK/seed$c/final" --policies "$pol" --quiet --simpy-exec --require-simgrid \
    --output "$out" > "$o/logs/$(basename "$out" .json).log" 2>&1
}
export -f run_one
jobs | xargs -P "${1:-14}" -L 1 bash -c 'run_one "$@"' _
