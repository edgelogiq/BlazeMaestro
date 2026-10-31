#!/usr/bin/env bash
# Retrain the DQN and Autoencoder models used in the paper (5 training seeds).
# Each model is trained in its own run so its checkpoint is saved as soon as it finishes.
# Output: out/campaign/train/seed<N>/{dqn,autoencoder}/final/*_checkpoint.pt
# Usage (from the repository root): scripts/retrain_models.sh [seed ...]   (default: 1 2 3 4 5)
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
seeds=${*:-1 2 3 4 5}
for seed in $seeds; do
  for model in dqn autoencoder; do
    out=out/campaign/train/seed$seed/$model
    mkdir -p "$out"
    PYTHONPATH=src "$PYTHON" src/campaign/train_seed.py \
      --enable-wifi --models "$model" --episodes-per-scenario 500 --scenarios all --enable-jitter \
      --platform-xml src/config/platform_sim.xml --services-xml src/config/services_sim.xml \
      --seed "$seed" --export-dir "$out" 2>&1 | tee "$out/train.log"
  done
done
