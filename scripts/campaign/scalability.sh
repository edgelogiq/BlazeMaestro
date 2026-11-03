#!/usr/bin/env bash
# Scalability study of the paper (Section IV-D, Table IX and supplementary tables), one run at a time
# so that decision times are measured without interference; a run is stopped after 30 min.
#  1) instances: |S| x |N| grid (1 robot, 3 clusters, depth 3) and robots/clusters/depth around (50, 25), 5 seeds
#  2) NSGA-II + Greedy Weighted on every instance (Greedy only at 200 x 5)
#  3) DQN + Autoencoder trained on six sub-grid points (instance seed 1, 800 episodes), 4 orchestrators, 5 run seeds
#  4) baselines on every instance (one-shot NSGA-II skipped at 200 x 5)
# Usage (from the repository root): scripts/campaign/scalability.sh   Output: out/campaign/scal
set -uo pipefail
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-.venv-linux/bin/python}
D=out/campaign/scal
mkdir -p $D/inst $D/runs $D/sub/train $D/baselines
JOBS=$D/jobs.txt   # tag nodes services robots clusters depth seed
: > $JOBS
for s in 10 25 50 100 200; do for n in 5 10 25 50; do for seed in 1 2 3 4 5; do echo "grid_n${n}_s$s $n $s 1 3 3 $seed" >> $JOBS; done; done; done
for seed in 1 2 3 4 5; do
  for r in 1 2 4 8; do echo "robots_r$r 25 50 $r 3 3 $seed" >> $JOBS; done
  for c in 3 6 12; do echo "clusters_c$c 25 50 1 $c 3 $seed" >> $JOBS; done
  for d in 1 2 4 8; do echo "depth_d$d 25 50 1 3 $d $seed" >> $JOBS; done
done
while read tag n s r c d seed; do
  [ -f $D/inst/${tag}_seed$seed/services.xml ] || "$PYTHON" src/campaign/gen_scalability.py --nodes $n --services $s \
    --robots $r --clusters $c --depth $d --seed $seed --out $D/inst/${tag}_seed$seed > /dev/null
done < $JOBS
run() { # tag seed outdir policies
  local inst=$D/inst/${1}_seed$2 out=$3/${1}_seed$2.json
  [ -f $out ] && return
  OMP_NUM_THREADS=1 PYTHONPATH=src timeout 1800 "$PYTHON" src/campaign/run_campaign.py --platform $inst/platform.xml \
    --services-xml $inst/services.xml --no-wifi --num-steps 10 --seed $2 --hysteresis 0.05 --quiet --simpy-exec \
    --require-simgrid --policies $4 --output $out > /dev/null 2>&1 || echo "$1 seed$2 $4 timeout/fail" >> $3/TIMEOUTS.txt
}
sort -k3,3n -k2,2n $JOBS | while read tag n s r c d seed; do
  if [ $tag = grid_n5_s200 ]; then run $tag $seed $D/runs GreedyWeighted; else run $tag $seed $D/runs NSGA-II,GreedyWeighted; fi
done
for pt in "10 10" "10 25" "10 50" "25 10" "25 25" "25 50"; do set -- $pt; tag=grid_n$1_s$2; t=$D/sub/train/$tag
  [ -f $t/final/AutoEncoder_checkpoint.pt ] || OMP_NUM_THREADS=1 PYTHONPATH=src "$PYTHON" src/campaign/train_seed.py \
    --episodes-per-scenario 50 --scenarios cpu50,ram50,mix50,gpu_hot,layer_hot --export-dir $t --enable-jitter \
    --platform-xml $D/inst/${tag}_seed1/platform.xml --services-xml $D/inst/${tag}_seed1/services.xml --seed 1 > $t.log 2>&1
  for seed in 1 2 3 4 5; do out=$D/sub/${tag}_run$seed.json; [ -f $out ] && continue
    OMP_NUM_THREADS=1 PYTHONPATH=src "$PYTHON" src/campaign/run_campaign.py --platform $D/inst/${tag}_seed1/platform.xml \
      --services-xml $D/inst/${tag}_seed1/services.xml --checkpoint-dir $t/final --no-wifi --num-steps 10 --seed $seed \
      --hysteresis 0.05 --quiet --simpy-exec --require-simgrid --policies NSGA-II,GreedyWeighted,Autoencoder,DQN --output $out > /dev/null 2>&1
  done
done
sort -k3,3n -k2,2n $JOBS | while read tag n s r c d seed; do
  if [ $tag = grid_n5_s200 ]; then run $tag $seed $D/baselines K8s-default,Static-initial,Kinitos-inspired,All-edge,All-fog
  else run $tag $seed $D/baselines OneShot-NSGA-II,K8s-default,Static-initial,Kinitos-inspired,All-edge,All-fog; fi
done
