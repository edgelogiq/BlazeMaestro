# BlazeMaestro

BlazeMaestro is the service-placement orchestrator of the BlazeAIoT platform. It places
the containers of a robotic application (ROS 2 services) on an edge/fog/cloud continuum
that includes a battery-powered mobile robot, and minimises energy and end-to-end latency
under period (deadline) and capacity constraints.

This repository contains the code used for the TPDS paper:

- four orchestrators sharing the same feasibility model V(x) and hysteresis:
  NSGA-II, Greedy weighted search with backtracking, DQN, and an autoencoder;
- reference baselines (static initial placement, one-shot NSGA-II, Kubernetes default,
  Kinitos-inspired, all-edge, all-fog);
- a SimPy/SimGrid simulator of the platform, with network jitter and WiFi mobility;
- the training, evaluation and revision-campaign scripts;
- the deployment mode that drives a real Kubernetes cluster from Prometheus metrics.

## Repository layout

```
src/
  EdgeLogiQ/BlazeAIoT/BlazeMaestro/
    Orchestrator/       NSGA-II, Greedy, DQN, autoencoder, shared utilities (V(x), hysteresis)
    Simulator/          platform topology, nodes, services, network jitter, latency matrix
    Topology/           platform, layer, node and service models; XML loaders
    Metrics/            simulator and Prometheus metrics facades
    ServiceAllocation/  simulator and Kubernetes allocators
    Managers/, Executors/, ConfigUpdater/, Config/   deployment mode (Kubernetes + Prometheus)
    Utils/              WiFi mobility model, CSV logging, logger mock
  campaign/             revision campaign: scenarios, baselines, scalability, statistics
  config/               platform and service definitions (XML), deployment configs (YAML/JSON)
  train.py              trains the DQN and autoencoder orchestrators
  compare_orchestrators.py, test.py, validate.py, visualize.py   simulation evaluation
  blaze_maestro_main.py entry point of the deployment mode
scripts/                paper figures, robot movement and WiFi degradation tools, Dockerfile
out/                    trained checkpoints, results and figures used in the paper
```

## Installation

Python 3.12 is used for the experiments.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
```

`blazeaiot-core-nodemanager` is only needed by the deployment mode; the simulation
scripts replace its logger with `Utils/blaze_logger_mock.py`.

The SimGrid Python bindings need CMake and Boost to build. They are only used at
start-up by `train.py`, `compare_orchestrators.py` and `test.py`; latencies come from
the platform XML links. On a machine without SimGrid, run training through
`src/campaign/train_seed.py`, which takes the same arguments as `train.py`.

All commands below are run from the repository root with `PYTHONPATH=src`.

## Reproducing the results

### Training

```bash
PYTHONPATH=src python src/train.py --enable-wifi --enable-jitter --episodes-per-scenario 500 \
    --scenarios all --platform-xml src/config/platform_sim.xml \
    --services-xml src/config/services_sim.xml --export-dir out/train
```

Checkpoints are written to `out/train/final/`.

The simulation results of the paper use five models per learned orchestrator, retrained
with seeds 1 to 5 (`src/campaign/train_seed.py`, one model per run):

```bash
PYTHONPATH=src python src/campaign/train_seed.py --models dqn --seed 1 --enable-wifi --enable-jitter \
    --episodes-per-scenario 500 --scenarios all --platform-xml src/config/platform_sim.xml \
    --services-xml src/config/services_sim.xml --export-dir out/campaign/train/seed1/dqn
```

They are provided in `out/campaign/train/seed<N>/{dqn,autoencoder}/final/`;
`out/campaign/train/seed<N>/final/` links both checkpoints of a seed into one folder.

### Simulation evaluation

```bash
PYTHONPATH=src python src/compare_orchestrators.py --checkpoint_dir out/train/final \
    --hysteresis-threshold 0.0 --enable-wifi --enable-jitter --duration 30 --time-step 5 \
    --seed 42 --output-dir out/compare
PYTHONPATH=src python src/visualize.py --results out/compare/dynamic_results.json --output-dir out/compare

PYTHONPATH=src python src/test.py --checkpoint-dir out/train/final --hysteresis-threshold 0.0 \
    --enable-wifi --enable-jitter --seed 42 --output-dir out/test
PYTHONPATH=src python src/validate.py --results out/test/dynamic_results.json --output-dir out/test
```

`compare_orchestrators.py` runs the dynamic ROS scenario and `visualize.py` plots it;
`test.py` runs the reassignment scenarios and `validate.py` measures their stability.

### Revision campaign

`src/campaign/run_campaign.py` replays the 600 s dynamic scenario of the paper (60
decisions, five workload phases, WiFi mobility and jitter) for all orchestrators and
baselines. It is deterministic for a given seed.

```bash
PYTHONPATH=src python src/campaign/run_campaign.py --seed 42 --hysteresis 0.0 \
    --checkpoint-dir out/campaign/train/seed1/final --output out/campaign/seed42_h0.json
PYTHONPATH=src python src/campaign/analyze.py out/campaign
```

Other scenarios are selected with `--scenario disruption` or `--scenario robot_pressure`.
The scalability study uses `gen_scalability.py` and `analyze_scal.py`, and the training
variance study uses `train_seed.py` and `analyze_training_variance.py`.

### Figures

`scripts/generate_paper_figures.py` builds the simulation figures from the evaluation
results. `scripts/generate_fig_dqn_energy.py` rebuilds the robot energy figure of the
real deployment from `out/20260319_080017/`.

## Deployment mode

`src/blaze_maestro_main.py` runs the orchestration loop against Kubernetes clusters,
reads node and service metrics from Prometheus, and applies placements through the
Kubernetes API. Configuration files are in `src/config/` (see `src/config/README.md`).

```bash
PYTHONPATH=src python src/blaze_maestro_main.py --layer_config_path src/config/layer_config.json \
    --node_config_path src/config/node_config.json \
    --service_config_path src/config/service_config_sim.json
```

## License

MIT, see [LICENSE](LICENSE).
