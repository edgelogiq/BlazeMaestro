# Maestro
source .venv/bin/activate
export LD_LIBRARY_PATH=/home/cmelancon/dev/BlazeAIoT/maestro/simgrid-v4.1/build/lib:$LD_LIBRARY_PATH

# Training with enhanced ML models (use matching platform and services XMLs)
# Increased episodes to 500 per scenario for better convergence with stricter period constraints
python src/train.py --enable-wifi --episodes-per-scenario 500 --scenarios all --export-dir out/train --enable-jitter --platform-xml src/config/platform_sim.xml --services-xml src/config/services_sim.xml

python src/compare_orchestrators.py --checkpoint_dir out/train/final --hysteresis-threshold 0.0 --enable-wifi --enable-jitter --duration 30 --time-step 5 --seed 42 --output-dir out/train

python src/test.py --checkpoint-dir out/train/final --hysteresis-threshold 0.0 --enable-wifi --enable-jitter --seed 42 --output-dir out/train

python src/validate.py --results out/train/dynamic_results.json --output-dir out/train
python src/visualize.py --results out/train/dynamic_results.json --output-dir out/train

python3 ./src/blaze_maestro_main.py --layer_config_path src/config/layer_config.json --node_config_path src/config/node_config.json --service_config_path src/config/service_config_sim.json

## Environment

- Activate the venv
```bash
source .venv/bin/activate
```

- Ensure SimGrid’s native libs are visible (add both lib and lib64 just in case)
```bash
export LD_LIBRARY_PATH="$(pwd)/.venv/lib:$(pwd)/.venv/lib64:${LD_LIBRARY_PATH:-}"
```

## Verify SimGrid loads (optional but recommended)

```bash
python - <<'PY'
from simgrid import Engine
e = Engine(["--cfg=plugin:energy"])  # should not raise
print("SimGrid OK")
PY
```

## Run the Simulator main

```bash
PYTHONPATH=src python3 src/EdgeLogiQ/BlazeAIoT/BlazeMaestro/Simulator/main.py
```

## Train/Test the RL orchestrator

- Quick smoke (prints service->node and node loads; no simulation)
```bash
PYTHONPATH=src python3 src/train_test_rl_orchestrator.py \
	--episodes 5 \
	--episodes-per-scenario 3 \
	--scenarios cpu50 \
	--no-sim \
	--log-every 1 \
	--export-dir out/smoke
```

- Use real link latencies via SimGrid (if available) and print latency stats
```bash
PYTHONPATH=src python3 src/train_test_rl_orchestrator.py \
	--scenarios cpu50 \
	--no-sim \
	--latency-mode simgrid \
	--latency-debug \
	--export-dir out/latency_check
```

### Latency sources and fallback

- The training script supports `--latency-mode`:
	- `auto` (default): tries SimGrid; if it yields an all-zero matrix, falls back to parsing `platform.xml` and computing shortest paths from link latencies.
	- `simgrid`: same as auto but warns if SimGrid is unavailable; still falls back to XML if SimGrid returns zeros.
	- `zero`: force a zero-latency matrix (for control experiments).

- With `--latency-debug`, you will see a line like:

	`Latency stats (SimGrid): nonzeros=462/484, min=0.002000s, mean=0.018312s, max=0.042000s`

	If SimGrid returns zeros, you’ll see:

	`WARN: SimGrid returned zero-latency matrix; falling back to XML route-based latency.`

	followed by stats labeled `(XML)` if that produced non-zero values.

## Troubleshooting

- simgrid import fails (e.g., `ImportError: libsimgrid.so...`)
	- Ensure you’re in the venv and that native libs are visible:
		```bash
		source .venv/bin/activate
		export LD_LIBRARY_PATH="$(pwd)/.venv/lib:$(pwd)/.venv/lib64:${LD_LIBRARY_PATH:-}"
		python - <<'PY'
		from simgrid import Engine
		Engine(["--cfg=plugin:energy"])  # should succeed
		print("SimGrid OK")
		PY
		```

- Latency stats show all zeros
	- Re-run in the same shell after setting `LD_LIBRARY_PATH` as above.
	- Use `--latency-mode simgrid --latency-debug`; if SimGrid still yields zeros, the script automatically falls back to XML-derived latencies and prints a warning. Non-zero stats should then appear.

