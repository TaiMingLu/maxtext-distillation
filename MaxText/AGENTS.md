# Repository Guidelines

## Project Structure & Module Organization
Key areas:
- `train.py`, `train_compile.py`, `elastic_train.py`, and helpers in `train_utils.py` / `pyconfig.py` drive training, XAOT compilation, and config validation.
- `configs/` stores YAML experiments (`base.yml`, hardware folders such as `v4/`, `v5e/`, `quantization/`); keep new knobs alongside similar hardware or workflows.
- Modeling code sits in `layers/`, `kernels/`, `optimizers.py`, and `multimodal_utils.py`, while inference plumbing lives in `maxengine.py`, `decode.py`, and `inference/`.
- Data ingestion lives in `input_pipeline/`, `data_loader.py`, `multihost_dataloading.py`, with supporting utilities in `utils/` and `profiler.py`.
- Tests reside in `tests/` plus `tests/integration_tests/` with fixtures in `test_assets/`.

## Build, Test, and Development Commands
- `python -m pip install -r inference_mlperf/requirements.txt` bootstraps the shared JAX/Flax/Orbax/JetStream stack used for training and inference.
- `python MaxText/train.py configs/base.yml run_name=my_run base_output_directory=gs://bucket/maxtext steps=1000` launches a reference training job; override YAML fields as `key=value` CLI pairs.
- `python MaxText/train_compile.py configs/base.yml compile_topology=v5e-256 compile_topology_num_slices=1` generates an ahead-of-time compiled train step to validate memory needs before reserving pods.
- `python MaxText/decode.py configs/inference.yml prompt="Hello, MaxText!" checkpoint_path=gs://bucket/ckpt` runs single-host inference via `MaxEngine`.
- `pytest tests -m "cpu_only and not integration_test"` is the fast regression gate; also run `pytest tests -m gpu_only` or `-m tpu_only` on the matching hardware before merging device-specific code.

## Coding Style & Naming Conventions
Follow Google-style Python: two-space indents, docstrings, expressive type hints, `snake_case` functions, `PascalCase` classes, `UPPER_SNAKE_CASE` constants, and alphabetized explicit imports. Run `pylint` and `pytype` locally because inline disables already assume their presence.

## Testing Guidelines
Use pytest and honor the `cpu_only`, `gpu_only`, `tpu_only`, and `integration_test` markers so CI shards predictably. Favor fixtures from `test_assets/` or synthetic batches from `input_pipeline/synthetic_data_processing.py`, and accompany changes touching training, inference, or checkpointing with an integration test or a documented `pytest ...` plus `python MaxText/train.py ... steps=5` smoke run.

## Commit & Pull Request Guidelines
History currently shows terse messages (for example, `update`), but reviewers expect `area: summary` lines such as `train: fix KD loss logging` capped at 72 characters. Each PR should link to an issue, note config or flag deltas, paste the commands that produced the results, and summarize key metrics (loss, throughput, latency). Update relevant docs (`configs/README.md`, `README_KD*.md`) whenever you introduce new knobs, and prefer logs over screenshots unless UI surfaces change.
