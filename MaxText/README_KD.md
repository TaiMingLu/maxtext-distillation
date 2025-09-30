## Knowledge Distillation (KD) in MaxText

This change adds standard knowledge distillation (last-layer logits) on top of the existing CE loss.

### What was added

- KD loss helper:
  - `max_utils.kl_divergence_between_logits(student_logits, teacher_logits, temperature)`

- Training integration (`train.py`):
  - `kd_loss_fn(...)`: computes CE (student vs labels) + KL(teacher || student) with temperature scaling.
  - State helpers `_split_kd_state(...)` / `_merge_kd_state(...)` to carry `teacher_params` alongside student params.
  - `train_step` / `eval_step`: select KD path when `use_kd=True`; logs `learning/kd_loss` and `evaluation/kd_loss`.
  - `setup_train_loop`: optionally load teacher parameters from a path if provided.

- Config validations (`pyconfig.py`):
  - Ensures `kd_alpha ∈ [0,1]` and `kd_temperature > 0` when `use_kd=True`.

### How it works

When `use_kd=True`, training uses:

```
loss = (1 - kd_alpha) * CE(student, labels) + kd_alpha * (T^2) * KL(teacher||student)
```

Teacher logits are computed via a separate forward pass with `teacher_params` (no dropout, stop_gradient), matching the student inputs and masking. Sequence padding is respected.

### How to enable

In your YAML or CLI overrides, set:

- `use_kd: true`
- `kd_alpha: 0.5`            # blend between CE and KD (0..1)
- `kd_temperature: 1.0`      # softmax temperature for teacher/student
- `kd_teacher_parameters_path: "/path/to/teacher/params"`  # optional; if omitted, teacher defaults to a copy of student at start

Example CLI overrides:

```
python MaxText/train.py base.yml use_kd=true kd_alpha=0.5 kd_temperature=2.0 kd_teacher_parameters_path=gs://bucket/teacher/params
```

Notes:
- If `kd_teacher_parameters_path` is provided, teacher params are restored once at startup and frozen.
- Without a path, teacher params default to a snapshot of the initialized student.
- KD is compatible with gradient accumulation and MoE/MTP features.
- KD and DPO are mutually exclusive at runtime; KD is used only when `use_dpo` is false.

### Metrics

- Train: `learning/kd_loss` (per-step), alongside existing losses.
- Eval: `evaluation/kd_loss`.

### Key code locations

- `MaxText/max_utils.py`: KD KL helper.
- `MaxText/train.py`: `kd_loss_fn`, KD state split/merge, train/eval wiring, teacher loading in `setup_train_loop`.
- `MaxText/pyconfig.py`: KD config validation.


