## Knowledge Distillation (KD) in MaxText

This change adds standard knowledge distillation (last-layer logits) on top of the existing CE loss.

### What was added

- KD loss helper:
  - `max_utils.kl_divergence_between_logits(student_logits, teacher_logits, temperature, top_k=None, top_p=None, use_other_bucket=False)`

- Training integration (`train.py`):
  - `kd_loss_fn(...)`: computes CE (student vs labels) + KL(teacher || student) with temperature scaling.
  - State helpers `_split_kd_state(...)` / `_merge_kd_state(...)` to carry `teacher_params` alongside student params.
  - `train_step` / `eval_step`: select KD path when `use_kd=True`; logs `learning/kd_loss` and `evaluation/kd_loss`.
  - `setup_train_loop`: optionally load teacher parameters from a path if provided.

- Config validations (`pyconfig.py`):
  - Ensures `kd_alpha ∈ [0,1]`, `kd_temperature > 0`, and mutually exclusive `kd_top_k` / `kd_top_p` when `use_kd=True`.

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
- `kd_top_k: 0`                  # optional; set >0 to keep only top-k teacher tokens
- `kd_top_p: 0.0`                # optional; set (0,1] to keep a nucleus by cumulative mass
- `kd_use_other_bucket: false`   # optional; when truncating, penalize student mass outside the kept set
- `kd_use_hard_labels: false`    # optional; switch KD term to teacher argmax cross-entropy

Example CLI overrides:

```
python MaxText/train.py base.yml use_kd=true kd_alpha=0.5 kd_temperature=2.0 kd_teacher_parameters_path=gs://bucket/teacher/params
```

Notes:
- If `kd_teacher_parameters_path` is provided, teacher params are restored once at startup and frozen.
- Without a path, teacher params default to a snapshot of the initialized student.
- KD is compatible with gradient accumulation and MoE/MTP features.
- KD and DPO are mutually exclusive at runtime; KD is used only when `use_dpo` is false.
- When `kd_top_k` or `kd_top_p` are set (mutually exclusive), the KD loss is computed on the truncated teacher distribution. `kd_use_other_bucket=true` aggregates the dropped probability mass into a single "OTHER" term so the student is penalized for probability left outside the kept set; otherwise the remaining mass is renormalized over the retained tokens.
- `kd_use_hard_labels=true` replaces the soft teacher distribution with hard argmax labels while still mixing with the standard NLL through `kd_alpha`.

### Metrics

- Train: `learning/kd_loss` (per-step), alongside existing losses.
- Eval: `evaluation/kd_loss`.

### Key code locations

- `MaxText/max_utils.py`: KD KL helper.
- `MaxText/train.py`: `kd_loss_fn`, KD state split/merge, train/eval wiring, teacher loading in `setup_train_loop`.
- `MaxText/pyconfig.py`: KD config validation.

