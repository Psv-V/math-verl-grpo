# Experiment Tracking Rules

This project uses Git for code provenance and SwanLab for
training curves. The tracking contract is intentionally small: every formal
run must be connected to one committed code revision, one saved resolved
configuration, and one SwanLab experiment.

## Fixed names

- SwanLab project: `math-verl-grpo`
- Primary runs:
  - `qwen25-05b-format-s17`
  - `qwen25-05b-format-s42`
  - `qwen25-05b-format-s2026`
- Exploratory ablation: `qwen25-05b-correctness-s42`
- Engineering runs must include `smoke` or `pilot` in their names and are not
  included in formal result aggregation.

## Required binding for a formal run

Before a formal run starts:

1. Commit the code and verify that `git status --short` is empty on the server.
2. Save the run's final resolved configuration under `configs/runs/` and commit
   it. The filename must match the SwanLab experiment name.
3. Record the full Git commit in SwanLab as `git_commit` and use the same run
   name for the configuration file, SwanLab experiment, and checkpoint
   directory.

The run record therefore needs only these links:

| Field | Example |
| --- | --- |
| Run name | `qwen25-05b-format-s42` |
| Git commit | output of `git rev-parse HEAD` |
| Resolved configuration | `configs/runs/qwen25-05b-format-s42.yaml` |
| SwanLab experiment | URL created when training starts |
| Checkpoint directory | directory named after the run |

## SwanLab use

Authenticate interactively on the rental server with `swanlab login`, or set
`SWANLAB_API_KEY` only in the server session. The API key is a secret: it must
not be committed, written into a configuration file, or pasted into project
documentation.

verl should be configured with the following logger identity:

```yaml
trainer:
  logger: [console, swanlab]
  project_name: math-verl-grpo
  experiment_name: <run-name>
```

Set `SWANLAB_LOG_DIR` to the persistent data disk; the launcher defaults it to
`$MATH_GRPO_OUTPUT_ROOT/swanlog`. If the server cannot reach SwanLab, set
`SWANLAB_MODE=offline` before training and synchronize the saved run after
network access is restored. The normal online mode is `cloud`.

At minimum, inspect correctness reward, format reward, total reward, policy
loss, KL, entropy, response length, validation accuracy, throughput, and GPU
memory. The complete metric requirements remain defined in the experiment
protocol.

Training checkpoint resume and SwanLab curve resume are separate. A resumed
training job must restore the intended verl checkpoint. To continue the same
SwanLab experiment, retain its experiment ID and set `SWANLAB_RUN_ID` together
with `SWANLAB_RESUME=must`; otherwise start a new SwanLab experiment and
document the relation.
