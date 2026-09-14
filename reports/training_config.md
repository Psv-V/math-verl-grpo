# Single-RTX-5090 GRPO Candidate Configuration

## Status

These are runnable candidate overrides for the pinned verl generation, not
frozen formal hyperparameters. They target one RTX 5090 with 32 GB VRAM and
Qwen2.5-0.5B-Instruct. Smoke and pilot measurements must validate or amend them
before any formal run.

The launcher is `scripts/run_grpo.sh`. It uses verl's FSDP actor backend and a
single-GPU vLLM rollout worker. It never passes the official GSM8K test split to
the trainer.

## Candidate parameters

| Parameter | Smoke | Pilot | Formal candidate |
| --- | ---: | ---: | ---: |
| Prompts per step | 2 | 16 | 32 |
| Rollouts per prompt | 2 | 4 | 8 |
| Trajectories per step | 4 | 64 | 256 |
| PPO mini-batch trajectories | 4 | 32 | 64 |
| Dynamic tokens/GPU | 4,096 | 8,192 | 12,288 |
| Training steps | 2 | 20 | derived from 2 epochs |

Shared candidates:

- maximum prompt length: 512 tokens;
- maximum response length: 512 tokens;
- rollout temperature: 1.0;
- actor learning rate: `1e-6`;
- actor update epochs per rollout batch: 1;
- KL in reward: disabled;
- KL loss: enabled, low-variance estimator, coefficient `0.001`;
- vLLM tensor parallel size: 1;
- vLLM GPU memory utilization: 0.50;
- actor and optimizer offload: disabled;
- reference-model parameter offload: enabled;
- gradient checkpointing and remove-padding optimization: initially disabled.

The launcher applies its seed to the training-data sampler, actor mini-batch
loader, actor and reference FSDP engines, and rollout sampler. A seed in the
run name is therefore an actual verl input, not only a label.

The formal candidate processes 32 prompts and 256 generated trajectories per
optimizer iteration. Dynamic token batching is used instead of a fixed
micro-batch count. If the first pilot has ample memory, token limits or prompt
batch size may be raised for throughput. On OOM, reduce dynamic tokens first,
then prompt batch size; do not change multiple controls at once.

## Why these are only candidates

VRAM capacity alone cannot determine the final batch sizes. Actual memory also
depends on response lengths, attention backend, PyTorch/vLLM kernels, optimizer
state, Ray process lifetime, KV cache, and whether hybrid-engine cache release
works on the selected package versions.

Pilot acceptance requires:

- no CUDA OOM, NaN, inf, or Ray worker restart;
- stable peak GPU and host memory with reserve;
- nonzero reward variance for at least some prompt groups;
- finite policy loss, KL, entropy, and gradients;
- actor parameters change and vLLM weights synchronize;
- projected four-run wall time fits the remaining 72-hour budget.

Only after those checks should the actual values be copied into committed files
under `configs/runs/` and described as frozen formal settings.

## Use on the server

Set the four server-specific paths without storing them in Git:

```bash
export MATH_GRPO_MODEL_PATH=<data-disk>/models/Qwen2.5-0.5B-Instruct
export MATH_GRPO_TRAIN_FILE=<data-disk>/gsm8k/train.parquet
export MATH_GRPO_DEV_FILE=<data-disk>/gsm8k/dev.parquet
export MATH_GRPO_OUTPUT_ROOT=<data-disk>/outputs/grpo
```

Then run one profile at a time:

```bash
bash scripts/run_grpo.sh smoke format 42
bash scripts/run_grpo.sh pilot format 42
bash scripts/run_grpo.sh formal format 17
bash scripts/run_grpo.sh formal correctness 42
```

The launcher refuses `test.parquet` as validation data, rejects unregistered
formal seeds, requires a clean Git worktree for formal runs, starts with resume
disabled, and avoids overwriting an existing run directory by default.
