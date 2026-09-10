# Math Reasoning RLVR/GRPO on a Single RTX 5090

This project studies whether GRPO post-training can improve the mathematical
reasoning accuracy of a small Qwen instruction model on GSM8K using one NVIDIA
RTX 5090 32 GB GPU.

The core target is Qwen2.5-0.5B-Instruct with verl, vLLM, and the FSDP actor
backend. Qwen2.5-1.5B-Instruct is an optional gated extension, not part of the
minimum completion criteria. The total rented-GPU budget is capped at 72 hours.

## Current status

Stage 0 is complete: research scope and experiment protocol were preregistered
before any dependencies, models, or datasets were installed or downloaded.
Stage 1 has recorded the rental vendor's reported hardware/image metadata and a
candidate dependency set; live-host validation is still pending. No training
result is claimed yet. Stage 2 fixes the minimal experiment naming and tracking
rules and provides a planning template; final run configurations will be frozen
only after the server audit and pilot. Stage 3 data preparation code is written
but has not been executed; no dataset has been downloaded locally. Stage 4
implements strict and conservative flexible numeric answer parsing; its tests
are written but have not yet been run.

The detailed protocol is in [reports/protocol.md](reports/protocol.md), and the
minimal Git/W&B binding rules are in
[reports/experiment_tracking.md](reports/experiment_tracking.md). The GSM8K
split and conversion contract is in [reports/data_card.md](reports/data_card.md).
The model response format and parsing rules are in
[reports/answer_contract.md](reports/answer_contract.md).

## Intended experiment

- Dataset: the official `openai/gsm8k` main split at a pinned revision.
- Core model: Qwen2.5-0.5B-Instruct at a pinned revision.
- Method: synchronous GRPO, rule-based outcome reward, vLLM rollouts, and a
  verl FSDP actor backend on one RTX 5090.
- Primary comparison: frozen base model versus GRPO checkpoints under one
  unchanged independent evaluation pipeline.
- Ablation: correctness-only reward versus correctness plus format reward.
- Reliability: three predefined seeds for the primary 0.5B condition, one
  explicitly exploratory correctness-only ablation, checkpoint export
  validation, test-set isolation, and failure analysis.

## Reproducibility principles

1. Training and tuning use only an official-train-derived train/dev split.
2. The official GSM8K test set is not used for training, early stopping,
   prompt design, hyperparameter selection, or checkpoint selection.
3. Every formal run records code revision, resolved configuration, environment,
   data hash, model revision, seed, metrics, and checkpoint provenance.
4. Mathematical correctness and output-format compliance are measured and
   logged separately.
5. Results are reported across all predefined seeds, not only the best run.
6. Only independently re-evaluated exported checkpoints may support final
   performance or resume claims.

## Relationship to verl

`references/verl-official` is a local sparse checkout of the upstream verl
repository at commit `1252cc71aa5bd82e5604322064d69bfe6454c660`. It is used
only to study public interfaces and verify configuration semantics. It is
ignored by this repository and is not part of the project's claimed
implementation.

The runnable project will depend on an exact, separately installed verl
version selected after an RTX 5090 compatibility audit. Project-owned data
processing, answer parsing, rewards, formula checks, evaluation, consistency
checks, statistics, and reporting will be implemented independently.

## Non-goals for the first version

- Agent Lightning, Calc-X, MCP, or external tools.
- Multi-node or multi-GPU scaling claims.
- Training on the official GSM8K test split.
- Reproducing an upstream benchmark number without matching its full protocol.
- Presenting copied upstream code as original implementation.

## License

Original project code and documentation are licensed under Apache-2.0. Upstream
dependencies, datasets, models, and the local reference checkout retain their
own licenses and attribution requirements. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
