# Experiment Protocol

- Protocol version: `0.1.0`
- Status: preregistered before environment setup, data download, model download,
  baseline evaluation, or training
- Core hardware target: one NVIDIA RTX 5090 with 32 GB VRAM
- Maximum rented-GPU wall-clock budget: 72 hours
- Core model: Qwen2.5-0.5B-Instruct
- Optional gated model: Qwen2.5-1.5B-Instruct

This document defines the claims that may be made, the information that must be
recorded, and the conditions that invalidate an experiment. Values that depend
on an unavailable rental server are explicitly marked `TBD-AUDIT`; values that
depend on a short pilot are marked `TBD-PILOT`. They must be frozen in a new
protocol revision before formal runs begin.

## 1. Scope and primary claim

The primary claim under test is:

> Under one frozen prompting and evaluation protocol, GRPO post-training of
> Qwen2.5-0.5B-Instruct improves GSM8K strict exact-match accuracy over the
> unmodified base checkpoint on a single RTX 5090.

An engineering pipeline that runs successfully is useful but is not evidence
for this performance claim. A change in training reward alone is not evidence
for improved mathematical reasoning.

## 2. Falsifiable research questions

### RQ1: Does GRPO improve answer accuracy?

- Primary outcome: strict numeric exact-match accuracy on the official GSM8K
  test split.
- Supporting outcome: flexible numeric exact-match accuracy.
- The claim is not supported if the predefined formal runs fail to show a
  consistent positive change, or if the apparent gain disappears under the
  independent evaluator.

### RQ2: Is a gain mathematical or merely formatting?

- Compare correctness-only reward with correctness plus format reward.
- Report strict accuracy, flexible accuracy, format compliance, and parse
  failure separately.
- A strict-only gain without a flexible-accuracy gain is described as format
  compliance improvement, not reasoning improvement.
- The correctness-only condition has one formal seed under the initial compute
  budget, so this ablation is descriptive and exploratory rather than a robust
  across-seed estimate. Additional seeds may be added only as a separately
  reported extension after the core matrix is complete.

### RQ3: Are rollout and optimized policies consistent?

- Compare rollout old log-probabilities with actor-side recomputation on sampled
  tokens, subject to a tolerance frozen after the numerical smoke test.
- Record actor and rollout weight versions across synchronization points.
- A stale rollout version, unexplained systematic log-probability discrepancy,
  or missing parameter update invalidates the affected run.

### RQ4: Is a 1.5B extension feasible within the resource budget?

- This is secondary and gated on completion of the 0.5B pipeline.
- Merely loading or starting the 1.5B model is not a successful experiment.

## 3. Dataset and leakage policy

The dataset will be `openai/gsm8k`, configuration `main`, at an exact revision
recorded in the data manifest.

- The official train and test split boundary is preserved.
- A deterministic 512-example development split will be selected from the
  official training split. The remaining official training examples form the
  RL training split.
- Split seed: `20260910`.
- Split assignment operates on stable source indices after verifying source
  cardinality and before any training-time filtering.
- Exact hashes of raw and normalized questions are checked across train, dev,
  and test. Near-duplicate findings are reported rather than silently removed.
- The official test split is never passed as verl validation data.

The test split may be used for the frozen base evaluation and final frozen-model
evaluation. It may not be used for prompt changes, reward changes,
hyperparameter tuning, early stopping, or checkpoint selection. Looking at test
errors after final evaluation is allowed only for final failure analysis; any
subsequent tuned result must be labeled exploratory or evaluated on a new held-
out benchmark.

Possible exposure of GSM8K during the model's original pretraining cannot be
eliminated by this project and will be disclosed as a limitation.

## 4. Models and software

Exact model revisions and the runnable verl/PyTorch/vLLM stack are
`TBD-AUDIT`. They will be chosen only after testing the rented RTX 5090 host.
The local upstream reference checkout is not the runtime dependency by default.

The runtime manifest must include:

- GPU name, VRAM, driver, CUDA runtime, operating system, CPU, RAM, and disk.
- Python, PyTorch, Transformers, Tokenizers, vLLM, Ray, verl, and attention
  backend versions.
- Package lock or immutable container digest.
- Model repository and exact revision.
- Dataset repository and exact revision.

## 5. Prompt and answer contract

The exact prompt and chat template are `TBD-AUDIT` until tokenizer inspection,
then frozen before the base evaluation.

The response will request a final numeric answer after `####`. Evaluation keeps
two distinct views:

- Strict parse: one valid final `#### <number>` answer in the required terminal
  form.
- Flexible parse: a conservative diagnostic extraction of the final valid
  number when strict formatting fails.

Numeric normalization must be independently tested for negative values,
commas, decimals, whitespace, trailing zeroes, ambiguous multiple answers,
malformed numbers, and truncated text. Parser uncertainty produces a parse
failure rather than a guessed answer.

## 6. Reward conditions

Both formal conditions use the same independently implemented correctness
component:

1. `correctness_only`: `R = correctness`.
2. `correctness_plus_format`: `R = correctness + 0.1 * format`.

`correctness` and `format` are binary and are always logged separately. An
incorrect but well-formatted answer therefore receives at most `0.1` in the
second condition. No reward is given for response length, verbosity, or merely
containing the ground-truth character sequence.

Any change to extraction, normalization, reward weights, or the adapter contract
creates a new protocol version and requires rerunning all affected baselines and
conditions.

## 7. Formal experiment matrix

The minimum credible 0.5B matrix is:

| Condition | Seeds | Test use |
| --- | --- | --- |
| Frozen base, greedy decoding | deterministic | Full test baseline |
| GRPO correctness-only | 42 | Exploratory frozen checkpoint |
| GRPO correctness+format | 17, 42, 2026 | Frozen selected checkpoint per seed |

Smoke tests and pilots are engineering runs and are excluded from formal result
aggregation. The 1.5B extension is outside the minimum matrix.

The optimizer-step/token budget, sampling temperature, group size, learning
rate, KL coefficient, batch sizes, and maximum lengths are `TBD-PILOT`. They
must be fixed before the four formal runs. Except for the reward condition and
seed, formal configurations must be identical.

## 8. Checkpoint selection and stopping

Checkpoints are selected without test results. For each seed:

1. Choose the checkpoint with highest development strict accuracy among valid
   checkpoints.
2. Break an exact tie using higher development flexible accuracy.
3. If still tied, select the earlier checkpoint; test performance and visual
   inspection are never tie-breakers.

KL/length safety limits and the checkpoint evaluation interval are
`TBD-PILOT`; they will be frozen before formal runs. Formal runs stop at their
fixed training budget unless a predefined invalidity or safety condition
occurs. A stopped invalid run is reported and rerun with the same configuration
only if the cause is external and documented.

## 9. Required metrics

Every formal run records:

- correctness, format, total reward, per-group reward standard deviation, and
  zero-variance-group fraction;
- policy loss, reference KL, approximate KL, clip fraction, entropy, gradient
  norm, and learning rate;
- response-length distribution, EOS rate, truncation rate, and length split by
  correctness;
- actor/rollout log-probability discrepancy and weight version;
- peak allocated/reserved GPU memory, host RAM, OOM/retry events;
- rollout, log-probability, update, synchronization, validation, and checkpoint
  time, plus generated tokens/second and end-to-end samples/second.

## 10. Independent evaluation

- Base and trained checkpoints use one frozen evaluation implementation and
  generation configuration.
- The evaluator does not call the trainer RewardManager.
- The exported Hugging Face checkpoint is loaded in a fresh process.
- Exported and source checkpoints must agree on fixed fixture outputs or logits
  within a documented numerical tolerance.
- Every per-example output, parsed value, score, and failure reason is retained.

The primary correctness-plus-format report includes per-seed results, mean and
standard deviation across seeds, and paired bootstrap 95% confidence intervals
across test examples. It does not report only the best seed. The single-seed
correctness-only ablation is reported descriptively and is not presented as a
stable estimate of training variance.

## 11. Invalid-run criteria

A run cannot support the main claim if any of the following applies:

- train/dev/test leakage or an untracked dataset transformation;
- wrong model revision, unresolved dirty code, missing resolved configuration,
  or accidental resume from another experiment;
- non-finite reward, loss, KL, gradients, or parameters;
- no verified parameter change after an intended optimizer step;
- stale rollout weights or a systematic actor/rollout discrepancy outside the
  frozen tolerance;
- a reward/parser implementation different from the registered condition;
- test results influenced prompt, hyperparameters, stopping, or checkpoint
  selection;
- exported checkpoint is not independently loadable or fails equivalence checks.

Invalid runs remain documented; they are not silently deleted or replaced.

## 12. Compute and storage budget

The hard rented-GPU budget is 72 wall-clock hours:

| Activity | Maximum planned time |
| --- | ---: |
| Environment, download, base evaluation, smoke | 8 h |
| Configuration calibration and pilot | 6 h |
| Four formal 0.5B runs | 32 h |
| Export and independent reevaluation | 6 h |
| Failure reserve or optional extension | 20 h |

Before formal runs, measured pilot step time must show that the frozen matrix
fits the remaining budget. The reserve is consumed by failed-run recovery
first. Extra correctness-only seeds or a 1.5B pilot are admitted only after the
0.5B core is complete and sufficient budget remains; neither may displace the
four-run core matrix or independent reevaluation.

The initial 50 GB data disk is considered insufficiently comfortable for
dependencies, caches, and repeated full optimizer checkpoints. The preferred
capacity is at least 100 GB, or an explicit retention/offload policy must be
frozen before training.

## 13. Reporting rules

- Upstream or vendor benchmark numbers are contextual references, not project
  results.
- Single-GPU FSDP is described accurately as use of the FSDP actor backend;
  no distributed sharding speedup is claimed at world size one.
- A strict-only gain without flexible-accuracy gain is reported as formatting
  improvement.
- A one-seed run is labeled a pilot or engineering demonstration.
- Resume, export, and reproducibility claims require direct verification.
- Every resume or performance number in a CV must map to an immutable run
  manifest and result artifact.

## 14. Amendment policy

Protocol amendments are allowed before formal runs when hardware audit or pilot
evidence requires them. Each amendment must:

1. increment the protocol version;
2. describe the change and reason;
3. state which previous runs are no longer comparable;
4. be committed before affected formal runs begin.

After official test results have been inspected, changes motivated by those
results are exploratory and cannot retroactively alter the preregistered claim.
