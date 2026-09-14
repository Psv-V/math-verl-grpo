# Independent Evaluation Protocol

## Purpose

The frozen base model and every exported trained checkpoint must be evaluated
with the same code, prompts, parser, and deterministic generation settings.
The evaluator does not call the training `RewardManager` or the project reward
functions.

## Model and generation

- Core model: `Qwen/Qwen2.5-0.5B-Instruct`
- Model and tokenizer revision: exact commit, frozen on the server
- Engine: vLLM offline `LLM.chat`
- Completions per question: 1
- Temperature: 0
- Top-p: 1
- Seed: 42
- Maximum tokens and model length: frozen after the dev smoke run

The model's own chat template is applied by vLLM. Base and trained checkpoints
must use the same prompt records and generation settings.

## Order of use

1. Inspect the tokenizer and chat template on the server.
2. Run a small development subset with `--limit`.
3. Run the complete 512-example development split.
4. Freeze the prompt and generation lengths.
5. Run the official 1,319-example test split once for the formal base result.

The official test result must not influence prompt design, hyperparameters,
checkpoint selection, or stopping. Development runs must never point at
`test.parquet`.

## Metrics

- `strict_accuracy`: normalized answer is correct under strict terminal format;
- `flexible_accuracy`: normalized answer is correct under conservative parsing;
- `format_compliance`: strict parser accepts the response;
- `parse_failure_rate`: flexible parser finds no unambiguous answer;
- `truncation_rate`: vLLM reports `finish_reason=length`;
- `mean_response_tokens`: average generated token count.

Strict accuracy is the primary metric. A strict-only improvement without a
flexible-accuracy improvement is reported as formatting improvement.

## Artifacts

Each evaluation writes a new output directory containing:

- `predictions.jsonl`: question, response, parsed answers, scores, token count,
  finish reason, source split, and source index for every example;
- `summary.json`: aggregate metrics and the exact generation settings.

Evaluation outputs are stored on the server under `outputs/` and are ignored by
Git. The formal result must be linked to its model revision or exported
checkpoint, data revision, evaluation configuration, and Git commit.

## Server command shape

Values marked with angle brackets are filled only after the server audit:

```bash
PYTHONPATH=src python -m math_grpo.evaluate \
  --revision <exact-model-commit> \
  --input-file <data-disk>/gsm8k/dev.parquet \
  --download-dir <data-disk>/huggingface \
  --output-dir <data-disk>/outputs/base-dev-smoke \
  --max-tokens <frozen-after-dev> \
  --max-model-len <frozen-after-dev> \
  --limit 16
```

Remove `--limit` only after the small dev run is inspected. Replace the input
and output paths with the official test paths only after the complete dev
configuration is frozen.

