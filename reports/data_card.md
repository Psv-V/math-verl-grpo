# GSM8K Data Preparation

## Status

The preparation code passed a temporary local end-to-end run at dataset
revision `740312add88f781978c0658806c59bc2815b9866`. It produced the expected
6,961/512/1,319 split counts; the temporary dataset and cache were then
deleted. No GSM8K data is stored in this local repository. Downloading and
conversion will be repeated on the rental server after its data-disk mount
point is confirmed.

## Source and split

- Dataset: `openai/gsm8k`
- Configuration: `main`
- Revision: required at runtime and recorded in generated `metadata.json`
- Official train: 7,473 examples
- Project train: 6,961 examples
- Project dev: 512 examples selected from official train
- Official test: 1,319 examples, kept unchanged
- Dev split seed: `20260910`

The official test split is written for independent evaluation only. It must
not be supplied to verl as training validation data or used to choose prompts,
hyperparameters, stopping points, or checkpoints.

## Server command

After replacing all three placeholders with real server values:

```bash
PYTHONPATH=src python -m math_grpo.data.gsm8k \
  --revision 740312add88f781978c0658806c59bc2815b9866 \
  --cache-dir <data-disk>/huggingface \
  --output-dir <data-disk>/math-verl-grpo/data/processed/gsm8k
```

The command creates `train.parquet`, `dev.parquet`, `test.parquet`, and a small
`metadata.json` file. Dataset files are ignored by Git.

## Stored row fields

Each row contains the verl-facing `data_source`, chat `prompt`, `ability`, and
rule-reward ground truth. `extra_info` retains the original question, original
worked answer, source split, source index, project split, and dataset revision
for debugging.

## Automatic checks

Preparation stops instead of silently producing data when:

- the official source counts differ from 7,473 train and 1,319 test rows;
- an answer does not end with a parseable `#### <number>` marker;
- the final project counts differ from 6,961 train, 512 dev, and 1,319 test;
- a normalized question appears in more than one project split.

Near-duplicate semantic detection is intentionally not included in this first,
small implementation. Exact normalized overlap is the blocking leakage check;
near-duplicate analysis can be reported separately without modifying the
official benchmark.
