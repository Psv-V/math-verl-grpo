# Reward Contract

## Formal reward conditions

The project has two reward conditions. Both use the same answer parser and
correctness calculation.

```text
correctness_plus_format = correctness + 0.1 * format
correctness_only        = correctness
```

`correctness` is 1 only when the conservative flexible parser produces the
normalized ground-truth number. `format` is 1 only when strict parsing accepts
exactly one terminal `#### <number>` answer. Both components are otherwise 0.

This separation means a mathematically correct but incorrectly formatted
answer can receive correctness credit, while a well-formatted wrong answer can
receive at most 0.1 in the primary condition.

| Response outcome | Correctness | Format | Primary total |
| --- | ---: | ---: | ---: |
| Correct and strict | 1 | 1 | 1.1 |
| Correct but non-strict | 1 | 0 | 1.0 |
| Incorrect but strict | 0 | 1 | 0.1 |
| Incorrect and non-strict | 0 | 0 | 0.0 |

## verl entry points

The implementation is in `src/math_grpo/rewards.py`:

- `compute_score`: correctness plus format;
- `compute_score_correctness_only`: correctness-only ablation.

Both follow verl's custom reward signature:

```python
reward_fn(data_source, solution_str, ground_truth, extra_info=None)
```

They return a dictionary whose `score` value is used for training. The numeric
`correctness`, `format`, and `parse_success` values are retained as separate
verl metrics for console and SwanLab reporting.

The formal run configuration must select the intended function explicitly.
Changing the parser, weights, or selected entry point creates a new reward
condition and requires a new protocol version.
