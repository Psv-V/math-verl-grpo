# Numeric Answer Contract

## Required model format

The model may write reasoning first, but its response must end with exactly one
final numeric answer in this form:

```text
#### 42
```

The strict parser accepts signed integers, correctly grouped thousands, and
decimals. It removes thousands separators and insignificant trailing decimal
zeroes before comparison. For example, `#### -1,200.50` becomes `-1200.5`.

Strict parsing fails when the marker is missing, appears more than once, has a
malformed number, or is followed by additional text. This makes strict accuracy
measure both mathematical correctness and compliance with the output contract.

## Flexible diagnostic parsing

Flexible parsing is kept separate from strict parsing. It tries, in order:

1. the complete strict format;
2. one unambiguous answer following `####`;
3. one explicit phrase such as `final answer is 42`;
4. a response containing only one number in total.

If more than one plausible answer remains, parsing fails with
`ambiguous_answers`; the parser never chooses the last number merely because it
appears last. Flexible accuracy is diagnostic and cannot replace the strict
primary metric.

## Parse result

Both parsers return a `ParseResult` containing:

- `value`: the normalized number, or `None` on failure;
- `source`: how a successful answer was found;
- `error`: a stable reason for failure;
- `success`: a convenience property.

These fields let later evaluation and reward code report format failures and
incorrect answers separately.

