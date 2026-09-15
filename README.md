# PairJudge

PairJudge learns a proposal-order-equivariant compatibility field from a short
paired support clip. It predicts a smooth barycentric mixture of frozen motion
proposals and abstains to uniform fusion when density or logit confidence is
insufficient.

## Status

Synthetic mixture, invariance, solver, and fallback tests pass. The corrected
real five-identity pilot failed the predeclared improvement gate over uniform
fusion, so this is a research mechanism rather than a validated talking-head
architecture. See `docs/architecture.md`.

## Install And Test

```bash
pip install -e .
python -m unittest -v
```

## Minimal Use

```python
from pairjudge import PairJudge

judge = PairJudge().fit(support_audio, support_proposals, support_real, region_masks)
motion, diagnostics = judge.predict(query_audio, query_proposals)
```

Query inference has no target-motion argument. Licensed under MIT.
