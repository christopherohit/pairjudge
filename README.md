# PairJudge

PairJudge is a support-calibrated compatibility field for softly mixing a
frozen bank of motion proposals. It derives per-frame, per-region barycentric
targets from a paired support clip, learns one proposal-order-equivariant
pairwise scoring rule, smooths query logits over time, and falls back to
uniform fusion when support density or logit spread is insufficient.

## Research Status

**The corrected real-data pilot is a negative result.** PairJudge is published
as an experimental mechanism and falsification record, not as a validated
talking-head architecture.

The five-identity, three-seed FLOAT pilot produced:

| Method | Correlation | MSE | Amplitude error |
|---|---:|---:|---:|
| PairJudge | `0.3441` | `0.39563` | `0.1680` |
| Uniform fusion | `0.3807` | `0.35412` | `0.2720` |
| Shuffled support | `0.3340` | `0.40967` | `0.1545` |
| Mismatched identity | `0.3379` | `0.37582` | `0.1551` |

PairJudge detects some support correspondence but loses `-0.0367` correlation
to uniform fusion, worsens MSE by `0.0415`, and wins on `0/5` identities. The
identity-bootstrap 95% interval for correlation gain over uniform is
`[-0.0490, -0.0251]`. The corrected run therefore fails the frozen utility
gate. A previous result used a non-converged oracle and is not authoritative.

See [docs/architecture.md](docs/architecture.md) for the solver audit,
protocol, controls, and permitted claim boundary.

## Problem and Non-Goals

PairJudge addresses a narrow situation: several frozen proposal generators or
modes already produce motion for the same query, and a short target support
clip may reveal when their errors cancel in a convex mixture.

It does not:

- train the proposal generators;
- rank human preference or learn a reward model;
- use query target motion;
- prove that the current FLOAT modes are independent teachers;
- render images or establish 3DGS quality;
- authorize a multilingual, personalization, or novelty claim.

## Method Overview

For support proposals `P` with shape `[T,K,D]`, target motion `Y`, and region
masks `M`, PairJudge first solves an entropy-regularized convex mixture for each
frame and region. This target can exploit error cancellation that winner-take-all
labels cannot represent.

The oracle weights become centered logits. Proposal descriptors combine audio
context, projected proposal motion, velocity, acceleration, deviation from the
proposal mean, and audio-motion interactions. A shared ridge model learns
pairwise logit margins from descriptor differences only; proposal IDs and bank
positions never enter the model. This construction is equivariant to proposal
permutation.

At query time PairJudge reconstructs logits, applies optional Gaussian temporal
smoothing, converts them to softmax weights, and blends proposals by anatomical
region. Regions outside the calibrated support-density radius or with nearly
flat logits receive uniform weights.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Runtime dependencies are NumPy and SciPy. The scaler, PCA, and weighted ridge
used by this module are implemented locally to keep the operator self-contained.

## Array Contract

| Input | Shape | Description |
|---|---|---|
| `support_audio` | `[T, F]` | Frame-aligned frozen audio features |
| `support_proposals` | `[T, K, D]` | `K` frozen motion proposals |
| `support_real` | `[T, D]` | Paired support target motion |
| `region_masks` | `[R, D]` | Non-negative anatomical masks |
| `support_confidence` | `[T]` | Optional support frame weights in `[0,1]` |
| `query_audio` | `[Tq, F]` | Query audio features |
| `query_proposals` | `[Tq, K, D]` | Query proposal bank with the same `K,D` |

Every mask row must have positive mass. Uncovered motion dimensions receive a
uniform regional fallback. Exact or proportional duplicate mask rows are
canonicalized so splitting a region into identical halves does not change the
result.

## Quick Start

```python
import numpy as np

from pairjudge import PairJudge

judge = PairJudge(
    audio_components=8,
    audio_radius=1,
    state_components=6,
    alpha=10.0,
    oracle_entropy=1e-4,
    temperature=1.0,
    temporal_sigma=1.0,
    flat_logit_threshold=0.02,
    density_quantile=0.99,
    density_scale=3.0,
)

fit_diagnostics = judge.fit(
    support_audio,
    support_proposals,
    support_real,
    region_masks,
    region_names=("lips", "eyes", "face", "jaw"),
)

motion, diagnostics = judge.predict(
    query_audio,
    query_proposals,
    proposal_ids=("neutral", "auto", "happy", "sad"),
)

print(fit_diagnostics["oracle"])
print(diagnostics["accepted_fraction"])
print(diagnostics["weights"].shape)  # [Tq, R, K]
```

`proposal_ids` are returned as diagnostic metadata only. They never affect
descriptors or predictions.

## Configuration Reference

- `audio_components`: PCA width for support audio.
- `audio_radius`: temporal audio-context radius.
- `state_components`: seeded orthogonal projection width for proposal state.
- `alpha`: weighted pairwise ridge penalty.
- `oracle_entropy`: entropy regularization in the support barycentric oracle.
- `oracle_target_sigma`: optional smoothing of support oracle logits.
- `temperature`: query softmax temperature.
- `temporal_sigma`: Gaussian smoothing applied to query logits.
- `flat_logit_threshold`: minimum per-region logit spread before acceptance.
- `density_quantile`, `density_scale`, `density_embargo`: support-only density
  calibration controls.
- `random_state`: deterministic projection and PCA seed.

`fit` returns oracle convergence information, pairwise fit MSE, density radius,
proposal/motion/region counts, region mass, and mean oracle entropy.

`predict` returns motion plus weights, logits, per-region acceptance, accepted
and fallback fractions, query density, radius, logit spread, and proposal IDs.

## Guarantees and Fallbacks

- Query inference has no target-motion argument.
- Output stays in the coordinate-wise convex hull of the proposal bank.
- Permuting proposals and applying the same inverse permutation to outputs does
  not change the blended motion.
- Low-density or flat-logit regions use exact uniform proposal weights.
- Mask normalization is invariant to common scaling and duplicate splitting.

These are mathematical and software properties, not empirical quality claims.

## Testing

```bash
python -m unittest -v
```

The suite covers:

- agreement of the Newton-KKT oracle with an independent SLSQP solve;
- recovery of a planted audio-conditioned mixture;
- error cancellation in barycentric targets;
- smoother soft mixtures than hard switching;
- shuffled and mismatched-support controls;
- density and flat-logit fallback;
- mask-scaling and duplicate-region invariance;
- proposal permutation equivariance and convex-hull containment;
- absence of query ground truth from the public API.

## Evaluation Guidance

Uniform fusion is a mandatory baseline. Also compare support-selected fixed
proposals, shuffled support, mismatched identity, a neutral anchor, and
non-deployable query oracles reported only as headroom. Controls must receive
the same search capacity as the aligned method.

Do not tune thresholds after reading query performance. If a future repair does
not beat uniform under predeclared gates, preserve the failure rather than
moving the method into a renderer because synthetic tests pass.

## Repository Layout

- `pairjudge.py` - oracle, mask normalization, equivariant head, and fallback.
- `test_pairjudge.py` - solver, invariance, control, and API tests.
- `docs/architecture.md` - dated protocol, corrected pilot, and repair options.

## License and Asset Responsibility

Code and documentation are MIT licensed. No media, identities, motion captures,
FLOAT outputs, datasets, checkpoints, renderers, or FLAME assets are included.
Those inputs remain governed by their own licenses, consent, privacy, and
biometric-data obligations.
