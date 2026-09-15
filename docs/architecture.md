# PairJudge: Support-Calibrated Proposal Compatibility Fields

> This document is a dated research record. The executable module in this
> repository is `pairjudge.py`; references to `probe/` and unpublished pilot
> artifacts are provenance only.

## Status

PairJudge is a motion-space research mechanism, not yet a validated 3DGS
architecture. The first real five-identity FLOAT pilot is deliberately recorded
as a falsifier: it fails the predeclared improvement gate over uniform fusion.
A solver audit then found that the initial exponentiated-gradient oracle had not
converged. The corrected Newton-KKT run fails by a larger margin; both artifacts
are retained.
The implementation and machine-readable protocol live in
`probe/pairjudge.py`, `probe/flame_regions.py`, and
`probe/run_pairjudge_probe.py`.

Evidence labels used here:

- **Synthetic:** planted mixtures and invariance/regression tests.
- **Pilot:** tracked motion from five identities and a frozen 15-second query.
- **Conference-scale:** not available yet; requires independent proposal routes,
  bilingual same-identity evaluation, renderer ablations, and human/sync tests.

## Narrow contribution claim

> PairJudge learns a target-specific, proposal-order-equivariant compatibility
> field over a frozen bank of facial-motion proposals using only a short paired
> support clip, then applies a temporally smoothed soft barycentric mixture at
> query time with explicit density/flat-logit abstention.

This claim intentionally does not say preference learning, reward learning,
human preference, multi-teacher learning, multilingual personalization, or
3DGS improvement. The current proposal bank contains four correlated FLOAT
branches (`neutral`, `auto`, `happy`, `sad`) from one generator. The pilot cannot
establish independent-teacher evidence.

## Motivation and failure boundary

The earlier static council exposed a useful but dangerous gap: the query
coordinate oracle can be much better than uniform fusion, while support-learned
static weights are not. Individual proposal cost margins are therefore not a
safe target: two proposals can each have large errors that cancel in their
mixture. Hard framewise switching is also temporally unstable.

PairJudge changes the target and the output representation:

1. Fit a convex mixture to the support target, rather than labeling one
   proposal as the winner.
2. Convert the fitted mixture to centered barycentric logits.
3. Predict pairwise logit margins with one shared low-capacity head.
4. Recover soft proposal weights, smooth logits over time, and abstain when the
   support manifold or the learned margins are unreliable.

If the real pilot does not beat uniform fusion, PairJudge remains a useful
negative result and must not be moved into 3DGS merely because its synthetic
tests pass.

## Mathematical specification

Let support proposals be `P_s in R^(T_s x K x D)`, target motion be
`Y_s in R^(T_s x D)`, query proposals be `P_q`, and audio features be `A_s,
A_q`. A region-mask matrix `M in R_+^(R x D)` is derived from local FLAME
assets. The four current regions are lips, eyes, remaining face, and jaw.

### Region normalization

The effective blend mask is

`B_{r,d} = M_{r,d} / sum_{r'} M_{r',d}`.

The descriptor mask uses the same anatomical evidence after region-mass
normalization. Zero-covered dimensions receive a uniform fallback. Exact or
proportional duplicate rows are canonicalized by summing them before
normalization. Consequently:

- common global scaling is invariant;
- common per-dimension scaling is invariant;
- splitting one region into identical halves is invariant;
- independently rescaling genuinely different regions is *not* invariant,
  because it changes their relative anatomical evidence.

The mask code computes expression influence as RMS displacement of the local
FLAME expression bases over vertex masks. The final three SMIRK jaw coordinates
are a separate jaw region. This is a proxy from the available 53-dimensional
SMIRK representation, not a claim that a 53-D coefficient has a unique vertex
support.

### Barycentric support target

For every support frame and region, solve

`w*_{t,r} = argmin_{w in simplex(K)} || sqrt(M_r) (sum_k w_k P_{s,t,k} - Y_{s,t}) ||^2`

`                         + eta sum_k w_k log(w_k + eps)`.

The implementation uses a batched Newton-KKT solver with a positivity-preserving
line search. Entropy `eta` makes cancellation representable, although the
current value is too weak to prevent near-boundary oracle weights. The oracle logits
are centered per frame:

`z*_{t,r,k} = log(w*_{t,r,k} + eps) - mean_j log(w*_{t,r,j} + eps)`.

The oracle is support-only supervision. It is never computed from query motion.

### Proposal descriptors and equivariant head

A descriptor for proposal `k` includes masked projected versions of proposal
motion, velocity, acceleration, proposal deviation from the framewise bank
mean, velocity deviation, and their interactions with context-stacked audio.
Projection is a fixed seeded orthogonal map, not a proposal-specific parameter.

For each proposal pair `(i,j)`, the shared head receives only

`x_{t,r,i,j} = phi(P_{t,r,i}) - phi(P_{t,r,j})`.

It is fit with weighted ridge regression to

`y_{t,r,i,j} = z*_{t,r,i} - z*_{t,r,j}`.

No proposal ID, mode name, or position in the input bank enters `phi`. The
pairwise construction makes the prediction equivariant to any permutation of
the proposal bank. Region/frame weights are proportional to the support benefit
of the oracle mixture over uniform, with a small positive floor.

### Query inference and abstention

The head predicts pairwise margins, recovers centered logits, applies optional
Gaussian temporal smoothing, and computes

`w_{q,t,r} = softmax(z_{q,t,r} / T)`.

The region output is `sum_k w_{q,t,r,k} P_{q,t,k}` and the final motion is
`sum_r B_r * output_{q,t,r}`.

Two support-only abstention checks are applied independently per frame/region:

- nearest-context distance must be below a radius calibrated from support
  leave-neighborhood distances;
- logit spread must exceed a flat-logit threshold.

Rejected regions use uniform weights. Thus the output remains in the proposal
  convex hull and cannot create a motion vector outside the bank.

The public inference API is exactly:

```python
predict(query_audio, query_proposals, proposal_ids=None)
```

There is no query target, query confidence, query motion, or hidden query oracle
argument.

## Protocol and controls

The pilot uses 125 support frames and 375 disjoint query frames per identity.
Each seed selects one contiguous 25-frame validation block and applies a
three-frame embargo; the final head is refit on all support frames only after
selection. The query is read only for final metrics.

Required controls:

1. Uniform proposal fusion.
2. Support-selected fixed proposal.
3. Support-audio correspondence shuffle.
4. Mismatched-identity support head.
5. Neutral anchor.
6. Query-coordinate/best-fixed oracle, reported as non-deployable headroom only.
7. Proposal permutation, identical proposals, convex-hull, and no-query-GT
   invariants.

Predeclared real gates are:

- mean correlation gain over uniform at least `+0.005`;
- at least 4/5 identities beat uniform;
- gain over shuffled support at least `+0.005`;
- gain over mismatched identity at least `+0.003`;
- MSE no worse than uniform;
- amplitude error no worse by more than `0.03`;
- match/beat support-selected fixed and neutral anchors;
- all invariants pass and accepted/fallback fractions are reported.

## Current implementation result

The first full pilot is in `probe/results/pairjudge_5id_3seed.json` and the
human-readable summary is in `probe/results/pairjudge_5id_3seed.md`.

The corrected-solver means were:

| Method | Corr | MSE | Amp. error |
|---|---:|---:|---:|
| PairJudge | 0.3441 | 0.39563 | 0.1680 |
| Uniform | 0.3807 | 0.35412 | 0.2720 |
| Shuffled support | 0.3340 | 0.40967 | 0.1545 |
| Mismatched identity | 0.3379 | 0.37582 | 0.1551 |

PairJudge improves correlation over shuffled support by `+0.0100` and over
mismatched identity by `+0.0061`, but loses `-0.0367` correlation to uniform,
loses `0.0415` MSE to uniform, and wins uniform on 0/5 identities. The
identity-bootstrap 95% interval for correlation gain over uniform is
`[-0.0490, -0.0251]`. The corrected pilot verdict is therefore **FAIL** under
the frozen gates. The accepted/fallback fraction is approximately
`0.9997 / 0.0003`.

The pre-repair artifact is retained as
`probe/results/pairjudge_5id_3seed_pre_repair.json`. It reported a smaller
`-0.0037` correlation gap, but its barycentric target was only an approximate,
non-converged solution and must not be used as the primary result.

The result is informative: support correspondence is detectable, but the
learned compatibility field is not yet a better aggregate than uniform. No
3DGS integration should be claimed from this run.

## Next falsification or repair

The next experiment should be a predeclared repair, not post-hoc tuning against
the query:

1. Freeze the current failed result.
2. Test whether support oracle logits are too noisy by fitting on block-averaged
   weights and comparing framewise versus 5-frame targets.
3. Test a shared monotone scalar compatibility head with Huber loss, keeping
   descriptor order equivariant and the same controls.
4. Add a proposal-independent audio-only baseline and a no-audio descriptor
   ablation to identify whether the signal comes from proposal geometry or
   audio context.
5. Stop if no predeclared repair beats uniform on the same 5x3 protocol.

Only a passing motion-space mechanism should proceed to independent proposal
routes (for example, a UniTalker-D1/FLAME inverse-fit path), InsTaG integration,
renderer ablations, and multilingual same-identity evaluation.

## Provenance and citations

The local FLAME model and masks are distributed with their own model-license
terms; the repository's `assets/FLAME2020/Readme.pdf` directs users to the
FLAME model license and the FLAME paper. The broader proposal/renderer threats
and dataset restrictions are tracked in `research/PairTalk_research_dossier.md`.
The current FLOAT pilot uses the existing locally generated files under
`probe/motion`, `probe/motion_bank`, and `probe/features_xlsr`; it is not a new
dataset release and does not establish the licensing status of the source
videos.
