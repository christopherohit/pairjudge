"""PairJudge support-calibrated proposal compatibility fields.

PairJudge learns a low-capacity, proposal-order-equivariant compatibility rule
from a short support clip.  The supervision target is an entropy-regularized
barycentric mixture of frozen motion proposals, not the cost of any individual
proposal.  Query inference accepts only audio and frozen proposals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d


class StandardScaler:
    """Small NumPy-only subset of sklearn's scaler used by this probe."""

    def __init__(self, with_mean: bool = True):
        self.with_mean = bool(with_mean)
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, value: np.ndarray) -> "StandardScaler":
        value = np.asarray(value, dtype=np.float64)
        self.mean_ = value.mean(axis=0) if self.with_mean else np.zeros(value.shape[1])
        scale = value.std(axis=0)
        self.scale_ = np.where(scale > 1e-12, scale, 1.0)
        return self

    def transform(self, value: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("scaler is not fitted")
        return (np.asarray(value, dtype=np.float64) - self.mean_) / self.scale_


class PCA:
    """Deterministic unwhitened PCA sufficient for the support audio path."""

    def __init__(self, n_components: int):
        self.n_components = int(n_components)
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None

    def fit(self, value: np.ndarray) -> "PCA":
        value = np.asarray(value, dtype=np.float64)
        self.mean_ = value.mean(axis=0)
        _, _, right = np.linalg.svd(value - self.mean_, full_matrices=False)
        self.components_ = right[:self.n_components]
        return self

    def transform(self, value: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA is not fitted")
        return (np.asarray(value, dtype=np.float64) - self.mean_) @ self.components_.T

    def fit_transform(self, value: np.ndarray) -> np.ndarray:
        return self.fit(value).transform(value)


class Ridge:
    """Weighted no-intercept ridge regression for pairwise scalar margins."""

    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.coef_: Optional[np.ndarray] = None

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "Ridge":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        weight = (
            np.ones(len(x), dtype=np.float64)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )
        root = np.sqrt(np.maximum(weight, 0.0))
        weighted_x = x * root[:, None]
        weighted_y = y * root
        system = weighted_x.T @ weighted_x
        system.flat[::system.shape[0] + 1] += self.alpha
        rhs = weighted_x.T @ weighted_y
        try:
            self.coef_ = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(system, rhs, rcond=None)[0]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("ridge model is not fitted")
        return np.asarray(x, dtype=np.float64) @ self.coef_


def context_stack(features: np.ndarray, radius: int) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"features must be [T,F], got {features.shape}")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return features.copy()
    padded = np.pad(features, ((radius, radius), (0, 0)), mode="edge")
    return np.concatenate(
        [padded[offset:offset + len(features)] for offset in range(2 * radius + 1)],
        axis=1,
    )


def _softmax(value: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = value - np.max(value, axis=axis, keepdims=True)
    exp = np.exp(np.clip(shifted, -60.0, 60.0))
    return exp / np.sum(exp, axis=axis, keepdims=True)


@dataclass(frozen=True)
class RegionMasks:
    """Two normalized views of the same non-negative anatomical masks."""

    blend: np.ndarray
    descriptor: np.ndarray
    mass: np.ndarray


def normalize_region_masks(masks: np.ndarray) -> RegionMasks:
    """Normalize masks while preserving the declared mask invariants.

    Scaling every region by a common scalar or by a common per-dimension vector
    leaves the result unchanged.  Splitting a region into identical half-masks
    leaves its total blend contribution and regression weight unchanged.
    """
    raw = np.asarray(masks, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] < 1:
        raise ValueError(f"region masks must be [R,D], got {raw.shape}")
    if not np.all(np.isfinite(raw)) or np.any(raw < 0.0):
        raise ValueError("region masks must be finite and non-negative")
    if np.any(raw.sum(axis=1) <= 0.0):
        raise ValueError("every region mask must contain positive mass")

    raw = raw.copy()
    # Canonicalize exact/proportional duplicate regions before normalizing.
    # This makes a region split into two identical halves equivalent to the
    # unsplit region without making arbitrary rescaling of distinct regions
    # invariant.
    row_norm = np.linalg.norm(raw, axis=1, keepdims=True)
    directions = raw / row_norm
    groups: list[list[int]] = []
    for index, direction in enumerate(directions):
        for group in groups:
            if np.allclose(direction, directions[group[0]], atol=1e-12, rtol=1e-12):
                group.append(index)
                break
        else:
            groups.append([index])
    raw = np.stack([raw[group].sum(axis=0) for group in groups], axis=0)
    uncovered = raw.sum(axis=0) <= 0.0
    raw[:, uncovered] = 1.0
    blend = raw / raw.sum(axis=0, keepdims=True)
    mass = blend.sum(axis=1)
    descriptor = blend / mass[:, None]
    return RegionMasks(blend=blend, descriptor=descriptor, mass=mass)


def _motion_derivatives(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    velocity = np.zeros_like(value)
    velocity[1:] = value[1:] - value[:-1]
    acceleration = np.zeros_like(value)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    return velocity, acceleration


def _orthogonal_projection(dims: int, components: int, seed: int) -> np.ndarray:
    components = min(max(1, int(components)), dims)
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(dims, components))
    q, _ = np.linalg.qr(matrix)
    return q[:, :components]


def barycentric_oracle(
    proposals: np.ndarray,
    target: np.ndarray,
    descriptor_masks: np.ndarray,
    entropy: float = 1e-4,
    steps: int = 400,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, dict]:
    """Solve per-frame, per-region entropy-regularized convex mixtures.

    The vectorized Newton-KKT solver minimizes masked squared error plus
    ``entropy * sum(w * log(w))`` over the proposal simplex.
    """
    bank = np.asarray(proposals, dtype=np.float64)
    real = np.asarray(target, dtype=np.float64)
    masks = np.asarray(descriptor_masks, dtype=np.float64)
    if bank.ndim != 3:
        raise ValueError(f"proposals must be [T,K,D], got {bank.shape}")
    if real.shape != (bank.shape[0], bank.shape[2]):
        raise ValueError("target must be frame-aligned with proposals")
    if masks.ndim != 2 or masks.shape[1] != bank.shape[2]:
        raise ValueError("descriptor masks must have the proposal motion dimension")
    if bank.shape[1] < 1:
        raise ValueError("at least one proposal is required")
    if entropy < 0.0:
        raise ValueError("entropy must be non-negative")

    frames, proposals_count, dims = bank.shape
    regions = masks.shape[0]
    sqrt_masks = np.sqrt(masks)
    weighted_bank = (
        bank[:, None, :, :] * sqrt_masks[None, :, None, :]
    ).reshape(frames * regions, proposals_count, dims)
    weighted_real = (
        real[:, None, :] * sqrt_masks[None, :, :]
    ).reshape(frames * regions, dims)

    weights = np.full(
        (frames * regions, proposals_count), 1.0 / proposals_count,
        dtype=np.float64,
    )
    iterations = 0
    if entropy > 0.0:
        # The quadratic objective is convex, and the entropy term makes its
        # simplex interior smooth. Batched Newton-KKT steps converge much faster
        # than a fixed mirror step when proposal scales differ by dimension.
        gram = np.einsum("nkd,njd->nkj", weighted_bank, weighted_bank)
        linear = np.einsum("nkd,nd->nk", weighted_bank, weighted_real)

        def objective(value: np.ndarray) -> np.ndarray:
            prediction = np.einsum("nkd,nk->nd", weighted_bank, value)
            error = np.sum((prediction - weighted_real) ** 2, axis=1)
            return error + entropy * np.sum(
                value * np.log(value + 1e-300), axis=1,
            )

        for iterations in range(1, int(steps) + 1):
            gradient = 2.0 * (
                np.einsum("nkj,nj->nk", gram, weights) - linear
            ) + entropy * (np.log(weights + 1e-300) + 1.0)
            hessian = 2.0 * gram.copy()
            diagonal = np.arange(proposals_count)
            hessian[:, diagonal, diagonal] += entropy / np.maximum(weights, 1e-12)
            kkt = np.zeros(
                (len(weights), proposals_count + 1, proposals_count + 1),
                dtype=np.float64,
            )
            kkt[:, :proposals_count, :proposals_count] = hessian
            kkt[:, :proposals_count, proposals_count] = 1.0
            kkt[:, proposals_count, :proposals_count] = 1.0
            rhs = np.zeros((len(weights), proposals_count + 1, 1), dtype=np.float64)
            rhs[:, :proposals_count, 0] = -gradient
            delta = np.linalg.solve(kkt, rhs)[:, :proposals_count, 0]

            step = np.ones(len(weights), dtype=np.float64)
            step = np.minimum(
                step,
                np.min(
                    np.where(
                        delta < 0.0,
                        -0.99 * weights / (delta - 1e-300),
                        np.inf,
                    ),
                    axis=1,
                ),
            )
            old_objective = objective(weights)
            for _ in range(40):
                candidate = weights + step[:, None] * delta
                candidate = np.maximum(candidate, 1e-15)
                candidate /= candidate.sum(axis=1, keepdims=True)
                new_objective = objective(candidate)
                bad = new_objective > old_objective + 1e-12
                if not np.any(bad):
                    break
                step[bad] *= 0.5
            update = candidate - weights
            weights = candidate
            if float(np.max(np.abs(update))) < tolerance:
                break
    else:
        # Entropy-free callers retain a stable projected mirror fallback.
        lipschitz = 2.0 * np.sum(weighted_bank * weighted_bank, axis=(1, 2))
        step_size = 0.5 / (lipschitz + 1e-10)
        for iterations in range(1, int(steps) + 1):
            prediction = np.einsum("nkd,nk->nd", weighted_bank, weights)
            gradient = 2.0 * np.einsum(
                "nkd,nd->nk", weighted_bank, prediction - weighted_real,
            )
            updated = _softmax(
                np.log(weights + 1e-12) - step_size[:, None] * gradient,
                axis=1,
            )
            delta = float(np.max(np.abs(updated - weights)))
            weights = updated
            if delta < tolerance:
                break

    prediction = np.einsum("nkd,nk->nd", weighted_bank, weights)
    uniform = np.mean(weighted_bank, axis=1)
    oracle_error = np.mean((prediction - weighted_real) ** 2, axis=1)
    uniform_error = np.mean((uniform - weighted_real) ** 2, axis=1)
    weights = weights.reshape(frames, regions, proposals_count)
    diagnostics = {
        "iterations": int(iterations),
        "mean_oracle_error": float(np.mean(oracle_error)),
        "mean_uniform_error": float(np.mean(uniform_error)),
        "mean_oracle_gain": float(np.mean(uniform_error - oracle_error)),
    }
    return weights, diagnostics


@dataclass
class PairJudgeState:
    audio_scaler: StandardScaler
    audio_pca: Optional[PCA]
    context_scaler: StandardScaler
    descriptor_scaler: StandardScaler
    model: Ridge
    projection: np.ndarray
    region_masks: RegionMasks
    support_context: np.ndarray
    density_radius: float
    proposals_count: int
    motion_dims: int


class PairJudge:
    """Learn support-specific soft compatibility among frozen proposals."""

    def __init__(
        self,
        *,
        audio_components: int = 8,
        audio_radius: int = 1,
        state_components: int = 6,
        alpha: float = 10.0,
        oracle_entropy: float = 1e-4,
        oracle_target_sigma: float = 0.0,
        temperature: float = 1.0,
        temporal_sigma: float = 1.0,
        flat_logit_threshold: float = 0.02,
        density_quantile: float = 0.99,
        density_scale: float = 3.0,
        density_embargo: int = 2,
        random_state: int = 0,
    ):
        if audio_components < 1 or state_components < 1:
            raise ValueError("component counts must be positive")
        if alpha < 0.0 or oracle_entropy < 0.0:
            raise ValueError("regularization values must be non-negative")
        if oracle_target_sigma < 0.0:
            raise ValueError("oracle_target_sigma must be non-negative")
        if temperature <= 0.0 or temporal_sigma < 0.0:
            raise ValueError("temperature must be positive and sigma non-negative")
        if not 0.0 < density_quantile <= 1.0:
            raise ValueError("density_quantile must be in (0,1]")
        self.audio_components = int(audio_components)
        self.audio_radius = int(audio_radius)
        self.state_components = int(state_components)
        self.alpha = float(alpha)
        self.oracle_entropy = float(oracle_entropy)
        self.oracle_target_sigma = float(oracle_target_sigma)
        self.temperature = float(temperature)
        self.temporal_sigma = float(temporal_sigma)
        self.flat_logit_threshold = float(flat_logit_threshold)
        self.density_quantile = float(density_quantile)
        self.density_scale = float(density_scale)
        self.density_embargo = int(density_embargo)
        self.random_state = int(random_state)
        self.state: Optional[PairJudgeState] = None
        self.fit_diagnostics: Optional[dict] = None

    @staticmethod
    def _validate_episode(
        audio: np.ndarray,
        proposals: np.ndarray,
        real: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        audio = np.asarray(audio, dtype=np.float64)
        proposals = np.asarray(proposals, dtype=np.float64)
        if audio.ndim != 2:
            raise ValueError(f"audio must be [T,F], got {audio.shape}")
        if proposals.ndim != 3:
            raise ValueError(f"proposals must be [T,K,D], got {proposals.shape}")
        if len(audio) != len(proposals):
            raise ValueError("audio and proposals must be frame-aligned")
        if not np.all(np.isfinite(audio)) or not np.all(np.isfinite(proposals)):
            raise ValueError("audio and proposals must be finite")
        if real is None:
            return audio, proposals, None
        real = np.asarray(real, dtype=np.float64)
        if real.shape != (len(audio), proposals.shape[2]):
            raise ValueError("real motion must be [T,D] and frame-aligned")
        if not np.all(np.isfinite(real)):
            raise ValueError("real motion must be finite")
        return audio, proposals, real

    def _fit_audio(self, audio: np.ndarray) -> tuple[np.ndarray, StandardScaler, Optional[PCA], StandardScaler]:
        scaler = StandardScaler().fit(audio)
        scaled = scaler.transform(audio)
        components = min(self.audio_components, len(audio) - 1, audio.shape[1])
        pca = None
        if components >= 1 and components < audio.shape[1]:
            pca = PCA(n_components=components)
            embedded = pca.fit_transform(scaled)
        else:
            embedded = scaled
        context = context_stack(embedded, self.audio_radius)
        context_scaler = StandardScaler().fit(context)
        return context_scaler.transform(context), scaler, pca, context_scaler

    def _transform_audio(self, audio: np.ndarray, state: PairJudgeState) -> np.ndarray:
        embedded = state.audio_scaler.transform(audio)
        if state.audio_pca is not None:
            embedded = state.audio_pca.transform(embedded)
        return state.context_scaler.transform(context_stack(embedded, self.audio_radius))

    @staticmethod
    def _descriptors(
        context: np.ndarray,
        proposals: np.ndarray,
        descriptor_masks: np.ndarray,
        projection: np.ndarray,
    ) -> np.ndarray:
        velocity, acceleration = _motion_derivatives(proposals)
        mean = proposals.mean(axis=1, keepdims=True)
        mean_velocity = velocity.mean(axis=1, keepdims=True)
        deviation = proposals - mean
        velocity_deviation = velocity - mean_velocity
        sources = (proposals, velocity, acceleration, deviation, velocity_deviation)
        rows = []
        for mask in descriptor_masks:
            root = np.sqrt(mask)[None, None, :]
            projected = [
                np.einsum("tkd,dq->tkq", source * root, projection)
                for source in sources
            ]
            norms = [
                np.sqrt(np.sum((source * root) ** 2, axis=2) + 1e-12)[..., None]
                for source in sources
            ]
            state = np.concatenate(projected + norms, axis=2)
            interaction_state = np.concatenate(
                [projected[0], projected[3], projected[4]], axis=2,
            )
            interaction = (
                context[:, None, :, None]
                * interaction_state[:, :, None, :]
            ).reshape(len(proposals), proposals.shape[1], -1)
            rows.append(np.concatenate([state, interaction], axis=2))
        return np.stack(rows, axis=1)

    @staticmethod
    def _pairwise_training(
        descriptors: np.ndarray,
        logits: np.ndarray,
        frame_region_weight: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, _, proposals_count, _ = descriptors.shape
        x_rows = []
        y_rows = []
        weights = []
        for left in range(proposals_count):
            for right in range(left + 1, proposals_count):
                x_rows.append(descriptors[:, :, left] - descriptors[:, :, right])
                y_rows.append(logits[:, :, left] - logits[:, :, right])
                weights.append(frame_region_weight)
        x = np.stack(x_rows, axis=2).reshape(-1, descriptors.shape[-1])
        y = np.stack(y_rows, axis=2).reshape(-1)
        sample_weight = np.stack(weights, axis=2).reshape(-1)
        return x, y, sample_weight

    def _density_radius(self, context: np.ndarray) -> tuple[float, np.ndarray]:
        frames = len(context)
        if frames < 2 * self.density_embargo + 2:
            return float("inf"), np.zeros(frames, dtype=np.float64)
        distance = np.sqrt(np.maximum(
            np.sum((context[:, None, :] - context[None, :, :]) ** 2, axis=2),
            0.0,
        ))
        indices = np.arange(frames)
        excluded = np.abs(indices[:, None] - indices[None, :]) <= self.density_embargo
        distance[excluded] = np.inf
        scores = np.min(distance, axis=1)
        finite = scores[np.isfinite(scores)]
        radius = float(np.quantile(finite, self.density_quantile) * self.density_scale)
        return radius, scores

    def fit(
        self,
        support_audio: np.ndarray,
        support_proposals: np.ndarray,
        support_real: np.ndarray,
        region_masks: np.ndarray,
        *,
        support_confidence: Optional[np.ndarray] = None,
        region_names: Optional[Sequence[str]] = None,
    ) -> dict:
        audio, proposals, real = self._validate_episode(
            support_audio, support_proposals, support_real,
        )
        assert real is not None
        regions = normalize_region_masks(region_masks)
        if regions.blend.shape[1] != proposals.shape[2]:
            raise ValueError("region masks and proposal motion dimensions differ")
        if region_names is not None and len(region_names) != regions.blend.shape[0]:
            raise ValueError("region_names must match the number of masks")

        context, audio_scaler, audio_pca, context_scaler = self._fit_audio(audio)
        projection = _orthogonal_projection(
            proposals.shape[2], self.state_components, self.random_state,
        )
        descriptors = self._descriptors(
            context, proposals, regions.descriptor, projection,
        )
        oracle_weights, oracle_diagnostics = barycentric_oracle(
            proposals,
            real,
            regions.descriptor,
            entropy=self.oracle_entropy,
        )
        oracle_logits = np.log(oracle_weights + 1e-9)
        oracle_logits -= oracle_logits.mean(axis=2, keepdims=True)
        if self.oracle_target_sigma > 0.0 and len(oracle_logits) > 1:
            oracle_logits = gaussian_filter1d(
                oracle_logits,
                sigma=self.oracle_target_sigma,
                axis=0,
                mode="nearest",
            )
            oracle_logits -= oracle_logits.mean(axis=2, keepdims=True)

        uniform_prediction = proposals.mean(axis=1)
        uniform_error = np.einsum(
            "td,rd->tr", (uniform_prediction - real) ** 2, regions.descriptor,
        )
        oracle_prediction = np.einsum("trk,tkd->trd", oracle_weights, proposals)
        oracle_error = np.einsum(
            "trd,rd->tr", (oracle_prediction - real[:, None, :]) ** 2,
            regions.descriptor,
        )
        benefit = np.maximum(uniform_error - oracle_error, 0.0)
        positive = benefit[benefit > 1e-12]
        benefit_scale = float(np.median(positive)) if len(positive) else 1.0
        frame_region_weight = 0.1 + np.clip(benefit / (benefit_scale + 1e-12), 0.0, 5.0)
        frame_region_weight *= regions.mass[None, :] / np.mean(regions.mass)
        if support_confidence is not None:
            confidence = np.asarray(support_confidence, dtype=np.float64)
            if confidence.shape != (len(audio),):
                raise ValueError("support_confidence must be [T]")
            frame_region_weight *= np.clip(confidence[:, None], 0.0, 1.0)

        descriptor_scaler = StandardScaler(with_mean=False).fit(
            descriptors.reshape(-1, descriptors.shape[-1]),
        )
        scaled_descriptors = descriptor_scaler.transform(
            descriptors.reshape(-1, descriptors.shape[-1]),
        ).reshape(descriptors.shape)
        pair_x, pair_y, pair_weight = self._pairwise_training(
            scaled_descriptors, oracle_logits, frame_region_weight,
        )
        model = Ridge(alpha=self.alpha).fit(
            pair_x, pair_y, sample_weight=pair_weight,
        )
        density_radius, support_density = self._density_radius(context)

        self.state = PairJudgeState(
            audio_scaler=audio_scaler,
            audio_pca=audio_pca,
            context_scaler=context_scaler,
            descriptor_scaler=descriptor_scaler,
            model=model,
            projection=projection,
            region_masks=regions,
            support_context=context,
            density_radius=density_radius,
            proposals_count=proposals.shape[1],
            motion_dims=proposals.shape[2],
        )
        fitted_pair = model.predict(pair_x)
        pair_mse = float(np.average((fitted_pair - pair_y) ** 2, weights=pair_weight))
        self.fit_diagnostics = {
            "oracle": oracle_diagnostics,
            "pairwise_mse": pair_mse,
            "density_radius": density_radius,
            "support_density_mean": float(np.mean(support_density)),
            "mean_oracle_weight_entropy": float(np.mean(
                -np.sum(oracle_weights * np.log(oracle_weights + 1e-9), axis=2)
            )),
            "oracle_target_sigma": self.oracle_target_sigma,
            "proposal_count": int(proposals.shape[1]),
            "motion_dims": int(proposals.shape[2]),
            "region_count": int(regions.blend.shape[0]),
            "region_names": list(region_names) if region_names is not None else None,
            "region_mass": [float(value) for value in regions.mass],
        }
        return self.fit_diagnostics

    def _query_density(self, context: np.ndarray, state: PairJudgeState) -> np.ndarray:
        squared = np.sum(
            (context[:, None, :] - state.support_context[None, :, :]) ** 2,
            axis=2,
        )
        return np.sqrt(np.maximum(np.min(squared, axis=1), 0.0))

    def predict(
        self,
        query_audio: np.ndarray,
        query_proposals: np.ndarray,
        proposal_ids: Optional[Sequence[str]] = None,
    ) -> tuple[np.ndarray, dict]:
        """Predict from query audio and proposals without query ground truth."""
        if self.state is None:
            raise RuntimeError("PairJudge must be fitted before predict")
        audio, proposals, _ = self._validate_episode(query_audio, query_proposals)
        state = self.state
        if proposals.shape[1:] != (state.proposals_count, state.motion_dims):
            raise ValueError("query proposal shape differs from support")
        if proposal_ids is not None and len(proposal_ids) != state.proposals_count:
            raise ValueError("proposal_ids must match the proposal count")

        context = self._transform_audio(audio, state)
        descriptors = self._descriptors(
            context, proposals, state.region_masks.descriptor, state.projection,
        )
        scaled = state.descriptor_scaler.transform(
            descriptors.reshape(-1, descriptors.shape[-1]),
        ).reshape(descriptors.shape)
        logits = state.model.predict(scaled.reshape(-1, scaled.shape[-1])).reshape(
            scaled.shape[:3]
        )
        logits -= logits.mean(axis=2, keepdims=True)
        if self.temporal_sigma > 0.0 and len(logits) > 1:
            logits = gaussian_filter1d(
                logits, sigma=self.temporal_sigma, axis=0, mode="nearest",
            )
            logits -= logits.mean(axis=2, keepdims=True)
        logits /= self.temperature

        density = self._query_density(context, state)
        spread = np.std(logits, axis=2)
        accepted = (
            (density[:, None] <= state.density_radius)
            & (spread >= self.flat_logit_threshold)
        )
        weights = _softmax(logits, axis=2)
        uniform = np.full_like(weights, 1.0 / state.proposals_count)
        weights = np.where(accepted[:, :, None], weights, uniform)
        region_prediction = np.einsum("trk,tkd->trd", weights, proposals)
        prediction = np.einsum(
            "rd,trd->td", state.region_masks.blend, region_prediction,
        )
        diagnostics = {
            "weights": weights,
            "logits": logits,
            "accepted": accepted,
            "accepted_fraction": float(np.mean(accepted)),
            "fallback_fraction": float(1.0 - np.mean(accepted)),
            "density": density,
            "density_radius": float(state.density_radius),
            "logit_spread": spread,
            # IDs are metadata only; they are never part of a descriptor.
            "proposal_ids": tuple(proposal_ids) if proposal_ids is not None else None,
        }
        return prediction, diagnostics
