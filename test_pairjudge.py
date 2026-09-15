import inspect
import unittest

import numpy as np
from scipy.optimize import minimize

from pairjudge import PairJudge, barycentric_oracle, normalize_region_masks


def make_episode(seed=7, frames=180, query_frames=80, opposite=False):
    rng = np.random.default_rng(seed)
    support_audio = rng.normal(size=(frames, 4))
    query_audio = rng.normal(size=(query_frames, 4))
    signatures = np.array([-1.4, 0.1, 1.3])
    if opposite:
        signatures = -signatures
    basis = rng.normal(size=(3, 6)) * 0.25

    def bank(audio):
        common = np.stack([
            np.sin(audio[:, 1]), np.cos(audio[:, 1]), audio[:, 2],
            audio[:, 3], np.sin(audio[:, 2] * 0.5), np.ones(len(audio)),
        ], axis=1) * 0.08
        return common[:, None, :] + basis[None, :, :]

    def target(audio, proposals):
        logits = audio[:, :1] * signatures[None, :]
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        return np.einsum("tk,tkd->td", weights, proposals)

    support_bank = bank(support_audio)
    query_bank = bank(query_audio)
    return {
        "support_audio": support_audio,
        "query_audio": query_audio,
        "support_bank": support_bank,
        "query_bank": query_bank,
        "support_real": target(support_audio, support_bank),
        "query_real": target(query_audio, query_bank),
    }


def make_judge(seed=3, **overrides):
    config = dict(
        audio_components=4,
        audio_radius=0,
        state_components=6,
        alpha=0.01,
        oracle_entropy=1e-6,
        temporal_sigma=0.0,
        flat_logit_threshold=0.0,
        density_scale=1e6,
        random_state=seed,
    )
    config.update(overrides)
    return PairJudge(**config)


class PairJudgeTest(unittest.TestCase):
    def test_barycentric_solver_matches_independent_slsqp(self):
        proposals = np.array([[[-1.0, 0.5], [0.2, -0.4], [1.5, 0.8]]])
        target = np.array([[0.7, 0.1]])
        masks = np.array([[0.7, 0.3]])
        entropy = 1e-3
        weights, diagnostics = barycentric_oracle(
            proposals, target, masks, entropy=entropy,
        )

        def objective(value):
            error = proposals[0].T @ value - target[0]
            return float(
                np.sum(masks[0] * error * error)
                + entropy * np.sum(value * np.log(value + 1e-15))
            )

        reference = minimize(
            objective,
            np.full(3, 1.0 / 3.0),
            method="SLSQP",
            bounds=[(1e-15, 1.0)] * 3,
            constraints={"type": "eq", "fun": lambda value: value.sum() - 1.0},
            options={"maxiter": 1000, "ftol": 1e-14},
        )
        self.assertTrue(reference.success)
        np.testing.assert_allclose(weights[0, 0], reference.x, atol=2e-6, rtol=2e-6)
        self.assertLess(diagnostics["iterations"], 100)

    def test_recovers_planted_audio_conditioned_mixture(self):
        episode = make_episode()
        masks = np.ones((1, 6))
        judge = make_judge()
        judge.fit(
            episode["support_audio"], episode["support_bank"],
            episode["support_real"], masks,
        )
        prediction, diagnostics = judge.predict(
            episode["query_audio"], episode["query_bank"],
        )
        uniform = episode["query_bank"].mean(axis=1)
        solved = np.mean((prediction - episode["query_real"]) ** 2)
        baseline = np.mean((uniform - episode["query_real"]) ** 2)
        self.assertLess(solved, baseline * 0.2)
        self.assertGreater(diagnostics["accepted_fraction"], 0.99)

    def test_barycentric_target_uses_error_cancellation(self):
        audio = np.linspace(-1.0, 1.0, 60)[:, None]
        proposals = np.zeros((60, 2, 1))
        proposals[:, 0, 0] = -1.0
        proposals[:, 1, 0] = 1.0
        real = np.zeros((60, 1))
        judge = make_judge(audio_components=1, state_components=1)
        judge.fit(audio, proposals, real, np.ones((1, 1)))
        prediction, _ = judge.predict(audio, proposals)
        hard_choice = proposals[:, 0]
        self.assertLess(np.mean(prediction ** 2), 1e-12)
        self.assertGreater(np.mean(hard_choice ** 2), 0.9)

    def test_soft_output_is_smoother_than_hard_switching(self):
        frames = 100
        audio = np.sin(np.linspace(0.0, 8.0, frames))[:, None]
        proposals = np.stack([
            np.sin(np.linspace(0.0, 5.0, frames)),
            np.cos(np.linspace(0.0, 5.0, frames)),
        ], axis=1)[:, :, None]
        real = proposals.mean(axis=1)
        judge = make_judge(audio_components=1, state_components=1, temporal_sigma=1.5)
        judge.fit(audio, proposals, real, np.ones((1, 1)))
        prediction, _ = judge.predict(audio, proposals)
        hard_index = np.argmin((proposals[:, :, 0] - real[:, None, 0]) ** 2, axis=1)
        hard = proposals[np.arange(frames), hard_index, 0]
        soft_jerk = np.mean(np.diff(prediction[:, 0], n=2) ** 2)
        hard_jerk = np.mean(np.diff(hard, n=2) ** 2)
        self.assertLess(soft_jerk, hard_jerk * 0.05)

    def test_shuffled_and_mismatched_support_lose_accuracy(self):
        aligned = make_episode(seed=13)
        judge = make_judge(seed=5)
        judge.fit(
            aligned["support_audio"], aligned["support_bank"],
            aligned["support_real"], np.ones((1, 6)),
        )
        prediction, _ = judge.predict(aligned["query_audio"], aligned["query_bank"])

        permutation = np.random.default_rng(31).permutation(len(aligned["support_real"]))
        shuffled = make_judge(seed=5)
        shuffled.fit(
            aligned["support_audio"], aligned["support_bank"],
            aligned["support_real"][permutation], np.ones((1, 6)),
        )
        shuffled_prediction, _ = shuffled.predict(
            aligned["query_audio"], aligned["query_bank"],
        )

        donor = make_episode(seed=13, opposite=True)
        mismatched = make_judge(seed=5)
        mismatched.fit(
            donor["support_audio"], donor["support_bank"],
            donor["support_real"], np.ones((1, 6)),
        )
        mismatch_prediction, _ = mismatched.predict(
            aligned["query_audio"], aligned["query_bank"],
        )
        target = aligned["query_real"]
        aligned_mse = np.mean((prediction - target) ** 2)
        self.assertLess(aligned_mse, np.mean((shuffled_prediction - target) ** 2) * 0.4)
        self.assertLess(aligned_mse, np.mean((mismatch_prediction - target) ** 2) * 0.4)

    def test_ood_and_flat_logits_fall_back_to_uniform(self):
        episode = make_episode(seed=17)
        judge = make_judge(density_scale=1.2, flat_logit_threshold=0.0)
        judge.fit(
            episode["support_audio"], episode["support_bank"],
            episode["support_real"], np.ones((1, 6)),
        )
        ood_audio = episode["query_audio"] * 100.0
        prediction, diagnostics = judge.predict(ood_audio, episode["query_bank"])
        self.assertAlmostEqual(diagnostics["fallback_fraction"], 1.0)
        np.testing.assert_allclose(prediction, episode["query_bank"].mean(axis=1))

        identical = np.repeat(episode["query_bank"][:, :1], 3, axis=1)
        prediction, diagnostics = judge.predict(episode["query_audio"], identical)
        np.testing.assert_allclose(prediction, identical[:, 0], atol=1e-12)
        np.testing.assert_allclose(diagnostics["weights"], 1.0 / 3.0, atol=1e-12)

    def test_mask_scaling_and_duplicate_split_are_invariant(self):
        episode = make_episode(seed=23)
        masks = np.array([
            [3.0, 2.0, 1.0, 0.2, 0.4, 0.5],
            [0.5, 0.3, 2.0, 2.0, 1.0, 0.1],
        ])

        def solve(value):
            judge = make_judge(seed=9)
            judge.fit(
                episode["support_audio"], episode["support_bank"],
                episode["support_real"], value,
            )
            return judge.predict(episode["query_audio"], episode["query_bank"])[0]

        original = solve(masks)
        per_dim_scaled = solve(masks * np.array([2.0, 4.0, 0.5, 3.0, 7.0, 1.5]))
        split = np.stack([masks[0] / 2.0, masks[0] / 2.0, masks[1]])
        duplicated = solve(split)
        np.testing.assert_allclose(original, per_dim_scaled, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(original, duplicated, atol=1e-8, rtol=1e-8)

    def test_proposal_permutation_equivariance(self):
        episode = make_episode(seed=29)
        permutation = np.array([2, 0, 1])

        first = make_judge(seed=11)
        first.fit(
            episode["support_audio"], episode["support_bank"],
            episode["support_real"], np.ones((1, 6)),
        )
        expected, _ = first.predict(episode["query_audio"], episode["query_bank"])

        second = make_judge(seed=11)
        second.fit(
            episode["support_audio"], episode["support_bank"][:, permutation],
            episode["support_real"], np.ones((1, 6)),
        )
        actual, _ = second.predict(
            episode["query_audio"], episode["query_bank"][:, permutation],
            proposal_ids=("third", "first", "second"),
        )
        np.testing.assert_allclose(expected, actual, atol=1e-8, rtol=1e-8)

    def test_prediction_stays_inside_coordinatewise_convex_hull(self):
        episode = make_episode(seed=37)
        masks = np.array([
            [1.0, 1.0, 0.2, 0.1, 0.5, 0.2],
            [0.1, 0.4, 1.0, 1.0, 0.2, 0.8],
        ])
        judge = make_judge(seed=12)
        judge.fit(
            episode["support_audio"], episode["support_bank"],
            episode["support_real"], masks,
        )
        prediction, _ = judge.predict(episode["query_audio"], episode["query_bank"])
        self.assertTrue(np.all(prediction >= episode["query_bank"].min(axis=1) - 1e-12))
        self.assertTrue(np.all(prediction <= episode["query_bank"].max(axis=1) + 1e-12))

    def test_predict_api_has_no_query_ground_truth(self):
        parameters = inspect.signature(PairJudge.predict).parameters
        self.assertEqual(
            tuple(parameters),
            ("self", "query_audio", "query_proposals", "proposal_ids"),
        )
        normalized = normalize_region_masks(np.ones((2, 3)))
        np.testing.assert_allclose(normalized.blend.sum(axis=0), 1.0)


if __name__ == "__main__":
    unittest.main()
