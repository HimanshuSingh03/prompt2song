import sys
import random
from typing import Iterable
import numpy as np


# Audio features to use for RLHF preference learning; values are lightly scaled
# so no single feature dominates the dot product.
FEATURE_KEYS = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "mode",
    "key",
    "duration_ms",
]


def extract_audio_features(metadata: dict) -> np.ndarray:
    """Return a scaled numeric feature vector for a song metadata entry.

    Missing or non-numeric features are treated as 0 so the model remains robust
    even if some metadata fields are absent in the artifacts.
    """
    def _to_float(value: object) -> float:
        try:
            if value is None:
                return 0.0
            return float(value)
        except Exception:
            return 0.0

    loudness = _to_float(metadata.get("loudness")) / 60.0  # roughly [-60,0]
    tempo = _to_float(metadata.get("tempo")) / 250.0  # common BPM range
    key = _to_float(metadata.get("key")) / 12.0
    duration = _to_float(metadata.get("duration_ms")) / 300_000.0  # 5 minutes

    vector = np.array(
        [
            _to_float(metadata.get("danceability")),
            _to_float(metadata.get("energy")),
            loudness,
            _to_float(metadata.get("speechiness")),
            _to_float(metadata.get("acousticness")),
            _to_float(metadata.get("instrumentalness")),
            _to_float(metadata.get("liveness")),
            _to_float(metadata.get("valence")),
            tempo,
            _to_float(metadata.get("mode")),
            key,
            duration,
        ],
        dtype=float,
    )
    return np.nan_to_num(vector)


def run_rlhf_session(
    candidates: list[dict],
    feature_vectors: list[np.ndarray],
    num_questions: int,
    learning_rate: float,
) -> np.ndarray:
    """Interactively learn a per-session preference vector from A/B feedback."""
    if num_questions <= 0 or len(candidates) < 2:
        return np.zeros(feature_vectors[0].shape if feature_vectors else (0,), dtype=float)
    if not sys.stdin or not sys.stdin.isatty():
        return np.zeros(feature_vectors[0].shape if feature_vectors else (0,), dtype=float)

    w = np.zeros(feature_vectors[0].shape, dtype=float)
    asked_pairs: set[tuple[int, int]] = set()
    rng = random.Random()
    answered = 0

    max_unique_pairs = len(candidates) * (len(candidates) - 1) // 2
    while answered < num_questions and len(asked_pairs) < max_unique_pairs:
        pair = None
        for _ in range(10):
            a, b = rng.sample(range(len(candidates)), 2)
            if a == b:
                continue
            if (a, b) in asked_pairs or (b, a) in asked_pairs:
                continue
            pair = (a, b)
            asked_pairs.add(pair)
            break
        if pair is None:
            break

        a_idx, b_idx = pair
        cand_a, cand_b = candidates[a_idx], candidates[b_idx]
        try:
            print(f"\nRLHF question {answered + 1}/{num_questions}:")
            print(f"A) \"{cand_a['name']}\" – {cand_a['artists']}")
            print(f"B) \"{cand_b['name']}\" – {cand_b['artists']}")
            choice = input("Which fits your vibe better? [A/B/skip]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            # Gracefully end preference gathering when stdin is unavailable.
            break

        if choice == "a":
            w = w + learning_rate * (feature_vectors[a_idx] - feature_vectors[b_idx])
            answered += 1
        elif choice == "b":
            w = w + learning_rate * (feature_vectors[b_idx] - feature_vectors[a_idx])
            answered += 1
        else:
            # Skipped; do not count toward answered questions.
            continue
    return w


def rerank_candidates(
    candidates: Iterable[dict],
    feature_vectors: list[np.ndarray],
    preference_vector: np.ndarray,
    preference_weight: float,
    final_top_k: int,
) -> list[dict]:
    """Combine base scores with RLHF preference scores and take the final top-K."""
    if final_top_k <= 0:
        return []
    reweighted = []
    for cand, feats in zip(candidates, feature_vectors):
        base_score = cand.get("score", 0.0)
        pref_score = float(preference_vector @ feats) if preference_vector.size else 0.0
        final_score = base_score + preference_weight * pref_score
        merged = dict(cand)
        merged["base_score"] = base_score
        merged["rlhf_score"] = pref_score
        merged["score"] = final_score
        reweighted.append(merged)
    reweighted.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return reweighted[:final_top_k]
