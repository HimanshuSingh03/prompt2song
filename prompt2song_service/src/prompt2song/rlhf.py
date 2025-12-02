import csv
import sys
import random
from pathlib import Path
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


class PreferenceVectorLogger:
    """Append preference vector values to a CSV per RLHF question."""

    def __init__(self, path: Path | str, session_id: str | None = None, feature_keys: list[str] | None = None):
        self.path = Path(path)
        self.session_id = session_id or ""
        self.feature_keys = feature_keys or FEATURE_KEYS
        self.fieldnames = ["session_id", "question"] + list(self.feature_keys)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, question_idx: int, weights: np.ndarray) -> None:
        row: dict[str, object] = {"session_id": self.session_id, "question": question_idx}
        for i, key in enumerate(self.feature_keys):
            value = float(weights[i]) if i < len(weights) else 0.0
            row[key] = value
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


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
    track_history: bool = False,
    weight_logger: PreferenceVectorLogger | None = None,
) -> np.ndarray | tuple[np.ndarray, list[dict]]:
    """Interactively learn a per-session preference vector from A/B feedback.

    When track_history is True, also return a per-question history containing the
    updated preference vector and the songs compared for each answered question.
    When a weight_logger is provided, the preference vector is persisted after each
    answered question (question index 0 is logged as the baseline).
    """
    if num_questions <= 0 or len(candidates) < 2:
        empty = np.zeros(feature_vectors[0].shape if feature_vectors else (0,), dtype=float)
        return (empty, []) if track_history else empty
    if not sys.stdin or not sys.stdin.isatty():
        empty = np.zeros(feature_vectors[0].shape if feature_vectors else (0,), dtype=float)
        return (empty, []) if track_history else empty

    w = np.zeros(feature_vectors[0].shape, dtype=float)
    if weight_logger:
        weight_logger.log(0, w)
    asked_pairs: set[tuple[int, int]] = set()
    rng = random.Random()
    answered = 0
    history: list[dict] = []

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
            if weight_logger:
                weight_logger.log(answered, w)
            if track_history:
                history.append(
                    {
                        "question": answered,
                        "choice": "a",
                        "song_a": cand_a.get("name"),
                        "artists_a": cand_a.get("artists"),
                        "score_a": cand_a.get("score"),
                        "song_b": cand_b.get("name"),
                        "artists_b": cand_b.get("artists"),
                        "score_b": cand_b.get("score"),
                        "preference_vector": w.copy(),
                    }
                )
        elif choice == "b":
            w = w + learning_rate * (feature_vectors[b_idx] - feature_vectors[a_idx])
            answered += 1
            if weight_logger:
                weight_logger.log(answered, w)
            if track_history:
                history.append(
                    {
                        "question": answered,
                        "choice": "b",
                        "song_a": cand_a.get("name"),
                        "artists_a": cand_a.get("artists"),
                        "score_a": cand_a.get("score"),
                        "song_b": cand_b.get("name"),
                        "artists_b": cand_b.get("artists"),
                        "score_b": cand_b.get("score"),
                        "preference_vector": w.copy(),
                    }
                )
        else:
            # Skipped; do not count toward answered questions.
            continue
    return (w, history) if track_history else w


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
