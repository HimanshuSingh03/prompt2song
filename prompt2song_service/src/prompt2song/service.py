import os
import uuid
from pathlib import Path
import numpy as np
import pandas as pd

from .config import Settings
from .encoder import PromptEncoder
from .data import SongMetadata, SongEmbeddings
from .recommender import Prompt2SongRecommender
from .exporter import RecommendationExporter
from .rlhf import (
    FEATURE_KEYS,
    PreferenceVectorLogger,
    extract_audio_features,
    run_rlhf_session,
    rerank_candidates,
)


def load_settings() -> Settings:
    override = os.environ.get("PROMPT2SONG_CONFIG")
    if override:
        return Settings(override)
    return Settings(Path(__file__).resolve().parents[2] / "config.yaml")


settings = load_settings()
encoder = PromptEncoder(settings.paths.text_encoder_dir, settings.retrieval.device)
metadata = SongMetadata(settings.paths.lyrics_metadata)
embeddings = SongEmbeddings(settings.paths.lyrics_embeddings, settings.retrieval.normalize_embeddings)
recommender = Prompt2SongRecommender(
    encoder=encoder,
    embeddings=embeddings,
    metadata=metadata,
    default_k=settings.retrieval.top_k,
    use_popularity_threshold=settings.retrieval.use_popularity_threshold,
    min_track_popularity=settings.retrieval.min_track_popularity,
)
default_csv = settings.output.csv_filename if settings.output else "recommendations.csv"
exporter = RecommendationExporter(settings.paths.output_dir, default_csv)


def classify_prompt(prompt: str) -> dict[str, object] | None:
    """Return the model's top label and probabilities for a prompt, if available."""
    try:
        return encoder.classify(prompt)
    except Exception:
        return None


def phase_filename(base_filename: str, phase_suffix: str) -> str:
    """Return a filename with a phase suffix inserted before the extension."""
    base = Path(base_filename)
    suffix = base.suffix or ".csv"
    return f"{base.stem}{phase_suffix}{suffix}"


def resolve_output_filenames(base_filename: str | None) -> tuple[str, str]:
    """Resolve phase-specific filenames using config overrides or a provided base."""
    base = base_filename or (settings.output.csv_filename if settings.output else "recommendations.csv")
    phase1 = settings.retrieval.output_csv or phase_filename(base, "_p1")
    phase2 = settings.rlhf.output_csv_base or phase_filename(base, "_p2")
    return phase1, phase2


def _save_rlhf_step(path: Path, rankings: list[dict]) -> None:
    """Write a per-question reranked list to CSV (drops heavy columns)."""
    frame = pd.DataFrame(rankings)
    frame = frame.drop(columns=["album_name", "feature_vector"], errors="ignore")
    if "lyrics" in frame.columns:
        cols = [c for c in frame.columns if c != "lyrics"] + ["lyrics"]
        frame = frame[cols]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def recommend(
    prompt: str,
    top_k: int | None = None,
    to_csv: bool = False,
    filename: str | None = None,
    question_log_dir: str | Path | None = None,
    session_id: str | None = None,
):
    rlhf_cfg = settings.rlhf
    phase1_name, phase2_name = resolve_output_filenames(filename)
    log_dir = Path(question_log_dir) if question_log_dir else None
    run_id = session_id or uuid.uuid4().hex
    weight_log_dir = Path(question_log_dir) if question_log_dir else settings.paths.output_dir / "rlhf_logs" / f"session_{run_id}"
    weight_logger: PreferenceVectorLogger | None = None

    if rlhf_cfg.num_rlhf_questions > 0:
        # Phase 1 pool size follows the requested/Phase 1 top_k; final list size follows RLHF final_top_k unless CLI overrides.
        pool_k = top_k or settings.retrieval.top_k
        final_k = top_k if top_k is not None else (rlhf_cfg.final_top_k or settings.retrieval.top_k)
        candidates = recommender.recommend(prompt, pool_k)
        phase1_csv_path = None
        if to_csv:
            phase1_csv_path = exporter.to_csv(candidates, filename=phase1_name)
        feature_vectors = [extract_audio_features(c) for c in candidates]
        if feature_vectors:
            weight_logger = PreferenceVectorLogger(weight_log_dir / "rlhf_weights.csv", session_id=run_id, feature_keys=FEATURE_KEYS)
        track_history = log_dir is not None
        result = run_rlhf_session(
            candidates,
            feature_vectors,
            num_questions=rlhf_cfg.num_rlhf_questions,
            learning_rate=rlhf_cfg.learning_rate,
            track_history=track_history,
            weight_logger=weight_logger,
            preference_weight=rlhf_cfg.preference_weight,
        )
        if track_history:
            preference_vector, question_history = result
        else:
            preference_vector = result  # type: ignore[assignment]
            question_history = []
        recommendations = rerank_candidates(
            candidates,
            feature_vectors,
            preference_vector=preference_vector,
            preference_weight=rlhf_cfg.preference_weight,
            final_top_k=final_k,
        )
        target_filename = phase2_name

        if log_dir is not None:
            zero_vector = np.zeros_like(feature_vectors[0]) if feature_vectors else np.zeros(0, dtype=float)
            baseline = rerank_candidates(
                candidates,
                feature_vectors,
                preference_vector=zero_vector,
                preference_weight=0.0,
                final_top_k=final_k,
            )
            _save_rlhf_step(log_dir / "rlhf_step_0.csv", baseline)
            for entry in question_history:
                w = entry.get("preference_vector")
                if w is None:
                    continue
                step_rankings = rerank_candidates(
                    candidates,
                    feature_vectors,
                    preference_vector=w,
                    preference_weight=rlhf_cfg.preference_weight,
                    final_top_k=final_k,
                )
                _save_rlhf_step(log_dir / f"rlhf_step_{entry.get('question', len(question_history))}.csv", step_rankings)
            # Also persist question summaries for reference.
            if question_history:
                log_dir.mkdir(parents=True, exist_ok=True)
                summary_frame = pd.DataFrame(question_history)
                summary_frame = summary_frame.drop(columns=["preference_vector"], errors="ignore")
                summary_frame.to_csv(log_dir / "rlhf_questions.csv", index=False)
    else:
        final_k = top_k or settings.retrieval.top_k
        recommendations = recommender.recommend(prompt, final_k)
        phase1_csv_path = None
        target_filename = phase1_name

    csv_path = None
    if to_csv:
        csv_path = exporter.to_csv(recommendations, filename=target_filename)
    return recommendations, csv_path
