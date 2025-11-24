import os
from pathlib import Path
from .config import Settings
from .encoder import PromptEncoder
from .data import SongMetadata, SongEmbeddings
from .recommender import Prompt2SongRecommender
from .exporter import RecommendationExporter
from .rlhf import extract_audio_features, run_rlhf_session, rerank_candidates


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


def recommend(prompt: str, top_k: int | None = None, to_csv: bool = False, filename: str | None = None):
    rlhf_cfg = settings.rlhf
    phase1_name, phase2_name = resolve_output_filenames(filename)

    if rlhf_cfg.num_rlhf_questions > 0:
        # Phase 1 pool size follows the requested/Phase 1 top_k; final list size follows RLHF final_top_k unless CLI overrides.
        pool_k = top_k or settings.retrieval.top_k
        final_k = top_k if top_k is not None else (rlhf_cfg.final_top_k or settings.retrieval.top_k)
        candidates = recommender.recommend(prompt, pool_k)
        phase1_csv_path = None
        if to_csv:
            phase1_csv_path = exporter.to_csv(candidates, filename=phase1_name)
        feature_vectors = [extract_audio_features(c) for c in candidates]
        preference_vector = run_rlhf_session(
            candidates,
            feature_vectors,
            num_questions=rlhf_cfg.num_rlhf_questions,
            learning_rate=rlhf_cfg.learning_rate,
        )
        recommendations = rerank_candidates(
            candidates,
            feature_vectors,
            preference_vector=preference_vector,
            preference_weight=rlhf_cfg.preference_weight,
            final_top_k=final_k,
        )
        target_filename = phase2_name
    else:
        final_k = top_k or settings.retrieval.top_k
        recommendations = recommender.recommend(prompt, final_k)
        phase1_csv_path = None
        target_filename = phase1_name

    csv_path = None
    if to_csv:
        csv_path = exporter.to_csv(recommendations, filename=target_filename)
    return recommendations, csv_path
