import os
from pathlib import Path
from .config import Settings
from .encoder import PromptEncoder
from .data import SongMetadata, SongEmbeddings
from .recommender import Prompt2SongRecommender
from .exporter import RecommendationExporter


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
exporter = RecommendationExporter(settings.paths.output_dir, settings.output.csv_filename)


def recommend(prompt: str, top_k: int | None = None, to_csv: bool = False, filename: str | None = None):
    recommendations = recommender.recommend(prompt, top_k)
    csv_path = None
    if to_csv:
        csv_path = exporter.to_csv(recommendations, filename=filename)
    return recommendations, csv_path
