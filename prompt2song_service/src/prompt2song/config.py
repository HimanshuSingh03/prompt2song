from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class PathConfig:
    text_encoder_dir: Path
    label_mapping: Path
    lyrics_embeddings: Path
    lyrics_metadata: Path
    output_dir: Path


@dataclass
class RetrievalConfig:
    top_k: int
    device: str
    normalize_embeddings: bool
    use_popularity_threshold: bool
    min_track_popularity: int


@dataclass
class OutputConfig:
    csv_filename: str


class Settings:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        raw = yaml.safe_load(self.config_path.read_text())
        base = self.config_path.parent
        self.paths = PathConfig(
            text_encoder_dir=(base / raw["paths"]["text_encoder_dir"]).resolve(),
            label_mapping=(base / raw["paths"]["label_mapping"]).resolve(),
            lyrics_embeddings=(base / raw["paths"]["lyrics_embeddings"]).resolve(),
            lyrics_metadata=(base / raw["paths"]["lyrics_metadata"]).resolve(),
            output_dir=(base / raw["paths"]["output_dir"]).resolve(),
        )
        retrieval = raw["retrieval"]
        self.retrieval = RetrievalConfig(
            top_k=retrieval["top_k"],
            device=retrieval["device"],
            normalize_embeddings=retrieval["normalize_embeddings"],
            use_popularity_threshold=retrieval["use_popularity_threshold"],
            min_track_popularity=retrieval["min_track_popularity"],
        )
        output = raw["output"]
        self.output = OutputConfig(csv_filename=output["csv_filename"])
