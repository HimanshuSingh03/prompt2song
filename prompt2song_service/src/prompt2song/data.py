import json
from pathlib import Path
import numpy as np


class SongMetadata:
    def __init__(self, metadata_path: Path):
        self.metadata_path = Path(metadata_path)
        with self.metadata_path.open() as f:
            self.entries = [json.loads(line) for line in f]
        self.popularity = [entry.get("track_popularity") for entry in self.entries]

    def __len__(self) -> int:
        return len(self.entries)

    def by_index(self, index: int) -> dict:
        return self.entries[index]


class SongEmbeddings:
    def __init__(self, embedding_path: Path, normalize: bool):
        self.embedding_path = Path(embedding_path)
        self.vectors = np.load(self.embedding_path)
        self.normalized = self._normalize(self.vectors) if normalize else self.vectors

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms
