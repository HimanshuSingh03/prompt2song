import numpy as np
from .encoder import PromptEncoder
from .data import SongEmbeddings, SongMetadata


class Prompt2SongRecommender:
    def __init__(
        self,
        encoder: PromptEncoder,
        embeddings: SongEmbeddings,
        metadata: SongMetadata,
        default_k: int,
        use_popularity_threshold: bool,
        min_track_popularity: int,
    ):
        self.encoder = encoder
        self.embeddings = embeddings
        self.metadata = metadata
        self.default_k = default_k
        self.use_popularity_threshold = use_popularity_threshold
        self.min_track_popularity = min_track_popularity

    def recommend(self, prompt: str, top_k: int | None = None) -> list[dict]:
        vector = self.encoder.embed(prompt)
        query = vector / np.linalg.norm(vector)
        scores = query @ self.embeddings.normalized.T
        k = top_k or self.default_k
        sorted_indices = np.argsort(scores)[::-1]
        indices = []
        for idx in sorted_indices:
            if self.use_popularity_threshold:
                pop = self.metadata.popularity[int(idx)]
                if pop is None or pop < self.min_track_popularity:
                    continue
            indices.append(idx)
            if len(indices) >= k:
                break
        recommendations = []
        for idx in indices:
            meta = self.metadata.by_index(int(idx))
            recommendations.append(
                {
                    "song_id": meta["song_id"],
                    "name": meta["name"],
                    "album_name": meta["album_name"],
                    "artists": meta["artists"],
                    "emotion_label": meta.get("emotion_label"),
                    "lyrics": meta["lyrics"],
                    "track_popularity": meta.get("track_popularity"),
                    "feature_vector": self.embeddings.vectors[idx].tolist(),
                    "score": float(scores[idx]),
                }
            )
        return recommendations
