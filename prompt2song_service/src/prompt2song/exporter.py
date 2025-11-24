from pathlib import Path
import pandas as pd


class RecommendationExporter:
    def __init__(self, output_dir: Path, default_filename: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.default_filename = default_filename

    def to_csv(self, recommendations: list[dict], filename: str | None = None) -> Path:
        target = self.output_dir / (filename or self.default_filename)
        frame = pd.DataFrame(recommendations)
        # Drop album data for slimmer exports.
        frame = frame.drop(columns=["album_name"], errors="ignore")
        if "lyrics" in frame.columns:
            cols = [c for c in frame.columns if c != "lyrics"] + ["lyrics"]
            frame = frame[cols]
        frame.to_csv(target, index=False)
        return target
