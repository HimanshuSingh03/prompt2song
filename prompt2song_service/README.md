# Prompt2Song Service

A lightweight package that serves prompt-to-song recommendations locally using the artifacts already produced by the notebooks. It loads the DistilBERT text encoder, lyric embeddings, and song metadata to return the top-k matches for any text prompt, with an optional popularity filter. There is no HTTP server—use the CLI or import it as a library.

## Layout
- `config.yaml` – all paths and runtime settings (paths are resolved relative to this file)
- `pyproject.toml` – dependencies and package metadata
- `src/prompt2song/` – implementation (config loader, encoder, data loaders, recommender, CSV exporter, CLI entrypoints)

## Setup
1. From the repo root: `cd prompt2song_service`
2. Install the package: `pip install -e .`

Set `PROMPT2SONG_CONFIG` if you want the code to read a different config file than the default `config.yaml` in this folder.

## Quickstart (CLI)
Use the built-in helper to write a CSV directly:
```bash
python -m prompt2song.cli "moody late-night pop" --k 10 --filename recommendations.csv
```
Results include title, artist, emotion label, lyrics, score, track popularity, and the feature vector per song (album is omitted from the CSV export). CSVs are written under `paths.output_dir` from `config.yaml`.

## Using as a library
```python
from prompt2song import recommend
songs, csv_path = recommend("energetic road trip", top_k=3, to_csv=True)
for song in songs:
    print(song["name"], song["score"])
print("CSV saved to", csv_path)
```

## Configuration notes
- `paths` should point to the existing notebook artifacts (text encoder, lyric embeddings, metadata)
- `retrieval.top_k` is the default k when none is provided
- `output.csv_filename` sets the default export name; files are written to `paths.output_dir`
- `retrieval.use_popularity_threshold` toggles filtering by the `track_popularity` field in your metadata
- `retrieval.min_track_popularity` is the minimum allowed popularity when the filter is enabled; entries without `track_popularity` are skipped when the filter is on
