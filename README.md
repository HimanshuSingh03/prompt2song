# Prompt2Song

Text-prompt to song retrieval system with an emotion-aware encoder, cosine-similarity search over lyric embeddings, and an optional RLHF reranker that learns listener preferences in-session. Ships with a CLI, a lightweight browser UI, and utilities to plot evaluation figures.

## Quick start
- Prereqs: Python 3.10+, `pip`, and the provided artifacts under `artifacts/` (HF text encoder, lyric embeddings/metadata). Run inside a virtualenv if you like.
- Install the backend from repo root so the CLI/frontend can import it:
  ```bash
  pip install -e prompt2song_service
  ```
- Launch the browser UI (serves `frontend/index.html` at http://localhost:8000):
  ```bash
  python frontend/server.py
  ```
- Or run the CLI to export a CSV:
  ```bash
  python -m prompt2song.cli "moody late-night pop" --k 15 --filename recs.csv --rlhf-log-dir outputs/rlhf_logs
  ```

## Repository layout
- `prompt2song_service/`: Python package with the prompt encoder, retrieval, RLHF, CLI, and config.
- `frontend/`: Static UI plus `server.py` (small HTTP API + file server that wraps the backend).
- `artifacts/`: Model and inference artifacts (HF emotion encoder; lyric embeddings/metadata; checkpoints).
- `datasets/`: Raw data used for training/eval (`emotions_NLP` splits; Spotify lyric/features CSVs).
- `outputs/`: Default write location for recommendation CSVs, RLHF logs, and evaluation figures.
- `notebooks/`: Experiment and training notebooks (emotion encoder fine-tuning, dataset prep).
- `generate_eval_plots.py`: Script to materialize eval visuals.
- `eval.md`: Notes on metrics, splits, and evaluation protocol.

## How it works
1. **Prompt encoding**: DistilBERT-based classifier fine-tuned on `datasets/emotions_NLP` embeds the prompt (mean pooling over final hidden state). Optional classification output (label + probabilities) is shown in the CLI/frontend.
2. **Phase 1 retrieval**: Cosine similarity between the prompt vector and normalized lyric embeddings (`artifacts/text_encoder/lyrics_retrieval/lyrics_embeddings.npy` + `lyrics_metadata.jsonl`). Optional popularity filter controlled via config.
3. **Phase 2 RLHF reranking (optional)**: User answers A/B questions over the top candidates. A preference vector over 12 scaled audio features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, mode, key, duration_ms) is learned online and blended with retrieval scores to produce the final top-k.
4. **Exports**: Recommendations (Phase 1 and final) are saved as CSVs; RLHF per-question reranks and weight logs are optionally persisted for plotting.

## Configuration
- Main config: `prompt2song_service/config.yaml`. Key sections:
  - `paths`: `text_encoder_dir`, `label_mapping`, `lyrics_embeddings`, `lyrics_metadata`, `output_dir`.
  - `retrieval`: `top_k`, `device`, `normalize_embeddings`, `use_popularity_threshold`, `min_track_popularity`, optional `output_csv`.
  - `rlhf`: `num_rlhf_questions`, `learning_rate`, `preference_weight`, `final_top_k`, optional `output_csv_base`.
  - `output`: default `csv_filename`.
- Override the config path with `PROMPT2SONG_CONFIG=/path/to/config.yaml`.
- Artifacts are referenced relative to the config file; defaults point at the checked-in `artifacts/` and `outputs/`.

## Usage
- **CLI** (`python -m prompt2song.cli ...`)
  - Args: `prompt`, `--k` (results), `--filename` (CSV name), `--rlhf-log-dir` (persist per-question reranks + summaries).
  - If `rlhf.num_rlhf_questions > 0`, the CLI writes both Phase 1 and Phase 2 CSVs. Without RLHF it writes a single CSV.
- **Library**:
  ```python
  from prompt2song import recommend
  songs, csv_path = recommend("energetic road trip", top_k=5, to_csv=True)
  ```
- **Frontend** (`python frontend/server.py`)
  - Loads the backend package (ensure `pip install -e prompt2song_service` was run).
  - POST `/api/start` with `{"prompt": "...", "top_k": 30}` begins a session and serves the first A/B question.
  - POST `/api/answer` with `{"sessionId": "...", "choice": "a"|"b"|"skip"}` records feedback and advances.
  - Streams logs, shows the current A/B pair, and displays the final CSV rows when done. Files save under `paths.output_dir`.

## Models and data
- **Emotion encoder**: DistilBERT sequence classifier trained via Hugging Face `Trainer`; artifacts under `artifacts/text_encoder/hf_model`, with label map in `artifacts/text_encoder/label2id.json`. Training notebook: `notebooks/text_emotion_encoder.ipynb`.
- **Lyric retrieval artifacts**: `artifacts/text_encoder/lyrics_retrieval/lyrics_embeddings.npy` (float32 vectors) and `lyrics_metadata.jsonl` (per-song metadata including lyrics, artists, popularity, audio features).
- **RLHF features**: Extracted from metadata and scaled in `prompt2song_service/src/prompt2song/rlhf.py` before dot-product blending.
- **Datasets**:
  - `datasets/emotions_NLP/{train,val,test}.txt` with 6-class emotion labels (see `eval.md` for split sizes/balance).
  - `datasets/song_features/*.csv` contain Spotify/lyrics attributes used to build the retrieval artifacts.

## Evaluation and plotting
- Generate figures (label balance, training curves, confusion matrix, RLHF rank shifts/trajectory/weights):
  ```bash
  python generate_eval_plots.py \
    --dataset-root datasets/emotions_NLP \
    --model-dir artifacts/text_encoder/hf_model \
    --trainer-state artifacts/text_encoder/checkpoints/checkpoint-3000/trainer_state.json \
    --phase1-csv outputs/recs_retrieval.csv \
    --phase2-csv outputs/recs_rlhf.csv \
    --rlhf-log-dir outputs/rlhf_logs \
    --out-dir outputs/figures
  ```
- `eval.md` documents the metric choices (macro-F1/accuracy), early stopping, and how RLHF outputs are inspected.

## Outputs
- Phase 1 CSV: `retrieval.output_csv` or `<base>_p1.csv` in `paths.output_dir`.
- Final CSV: `rlhf.output_csv_base` or `<base>_p2.csv`.
- RLHF logs (optional): `--rlhf-log-dir` or `outputs/rlhf_logs/session_<id>/` with `rlhf_step_*.csv`, `rlhf_questions.csv`, and `rlhf_weights.csv`.

## Development notes
- Dependencies are declared in `prompt2song_service/pyproject.toml`; `pip install -e prompt2song_service[dev]` adds notebook kernels.
- Set `retrieval.device` in config to `cuda` if you have a GPU and the HF model is GPU-capable.
- If you relocate artifacts, update `config.yaml` or set `PROMPT2SONG_CONFIG` to a copy with the new paths.

