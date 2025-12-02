# Prompt2Song

Local prompt-to-song recommendation pipeline with optional RLHF reranking. Ships with both a CLI and a lightweight browser UI.

## Prerequisites
- Python 3.10+
- Project artifacts available (text encoder, lyric embeddings, metadata). Paths are set in `config.yaml`.
- From repo root, install the backend:  
  ```bash
  pip install -e prompt2song_service
  ```
  (Run inside a virtualenv if desired.)

## Configure
`prompt2song_service/config.yaml` defines everything. Key fields:
- `paths`: `text_encoder_dir`, `label_mapping`, `lyrics_embeddings`, `lyrics_metadata`, `output_dir` (all resolved relative to this file).
- `retrieval`: `top_k`, `device`, `normalize_embeddings`, optional popularity filter (`use_popularity_threshold`, `min_track_popularity`), optional `output_csv` for Phase 1.
- `rlhf`: `num_rlhf_questions`, `learning_rate`, `preference_weight`, `final_top_k`, optional `output_csv_base` for the RLHF CSV.
- `output`: default `csv_filename` fallback.

Override config with `PROMPT2SONG_CONFIG=/path/to/config.yaml` if needed.

## Run: CLI
```bash
python -m prompt2song.cli "moody late-night pop" --k 15 --filename recs.csv --rlhf-log-dir outputs/rlhf_logs
```
Notes:
- Phase 1 CSV: `retrieval.output_csv` or `<base>_p1.csv` under `paths.output_dir`.
- Phase 2 CSV (RLHF): `rlhf.output_csv_base` or `<base>_p2.csv`.
- RLHF behavior: after each answer it reranks and asks about the two highest-ranked unseen songs until questions or fresh songs run out. Preference vector updates: `w += lr * (chosen - other)`.
- Weight logs (for plotting): `outputs/rlhf_logs/session_<id>/rlhf_weights.csv` unless `--rlhf-log-dir` is provided.

## Run: Frontend
```bash
python frontend/server.py
```
Then open http://localhost:8000
- Matches CLI behavior (same config, same outputs).
- Streams logs, shows the current A/B cards (top-2 unseen per step), and displays final CSV rows.
- Saves Phase 1/2 CSVs to `paths.output_dir`; weight logs to `outputs/rlhf_logs/session_<sessionId>/rlhf_weights.csv`.

## Library usage
```python
from prompt2song import recommend
songs, csv_path = recommend("energetic road trip", top_k=5, to_csv=True)
for s in songs:
    print(s["name"], s["score"])
print("CSV saved to", csv_path)
```

## Evaluation plots
Generate figures (label distribution, training curves, confusion matrix, RLHF rank shifts, rank trajectory, weight changes):
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
- If multiple RLHF sessions exist, pick one with `--rlhf-session-id <id>`.
- Outputs include `rlhf_weight_changes.png` (preference-vector values per question).

## Outputs at a glance
- Phase 1: `<base>_p1.csv` (or `retrieval.output_csv`)
- RLHF final: `<base>_p2.csv` (or `rlhf.output_csv_base`)
- Per-step reranks (optional): `--rlhf-log-dir` / `rlhf_step_*.csv`
- RLHF weight log: `outputs/rlhf_logs/session_<id>/rlhf_weights.csv`

## Troubleshooting
- Empty results: verify artifact paths in `config.yaml`.
- RLHF not running: ensure `rlhf.num_rlhf_questions > 0`.
- Frontend import errors: confirm `pip install -e prompt2song_service` was run from repo root.
