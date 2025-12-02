# Evaluation

This section documents how we evaluate the prompt-to-song system using the artifacts and code in this repository.

## Data splits and label balance
- Emotion classification uses fixed splits from `datasets/emotions_NLP/*.txt`: 16,000 train, 2,000 validation, and 2,000 test examples (`load_split` in `notebooks/text_emotion_encoder.ipynb`). Label counts remain consistent across splits (train: joy 5,362; sadness 4,666; anger 2,159; fear 1,937; love 1,304; surprise 572; validation/test mirror those ratios), so macro-sensitive metrics are needed.
- Retrieval relies on the exported lyric metadata and embeddings (`artifacts/text_encoder/lyrics_retrieval/lyrics_metadata.jsonl` and `.../lyrics_embeddings.npy`) purely for inference; no further split is introduced because downstream ranking lacks ground-truth relevance labels.

## Metrics and validation protocol
- The DistilBERT emotion encoder is trained with Hugging Face `Trainer` using `accuracy`, `f1_macro`, and `f1_weighted` (`compute_metrics` block in `notebooks/text_emotion_encoder.ipynb`). We load the best checkpoint by `f1_macro` and apply early stopping (patience=2, threshold=5e-4) to avoid overfitting on the minority classes.
- Each epoch evaluates on the 2k validation set (`eval_strategy="epoch"`); final numbers are taken on the held-out 2k test set, accompanied by a full `classification_report` and confusion matrix to surface per-class weaknesses.
- Loss curves are logged per epoch (and every 50 steps) to confirm convergence; smoothing is applied only for visualization so raw validation metrics remain exact.

## Retrieval and RLHF assessment
- Phase 1 retrieval scores cosine similarity between mean-pooled prompt embeddings and normalized lyric vectors (`prompt2song_service/src/prompt2song/recommender.py`), with an optional track-popularity filter and a deliberately generous default pool size (`top_k=100` in `config.yaml`) to preserve recall before reranking.
- In the absence of labeled relevance judgments, Phase 2 uses human feedback as the evaluation signal: users answer A/B preference prompts over 12 scaled audio features (`rlhf.py`), producing a session-specific preference vector that reweights the Phase 1 candidates (`rerank_candidates`). We compare Phase 1 vs. Phase 2 CSVs in `outputs/` to verify that human-preferred songs rise in the final top-k.
- Exported scores include the base retrieval similarity, RLHF contribution, and final blended score (`service.py` and `exporter.py`), enabling manual sanity checks of how much the learned preferences shift rankings.

## Adequacy of the chosen metrics
- Macro-F1 drives model selection to keep rare labels (e.g., surprise) from being swamped by majority classes, while accuracy and weighted-F1 offer complementary sanity checks.
- We do not use cross-validation; the fixed train/val/test splits plus early stopping provide a stable estimate of generalization. For retrieval, interactive RLHF acts as the primary quality gate until we acquire labeled relevance data to support offline ranking metrics.
