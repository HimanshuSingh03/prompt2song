# Prompt2Song Notebook Pipeline Overview

This document explains how the three notebooks in this project collaborate to turn raw text prompts and song metadata into a working prompt-to-song retrieval system. It also summarizes the key models involved so you have an intuition for what each component does and why it is needed.

---

## Pipeline at a Glance

- **01_text_emotion_encoder.ipynb**  
  Fine-tunes a text classifier (DistilBERT) on emotion-labeled prompts and exports a reusable encoder for turning text into emotion-aware embeddings.

- **02_audio_encoder_and_fusion.ipynb**  
  Uses the text encoder to embed song lyrics, learns an audio model that predicts those embeddings from acoustic features, and blends both modalities into a single fused song representation.

- **03_contrastive_retrieval.ipynb**  
  Learns projection layers that map prompt and song embeddings into a shared space, builds a FAISS similarity index, and exposes helper functions for prompt-to-song search.

Run the notebooks in sequence; each one produces artifacts that the next notebook consumes.

---

## Notebook Walkthrough

### 1. Text Emotion Encoder

**Goal:** Teach a language model to recognize emotions in short text prompts and turn those prompts into dense vectors that capture the emotional signal.

**Key steps**

1. Load the emotion dataset (`datasets/emotions_NLP/*.txt`) and normalize labels.
2. Fine-tune DistilBERT as a 6-class classifier (emotion labels) using Hugging Face `Trainer`.
3. Evaluate the model, then save:
   - Hugging Face model directory (`artifacts/text_encoder/hf_model`)
   - Tokenizer and label mappings
   - Example embeddings of training prompts for quick reuse
4. Provide a `TextEmotionEncoder` helper class that performs mean pooling over DistilBERT outputs to generate embeddings for arbitrary text.

**Outputs consumed later**

- `artifacts/text_encoder/hf_model/` (model + tokenizer)
- `artifacts/text_encoder/label2id.json`
- Optional `train_prompt_embeddings.npy` for debugging downstream components

### 2. Audio Encoder & Gated Fusion

**Goal:** Derive emotion-aware song vectors by blending lyric semantics with acoustic attributes so we can query the catalog even when lyrics are missing or incomplete.

**Key steps**

1. Load the song metadata + acoustic feature CSV (`datasets/song_features/...csv`).
2. Reuse `TextEmotionEncoder` to embed each song's lyrics; cache the results to `np.npy`.
3. Select and scale acoustic features (danceability, energy, tempo, etc.), persisting the scaler params.
4. Train (or outline training for) two neural modules:
   - `AudioEmotionEncoder`: a feed-forward network that predicts lyric embeddings directly from acoustic features.
   - `GatedFusion`: a gating mechanism that learns how to combine lyric and audio embeddings into a unified representation.
5. Export fused song embeddings, gating weights, and a metadata JSON that keeps song IDs, titles, artists, and lyrics aligned with the embeddings.

**Outputs consumed later**

- `artifacts/fusion/lyric_embeddings.npy`
- `artifacts/fusion/fused_song_embeddings.npy`
- `artifacts/fusion/song_metadata.json`
- `artifacts/fusion/audio_feature_*` (scaler stats) for reproducibility

### 3. Contrastive Retrieval & FAISS Indexing

**Goal:** Map prompt embeddings and fused song embeddings into a shared space where cosine similarity reflects emotional and contextual alignment, then index songs for fast nearest-neighbor search.

**Key steps**

1. Load prompt splits, the saved text encoder, and label mapping.
2. Encode prompts into arrays, saving embeddings for analysis.
3. Load fused song embeddings, lyric embeddings, and metadata from Notebook 02.
4. Use the classifier head to estimate emotion distributions for songs, aligning them with prompt labels.
5. Define projection heads (`prompt_projector` and `song_projector`)—small neural networks that learn a contrastive embedding space.
6. Sample matched prompt/song pairs by shared emotion label and train the projectors with an InfoNCE loss.
7. (After training) Save the projectors, project all songs, and build a FAISS inner-product index for efficient retrieval.
8. Provide helper functions to reload the index and `recommend(prompt_text, top_k)` to fetch top-k songs for a new prompt.

**Outputs**

- `artifacts/retrieval/*.pt` (optional projector weights)
- `artifacts/retrieval/projected_song_embeddings.npy`
- `artifacts/retrieval/faiss_song.index`
- Search helper utilities defined in the notebook

---

## Model Cheat Sheet

- **DistilBERT**  
  A compressed version of the BERT transformer that keeps most of BERT’s language understanding ability with ~40% fewer parameters. In Notebook 01 it is fine-tuned to classify emotions in prompts. Internally it produces contextual embeddings for each token. We mean-pool these token vectors to create fixed-length representations for prompts/lyrics.

- **EmotionDataset (PyTorch Dataset)**  
  Wraps text samples and uses the tokenizer to produce padded, truncated token tensors plus label IDs for the Trainer.

- **AudioEmotionEncoder**  
  A fully connected neural network (`nn.Sequential`) that maps acoustic feature vectors (danceability, energy, etc.) into the same latent space as lyric embeddings. It minimizes mean-squared error between its predictions and the lyric encoder embeddings so the model learns a shared emotional representation for songs using only audio data.

- **GatedFusion**  
  A small neural module that concatenates lyric and audio embeddings, runs them through a gating network, and outputs a convex combination of the two. The gate learns when to trust lyrics vs. audio features, producing a richer fused embedding.

- **Contrastive Projection Heads** (`prompt_projector`, `song_projector`)  
  Two identical MLPs that project prompt and song embeddings into a lower-dimensional space where cosine similarity is meaningful. They are trained with contrastive learning (InfoNCE loss) so matching prompt–song pairs cluster together while mismatched pairs repel.

- **FAISS Index**  
  A high-performance similarity search library from Facebook AI. We store L2-normalized projected song embeddings in a FAISS `IndexFlatIP` (inner product) so we can retrieve top-k songs for a given prompt in milliseconds.

---

## Putting It All Together

1. **Train the text encoder** (Notebook 01) and export its artifacts.
2. **Embed songs & build fused representations** (Notebook 02) leveraging both lyrics and acoustic features.
3. **Fit contrastive projection heads & index songs** (Notebook 03) to enable real-time prompt-to-song retrieval.
4. **Serve recommendations** by loading the projection heads, FAISS index, and metadata, then calling the `recommend` helper with any new prompt.

Each notebook is self-contained but depends on artifacts produced earlier in the pipeline. Rerun downstream notebooks whenever upstream artifacts change (e.g., new text encoder weights or updated song metadata).

---

## Tips for Exploration

- Want to inspect embeddings? Load the saved `.npy` files into a Python session and visualize with PCA/UMAP.
- To experiment with new datasets, swap out the CSV/label files but keep artifact directory structure consistent.
- If you retrain the audio or contrastive modules, regenerate the FAISS index to keep retrieval results up to date.

