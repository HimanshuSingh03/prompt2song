"""
Utility script to materialize evaluation figures for the prompt-to-song project.

Outputs (saved under --out-dir, default: outputs/figures):
- label_distribution.png: class balance across train/val/test splits.
- training_curves.png: train loss and validation metrics from Trainer log history.
- confusion_matrix.png: test-set confusion matrix from the fine-tuned classifier.
- rlhf_rank_shifts.png: rank movement after RLHF reranking (if Phase 1/2 CSVs exist).
- rlhf_rank_trajectory.png: rank trajectory per RLHF question using per-question logs (if available).
- rlhf_weight_changes.png: preference-vector weights over RLHF questions (if logs exist).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def plot_label_distribution(dataset_root: Path, out_dir: Path) -> Path:
    splits: Dict[str, pd.Series] = {}
    for split in ["train", "val", "test"]:
        df = pd.read_csv(dataset_root / f"{split}.txt", sep=";", names=["text", "label"], encoding="utf-8")
        splits[split] = df["label"].value_counts().sort_index()

    labels = sorted({label for series in splits.values() for label in series.index})
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, split in enumerate(["train", "val", "test"]):
        counts = [splits[split].get(label, 0) for label in labels]
        ax.bar(x + i * width, counts, width, label=split)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_ylabel("Count")
    ax.set_title("Emotion label distribution by split")
    ax.legend()
    out_path = out_dir / "label_distribution.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _load_trainer_history(trainer_state_path: Path) -> Tuple[List[Tuple[int, float]], List[Tuple[float, float, float]]]:
    state = json.loads(trainer_state_path.read_text())
    history = state.get("log_history", [])
    train_loss = []
    eval_metrics = []
    for entry in history:
        if "loss" in entry and "eval_loss" not in entry:
            step = entry.get("step") or entry.get("global_step")
            train_loss.append((int(step), float(entry["loss"])))
        if "eval_f1_macro" in entry:
            eval_metrics.append(
                (
                    float(entry.get("epoch", 0.0)),
                    float(entry["eval_f1_macro"]),
                    float(entry.get("eval_accuracy", np.nan)),
                )
            )
    return train_loss, eval_metrics


def plot_training_curves(trainer_state_path: Path, out_dir: Path) -> Path:
    train_loss, eval_metrics = _load_trainer_history(trainer_state_path)
    if not train_loss and not eval_metrics:
        raise RuntimeError(f"No log history found in {trainer_state_path}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    if train_loss:
        steps, losses = zip(*train_loss)
        axes[0].plot(steps, losses, label="Train loss")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training loss over steps")
        axes[0].grid(alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No train loss found", ha="center", va="center")

    if eval_metrics:
        epochs, f1_macro, acc = zip(*eval_metrics)
        axes[1].plot(epochs, f1_macro, marker="o", label="Val macro-F1")
        axes[1].plot(epochs, acc, marker="s", label="Val accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].set_ylim(0, 1)
        axes[1].set_title("Validation metrics")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No eval metrics found", ha="center", va="center")

    out_path = out_dir / "training_curves.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_confusion(model_dir: Path, dataset_root: Path, split: str, out_dir: Path) -> Path | None:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
    except ImportError:
        print("transformers/torch not installed; skipping confusion matrix.")
        return None

    df = pd.read_csv(dataset_root / f"{split}.txt", sep=";", names=["text", "label"], encoding="utf-8")
    labels = sorted(df["label"].unique())
    label2id = {label: idx for idx, label in enumerate(labels)}
    y_true = np.array([label2id[lbl] for lbl in df["label"]], dtype=int)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval()

    preds: List[int] = []
    batch_size = 32
    for i in range(0, len(df), batch_size):
        batch = df["text"].iloc[i : i + batch_size].tolist()
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        preds.extend(logits.argmax(dim=-1).tolist())

    cm = confusion_matrix(y_true, preds, labels=list(range(len(labels))))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{split.capitalize()} confusion matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, shrink=0.8)

    out_path = out_dir / f"{split}_confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_rlhf_rank_shifts(phase1_csv: Path, phase2_csv: Path, out_dir: Path) -> Path | None:
    if not phase1_csv.exists() or not phase2_csv.exists():
        print("RLHF CSVs missing; skipping rank shift plot.")
        return None

    p1 = pd.read_csv(phase1_csv)
    p2 = pd.read_csv(phase2_csv)

    def make_key(frame: pd.DataFrame) -> pd.Series:
        return frame["name"].astype(str) + " – " + frame["artists"].astype(str)

    p1 = p1.reset_index().rename(columns={"index": "rank_phase1"})
    p2 = p2.reset_index().rename(columns={"index": "rank_phase2"})
    p1["rank_phase1"] += 1
    p2["rank_phase2"] += 1
    p1["key"] = make_key(p1)
    p2["key"] = make_key(p2)

    merged = p1.merge(p2[["key", "rank_phase2"]], on="key", how="inner")
    if merged.empty:
        print("No overlapping songs between Phase 1 and Phase 2; skipping rank shift plot.")
        return None

    merged["delta"] = merged["rank_phase1"] - merged["rank_phase2"]
    merged = merged.sort_values("delta", ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(merged["key"], merged["delta"])
    ax.set_xlabel("Rank improvement (+ means moved up)")
    ax.set_title("Top rank gains after RLHF reranking (Phase 1 vs Phase 2)")
    ax.invert_yaxis()

    out_path = out_dir / "rlhf_rank_shifts.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_rlhf_trajectory(log_dir: Path, out_dir: Path, top_n: int = 10) -> Path | None:
    all_step_files = sorted(log_dir.rglob("rlhf_step_*.csv")) if log_dir.exists() else []
    if not all_step_files:
        print("No per-question RLHF logs found; skipping trajectory plot.")
        return None

    by_parent: dict[Path, list[Path]] = {}
    for f in all_step_files:
        by_parent.setdefault(f.parent, []).append(f)
    # Prefer the most recent session directory.
    step_files: list[Path] = []
    if by_parent:
        parent, files = max(
            by_parent.items(),
            key=lambda item: max(p.stat().st_mtime for p in item[1]),
        )
        step_files = sorted(files, key=lambda p: int(p.stem.split("_")[-1]))
    if not step_files:
        print("No per-question RLHF logs found; skipping trajectory plot.")
        return None

    step_indices = [int(f.stem.split("_")[-1]) for f in step_files]
    frames = []
    for f in step_files:
        df = pd.read_csv(f)
        df["key"] = df["name"].astype(str) + " – " + df["artists"].astype(str)
        frames.append(df)

    focus_keys = frames[-1].sort_values("score", ascending=False).head(top_n)["key"].tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    for key in focus_keys:
        ranks = []
        for df in frames:
            rank_map = {k: idx + 1 for idx, k in enumerate(df["key"])}
            ranks.append(rank_map.get(key, np.nan))
        ax.plot(step_indices, ranks, marker="o", label=key)

    ax.set_xlabel("RLHF step (0 = baseline)")
    ax.set_ylabel("Rank (lower is better)")
    ax.set_title("Ranking trajectory per RLHF question")
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    out_path = out_dir / "rlhf_rank_trajectory.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _load_weight_log(log_root: Path, session_id: str | None) -> tuple[pd.DataFrame | None, Path | None]:
    if log_root.is_file():
        candidates = [log_root]
    elif log_root.exists():
        candidates = sorted(log_root.rglob("rlhf_weights.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        candidates = []

    if not candidates:
        print("No RLHF weight logs found; skipping weight change plot.")
        return None, None

    for path in candidates:
        df = pd.read_csv(path)
        if "question" not in df.columns:
            continue
        if session_id and "session_id" in df.columns:
            df = df[df["session_id"] == session_id]
        elif "session_id" in df.columns and df["session_id"].nunique() > 1:
            latest_sid = df.iloc[-1]["session_id"]
            df = df[df["session_id"] == latest_sid]
        df = df.sort_values("question")
        if not df.empty:
            return df, path

    print("No RLHF weight entries matched the requested session; skipping weight change plot.")
    return None, None


def plot_rlhf_weight_changes(log_dir: Path, out_dir: Path, session_id: str | None = None) -> Path | None:
    df, src_path = _load_weight_log(log_dir, session_id)
    if df is None or src_path is None:
        return None

    feature_cols = [c for c in df.columns if c not in {"session_id", "question"}]
    if not feature_cols:
        print("RLHF weight log missing feature columns; skipping weight change plot.")
        return None

    questions = df["question"].astype(int)
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in feature_cols:
        ax.plot(questions, df[col], marker="o", label=col)

    title = "Preference vector over RLHF questions"
    if session_id:
        title += f" (session {session_id})"
    elif "session_id" in df.columns and not df["session_id"].empty:
        title += f" (session {df['session_id'].iloc[0]})"
    ax.set_title(title)
    ax.set_xlabel("RLHF question (0 = baseline)")
    ax.set_ylabel("Weight value")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    out_path = out_dir / "rlhf_weight_changes.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation figures.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/emotions_NLP"))
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/text_encoder/hf_model"))
    parser.add_argument(
        "--trainer-state",
        type=Path,
        default=Path("artifacts/text_encoder/checkpoints/checkpoint-3000/trainer_state.json"),
    )
    parser.add_argument("--phase1-csv", type=Path, default=Path("outputs/recs_retrieval.csv"))
    parser.add_argument("--phase2-csv", type=Path, default=Path("outputs/recs_rlhf.csv"))
    parser.add_argument("--rlhf-log-dir", type=Path, default=Path("outputs/rlhf_logs"))
    parser.add_argument(
        "--rlhf-session-id",
        type=str,
        default=None,
        help="Optional session_id to select when multiple RLHF weight logs exist.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Saving label distribution...")
    ld_path = plot_label_distribution(args.dataset_root, args.out_dir)
    print(f"  - {ld_path}")

    print("Saving training curves...")
    tc_path = plot_training_curves(args.trainer_state, args.out_dir)
    print(f"  - {tc_path}")

    print("Saving confusion matrix...")
    cm_path = plot_confusion(args.model_dir, args.dataset_root, "test", args.out_dir)
    if cm_path:
        print(f"  - {cm_path}")

    print("Saving RLHF rank shifts (if CSVs exist)...")
    rs_path = plot_rlhf_rank_shifts(args.phase1_csv, args.phase2_csv, args.out_dir)
    if rs_path:
        print(f"  - {rs_path}")

    print("Saving RLHF rank trajectory (per question, if logs exist)...")
    rt_path = plot_rlhf_trajectory(args.rlhf_log_dir, args.out_dir)
    if rt_path:
        print(f"  - {rt_path}")

    print("Saving RLHF weight changes (if logs exist)...")
    rw_path = plot_rlhf_weight_changes(args.rlhf_log_dir, args.out_dir, session_id=args.rlhf_session_id)
    if rw_path:
        print(f"  - {rw_path}")


if __name__ == "__main__":
    main()
