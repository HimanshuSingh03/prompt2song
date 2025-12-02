import argparse
from .service import classify_prompt, recommend, settings, resolve_output_filenames


def main() -> None:
    parser = argparse.ArgumentParser(description="Export prompt-to-song recommendations to CSV.")
    parser.add_argument("prompt", type=str, help="Text prompt to search with")
    parser.add_argument("--k", type=int, default=None, help="Number of songs to return")
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Optional filename for the CSV (defaults to config output.csv_filename)",
    )
    parser.add_argument(
        "--rlhf-log-dir",
        type=str,
        default=None,
        help="Optional directory to write per-question RLHF rerank CSVs (step_*.csv) and question summaries.",
    )
    args = parser.parse_args()
    phase1_name, phase2_name = resolve_output_filenames(args.filename)

    classification = classify_prompt(args.prompt)
    if classification:
        label = classification.get("label")
        score = classification.get("score")
        probs = classification.get("probabilities") or {}
        if label is not None and score is not None:
            print(f"Prompt classification: {label} ({score:.2f} confidence)")
        elif label:
            print(f"Prompt classification: {label}")
        if probs:
            ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            prob_str = ", ".join(f"{name}: {p:.2f}" for name, p in ranked)
            print(f"Class probabilities: {prob_str}")

    recommendations, csv_path = recommend(
        args.prompt,
        args.k,
        to_csv=True,
        filename=args.filename,
        question_log_dir=args.rlhf_log_dir,
    )
    if settings.rlhf.num_rlhf_questions > 0:
        phase1_path = settings.paths.output_dir / phase1_name
        print(f"Wrote Phase 1 recommendations to {phase1_path}")
    print(f"Wrote final recommendations to {csv_path}")


if __name__ == "__main__":
    main()
