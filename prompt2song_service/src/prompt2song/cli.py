import argparse
from .service import recommend, settings, resolve_output_filenames


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
    args = parser.parse_args()
    phase1_name, phase2_name = resolve_output_filenames(args.filename)
    recommendations, csv_path = recommend(args.prompt, args.k, to_csv=True, filename=args.filename)
    if settings.rlhf.num_rlhf_questions > 0:
        phase1_path = settings.paths.output_dir / phase1_name
        print(f"Wrote Phase 1 recommendations to {phase1_path}")
    print(f"Wrote final recommendations to {csv_path}")


if __name__ == "__main__":
    main()
