import argparse
from .service import recommend


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
    recommendations, csv_path = recommend(args.prompt, args.k, to_csv=True, filename=args.filename)
    print(f"Wrote {len(recommendations)} recommendations to {csv_path}")


if __name__ == "__main__":
    main()
