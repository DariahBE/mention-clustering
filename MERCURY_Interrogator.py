"""Interactive query engine CLI script for MERCURY."""

import argparse
import sys
from MERCURY import MERCURY


def main() -> None:
    """CLI entry point for MERCURY_Interrogator.py."""
    parser = argparse.ArgumentParser(
        description="MERCURY_Interrogator: Interactive query engine for pre-indexed dataset."
    )
    parser.add_argument(
        "searchterm",
        nargs="?",
        type=str,
        default=None,
        help="Initial search term or entity phrase (e.g., 'Alexander de grote').",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Initial search term or entity phrase (alternative flag).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of relevant matching sentences to retrieve.",
    )
    parser.add_argument(
        "--pair",
        type=str,
        default=None,
        help="Dataset/cache pair ID to interrogate.",
    )
    parser.add_argument(
        "--list-pairs",
        action="store_true",
        help="List all registered dataset/cache pairs in cache_catalog.json.",
    )
    parser.add_argument(
        "--cross-encoder-model",
        type=str,
        default=None,
        help="Cross-encoder model for Stage 2 re-ranking (e.g. 'cross-encoder/ms-marco-MiniLM-L-6-v2' or 'None').",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="settings.json",
        help="Path to JSON configuration file.",
    )

    args = parser.parse_args()
    search_term = args.searchterm or args.query

    engine = MERCURY(config_path=args.config)

    print(engine.ascii_greeting)

    if args.list_pairs:
        pairs = engine.list_pairs()
        print("\n=== REGISTERED DATASET/CACHE PAIRS ===")
        for pid, meta in pairs.items():
            active_marker = " (ACTIVE)" if pid == engine.active_pair_id else ""
            print(f"- [{pid}]{active_marker} {meta.get('name', pid)}")
            print(f"    Dataset: {meta.get('dataset_path')}")
            print(f"    Model: {meta.get('model_name')} | Method: {meta.get('clustering_method')}")
            print(f"    Cache: {meta.get('cache_dir')} | Last updated: {meta.get('last_updated')}\n")
        return

    if args.pair:
        engine.select_pair(args.pair)

    if args.cross_encoder_model is not None:
        engine.load_model(cross_encoder_model=args.cross_encoder_model)

    if not engine.load_index():
        sys.exit(1)

    engine.interrogate(initial_query=search_term, top_k=args.top_k)


if __name__ == "__main__":
    main()
