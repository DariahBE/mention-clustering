"""Offline index generator CLI script for MERCURY."""

import argparse
import sys
from MERCURY import MERCURY


def main() -> None:
    """CLI entry point for MERCURY_Generator.py."""
    parser = argparse.ArgumentParser(
        description="MERCURY_Generator: Generate dual vector embeddings and FAISS index."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="settings.json",
        help="Path to JSON configuration file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to CSV dataset file.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Transformer model name (e.g., 'sentence-transformers/LaBSE').",
    )
    parser.add_argument(
        "--clustering-method",
        type=str,
        choices=["cugraph_leiden", "leiden", "agglomerative", "faiss_topk", "faiss_leader", "faiss_kmeans"],
        default="cugraph_leiden",
        help="Clustering algorithm ('cugraph_leiden', 'faiss_topk', 'faiss_leader', 'faiss_kmeans', or 'agglomerative').",
    )
    parser.add_argument(
        "--pair",
        type=str,
        default=None,
        help="Dataset/cache pair ID.",
    )
    parser.add_argument(
        "--list-pairs",
        action="store_true",
        help="List all registered dataset/cache pairs in cache_catalog.json.",
    )
    parser.add_argument(
        "--create-pair",
        action="store_true",
        help="Create a new dataset/cache pair entry.",
    )
    parser.add_argument(
        "--force-update",
        action="store_true",
        help="Force re-indexing of an existing dataset/cache pair.",
    )
    parser.add_argument(
        "--delete-pair",
        action="store_true",
        help="Delete a dataset/cache pair and remove its cache directory.",
    )
    parser.add_argument(
        "--column-mapping",
        type=str,
        default=None,
        help="JSON string mapping raw dataset column names to internal standard schema.",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=None,
        help="Dataset delimiter (e.g. ',' or '\\t'). Auto-detected if None.",
    )
    parser.add_argument(
        "--map-only",
        action="store_true",
        help="Run Stage 1 dataset loading, auto-detection, and column mapping only.",
    )
    parser.add_argument(
        "--alpha-span-weight",
        type=float,
        default=None,
        help="Weight for entity mention vector vs sentence context vector (0.0 to 1.0).",
    )
    parser.add_argument(
        "--notebook",
        action="store_true",
        help="Launch interactive ipywidgets generator UI in Jupyter Notebooks.",
    )
    args = parser.parse_args()

    import json
    col_mapping = json.loads(args.column_mapping) if args.column_mapping else None

    engine = MERCURY(config_path=args.config)

    print(engine.ascii_greeting)

    if args.notebook:
        engine.get_generator_notebook_ui()
        return

    if args.map_only:
        dataset_path = args.dataset or engine.config.get("dataset_path", "data/prepared_data.tsv")
        df_mapped, mapping = engine.map_dataset(dataset_path, column_mapping=col_mapping, delimiter=args.delimiter)
        print("\n=== STAGE 1 MAPPING RESULTS ===")
        for k, v in mapping.items():
            print(f"  '{k}' -> '{v}'")
        print(f"\nMapped dataset shape: {df_mapped.shape}")
        return

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

    target_pair = args.pair or engine.active_pair_id

    if args.delete_pair:
        if not args.pair:
            print("[ERROR] Please specify --pair <pair_id> to delete.")
            sys.exit(1)
        engine.delete_pair(args.pair)
        return

    if args.create_pair:
        if not args.pair or not args.dataset:
            print("[ERROR] Please specify both --pair <pair_id> and --dataset <path> to create a pair.")
            sys.exit(1)
        engine.create_pair(
            pair_id=args.pair,
            dataset_path=args.dataset,
            model_name=args.model_name,
            clustering_method=args.clustering_method,
            force_update=True,
        )
        return

    if args.force_update:
        if not target_pair:
            print("[ERROR] No pair specified for update.")
            sys.exit(1)
        engine.update_pair(target_pair, force_update=True)
        return

    if args.pair:
        engine.select_pair(args.pair)

    try:
        engine.generate_indexes(
            dataset_path=args.dataset,
            clustering_method=args.clustering_method,
            model_name=args.model_name,
            alpha_span_weight=args.alpha_span_weight,
        )

    except FileNotFoundError as e:
        print(f"[WARN] {e}")
        print("[INFO] MERCURY ready. Place dataset CSV in data directory to run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
