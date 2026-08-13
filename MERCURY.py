"""Core MERCURY utility engine for multilingual entity mention clustering and retrieval.

This module encapsulates all GPU management, transformer vector encoding, dual-encoder span
pooling, FAISS indexing, mutual k-NN graph community clustering, and two-stage query retrieval.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from utilities import gpu_manager

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch


class MERCURY:
    """Unified MERCURY engine for minimum-code exposure in Jupyter Notebooks and CLI scripts."""

    def __init__(self, config_path: str = "settings.json") -> None:
        """Initialize the MERCURY engine and load GPU settings.

        Args:
            config_path: Path to JSON configuration file.
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)

        # Setup GPU
        config = self.config
        max_gpus = config.get("max_gpus", 1)
        gpu_manager.pick_gpu(1, 'auto', int(max_gpus))
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        devices = gpu_manager.pick_gpu(mode = 'report', verbosity=0)
        if not bool(devices): 
            device = 'cpu'
        else:
            device = 'cuda'

        print(f"Using {device}")
        self.device = torch.device(device)

        self.model_name = self.config.get(
            "model_name", "sentence-transformers/LaBSE"
        )
        self.batch_size = int(self.config.get("batch_size", 32))
        self.cache_dir = Path(self.config.get("index_cache_dir", "cache"))
        self.similarity_threshold = float(self.config.get("similarity_threshold", 0.70))
        self.min_score = float(self.config.get("min_similarity_score", 0.50))
        self.alpha_span_weight = float(self.config.get("alpha_span_weight", 0.60))
        self.top_k = int(self.config.get("top_k", 10))
        self.reduced_dimensions = self.config.get("reduced_dimensions", 128)
        self.available_models = self.config.get("available_models", [self.model_name])
        self.cross_encoder_model = self.config.get("cross_encoder_model", None)
        self.available_cross_encoder_models = self.config.get(
            "available_cross_encoder_models",
            ["None", "cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-mMiniLM-L-6-v2", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"],
        )

        self.model = None
        self.cross_encoder = None
        self.pca_model = None
        self.df: Optional[pd.DataFrame] = None
        self.combined_embeddings: Optional[np.ndarray] = None
        self.mention_embeddings: Optional[np.ndarray] = None
        self.context_embeddings: Optional[np.ndarray] = None
        self.faiss_index = None
        self.is_faiss_available = False
        self.cluster_diagnostics: Optional[dict] = None

        #ASCII art: 
        self.ascii_greeting = """
                ##     ## ######## ########   ######  ##     ## ########  ##    ## 
                ###   ### ##       ##     ## ##    ## ##     ## ##     ##  ##  ##  
                #### #### ##       ##     ## ##       ##     ## ##     ##   ####   
                ## ### ## ######   ########  ##       ##     ## ########     ##    
                ##     ## ##       ##   ##   ##       ##     ## ##   ##      ##    
                ##     ## ##       ##    ##  ##    ## ##     ## ##    ##     ##    
                ##     ## ######## ##     ##  ######   #######  ##     ##    ##    
        """

        # Initialize catalog and active pair
        self.active_pair_id = "default"
        catalog = self._load_catalog()
        active_id = catalog.get("active_pair", "default")
        pairs = catalog.get("pairs", {})
        if active_id in pairs:
            self.active_pair_id = active_id
            meta = pairs[active_id]
            self.cache_dir = Path(meta.get("cache_dir", f"cache/{active_id}"))
            if "dataset_path" in meta:
                self.config["dataset_path"] = meta["dataset_path"]
            if "model_name" in meta:
                self.model_name = meta["model_name"]
        else:
            self.cache_dir = Path(self.config.get("index_cache_dir", "cache/default"))

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _catalog_path(self) -> Path:
        """Get path to cache catalog JSON file."""
        return Path("cache_catalog.json")

    def _load_catalog(self) -> dict:
        """Load dataset/cache pairs catalog from JSON file."""
        cat_file = self._catalog_path()
        if cat_file.exists():
            try:
                with open(cat_file, "r", encoding="utf-8") as file:
                    return json.load(file)
            except Exception as ex:
                print(f"[WARN] Failed to load catalog from '{cat_file}': {ex}.")

        default_catalog = {
            "active_pair": "default",
            "pairs": {
                "default": {
                    "name": "Default dataset pair",
                    "dataset_path": self.config.get("dataset_path", "data/train_language_aware.csv"),
                    "cache_dir": "cache/default",
                    "model_name": self.config.get("model_name", "sentence-transformers/LaBSE"),
                    "clustering_method": "agglomerative",
                    "created_at": pd.Timestamp.now().isoformat(),
                    "last_updated": pd.Timestamp.now().isoformat(),
                }
            },
        }
        self._save_catalog(default_catalog)
        return default_catalog

    def _save_catalog(self, catalog: dict) -> None:
        """Save dataset/cache pairs catalog to JSON file.

        Args:
            catalog: Catalog dictionary.
        """
        cat_file = self._catalog_path()
        with open(cat_file, "w", encoding="utf-8") as file:
            json.dump(catalog, file, indent=2)

    def list_pairs(self) -> Dict[str, dict]:
        """List all registered dataset/cache pairs.

        Returns:
            Dictionary mapping pair IDs to pair metadata dictionaries.
        """
        catalog = self._load_catalog()
        return catalog.get("pairs", {})

    def select_pair(self, pair_id: str) -> bool:
        """Select active dataset/cache pair and update engine state.

        Args:
            pair_id: Identifier of pair to select.

        Returns:
            Boolean indicating whether selection succeeded.
        """
        catalog = self._load_catalog()
        pairs = catalog.get("pairs", {})
        if pair_id not in pairs:
            print(f"[ERROR] Pair '{pair_id}' not found in catalog.")
            return False

        catalog["active_pair"] = pair_id
        self._save_catalog(catalog)

        meta = pairs[pair_id]
        self.active_pair_id = pair_id
        self.cache_dir = Path(meta.get("cache_dir", f"cache/{pair_id}"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if "dataset_path" in meta:
            self.config["dataset_path"] = meta["dataset_path"]
        if "model_name" in meta:
            self.load_model(meta["model_name"])

        # Reset loaded in-memory matrices so next load_index loads selected pair
        self.df = None
        self.combined_embeddings = None
        self.mention_embeddings = None
        self.context_embeddings = None
        self.faiss_index = None
        self.pca_model = None

        print(f"[INFO] Selected active dataset/cache pair '{pair_id}' ('{meta.get('name', pair_id)}').")
        return True

    def create_pair(
        self,
        pair_id: str,
        dataset_path: str,
        name: Optional[str] = None,
        model_name: Optional[str] = None,
        clustering_method: str = "agglomerative",
        force_update: bool = True,
    ) -> bool:
        """Create new dataset/cache pair and generate indexes.

        Args:
            pair_id: Identifier for new pair.
            dataset_path: Path to dataset CSV/TSV file.
            name: Optional human-readable name.
            model_name: Optional transformer model name.
            clustering_method: Clustering algorithm ('agglomerative', 'faiss_topk', or 'faiss_kmeans').
            force_update: Whether to run index generation immediately.

        Returns:
            Boolean indicating whether creation succeeded.
        """
        clean_id = pair_id.strip().lower().replace(" ", "_")
        if not clean_id:
            print("[ERROR] Pair ID cannot be empty.")
            return False

        catalog = self._load_catalog()
        pairs = catalog.get("pairs", {})

        selected_model = model_name or self.model_name
        cache_subdir = f"cache/{clean_id}"

        now_iso = pd.Timestamp.now().isoformat()
        pairs[clean_id] = {
            "name": name or pair_id.strip(),
            "dataset_path": dataset_path,
            "cache_dir": cache_subdir,
            "model_name": selected_model,
            "clustering_method": clustering_method,
            "created_at": now_iso,
            "last_updated": now_iso,
        }
        catalog["pairs"] = pairs
        catalog["active_pair"] = clean_id
        self._save_catalog(catalog)

        self.select_pair(clean_id)

        if force_update:
            self.generate_indexes(dataset_path=dataset_path, clustering_method=clustering_method, model_name=selected_model)

        return True

    def update_pair(self, pair_id: str, force_update: bool = True) -> bool:
        """Force update and re-index an existing dataset/cache pair.

        Args:
            pair_id: Identifier of pair to update.
            force_update: Whether to re-run index generation.

        Returns:
            Boolean indicating whether update succeeded.
        """
        catalog = self._load_catalog()
        pairs = catalog.get("pairs", {})
        if pair_id not in pairs:
            print(f"[ERROR] Pair '{pair_id}' not found in catalog.")
            return False

        meta = pairs[pair_id]
        self.select_pair(pair_id)

        if force_update:
            self.generate_indexes(
                dataset_path=meta.get("dataset_path"),
                clustering_method=meta.get("clustering_method", "agglomerative"),
                model_name=meta.get("model_name"),
            )

        pairs[pair_id]["last_updated"] = pd.Timestamp.now().isoformat()
        catalog["pairs"] = pairs
        self._save_catalog(catalog)
        print(f"[INFO] Pair '{pair_id}' successfully updated.")
        return True

    def delete_pair(self, pair_id: str) -> bool:
        """Delete dataset/cache pair and remove associated cache directory from disk.

        Args:
            pair_id: Identifier of pair to delete.

        Returns:
            Boolean indicating whether deletion succeeded.
        """
        import gc
        import shutil

        catalog = self._load_catalog()
        pairs = catalog.get("pairs", {})
        if pair_id not in pairs:
            print(f"[ERROR] Pair '{pair_id}' not found in catalog.")
            return False

        meta = pairs[pair_id]
        cache_dir_path = Path(meta.get("cache_dir", f"cache/{pair_id}"))

        if self.active_pair_id == pair_id:
            self.df = None
            self.combined_embeddings = None
            self.mention_embeddings = None
            self.context_embeddings = None
            self.faiss_index = None
            self.pca_model = None
            gc.collect()

        if cache_dir_path.exists():
            try:
                shutil.rmtree(cache_dir_path)
                print(f"[INFO] Deleted cache directory '{cache_dir_path}'.")
            except Exception as ex:
                print(f"[WARN] Could not remove cache directory '{cache_dir_path}': {ex}.")

        del pairs[pair_id]
        catalog["pairs"] = pairs

        if catalog.get("active_pair") == pair_id:
            catalog["active_pair"] = list(pairs.keys())[0] if pairs else ""
            if catalog["active_pair"]:
                self.select_pair(catalog["active_pair"])
            else:
                self.active_pair_id = None

        self._save_catalog(catalog)
        print(f"[INFO] Deleted pair '{pair_id}' from catalog.")
        return True

    def _load_config(self, config_path: str) -> dict:
        """Load configuration options from JSON file.

        Args:
            config_path: Path to configuration file.

        Returns:
            Dictionary containing configuration options.
        """
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as file:
                return json.load(file)
        return {}

    def load_model(self, model_name: Optional[str] = None, cross_encoder_model: Optional[str] = None) -> None:
        """Load bi-encoder and optional cross-encoder models into memory.

        Args:
            model_name: Optional transformer model name to switch to.
            cross_encoder_model: Optional cross-encoder model name to switch to ("None" or None to disable).
        """
        if model_name and model_name != self.model_name:
            self.model_name = model_name
            self.model = None

        if cross_encoder_model is not None:
            if str(cross_encoder_model).lower() in ("none", "", "null"):
                self.cross_encoder_model = None
                self.cross_encoder = None
            else:
                self.cross_encoder_model = cross_encoder_model
                self.cross_encoder = None

        if self.model is None:
            print(f"[INFO] Loading MERCURY transformer model '{self.model_name}' on {self.device}...")
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(self.model_name, device=str(self.device))
            except ImportError:
                from transformers import AutoModel, AutoTokenizer

                class HFWrapper:
                    """Wrapper for Hugging Face transformers model."""

                    def __init__(self, model_name: str, device: torch.device) -> None:
                        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                        self.hf_model = AutoModel.from_pretrained(model_name).to(device)
                        self.device = device

                    def encode(
                        self,
                        sentences: List[str],
                        batch_size: int = 32,
                        show_progress_bar: bool = False,
                        normalize_embeddings: bool = True,
                    ) -> np.ndarray:
                        all_embeddings = []
                        iterator = range(0, len(sentences), batch_size)
                        if show_progress_bar:
                            iterator = tqdm(iterator, desc="Encoding sentences")

                        self.hf_model.eval()
                        with torch.no_grad():
                            for idx in iterator:
                                batch = sentences[idx : idx + batch_size]
                                encoded = self.tokenizer(
                                    batch,
                                    padding=True,
                                    truncation=True,
                                    max_length=256,
                                    return_tensors="pt",
                                ).to(self.device)
                                outputs = self.hf_model(**encoded)
                                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                                token_embeddings = outputs.last_hidden_state
                                sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
                                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                                pooled = sum_embeddings / sum_mask

                                if normalize_embeddings:
                                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

                                all_embeddings.append(pooled.cpu().numpy())

                        return np.vstack(all_embeddings)

                self.model = HFWrapper(self.model_name, self.device)

        if self.cross_encoder_model and str(self.cross_encoder_model).lower() not in ("none", "", "null") and self.cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder

                print(f"[INFO] Loading cross-encoder model '{self.cross_encoder_model}'...")
                self.cross_encoder = CrossEncoder(self.cross_encoder_model, device=str(self.device))
            except Exception as e:
                print(f"[WARN] Could not load cross-encoder '{self.cross_encoder_model}': {e}.")
                self.cross_encoder = None

    def map_dataset(
        self,
        dataset_path: str,
        column_mapping: Optional[Dict[str, str]] = None,
        delimiter: Optional[str] = None,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Stage 1: Load raw CSV/TSV dataset, auto-detect columns/delimiter, and map to MERCURY standard internal schema.

        Args:
            dataset_path: Path to dataset file (CSV or TSV).
            column_mapping: Optional dict mapping raw column names to standard internal names.
            delimiter: Optional delimiter string (e.g. ',' or '\t'). Auto-detected if None.
            verbose: Whether to print progress log messages.

        Returns:
            Tuple of (mapped_dataframe, active_column_mapping).
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found at '{dataset_path}'.")

        if delimiter is None:
            if dataset_path.lower().endswith(".tsv"):
                delimiter = "\t"
            else:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    first_line = f.readline()
                delimiter = "\t" if "\t" in first_line and "," not in first_line else ","

        if verbose:
            print(f"[INFO] Stage 1: Loading raw dataset from '{dataset_path}' using delimiter '{repr(delimiter)}'...")
        df_raw = pd.read_csv(dataset_path, sep=delimiter, encoding="utf-8")
        df_raw = df_raw.loc[:, ~df_raw.columns.str.contains("^Unnamed")]

        mapped_dict: Dict[str, str] = {}
        sentence_aliases = ["clean_mention_sentence", "sentence", "text", "clean_sentence", "mention_sentence"]
        entity_aliases = ["clean_entity_mention", "entity", "mention", "clean_entity", "entity_mention"]
        start_aliases = ["mention_start", "start", "start_idx", "start_pos"]
        end_aliases = ["mention_end", "end", "end_idx", "end_pos"]
        type_aliases = ["entity_type", "type", "spacy_type"]
        file_aliases = ["source_file", "document source", "source_article", "filename"]

        raw_cols_lower = {col.strip().lower(): col for col in df_raw.columns}

        def find_col(aliases: List[str]) -> Optional[str]:
            for alias in aliases:
                if alias.lower() in raw_cols_lower:
                    return raw_cols_lower[alias.lower()]
            return None

        if column_mapping:
            for raw_col, std_target in column_mapping.items():
                if raw_col in df_raw.columns:
                    mapped_dict[raw_col] = std_target
        else:
            col_sentence = find_col(sentence_aliases)
            col_entity = find_col(entity_aliases)
            col_start = find_col(start_aliases)
            col_end = find_col(end_aliases)
            col_type = find_col(type_aliases)
            col_file = find_col(file_aliases)

            if col_sentence:
                mapped_dict[col_sentence] = "clean_mention_sentence"
            if col_entity:
                mapped_dict[col_entity] = "clean_entity_mention"
            if col_start:
                mapped_dict[col_start] = "mention_start"
            if col_end:
                mapped_dict[col_end] = "mention_end"
            if col_type:
                mapped_dict[col_type] = "entity_type"
            if col_file:
                mapped_dict[col_file] = "source_file"

        df_mapped = df_raw.rename(columns=mapped_dict)

        if "clean_mention_sentence" not in df_mapped.columns:
            raise ValueError(f"Could not identify sentence column in dataset. Raw columns: {list(df_raw.columns)}")

        self.df = df_mapped
        if verbose:
            print(f"[INFO] Stage 1 complete. Successfully mapped {len(self.df)} dataset rows.")
        return df_mapped, mapped_dict

    def load_dataset(
        self,
        dataset_path: Optional[str] = None,
        column_mapping: Optional[Dict[str, str]] = None,
        delimiter: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load and map corpus dataset from CSV/TSV file.

        Args:
            dataset_path: File path to CSV/TSV dataset.
            column_mapping: Optional column mapping dict.
            delimiter: Optional delimiter character.

        Returns:
            Mapped DataFrame.
        """
        path = dataset_path or self.config.get("dataset_path", "data/prepared_data.tsv")
        df_mapped, _ = self.map_dataset(path, column_mapping=column_mapping, delimiter=delimiter)
        return df_mapped

    def _extract_mention_and_context(self, row: pd.Series) -> Tuple[str, str]:
        """Extract isolated mention string and sentence context.

        Args:
            row: Single row from dataset DataFrame.

        Returns:
            Tuple containing (mention_text, context_text).
        """
        sentence = str(row.get("clean_mention_sentence", ""))
        start = row.get("mention_start")
        end = row.get("mention_end")
        entity = str(row.get("clean_entity_mention", ""))

        mention_str = entity if (entity and entity != "nan") else sentence
        context_str = sentence

        if pd.notna(start) and pd.notna(end):
            try:
                start_idx = int(start)
                end_idx = int(end)
                if 0 <= start_idx < end_idx <= len(sentence):
                    mention_str = sentence[start_idx:end_idx]
                    prefix = sentence[:start_idx]
                    suffix = sentence[end_idx:]
                    context_str = f"{prefix} <mention> {mention_str} </mention> {suffix}"
            except (ValueError, TypeError):
                pass

        return mention_str, context_str

    def generate_indexes(
        self,
        dataset_path: Optional[str] = None,
        clustering_method: str = "agglomerative",
        model_name: Optional[str] = None,
        alpha_span_weight: Optional[float] = None,
    ) -> None:
        """Generate dual vector embeddings, build FAISS index, and cluster medoids.

        Args:
            dataset_path: Optional file path to CSV dataset.
            clustering_method: Clustering algorithm ('agglomerative', 'faiss_topk', or 'faiss_kmeans').
            model_name: Optional transformer model name.
            alpha_span_weight: Optional float weight for entity mention vector (0.0 to 1.0).
        """
        if alpha_span_weight is not None:
            self.alpha_span_weight = float(alpha_span_weight)

        if model_name:
            self.model_name = model_name
            self.model = None

        if self.df is None:
            self.load_dataset(dataset_path)

        self.load_model()

        print("[INFO] Extracting mention spans and context strings...")
        mentions = []
        contexts = []
        for _, row in self.df.iterrows():
            m_str, c_str = self._extract_mention_and_context(row)
            mentions.append(m_str)
            contexts.append(c_str)

        print(f"[INFO] Pass 1/2: Computing mention span vector embeddings for {len(mentions):,} items...")
        self.mention_embeddings = self.model.encode(
            mentions,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        print(f"[INFO] Pass 2/2: Computing sentence context vector embeddings for {len(contexts):,} items...")
        self.context_embeddings = self.model.encode(
            contexts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        alpha = self.alpha_span_weight
        combined = alpha * self.mention_embeddings + (1.0 - alpha) * self.context_embeddings
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        self.combined_embeddings = np.ascontiguousarray(combined / norms, dtype=np.float32)

        if self.reduced_dimensions and self.reduced_dimensions < self.combined_embeddings.shape[1]:
            print(f"[INFO] Reducing vector dimensions from {self.combined_embeddings.shape[1]} to {self.reduced_dimensions} using PCA...")
            from sklearn.decomposition import PCA

            self.pca_model = PCA(n_components=self.reduced_dimensions, random_state=42)
            reduced = self.pca_model.fit_transform(self.combined_embeddings)
            norms = np.linalg.norm(reduced, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-9, norms)
            self.combined_embeddings = np.ascontiguousarray(reduced / norms, dtype=np.float32)

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            pca_path = self.cache_dir / "pca_model.pkl"
            with open(pca_path, "wb") as file:
                pickle.dump(self.pca_model, file)
            print(f"[INFO] Saved PCA dimension reducer to '{pca_path}'.")

        dimension = self.combined_embeddings.shape[1]
        print(f"[INFO] Building FAISS index for vector dimension {dimension}...")

        try:
            import faiss

            index = faiss.IndexFlatIP(dimension)
            index.add(self.combined_embeddings)
            self.faiss_index = index

            self.cache_dir.mkdir(parents=True, exist_ok=True)
            faiss_path = self.cache_dir / "faiss_index.index"
            faiss.write_index(index, str(faiss_path))
            print(f"[INFO] Saved FAISS index to '{faiss_path}'.")
        except Exception as e:
            print(f"[WARN] FAISS unavailable ({e}). Saving raw vector matrices instead.")

        self._compute_graph_clusters_and_medoids(clustering_method=clustering_method)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.cache_dir / "embeddings_combined.npy", self.combined_embeddings)
        np.save(self.cache_dir / "embeddings_mention.npy", self.mention_embeddings)
        np.save(self.cache_dir / "embeddings_context.npy", self.context_embeddings)

        meta_path = self.cache_dir / "metadata.pkl"
        with open(meta_path, "wb") as file:
            pickle.dump(self.df, file)
        print(f"[INFO] Saved dataset metadata to '{meta_path}'.")
        print("[INFO] MERCURY indexing completed successfully.")

    def _compute_graph_clusters_and_medoids(self, clustering_method: str = "agglomerative") -> None:
        """Perform community clustering and calculate medoid mentions.

        Args:
            clustering_method: Algorithm selection ('agglomerative', 'faiss_topk', or 'faiss_kmeans').
        """
        if self.combined_embeddings is None or self.df is None:
            return

        num_samples = len(self.combined_embeddings)
        if num_samples == 0:
            return

        if num_samples == 1:
            m_text, _ = self._extract_mention_and_context(self.df.iloc[0])
            self.df["cluster_id"] = [0]
            self.df["cluster_canonical_mention"] = [m_text]
            return

        method = clustering_method.lower().strip()

        if method == "faiss_topk":
            print("[INFO] Computing clusters using FAISS top-k mutual nearest-neighbor graph...")
            k_neighbors = min(30, num_samples)
            if self.faiss_index is not None:
                _, indices = self.faiss_index.search(self.combined_embeddings, k_neighbors)
            else:
                sim_matrix = np.dot(self.combined_embeddings, self.combined_embeddings.T)
                indices = np.argsort(sim_matrix, axis=1)[:, ::-1][:, :k_neighbors]

            parent = list(range(num_samples))

            def find(i: int) -> int:
                path = []
                while parent[i] != i:
                    path.append(i)
                    i = parent[i]
                for node in path:
                    parent[node] = i
                return i

            def union(i: int, j: int) -> None:
                root_i = find(i)
                root_j = find(j)
                if root_i != root_j:
                    parent[root_i] = root_j

            thresh = float(self.similarity_threshold)
            for i in range(num_samples):
                neighbors_i = indices[i]
                for n_idx in neighbors_i:
                    if n_idx != i:
                        sim = float(np.dot(self.combined_embeddings[i], self.combined_embeddings[n_idx]))
                        if sim >= thresh and i in indices[n_idx]:
                            union(i, n_idx)

            cluster_groups: Dict[int, List[int]] = {}
            for i in range(num_samples):
                root = find(i)
                if root not in cluster_groups:
                    cluster_groups[root] = []
                cluster_groups[root].append(i)

            cluster_ids = np.zeros(num_samples, dtype=int)
            for c_idx, (root, members) in enumerate(cluster_groups.items()):
                for m_idx in members:
                    cluster_ids[m_idx] = c_idx

        elif method == "faiss_leader":
            print("[INFO] Computing clusters using FAISS leader-follower canopy clustering (self-determined cluster count)...")
            thresh = float(self.similarity_threshold)
            cluster_ids = np.full(num_samples, -1, dtype=int)
            leader_indices: List[int] = []

            try:
                import faiss
                d = self.combined_embeddings.shape[1]
                if torch.cuda.is_available() and hasattr(faiss, "StandardGpuResources"):
                    res = faiss.StandardGpuResources()
                    gpu_index = faiss.GpuIndexFlatIP(res, d)
                else:
                    gpu_index = faiss.IndexFlatIP(d)

                for i in range(num_samples):
                    vec = self.combined_embeddings[i : i + 1]
                    if len(leader_indices) == 0:
                        leader_indices.append(i)
                        gpu_index.add(vec)
                        cluster_ids[i] = 0
                    else:
                        scores, sim_leaders = gpu_index.search(vec, 1)
                        best_sim = float(scores[0][0])
                        best_leader_id = int(sim_leaders[0][0])

                        if best_sim >= thresh:
                            cluster_ids[i] = best_leader_id
                        else:
                            new_leader_id = len(leader_indices)
                            leader_indices.append(i)
                            gpu_index.add(vec)
                            cluster_ids[i] = new_leader_id
            except Exception as ex:
                print(f"[WARN] FAISS leader clustering fallback ({ex})...")
                cluster_ids = np.arange(num_samples)

        elif method == "faiss_kmeans":
            print("[INFO] Computing clusters using FAISS k-means GPU/CPU (fixed target K)...")
            num_clusters = max(1, min(num_samples // 10, 50000))
            d = self.combined_embeddings.shape[1]

            try:
                import faiss

                kmeans = faiss.Clustering(d, num_clusters)
                kmeans.niter = 20
                kmeans.max_points_per_centroid = 10000
                kmeans.verbose = False

                if torch.cuda.is_available() and hasattr(faiss, "StandardGpuResources"):
                    res = faiss.StandardGpuResources()
                    index_flat = faiss.GpuIndexFlatIP(res, d)
                else:
                    index_flat = faiss.IndexFlatIP(d)

                kmeans.train(self.combined_embeddings, index_flat)
                _, cluster_ids = index_flat.search(self.combined_embeddings, 1)
                cluster_ids = cluster_ids.squeeze()
            except Exception as e:
                print(f"[WARN] FAISS k-means GPU failed ({e}), falling back to MiniBatchKMeans...")
                from sklearn.cluster import MiniBatchKMeans

                kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=1024)
                cluster_ids = kmeans.fit_predict(self.combined_embeddings)

        else:
            print("[INFO] Computing community clusters using agglomerative average-linkage clustering...")
            from sklearn.cluster import AgglomerativeClustering

            dist_threshold = max(0.01, 1.0 - float(self.similarity_threshold))
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=dist_threshold,
            )
            cluster_ids = clustering.fit_predict(self.combined_embeddings)

        unique_clusters = np.unique(cluster_ids)
        print(f"[INFO] Formed {len(unique_clusters)} distinct clusters. Computing cluster medoids...")

        canonical_mentions = [""] * num_samples

        for c_id in unique_clusters:
            member_indices = np.where(cluster_ids == c_id)[0]
            cluster_vecs = self.combined_embeddings[member_indices]
            centroid = np.mean(cluster_vecs, axis=0, keepdims=True)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 0:
                centroid = centroid / centroid_norm

            sims_to_centroid = np.dot(cluster_vecs, centroid.T).squeeze()
            medoid_local_idx = np.argmax(sims_to_centroid) if len(member_indices) > 1 else 0
            medoid_global_idx = member_indices[medoid_local_idx]

            medoid_row = self.df.iloc[medoid_global_idx]
            m_text, _ = self._extract_mention_and_context(medoid_row)

            for member_i in member_indices:
                canonical_mentions[member_i] = m_text

        self.df["cluster_id"] = cluster_ids
        self.df["cluster_canonical_mention"] = canonical_mentions

    def load_index(self) -> bool:
        """Load pre-computed FAISS index, dual embeddings, and dataset metadata from cache.

        Returns:
            Boolean indicating whether loading succeeded.
        """
        meta_path = self.cache_dir / "metadata.pkl"
        comb_path = self.cache_dir / "embeddings_combined.npy"
        ment_path = self.cache_dir / "embeddings_mention.npy"
        cont_path = self.cache_dir / "embeddings_context.npy"
        faiss_path = self.cache_dir / "faiss_index.index"

        if not meta_path.exists() or not (comb_path.exists() or (self.cache_dir / "embeddings.npy").exists()):
            print("[ERROR] Index cache files not found. Please call 'generate_indexes()' first.")
            return False

        print(f"[INFO] Loading cached dataset metadata from '{meta_path}'...")
        with open(meta_path, "rb") as file:
            self.df = pickle.load(file)

        print(f"[INFO] Loading cached vector matrices from '{self.cache_dir}'...")
        if comb_path.exists():
            self.combined_embeddings = np.load(comb_path)
            if ment_path.exists():
                self.mention_embeddings = np.load(ment_path)
            if cont_path.exists():
                self.context_embeddings = np.load(cont_path)
        else:
            legacy_path = self.cache_dir / "embeddings.npy"
            self.combined_embeddings = np.load(legacy_path)

        pca_path = self.cache_dir / "pca_model.pkl"
        if pca_path.exists():
            with open(pca_path, "rb") as file:
                self.pca_model = pickle.load(file)
            print(f"[INFO] Loaded cached PCA dimension reducer from '{pca_path}'.")

        if faiss_path.exists():
            try:
                import faiss

                print(f"[INFO] Loading FAISS index from '{faiss_path}'...")
                self.faiss_index = faiss.read_index(str(faiss_path))
                self.is_faiss_available = True
            except ImportError:
                print("[WARN] FAISS library not installed. Using matrix cosine similarity fallback.")
                self.is_faiss_available = False
        else:
            print("[WARN] FAISS index file not found. Using matrix cosine similarity fallback.")
            self.is_faiss_available = False

        self.load_model()
        print(f"[INFO] MERCURY ready! Loaded {len(self.df)} sentences into memory.")
        return True

    def query(
        self,
        search_term: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        use_entity_vector_only: bool = True,
    ) -> pd.DataFrame:
        """Query pre-indexed dataset using entity vector embeddings.

        Args:
            search_term: Entity name, reference, or query phrase (e.g. "France").
            top_k: Number of matching sentences to return.
            min_score: Minimum similarity score threshold.
            use_entity_vector_only: Whether to calculate similarity strictly against entity vectors.

        Returns:
            DataFrame containing top matching sentences and similarity scores.
        """
        if self.df is None or self.combined_embeddings is None:
            if not self.load_index():
                raise RuntimeError("Failed to load pre-computed index cache.")

        k = top_k or self.top_k
        cutoff = min_score if min_score is not None else self.min_score

        raw_query_vec = self.model.encode(
            [search_term], show_progress_bar=False, normalize_embeddings=True
        )

        candidate_pool_size = min(k * 4, len(self.df))

        if use_entity_vector_only and self.mention_embeddings is not None:
            sims = np.dot(self.mention_embeddings, raw_query_vec.T).squeeze()
            stage1_indices = np.argsort(sims)[::-1][:candidate_pool_size]
            stage1_scores = sims[stage1_indices]
        elif self.is_faiss_available and self.faiss_index is not None:
            if self.pca_model is not None:
                query_vec = self.pca_model.transform(raw_query_vec)
                q_norm = np.linalg.norm(query_vec)
                if q_norm > 0:
                    query_vec = query_vec / q_norm
            else:
                query_vec = raw_query_vec
            query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)
            stage1_scores, stage1_indices = self.faiss_index.search(query_vec, candidate_pool_size)
            stage1_scores = stage1_scores[0]
            stage1_indices = stage1_indices[0]
        else:
            sims = np.dot(self.combined_embeddings, raw_query_vec.T).squeeze()
            stage1_indices = np.argsort(sims)[::-1][:candidate_pool_size]
            stage1_scores = sims[stage1_indices]

        if self.cross_encoder is not None:
            pairs = [(search_term, str(self.df.iloc[idx].get("clean_mention_sentence", ""))) for idx in stage1_indices]
            rerank_scores = self.cross_encoder.predict(pairs)
            if len(rerank_scores) > 0:
                min_s = float(np.min(rerank_scores))
                max_s = float(np.max(rerank_scores))
                rerank_scores = (rerank_scores - min_s) / (max_s - min_s) if max_s > min_s else stage1_scores
        else:
            rerank_scores = stage1_scores

        sort_order = np.argsort(rerank_scores)[::-1]
        sorted_indices = stage1_indices[sort_order]
        sorted_scores = rerank_scores[sort_order]

        valid_mask = sorted_scores >= cutoff
        filtered_indices = sorted_indices[valid_mask][:k]
        filtered_scores = sorted_scores[valid_mask][:k]

        if len(filtered_indices) == 0:
            filtered_indices = sorted_indices[:k]
            filtered_scores = sorted_scores[:k]

        results = self.df.iloc[filtered_indices].copy()
        results["similarity_score"] = filtered_scores
        return results

    def get_co_occurring_entities(self, query_results: pd.DataFrame, ignore_case: bool = False) -> pd.DataFrame:
        """Extract and rank co-occurring entity mentions in matching query sentences.

        Args:
            query_results: DataFrame of search query results or subset of dataset.
            ignore_case: If True, merges entity mentions that differ only in casing.

        Returns:
            DataFrame containing co-occurring entities, absolute counts, and frequencies.
        """
        if query_results is None or query_results.empty or self.df is None:
            return pd.DataFrame(columns=["Entity mention", "Absolute count", "Frequency (%)"])

        result_sentences = query_results["clean_mention_sentence"].dropna().unique()
        total_sentences = len(result_sentences)
        if total_sentences == 0:
            return pd.DataFrame(columns=["Entity mention", "Absolute count", "Frequency (%)"])

        matching_rows = self.df[self.df["clean_mention_sentence"].isin(result_sentences)]

        entity_sentence_map: Dict[str, set] = {}
        raw_case_freq: Dict[str, Dict[str, int]] = {}

        for _, row in matching_rows.iterrows():
            m_text, _ = self._extract_mention_and_context(row)
            sentence = str(row.get("clean_mention_sentence", ""))
            start = row.get("mention_start")
            end = row.get("mention_end")

            has_span = (pd.notna(start) and pd.notna(end)) or (m_text and m_text != sentence)
            if not has_span or not m_text or m_text == "nan":
                continue

            norm_key = m_text.lower() if ignore_case else m_text

            if norm_key not in raw_case_freq:
                raw_case_freq[norm_key] = {}
            raw_case_freq[norm_key][m_text] = raw_case_freq[norm_key].get(m_text, 0) + 1

            if norm_key not in entity_sentence_map:
                entity_sentence_map[norm_key] = set()

            entity_sentence_map[norm_key].add(sentence)

        records = []
        for norm_key, sentences in entity_sentence_map.items():
            count = len(sentences)
            percentage = (count / total_sentences) * 100.0
            display_entity = max(raw_case_freq[norm_key].keys(), key=lambda k: raw_case_freq[norm_key][k]) if ignore_case else norm_key
            records.append({
                "Entity mention": display_entity,
                "Absolute count": count,
                "Frequency (%)": f"{percentage:.1f}%",
                "_raw_count": count,
                "_raw_pct": percentage,
            })

        if not records:
            return pd.DataFrame(columns=["Entity mention", "Absolute count", "Frequency (%)"])

        df_co = pd.DataFrame(records)
        df_co = df_co.sort_values(by=["_raw_count", "_raw_pct"], ascending=False)
        df_co = df_co.drop(columns=["_raw_count", "_raw_pct"])
        return df_co.reset_index(drop=True)

    def plot_co_occurrence_network(
        self,
        query_results: pd.DataFrame,
        max_nodes: int = 25,
        use_medoids: bool = False,
        ignore_case: bool = False,
    ) -> None:
        """Render network graph visualization showing co-occurrences of entities in sentences.

        Args:
            query_results: DataFrame of search query results or subset of dataset.
            max_nodes: Maximum number of top co-occurring nodes to display in the graph.
            use_medoids: If True, renders cluster canonical medoid labels; if False, renders raw clean entity mention labels.
            ignore_case: If True, treats entity mention labels as case-insensitive.
        """
        if query_results is None or query_results.empty or self.df is None:
            print("[WARN] No search query results available to generate co-occurrence network.")
            return

        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError:
            print("[WARN] networkx or matplotlib is not installed. Please install networkx and matplotlib to view network graphs.")
            return

        result_sentences = query_results["clean_mention_sentence"].dropna().unique()
        if len(result_sentences) == 0:
            return

        matching_rows = self.df[self.df["clean_mention_sentence"].isin(result_sentences)]

        sentence_entities_map: Dict[str, set] = {}
        raw_case_freq: Dict[str, Dict[str, int]] = {}

        col_label = "cluster_canonical_mention" if (use_medoids and "cluster_canonical_mention" in matching_rows.columns) else "clean_entity_mention"

        for _, row in matching_rows.iterrows():
            entity_label = str(row.get(col_label, "")).strip()
            sentence = str(row.get("clean_mention_sentence", "")).strip()
            if not entity_label or entity_label == "nan":
                continue

            norm_key = entity_label.lower() if ignore_case else entity_label

            if norm_key not in raw_case_freq:
                raw_case_freq[norm_key] = {}
            raw_case_freq[norm_key][entity_label] = raw_case_freq[norm_key].get(entity_label, 0) + 1

            if sentence not in sentence_entities_map:
                sentence_entities_map[sentence] = set()

            sentence_entities_map[sentence].add(norm_key)

        display_label_map: Dict[str, str] = {
            k: (max(v.keys(), key=lambda orig: v[orig]) if ignore_case else k)
            for k, v in raw_case_freq.items()
        }

        entity_counts: Dict[str, int] = {}
        for sentence, keys in sentence_entities_map.items():
            for k in keys:
                entity_counts[k] = entity_counts.get(k, 0) + 1

        if not entity_counts:
            print("[WARN] No entity co-occurrences found to plot.")
            return

        top_keys = set(sorted(entity_counts.keys(), key=lambda k: entity_counts[k], reverse=True)[:max_nodes])

        edge_weights: Dict[Tuple[str, str], int] = {}
        for sentence, keys in sentence_entities_map.items():
            valid_keys = list(keys.intersection(top_keys))
            for i in range(len(valid_keys)):
                for j in range(i + 1, len(valid_keys)):
                    k1, k2 = sorted([valid_keys[i], valid_keys[j]])
                    edge_weights[(k1, k2)] = edge_weights.get((k1, k2), 0) + 1

        G = nx.Graph()
        for k in top_keys:
            disp_label = display_label_map[k]
            G.add_node(disp_label, weight=entity_counts[k])

        for (k1, k2), weight in edge_weights.items():
            u = display_label_map[k1]
            v = display_label_map[k2]
            G.add_edge(u, v, weight=weight)

        if G.number_of_nodes() == 0:
            print("[WARN] Network graph has no nodes.")
            return

        plt.figure(figsize=(10, 7), dpi=100)
        label_type_str = "canonical cluster medoids" if (use_medoids and "cluster_canonical_mention" in matching_rows.columns) else "clean entity mentions"
        case_str = " (case-insensitive)" if ignore_case else ""
        plt.title(f"Co-occurrence network of {label_type_str}{case_str} ({len(G.nodes())} nodes)", fontsize=13, fontweight="bold")

        pos = nx.spring_layout(G, k=0.5, seed=42)

        node_sizes = [max(300, min(3000, G.nodes[n]["weight"] * 300)) for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#1f77b4", alpha=0.85)

        if G.number_of_edges() > 0:
            weights = [G[u][v]["weight"] for u, v in G.edges()]
            max_w = max(weights) if weights else 1
            widths = [max(1.0, (w / max_w) * 4.0) for w in weights]
            nx.draw_networkx_edges(G, pos, width=widths, edge_color="#8C8C8C", alpha=0.6)

        nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", font_family="sans-serif")

        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def find_similar_sentences(self, target_sentence: str, top_k: int = 20) -> pd.DataFrame:
        """Find sentences with highest context vector similarity to target_sentence.

        Args:
            target_sentence: Sentence text to compare against dataset.
            top_k: Number of similar sentences to return.

        Returns:
            DataFrame containing similar sentences and similarity scores.
        """
        if self.df is None or self.context_embeddings is None:
            if not self.load_index():
                raise RuntimeError("Failed to load pre-computed index cache.")

        sent_vec = self.model.encode([target_sentence], show_progress_bar=False, normalize_embeddings=True)
        sims = np.dot(self.context_embeddings, sent_vec.T).squeeze()
        top_indices = np.argsort(sims)[::-1][:top_k]
        top_scores = sims[top_indices]

        results = self.df.iloc[top_indices].copy()
        results["similarity_score"] = top_scores
        cols = ["similarity_score", "clean_mention_sentence", "clean_entity_mention", "source_language", "cluster_canonical_mention"]
        display_cols = [c for c in cols if c in results.columns]
        return results[display_cols]

    def get_pair_selection_notebook_ui(self) -> None:
        """Create and display Stage 1 cache pair selection and index loading UI for Jupyter Notebooks."""
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError:
            print("[WARN] ipywidgets is not installed. Please install ipywidgets to use the pair selection UI.")
            return

        pairs = self.list_pairs()
        pair_options = [(f"{meta.get('name', pid)} ({pid})", pid) for pid, meta in pairs.items()]
        if not pair_options:
            pair_options = [("Default pair (default)", "default")]

        active_pair_id = self.active_pair_id if self.active_pair_id in pairs else pair_options[0][1]

        pair_dropdown = widgets.Dropdown(
            options=pair_options,
            value=active_pair_id,
            description="Cache pair:",
            disabled=False,
        )

        pair_info_html = widgets.HTML(value="")

        def format_pair_info(pid: str) -> str:
            meta = pairs.get(pid, {})
            c_dir = Path(meta.get("cache_dir", f"cache/{pid}"))
            is_indexed = (c_dir / "metadata.pkl").exists() and (c_dir / "embeddings_combined.npy").exists()
            status_text = "<span style='color: green; font-weight: bold;'>Indexed</span>" if is_indexed else "<span style='color: red; font-weight: bold;'>Not Indexed</span>"
            return (
                f"<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px; background: #f9f9f9; margin-bottom: 10px;'>"
                f"<b>Pair ID:</b> {pid}<br>"
                f"<b>Name:</b> {meta.get('name', pid)}<br>"
                f"<b>Dataset path:</b> {meta.get('dataset_path', 'N/A')}<br>"
                f"<b>Vector model:</b> {meta.get('model_name', 'N/A')}<br>"
                f"<b>Clustering method:</b> {meta.get('clustering_method', 'N/A')}<br>"
                f"<b>Last updated:</b> {meta.get('last_updated', 'N/A')}<br>"
                f"<b>Cache status:</b> {status_text} (Folder: <code>{c_dir}</code>)"
                f"</div>"
            )

        pair_info_html.value = format_pair_info(active_pair_id)

        def on_pair_change(change):
            if change["type"] == "change" and change["name"] == "value":
                pair_info_html.value = format_pair_info(change["new"])

        pair_dropdown.observe(on_pair_change)

        load_button = widgets.Button(description="Select & load cache pair", button_style="primary")
        output_area = widgets.Output()

        def on_load_click(_b=None):
            with output_area:
                output_area.clear_output()
                pid = pair_dropdown.value
                print(f"[NOTICE] Wait with running cell2 untill you see the SUCCESS message")
                print(f"[INFO] Selecting cache pair '{pid}'...")
                if self.select_pair(pid):
                    if self.load_index():
                        print(f"[SUCCESS] Loaded cache pair '{pid}' ({len(self.df):,} rows indexed). You can now run Cell 2 to query.")
                    else:
                        print(f"[WARN] Failed to load index files for pair '{pid}'. Generating indexes may be required.")

        load_button.on_click(on_load_click)

        ui_box = widgets.VBox([
            widgets.HTML("<h3>Stage 1: Select cached dataset</h3>"),
            widgets.HBox([pair_dropdown, load_button]),
            pair_info_html,
            output_area,
        ])
        display(ui_box)

    def get_notebook_ui(self) -> None:
        """Create and display Stage 2 entity interrogation UI for Jupyter Notebooks."""
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError:
            print("[WARN] ipywidgets is not installed. Please install ipywidgets to use the tabbed Jupyter UI.")
            return

        available_ce_models = self.config.get(
            "available_cross_encoder_models",
            ["None", "cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-mMiniLM-L-6-v2", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"],
        )
        current_ce_model = self.cross_encoder_model or "None"
        if current_ce_model not in available_ce_models:
            available_ce_models.insert(0, current_ce_model)

        ce_dropdown = widgets.Dropdown(
            options=available_ce_models,
            value=current_ce_model,
            description="Cross-encoder:",
            disabled=False,
        )

        def on_ce_change(change):
            if change["type"] == "change" and change["name"] == "value":
                val = None if change["new"] == "None" else change["new"]
                self.load_model(cross_encoder_model=val)

        ce_dropdown.observe(on_ce_change)

        if self.df is None:
            print(f"[INFO] Active pair: '{self.active_pair_id}'. Loading index...")
            if not self.load_index():
                print("[WARN] No cache index loaded yet. Please select and load a pair in Cell 1 first.")

        query_input = widgets.Text(
            value="",
            placeholder="Enter entity name or search phrase...",
            description="Search:",
            disabled=False,
        )
        top_k_input = widgets.BoundedIntText(
            value=self.top_k,
            min=1,
            max=50000,
            step=1,
            description="Top-k hits:",
            disabled=False,
        )
        graph_label_mode = widgets.Dropdown(
            options=[("Clean entity mentions", False), ("Cluster canonical medoids", True)],
            value=False,
            description="Graph labels:",
            disabled=False,
        )
        ignore_case_checkbox = widgets.Checkbox(
            value=True,
            description="Ignore case",
            disabled=False,
        )
        search_button = widgets.Button(description="Search", button_style="primary")

        extra_columns = [col for col in self.df.columns if col not in ("clean_mention_sentence", "similarity_score")] if self.df is not None else []
        column_selector = widgets.SelectMultiple(
            options=extra_columns,
            value=[col for col in ("clean_entity_mention", "source_language", "cluster_id", "cluster_canonical_mention") if col in extra_columns],
            description="Extra columns:",
            disabled=False,
            rows=min(5, max(1, len(extra_columns))),
        )

        tab1_output = widgets.Output()
        tab2_output = widgets.Output()
        tab3_output = widgets.Output()

        tab_container = widgets.Tab()
        tab_container.children = [tab1_output, tab2_output, tab3_output]
        tab_container.set_title(0, "Search results")
        tab_container.set_title(1, "Co-occurring entities & Network")
        tab_container.set_title(2, "Similar sentences")

        state = {"results": None, "term": ""}

        def render_tab2():
            if state["results"] is None or state["results"].empty:
                return
            results = state["results"]
            search_term = state["term"]

            use_medoids = graph_label_mode.value
            ignore_case = ignore_case_checkbox.value

            co_entities = self.get_co_occurring_entities(results, ignore_case=ignore_case)

            with tab2_output:
                tab2_output.clear_output(wait=True)
                case_info = " (case-insensitive)" if ignore_case else ""
                print(f"Co-occurring entities in sentences matching: '{search_term}' ({len(co_entities)} entities){case_info}\n")
                if not co_entities.empty:
                    display(co_entities)
                    print("\n" + "=" * 60)
                    label_type_label = "CANONICAL CLUSTER MEDOIDS" if use_medoids else "RAW CLEAN ENTITY MENTIONS"
                    print(f"CO-OCCURRENCE NETWORK GRAPH ({label_type_label}{case_info.upper()})")
                    print("=" * 60)
                    self.plot_co_occurrence_network(
                        results,
                        max_nodes=25,
                        use_medoids=use_medoids,
                        ignore_case=ignore_case,
                    )
                else:
                    print("No co-occurring entities found in matching sentences.\n")

        def on_setting_change(_change):
            render_tab2()

        graph_label_mode.observe(on_setting_change, names="value")
        ignore_case_checkbox.observe(on_setting_change, names="value")

        def run_search(_b=None):
            search_term = query_input.value.strip()
            if not search_term:
                return

            requested_k = top_k_input.value
            results = self.query(search_term, top_k=requested_k)
            state["results"] = results
            state["term"] = search_term

            selected_cols = list(column_selector.value)
            display_cols = ["similarity_score", "clean_mention_sentence"] + [c for c in selected_cols if c not in ("similarity_score", "clean_mention_sentence")]

            with tab1_output:
                tab1_output.clear_output()
                print(f"Search results for: '{search_term}' ({len(results)} hits)\n")
                display(results[display_cols])

                hit_options = [(f"Hit #{i+1}: {row.get('clean_mention_sentence', '')[:80]}...", row.get("clean_mention_sentence", "")) for i, (_, row) in enumerate(results.iterrows())]
                if hit_options:
                    hit_dropdown = widgets.Dropdown(
                        options=hit_options,
                        description="Select hit:",
                        layout=widgets.Layout(width="70%"),
                    )
                    similar_btn = widgets.Button(
                        description="Find similar sentences",
                        button_style="info",
                        icon="search",
                    )

                    def on_similar_click(_b=None):
                        selected_sentence = hit_dropdown.value
                        if not selected_sentence:
                            return
                        with tab3_output:
                            tab3_output.clear_output()
                            print(f"Sentences similar to: '{selected_sentence}' (Top-20 hits)\n")
                            df_sim = self.find_similar_sentences(selected_sentence, top_k=20)
                            display(df_sim)
                        tab_container.selected_index = 2

                    similar_btn.on_click(on_similar_click)
                    print("\n--- FIND SIMILAR SENTENCES FOR A HIT ---")
                    display(widgets.HBox([hit_dropdown, similar_btn]))

            render_tab2()

            with tab3_output:
                tab3_output.clear_output()
                print("Select a search hit in Tab 1 and click 'Find similar sentences' to display results here.")

        search_button.on_click(run_search)
        query_input.on_submit(run_search)

        ui_box = widgets.VBox([
            ce_dropdown,
            widgets.HBox([query_input, top_k_input, graph_label_mode, ignore_case_checkbox, search_button]),
            column_selector,
            tab_container,
        ])
        display(ui_box)

    def interrogate(self, initial_query: Optional[str] = None, top_k: int = 10) -> None:
        """Run interactive CLI prompt loop for continuous querying.

        Args:
            initial_query: Optional search term to run immediately before prompting.
            top_k: Default top_k results to return for queries.
        """
        if self.df is None:
            if not self.load_index():
                return

        if initial_query:
            results = self.query(initial_query, top_k=top_k)
            self._print_results(initial_query, results)

        print("\n" + "=" * 60)
        print("MERCURY INTERROGATOR SESSION")
        print(f"Active pair: '{self.active_pair_id}' ({self.cache_dir})")
        print("Type a search term to find related sentences across languages.")
        print("Type ':k N' or 'top_k=N' to change the number of displayed results.")
        print("Type 'exit' or 'quit' to end the session.")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("Enter search term (or 'exit' to quit): ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit", "q"):
                    print("[INFO] Ending MERCURY session.")
                    break

                if user_input.lower().startswith(":k ") or user_input.lower().startswith("top_k="):
                    try:
                        val = int(user_input.replace(":k ", "").replace("top_k=", "").strip())
                        if val > 0:
                            top_k = val
                            print(f"[INFO] Updated default top-k results limit to {top_k}.\n")
                            continue
                    except ValueError:
                        pass

                results = self.query(user_input, top_k=top_k)
                self._print_results(user_input, results)

            except (KeyboardInterrupt, EOFError):
                print("\n[INFO] Session interrupted by user. Exiting.")
                break

    def _print_results(self, search_term: str, results: pd.DataFrame) -> None:
        """Display query results in terminal format across two tabs.

        Args:
            search_term: Query search term.
            results: DataFrame of matching results.
        """
        co_entities = self.get_co_occurring_entities(results)

        print(f"\n==================================================================")
        print(f"--- TAB 1: SEARCH RESULTS FOR: '{search_term}' ({len(results)} hits) ---")
        print(f"==================================================================")
        for idx, row in results.iterrows():
            score = row.get("similarity_score", 0.0)
            sentence = row.get("clean_mention_sentence", "")
            entity = row.get("clean_entity_mention", "")
            medoid = row.get("cluster_canonical_mention", entity or "N/A")
            lang = row.get("source_language", "")
            cluster = row.get("cluster_id", "N/A")
            print(f"[{score:.4f}] (Lang: {lang} | Cluster: {cluster} | Medoid: '{medoid}')")
            print(f"       Entity Mention: {entity}")
            print(f"       Sentence: {sentence}\n")

        print(f"==================================================================")
        print(f"--- TAB 2: CO-OCCURRING ENTITIES FOR: '{search_term}' ({len(co_entities)} entities) ---")
        print(f"==================================================================")
        if co_entities.empty:
            print("No co-occurring entities found in matching sentences.\n")
        else:
            print(co_entities.to_string(index=False))
            print()

    def get_mapping_notebook_ui(self) -> None:
        """Create and display Stage 1 dataset mapping and inspection UI for Jupyter Notebooks."""
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError:
            print("[WARN] ipywidgets is not installed. Please install ipywidgets to use the Jupyter mapping UI.")
            return

        data_dir = Path("data")
        file_options = []
        if data_dir.exists():
            for f in data_dir.glob("*"):
                if f.suffix.lower() in (".csv", ".tsv", ".txt"):
                    file_options.append(str(f))
        if not file_options:
            file_options = ["data/prepared_data.tsv", "data/dataset.csv"]

        dataset_dropdown = widgets.Dropdown(
            options=file_options,
            value=file_options[0],
            description="Dataset file:",
            disabled=False,
        )

        mapping_container = widgets.VBox()
        output_area = widgets.Output()

        last_path = [None]

        def inspect_and_build_mapping_form(path: str, force: bool = False):
            if not force and last_path[0] == path:
                return
            last_path[0] = path

            mapping_container.children = []
            if not os.path.exists(path):
                with output_area:
                    output_area.clear_output()
                    print(f"[ERROR] File '{path}' does not exist.")
                return

            try:
                delimiter = "\t" if path.lower().endswith(".tsv") else None
                if delimiter is None:
                    with open(path, "r", encoding="utf-8") as f:
                        first_line = f.readline()
                    delimiter = "\t" if "\t" in first_line and "," not in first_line else ","

                df_raw = pd.read_csv(path, sep=delimiter, nrows=5, encoding="utf-8")
                raw_cols = [c for c in df_raw.columns if not c.startswith("Unnamed")]
                raw_cols_with_none = ["None"] + raw_cols

                _, auto_mapped = self.map_dataset(path, delimiter=delimiter, verbose=False)
                inv_auto = {v: k for k, v in auto_mapped.items()}

                sentence_col_dd = widgets.Dropdown(
                    options=raw_cols_with_none,
                    value=inv_auto.get("clean_mention_sentence", "None"),
                    description="Sentence col:",
                )
                entity_col_dd = widgets.Dropdown(
                    options=raw_cols_with_none,
                    value=inv_auto.get("clean_entity_mention", "None"),
                    description="Entity col:",
                )
                start_col_dd = widgets.Dropdown(
                    options=raw_cols_with_none,
                    value=inv_auto.get("mention_start", "None"),
                    description="Start col:",
                )
                end_col_dd = widgets.Dropdown(
                    options=raw_cols_with_none,
                    value=inv_auto.get("mention_end", "None"),
                    description="End col:",
                )

                apply_button = widgets.Button(description="Apply & save mapping", button_style="success")
                is_applying = [False]

                def on_apply_click(_b=None):
                    if is_applying[0]:
                        return
                    is_applying[0] = True
                    try:
                        with output_area:
                            output_area.clear_output(wait=True)
                            user_mapping = {}
                            if sentence_col_dd.value != "None":
                                user_mapping[sentence_col_dd.value] = "clean_mention_sentence"
                            if entity_col_dd.value != "None":
                                user_mapping[entity_col_dd.value] = "clean_entity_mention"
                            if start_col_dd.value != "None":
                                user_mapping[start_col_dd.value] = "mention_start"
                            if end_col_dd.value != "None":
                                user_mapping[end_col_dd.value] = "mention_end"

                            print(f"[INFO] Applying mapping on '{path}'...")
                            df_mapped, final_map = self.map_dataset(path, column_mapping=user_mapping, delimiter=delimiter, verbose=True)
                            print(f"[INFO] Final active column mappings:")
                            for k, v in final_map.items():
                                print(f"  - '{k}' -> '{v}'")
                            print(f"\n[INFO] Sample mapped rows ({len(df_mapped)} total rows):\n")
                            display(df_mapped.head(5))
                    except Exception as ex:
                        with output_area:
                            print(f"[ERROR] Mapping failed: {ex}")
                    finally:
                        is_applying[0] = False

                apply_button.on_click(on_apply_click)

                form_box = widgets.VBox([
                    widgets.HTML("<p><b>Correct auto-detected column mappings if needed:</b></p>"),
                    widgets.HBox([sentence_col_dd, entity_col_dd]),
                    widgets.HBox([start_col_dd, end_col_dd]),
                    apply_button,
                ])
                mapping_container.children = [form_box]

                with output_area:
                    output_area.clear_output()
                    print(f"[INFO] Loaded header for '{path}'. Auto-detected mappings pre-selected below.")

            except Exception as ex:
                with output_area:
                    output_area.clear_output()
                    print(f"[ERROR] Could not read headers from '{path}': {ex}")

        def on_dataset_change(change):
            if change["type"] == "change" and change["name"] == "value":
                if change["new"] != last_path[0]:
                    inspect_and_build_mapping_form(change["new"], force=True)

        dataset_dropdown.observe(on_dataset_change)

        inspect_button = widgets.Button(description="Inspect columns", button_style="info")
        inspect_button.on_click(lambda _b: inspect_and_build_mapping_form(dataset_dropdown.value, force=True))

        inspect_and_build_mapping_form(dataset_dropdown.value)

        ui_box = widgets.VBox([
            widgets.HTML("<h3>Stage 1: Dataset Inspection & Schema Mapping</h3>"),
            widgets.HBox([dataset_dropdown, inspect_button]),
            mapping_container,
            output_area,
        ])
        display(ui_box)

    def get_generator_notebook_ui(self) -> None:
        """Create and display index generator UI with full dataset/cache pair CRUD capabilities."""
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError:
            print("[WARN] ipywidgets is not installed. Please install ipywidgets to use the generator Jupyter UI.")
            return

        pairs = self.list_pairs()
        pair_options = [(f"{meta.get('name', pid)} ({pid})", pid) for pid, meta in pairs.items()]
        if not pair_options:
            pair_options = [("Default pair (default)", "default")]

        active_pair_id = self.active_pair_id if self.active_pair_id in pairs else pair_options[0][1]

        pair_dropdown = widgets.Dropdown(
            options=pair_options,
            value=active_pair_id,
            description="Active pair:",
            disabled=False,
        )

        pair_info_html = widgets.HTML(value="")

        def format_pair_info(pid: str) -> str:
            if not pid:
                return "<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px; background: #f9f9f9; margin-bottom: 10px; color: #777;'>No dataset/cache pairs available.</div>"
            meta = self.list_pairs().get(pid, {})
            if not meta:
                return "<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px; background: #f9f9f9; margin-bottom: 10px; color: #777;'>Pair details not available.</div>"
            c_dir = Path(meta.get("cache_dir", f"cache/{pid}"))
            is_indexed = (c_dir / "metadata.pkl").exists() and (c_dir / "embeddings_combined.npy").exists()
            status_text = "<span style='color: green; font-weight: bold;'>Indexed</span>" if is_indexed else "<span style='color: red; font-weight: bold;'>Not Indexed</span>"
            return (
                f"<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px; background: #f9f9f9; margin-bottom: 10px;'>"
                f"<b>Pair ID:</b> {pid}<br>"
                f"<b>Name:</b> {meta.get('name', pid)}<br>"
                f"<b>Dataset path:</b> {meta.get('dataset_path', 'N/A')}<br>"
                f"<b>Vector model:</b> {meta.get('model_name', 'N/A')}<br>"
                f"<b>Clustering method:</b> {meta.get('clustering_method', 'N/A')}<br>"
                f"<b>Last updated:</b> {meta.get('last_updated', 'N/A')}<br>"
                f"<b>Cache status:</b> {status_text} (Folder: <code>{c_dir}</code>)"
                f"</div>"
            )

        pair_info_html.value = format_pair_info(active_pair_id)

        def refresh_pair_dropdown(selected_id: Optional[str] = None):
            all_p = self.list_pairs()
            opts = [(f"{meta.get('name', pid)} ({pid})", pid) for pid, meta in all_p.items()]
            if not opts:
                opts = [("No pairs available", "")]
            pair_dropdown.options = opts
            if selected_id and selected_id in all_p:
                target = selected_id
            elif self.active_pair_id and self.active_pair_id in all_p:
                target = self.active_pair_id
            else:
                target = opts[0][1] if opts else ""
            pair_dropdown.value = target
            pair_info_html.value = format_pair_info(target)

        def on_pair_dropdown_change(change):
            if change["type"] == "change" and change["name"] == "value":
                pid = change["new"]
                pair_info_html.value = format_pair_info(pid)

        pair_dropdown.observe(on_pair_dropdown_change)

        select_button = widgets.Button(description="Select active pair", button_style="info")
        update_button = widgets.Button(description="Force update pair", button_style="warning")
        delete_button = widgets.Button(description="Delete pair", button_style="danger")

        pair_id_input = widgets.Text(
            value="",
            placeholder="e.g. news_dataset_2026",
            description="Pair ID:",
            disabled=False,
        )
        pair_name_input = widgets.Text(
            value="",
            placeholder="e.g. Multilingual News Corpus 2026",
            description="Pair name:",
            disabled=False,
        )
        dataset_path_input = widgets.Text(
            value=self.config.get("dataset_path", "data/train_language_aware.csv"),
            placeholder="Path to CSV dataset...",
            description="Dataset path:",
            disabled=False,
        )

        current_model = self.model_name if self.model_name in self.available_models else self.available_models[0]
        model_selector = widgets.Dropdown(
            options=self.available_models,
            value=current_model,
            description="Vector model:",
            disabled=False,
        )

        model_descriptions = {
            "sentence-transformers/LaBSE": (
                "<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px; background: #f9f9f9;'>"
                "<b>Language-Agnostic BERT Sentence Embedding (LaBSE)</b><br>"
                "Produces 768-dimensional embeddings tuned for 109+ languages. Highly effective for cross-lingual entity mention mapping."
                "</div>"
            ),
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": (
                "<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px; background: #f9f9f9;'>"
                "<b>Paraphrase Multilingual MPNet Base v2</b><br>"
                "Produces 768-dimensional embeddings mapped across 50+ languages. Excellent balance of semantic representation and multilingual accuracy."
                "</div>"
            ),
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": (
                "<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px; background: #f9f9f9;'>"
                "<b>Paraphrase Multilingual MiniLM L12 v2</b><br>"
                "Produces 384-dimensional compact embeddings for 50+ languages. Optimized for fast inference speed and reduced memory usage."
                "</div>"
            ),
            "sentence-transformers/all-mpnet-base-v2": (
                "<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px; background: #f9f9f9;'>"
                "<b>All MPNet Base v2</b><br>"
                "Produces 768-dimensional high-quality sentence embeddings. Highly accurate for monolingual English corpora."
                "</div>"
            ),
        }

        default_model_desc = model_descriptions.get(
            model_selector.value,
            f"<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px; background: #f9f9f9;'><b>{model_selector.value}</b><br>Transformer model for vector encoding.</div>"
        )
        model_desc_html = widgets.HTML(value=default_model_desc)

        def on_model_change(change):
            if change["type"] == "change" and change["name"] == "value":
                model_desc_html.value = model_descriptions.get(
                    change["new"],
                    f"<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px; background: #f9f9f9;'><b>{change['new']}</b><br>Transformer model for vector encoding.</div>"
                )

        model_selector.observe(on_model_change)

        method_options = [
            ("FAISS top-k mutual graph (Self-determined K | Large datasets 50k - 5M+)", "faiss_topk"),
            ("FAISS leader-follower canopy (Self-determined K | Massive datasets 100k - 5M+)", "faiss_leader"),
            ("FAISS k-means GPU (Fixed target K = N // 10 | Massive datasets 1M+)", "faiss_kmeans"),
            ("Agglomerative average-linkage (Self-determined K | Small datasets < 50k)", "agglomerative"),
        ]

        method_selector = widgets.Dropdown(
            options=method_options,
            value="faiss_topk",
            description="Clustering method:",
            disabled=False,
        )

        method_descriptions = {
            "faiss_topk": (
                "<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px; background: #f9f9f9;'>"
                "<b>FAISS top-k mutual graph clustering</b><br>"
                "<b>Cluster Count:</b> <span style='color: green; font-weight: bold;'>Self-determined algorithmically</span> via similarity threshold cutoff and graph component connectivity.<br>"
                "<b>Scale:</b> Optimized for large and massive datasets between 50,000 and 5,000,000+ rows on GPU."
                "</div>"
            ),
            "faiss_leader": (
                "<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px; background: #f9f9f9;'>"
                "<b>FAISS leader-follower canopy clustering</b><br>"
                "<b>Cluster Count:</b> <span style='color: green; font-weight: bold;'>Self-determined algorithmically</span> via cosine similarity threshold (items within threshold match existing leader; otherwise create a new leader).<br>"
                "<b>Scale:</b> Extremely fast O(N) GPU index search for massive datasets between 100,000 and 5,000,000+ rows."
                "</div>"
            ),
            "faiss_kmeans": (
                "<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px; background: #f9f9f9;'>"
                "<b>FAISS k-means GPU</b><br>"
                "<b>Cluster Count:</b> <span style='color: #d9534f; font-weight: bold;'>Fixed target K set by MERCURY logic</span> (<code>num_samples // 10</code>, capped at 50,000 clusters).<br>"
                "<b>Scale:</b> Fast GPU centroid partitioning for massive datasets above 1,000,000+ rows."
                "</div>"
            ),
            "agglomerative": (
                "<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px; background: #f9f9f9;'>"
                "<b>Agglomerative average-linkage</b><br>"
                "<b>Cluster Count:</b> <span style='color: green; font-weight: bold;'>Self-determined algorithmically</span> via distance threshold.<br>"
                "<b>Scale:</b> Best for small to medium datasets between 1 and 50,000 rows."
                "</div>"
            ),
        }

        method_desc_html = widgets.HTML(value=method_descriptions["faiss_topk"])

        def on_method_change(change):
            if change["type"] == "change" and change["name"] == "value":
                method_desc_html.value = method_descriptions.get(change["new"], "")

        method_selector.observe(on_method_change)

        alpha_slider = widgets.FloatSlider(
            value=self.alpha_span_weight,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Entity weight:",
            readout_format=".2f",
            disabled=False,
            layout=widgets.Layout(width="50%"),
        )

        alpha_info_html = widgets.HTML()

        def update_alpha_info(val: float):
            ent_pct = int(round(val * 100))
            ctx_pct = int(round((1.0 - val) * 100))
            alpha_info_html.value = (
                f"<div style='margin-left: 10px; font-size: 13px; color: #444; margin-top: 5px;'>"
                f"<b>Entity mention weight:</b> {ent_pct}% &nbsp;|&nbsp; "
                f"<b>Sentence context weight:</b> {ctx_pct}%"
                f"</div>"
            )

        update_alpha_info(alpha_slider.value)

        def on_alpha_change(change):
            if change["type"] == "change" and change["name"] == "value":
                update_alpha_info(change["new"])

        alpha_slider.observe(on_alpha_change)

        create_button = widgets.Button(description="Create and index pair", button_style="primary")
        cli_command_button = widgets.Button(description="get CLI command", button_style="")
        output_area = widgets.Output()

        def on_select_click(_b=None):
            with output_area:
                output_area.clear_output()
                pid = pair_dropdown.value
                if not pid:
                    print("[WARN] No pair selected.")
                    return
                self.select_pair(pid)
                refresh_pair_dropdown(pid)

        def on_update_click(_b=None):
            with output_area:
                output_area.clear_output()
                pid = pair_dropdown.value
                if not pid:
                    print("[WARN] No pair selected to update.")
                    return
                print(f"[INFO] Force updating index for pair '{pid}'...\n")
                self.alpha_span_weight = alpha_slider.value
                if self.update_pair(pid, force_update=True):
                    refresh_pair_dropdown(pid)
                    print(f"\n[INFO] Force update for pair '{pid}' completed successfully.")

        def on_delete_click(_b=None):
            with output_area:
                output_area.clear_output()
                pid = pair_dropdown.value
                if not pid:
                    print("[WARN] No pair selected to delete.")
                    return
                print(f"[INFO] Deleting pair '{pid}'...\n")
                if self.delete_pair(pid):
                    refresh_pair_dropdown()
                    print(f"[INFO] Pair '{pid}' deleted.")

        def on_create_click(_b=None):
            with output_area:
                output_area.clear_output()
                pid = pair_id_input.value.strip()
                pname = pair_name_input.value.strip()
                path = dataset_path_input.value.strip()
                selected_model = model_selector.value
                selected_method = method_selector.value
                selected_alpha = alpha_slider.value

                if not pid:
                    print("[ERROR] Pair ID is required.")
                    return

                self.alpha_span_weight = selected_alpha
                print(f"[INFO] Creating pair '{pid}' (Entity weight α={selected_alpha:.2f}) using model '{selected_model}' and method '{selected_method}' on '{path}'...\n")
                try:
                    if self.create_pair(
                        pair_id=pid,
                        dataset_path=path,
                        name=pname or pid,
                        model_name=selected_model,
                        clustering_method=selected_method,
                        force_update=True,
                    ):
                        refresh_pair_dropdown(pid)
                        print("\n[INFO] Pair creation and indexing completed successfully.")
                except Exception as ex:
                    print(f"\n[ERROR] Pair creation failed: {ex}")

        def on_cli_command_click(_b=None):
            with output_area:
                output_area.clear_output()
                pid = pair_id_input.value.strip() or (pair_dropdown.value if pair_dropdown.value else "")
                path = dataset_path_input.value.strip()
                selected_model = model_selector.value
                selected_method = method_selector.value
                selected_alpha = alpha_slider.value

                cmd_parts = ["python MERCURY_Generator.py"]
                if pid:
                    cmd_parts.append(f'--pair "{pid}"')
                if path:
                    cmd_parts.append(f'--dataset "{path}"')
                if selected_model:
                    cmd_parts.append(f'--model-name "{selected_model}"')
                if selected_method:
                    cmd_parts.append(f'--clustering-method "{selected_method}"')
                if selected_alpha is not None:
                    cmd_parts.append(f'--alpha-span-weight {selected_alpha:.2f}')
                if pair_id_input.value.strip():
                    cmd_parts.append("--create-pair")
                else:
                    cmd_parts.append("--force-update")

                cli_cmd = " ".join(cmd_parts)
                print("[INFO] Equivalent CLI command based on current UI settings:\n")
                print(f"  {cli_cmd}\n")

        select_button.on_click(on_select_click)
        update_button.on_click(on_update_click)
        delete_button.on_click(on_delete_click)
        create_button.on_click(on_create_click)
        cli_command_button.on_click(on_cli_command_click)

        ui_box = widgets.VBox([
            widgets.HTML("<h3>Manage dataset/cache pairs</h3>"),
            widgets.HBox([pair_dropdown, select_button, update_button, delete_button]),
            pair_info_html,
            widgets.HTML("<hr><h3>Create new dataset/cache pair</h3>"),
            widgets.HBox([pair_id_input, pair_name_input]),
            dataset_path_input,
            widgets.VBox([model_selector, model_desc_html]),
            widgets.VBox([method_selector, method_desc_html]),
            widgets.VBox([widgets.HTML("<b>Entity vs Sentence Weight (α):</b>"), alpha_slider, alpha_info_html]),
            widgets.HBox([create_button, cli_command_button]),
            output_area,
        ])
        display(ui_box)
