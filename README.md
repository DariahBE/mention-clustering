# MERCURY: Multilingual Entity Retrieval & Clustering Utility

MERCURY is a dataset-agnostic NLP tool for unsupervised entity mention clustering, dual-encoder span vector encoding, and cross-lingual query retrieval. All capabilities are encapsulated in a reusable core file (`MERCURY.py`), supporting a Jupyter Notebook UIs as well as command line execution for large datasets that require multi-hour computes.


## Installation 
- Pull this repository onto a machine that has a CUDA capable GPU. 
- Create a python virtual environment: ```python -m venv .venv ```
- Activate the virtual environment for your system
- Install required modules into your activate virtual environment: ```pip install -r requirements.txt```

## Data preprocessing: 
A preprocessing notebook (`preprocessing.ipynb`) is included in this repository. It takes a raw dump of data with sentences, extracts entities out of these sentences and marks them by type and provides indexes. Each entity needs to be on it's own line in the resulting `prepared_data.tsv` file. Optionally the provenance of each sentence can be provided. 
Valid output examlpe:
```
115724,"Ukraine Facility program In February 2024, the European Parliament approved the regulations of the Ukraine Facility program.",European Parliament,47,66,ORG,eu-council-approves-plan-under-ukraine-facility-1715684593.txt
```

## Features
- **Multi-format dataset support (CSV & TSV)**: Automatically handles datasets with different delimiters (comma, tab) and non-standard column ordering/names (e.g. `data/dataset.csv` vs `data/prepared_data.tsv`).
- **Two-stage generator pipeline**:
  - **Stage 1 (Dataset inspection & schema creation)**: Auto-detects delimiters and maps raw column names to MERCURY's internal standard schema with interactive UI for overriding automatic picks.
  - **Stage 2 (vectorization, indexing & clustering)**: Generates dual-encoder span vectors, applies PCA 128D reduction, builds FAISS indexes, and clusters entity mentions. FAISS is used so later inquiries into a calculate dataset can happen instantly.

- **Two-stage interrogator pipeline**:
  - **Stage 1 (cache selection)**: Allows picking a pre-indexed cache from disk, Once loaded you can write a search query against the index of the dataset.
  - **Stage 2 (querying)**: Interactively queries entity mentions across languages with a two-tab interface (Search results and Co-occurring entities). With a network-view you can see what other concepts are mentioned togehter with the query.
- **Dataset caching system (`cache_catalog.json`)**: Multi-dataset cache tracking file.
- **Interactive notebooks**: Pre-built notebook interfaces for index generation and storage: (`MERCURY_Generator.ipynb`). Entity interrogation (`MERCURY_Interrogator.ipynb`) tool for querying and exploring the dataset.


## Command line interface (CLI) capabilities

### 1. Index generator (`MERCURY_Generator.py`)

```bash
# Create a pair and generate dual-encoder vector indexes
python MERCURY_Generator.py --create-pair --pair <corpus_name> --dataset <path_to_dataset>
```

### 2. Entity interrogator (`MERCURY_Interrogator.py`)

```bash
# Interrogate a specific cached dataset pair with a cross-encoder
python MERCURY_Interrogator.py --pair <corpus_name> --query <"Your search term"> --cross-encoder-model <cross encoder e.g.: cross-encoder/ms-marco-mMiniLM-L-6-v2> --top-k <Numerically limit searhc to x hits >
```
