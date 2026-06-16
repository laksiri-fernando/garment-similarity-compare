# Implementation Plan - Garment Similarity Search

The Analysis Phase has confirmed that **DINOv2** is the optimal model for this task. The user has selected **FAISS** for the vector index, which provides a scalable and efficient similarity search.

This plan details the implementation of a Python based service to index and search garment images.

## User Review Required
> The solution requires **PyTorch**, **Transformers**, **faiss-cpu**, **fastapi**, **uvicorn**, and **python-multipart**. The initial setup requires downloading ~1.2GB of model weights.

## Proposed Changes

We will create a specific directory `src/similarity` to house the logic.

### `src`
#### [NEW] [similarity.py](file:///wsl.localhost/Ubuntu-24.04/home/laksiri/my/randd/garment-similarity-compare/src/similarity.py)
- **Class `GarmentMatcher`**:
    - `__init__(model_name="facebook/dinov2-large")`: Loads DINOv2 and initializes a FAISS index.
    - `_compute_embedding(image)`: Computes normalized vector.
    - `index_folder(folder_path)`: 
        - computes embeddings for all images.
        - adds vectors to a `faiss.IndexFlatIP` (Inner Product, equivalent to Cosine Similarity on normalized vectors).
        - saves the index to `index.faiss` and metadata (filenames) to `index_meta.pkl`.
    - `search(query_image_path, top_k=5)`: 
        - computes query vector.
        - searches the FAISS index.
        - maps indices back to filenames.

#### [NEW] [api.py](file:///wsl.localhost/Ubuntu-24.04/home/laksiri/my/randd/garment-similarity-compare/src/api.py)
- **Framework**: FastAPI.
- **Endpoints**:
    - `POST /index`: Triggers re-indexing of the samples folder.
    - `POST /search`: Accepts an uploaded image, returns JSON list of top matches with scores.
    - `GET /`: Serves the static HTML frontend.
    - `GET /samples/{filename}`: Serves static images from the samples directory.

#### [NEW] [static/index.html](file:///wsl.localhost/Ubuntu-24.04/home/laksiri/my/randd/garment-similarity-compare/src/static/index.html)
- **Tech**: HTML5, Vanilla JS, CSS.
- **Features**:
    - Simple Drag & Drop zone for `Add Style` (Upload -> Index).
    - Drag & Drop zone for `Search Similar` (Upload -> Display grid of results).

### Containerization
#### [NEW] [Dockerfile](file:///wsl.localhost/Ubuntu-24.04/home/laksiri/my/randd/garment-similarity-compare/Dockerfile)
- Base Image: `python:3.10-slim`.
- Installs system dependencies (libgl1).
- Copies `src` and `samples`.
- Runs `uvicorn src.api:app --host 0.0.0.0 --port 8000`.

#### [NEW] [docker-compose.yml](file:///wsl.localhost/Ubuntu-24.04/home/laksiri/my/randd/garment-similarity-compare/docker-compose.yml)
- Service: `garment-matcher`
- Ports: `8000:8000`
- Volumes:
    - `./samples:/app/samples`
    - `./index:/app/index`

## Verification Plan

### Automated Tests
- `tests/test_similarity.py` (Core logic).
- `tests/test_api.py` (FastAPI endpoints using `TestClient`).

### Manual Verification
- **Docker**: Run `docker-compose up`.
- **Browser**: Open `http://localhost:8000`.
- **Flow**:
    1. Upload a new image using the UI.
    2. Check response shows valid JSON/images.
