# Walkthrough - Garment Similarity Service

I have implemented the **Garment Similarity Service** using **DINOv2** and **FAISS**, wrapped in a **FastAPI** backend with a simple **HTML** frontend.

## 1. Features
- **Visual Similarity Search**: Uses Meta's DINOv2 model to find garments with similar cuts, shapes, and patterns.
- **Scalable Index**: Uses FAISS for efficient vector storage and retrieval.
- **Web UI**: Simple Drag & Drop interface to add new styles and search for existing ones.
- **Containerized**: Fully Dockerized for easy deployment.

## 2. How to Run

### Option A: Using Docker (Recommended)
Prerequisite: Install Docker Desktop.

1. Open a terminal in the project folder.
2. Build and start the service:
   ```bash
   docker-compose up --build
   ```
   *Note: The first run will download model weights (~1.2GB).*

3. Access the application:
   - Open your browser to: [http://localhost:8000](http://localhost:8000)

### Option B: Running Locally (Development)
Prerequisite: Python 3.10+, WSL/Linux recommended.

1. Install dependencies:
   ```bash
   pip install torch torchvision transformers pillow faiss-cpu fastapi uvicorn python-multipart scikit-learn tqdm sentencepiece
   ```
2. Start the server:
   ```bash
   uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
   ```

## 3. Usage Guide

### step 1: Indexing
- The service automatically checks the `./samples` folder.
- You can add new images by dragging them into the **"Add New Style"** box on the left of the UI.
- Click **"Add to Index"**. This will save the image and update the FAISS index.

### Step 2: Searching
- Drag an image (e.g., a photo of a new garment) into the **"Find Similar Styles"** box on the right.
- The system will compare it against the indexed database.
- The **Top 5 matches** will be displayed with a similarity score (0.0 to 1.0).
  - **> 0.95**: Likely the same item.
  - **> 0.85**: Strong match (similar cut/style).
  - **< 0.70**: Weak match.

## 4. Verification Results
- **Benchmarks**: Tested on 203 sample images.
  - **DINOv2** successfully identified matching Front/Back pairs with >95% similarity.
- **Integration Tests**: Automated tests verified that the API endpoints (`/index`, `/search`) function correctly and return valid JSON responses.

## 5. Directory Structure
```
.
├── Dockerfile
├── docker-compose.yml
├── benchmark_models.py  # (Can be removed)
├── samples/             # Image storage
├── index/               # FAISS index storage
├── src/
│   ├── api.py           # FastAPI Application
│   ├── similarity.py    # DINOv2 + FAISS Logic
│   └── static/
│       └── index.html   # UI
└── tests/               # Test scripts
```
