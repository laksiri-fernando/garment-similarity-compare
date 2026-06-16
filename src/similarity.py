import os
import torch
import faiss
import pickle
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

class GarmentMatcher:
    def __init__(self, model_name="facebook/dinov2-large", index_path="index/index.faiss", meta_path="index/index_meta.pkl"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing GarmentMatcher on {self.device}...")
        
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.index_path = index_path
        self.meta_path = meta_path
        
        self.index = None
        self.image_paths = []
        
        # Load existing index if available
        if os.path.exists(index_path) and os.path.exists(meta_path):
            self.load_index()
        else:
            print("No existing index found. Starting fresh.")

    def _compute_embedding(self, image_path):
        """Computes normalized embedding for a single image."""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # DINOv2 uses the [CLS] token (index 0)
                emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
            # Normalize for Cosine Similarity (Inner Product)
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / (norm + 1e-10)
            return emb
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def index_folder(self, folder_path):
        """Indexes all images in a folder and saves the index."""
        valid_exts = {".png", ".jpg", ".jpeg", ".webp"}
        files_to_process = []
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in valid_exts:
                    files_to_process.append(os.path.join(root, file))
        
        if not files_to_process:
            print("No images found to index.")
            return

        print(f"Found {len(files_to_process)} images. Computing embeddings...")
        
        embeddings = []
        valid_paths = []
        
        for path in tqdm(files_to_process):
            emb = self._compute_embedding(path)
            if emb is not None:
                embeddings.append(emb[0])
                valid_paths.append(path)
        
        if not embeddings:
            print("Failed to compute any embeddings.")
            return
            
        embeddings_np = np.array(embeddings).astype('float32')
        
        # Dimension of DINOv2-large is 1024
        d = embeddings_np.shape[1]
        
        # Create FAISS Index (Inner Product for Cosine Similarity)
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings_np)
        self.image_paths = valid_paths
        
        self.save_index()
        print(f"Indexed {len(valid_paths)} images successfully.")

    def save_index(self):
        """Saves FAISS index and metadata to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.image_paths, f)
        print("Index saved to disk.")

    def load_index(self):
        """Loads FAISS index and metadata from disk."""
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.image_paths = pickle.load(f)
            print(f"Loaded index with {self.index.ntotal} vectors.")
        except Exception as e:
            print(f"Failed to load index: {e}")

    def search(self, query_image_path, top_k=5):
        """Searches for similar images."""
        if not self.index or self.index.ntotal == 0:
            print("Index is empty.")
            return []
            
        query_emb = self._compute_embedding(query_image_path)
        if query_emb is None:
            return []
            
        distances, indices = self.index.search(query_emb.astype('float32'), top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.image_paths):
                results.append({
                    "path": self.image_paths[idx],
                    "filename": os.path.basename(self.image_paths[idx]),
                    "score": float(distances[0][i])
                })
                
        return results

if __name__ == "__main__":
    # Test run
    matcher = GarmentMatcher()
    # Uncomment to test manually
    # matcher.index_folder("samples")
    # print(matcher.search("samples/image001.png"))
