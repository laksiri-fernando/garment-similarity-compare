import os
import torch
import random
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel, AutoProcessor
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Configuration
SAMPLE_DIR = "./samples"
TOP_K = 5
NUM_QUERIES = 5  # Increased to get a better sense

# Models to Benchmark
MODELS = {
    "DINOv2": {
        "name": "facebook/dinov2-large",
        "type": "dinov2"
    },
    "SigLIP": {
        "name": "google/siglip-so400m-patch14-384",
        "type": "siglip"
    },
    "CLIP": {
        "name": "openai/clip-vit-large-patch14",
        "type": "clip"
    }
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def load_images(directory):
    image_paths = []
    valid_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)

def get_model_and_processor(model_key):
    config = MODELS[model_key]
    print(f"Loading {model_key} ({config['name']})...")
    
    try:
        if config['type'] == 'dinov2':
            processor = AutoImageProcessor.from_pretrained(config['name'])
            model = AutoModel.from_pretrained(config['name']).to(device)
        elif config['type'] == 'siglip':
            processor = AutoProcessor.from_pretrained(config['name'])
            model = AutoModel.from_pretrained(config['name']).to(device)
        elif config['type'] == 'clip':
            processor = CLIPProcessor.from_pretrained(config['name'])
            model = CLIPModel.from_pretrained(config['name']).to(device)
            
        return processor, model
    except Exception as e:
        print(f"Failed to load {model_key}: {e}")
        return None, None

def compute_embeddings(model_key, image_paths):
    processor, model = get_model_and_processor(model_key)
    if not model:
        return None

    embeddings = []
    
    print(f"Computing embeddings for {model_key}...")
    model.eval()
    
    with torch.no_grad():
        for path in tqdm(image_paths):
            try:
                image = Image.open(path).convert("RGB")
                
                if MODELS[model_key]['type'] == 'dinov2':
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    outputs = model(**inputs)
                    # DINOv2: use last_hidden_state class token [CLS]
                    emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    
                elif MODELS[model_key]['type'] == 'siglip':
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    outputs = model.get_image_features(**inputs)
                    emb = outputs.cpu().numpy()
                    
                elif MODELS[model_key]['type'] == 'clip':
                    inputs = processor(images=image, return_tensors="pt").to(device)
                    outputs = model.get_image_features(**inputs)
                    emb = outputs.cpu().numpy()
                
                embeddings.append(emb[0])
            except Exception as e:
                print(f"Error processing {path}: {e}")
                embeddings.append(np.zeros(model.config.hidden_size if hasattr(model.config, 'hidden_size') else 768))

    return np.array(embeddings)

def main():
    image_paths = load_images(SAMPLE_DIR)
    if not image_paths:
        print(f"No images found in {SAMPLE_DIR}")
        return

    print(f"Found {len(image_paths)} images.")
    
    # Pick random queries for demonstration
    # ensuring we have enough images
    num_queries = min(NUM_QUERIES, len(image_paths))
    query_indices = random.sample(range(len(image_paths)), num_queries)
    
    results = {}

    for model_key in MODELS.keys():
        embeddings = compute_embeddings(model_key, image_paths)
        if embeddings is None:
            continue
            
        # Normalize embeddings for cosine similarity
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm[norm == 0] = 1e-10
        embeddings = embeddings / norm
        
        sim_matrix = cosine_similarity(embeddings)
        results[model_key] = sim_matrix

    # Generate Report
    print("\n" + "="*50)
    print("BENCHMARK REPORT")
    print("="*50)
    
    for idx, query_idx in enumerate(query_indices):
        q_path = image_paths[query_idx]
        q_name = os.path.basename(q_path)
        print(f"\nQuery Image: {q_name}")
        
        for model_key in results.keys():
            print(f"  Model: {model_key}")
            sim_scores = results[model_key][query_idx]
            
            # Get top K indices
            top_indices = np.argsort(sim_scores)[::-1]
            
            count = 0
            for k in top_indices:
                if k == query_idx: continue # Skip self
                
                match_name = os.path.basename(image_paths[k])
                score = sim_scores[k]
                print(f"    - {match_name} (Score: {score:.4f})")
                
                count += 1
                if count >= TOP_K: break

if __name__ == "__main__":
    main()
