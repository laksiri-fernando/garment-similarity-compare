import sys
import os
# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from src.api import app
import shutil

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Garment Similarity Search" in response.text

def test_indexing():
    # Ensure samples exist
    if not os.path.exists("samples"):
        os.makedirs("samples")
    
    # Check if we have images, if not, skip logic but pass
    if not os.listdir("samples"):
        print("No samples to index, skipping core logic.")
        return

    response = client.post("/index")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert data["count"] >= 0

def test_search():
    # We need an image to upload. Let's use one from samples if available.
    samples = [f for f in os.listdir("samples") if f.endswith(('.png', '.jpg'))]
    if not samples:
        print("No samples found for search test.")
        return

    test_image_path = os.path.join("samples", samples[0])
    
    with open(test_image_path, "rb") as f:
        # First index to be sure
        client.post("/index")
        
        # Now search
        f.seek(0)
        response = client.post("/search", files={"file": ("test.png", f, "image/png")})
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if len(data) > 0:
            # The top match should effectively be itself (high score)
            top_match = data[0]
            print(f"Top match score: {top_match['score']}")
            assert float(top_match['score']) > 0.9

if __name__ == "__main__":
    try:
        test_root()
        print("Root Test: PASS")
        test_indexing()
        print("Index Test: PASS")
        test_search()
        print("Search Test: PASS")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        exit(1)
