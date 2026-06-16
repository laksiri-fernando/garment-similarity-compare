import os
import shutil
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from src.similarity import GarmentMatcher

app = FastAPI(title="Garment Similarity Service")

# Initialize Logic
# Ensure directories exist
os.makedirs("samples", exist_ok=True)
os.makedirs("temp_uploads", exist_ok=True)

matcher = GarmentMatcher()

# Mount Static Files
app.mount("/static", StaticFiles(directory="src/static"), name="static")
app.mount("/samples-files", StaticFiles(directory="samples"), name="samples")

@app.get("/")
async def read_root():
    return FileResponse('src/static/index.html')

@app.post("/index")
async def trigger_indexing():
    try:
        matcher.index_folder("samples")
        return {"message": "Indexing completed successfully.", "count": len(matcher.image_paths)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-sample")
async def upload_sample(file: UploadFile = File(...)):
    """Uploads a new sample to the library and re-indexes."""
    file_path = f"samples/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Optional: Re-index immediately or let user trigger it
    # For POC, let's re-index immediately for better UX
    matcher.index_folder("samples")
    
    return {"message": f"Uploaded {file.filename} and updated index."}

@app.post("/search")
async def search_similar(file: UploadFile = File(...)):
    temp_path = f"temp_uploads/{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        results = matcher.search(temp_path, top_k=5)
        
        # Format results for frontend
        response_data = []
        for res in results:
            response_data.append({
                "filename": res["filename"],
                "score": f"{res['score']:.4f}",
                "url": f"/samples-files/{res['filename']}"
            })
            
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
