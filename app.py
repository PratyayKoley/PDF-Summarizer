from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
import os
import shutil
import traceback
from inference import process_pdf_and_summarize

app = FastAPI(
    title="Academic PDF Summarization API",
    description="Extract coherent summaries from academic papers and long documents",
    version="2.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {
        "message": "Academic PDF Summarization API v2.0",
        "status": "running",
        "endpoints": {
            "POST /upload": "Upload PDF and get extractive summary",
            "GET /health": "Health check"
        },
        "features": [
            "Keyphrase-weighted embeddings",
            "Smart sentence-based chunking",
            "Multi-factor sentence scoring",
            "MMR-based diversity selection",
            "Academic paper optimized"
        ]
    }

@app.get("/health")
def health_check():
    import torch
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    summary_sentences: int = 5
):
    """
    Upload a PDF and receive an extractive summary
    
    Parameters:
    - file: PDF document (max 50MB)
    - summary_sentences: Number of sentences (3-10, default: 5)
    
    Returns:
    - summary: Coherent extractive summary
    - keyphrases: Key concepts from document
    - stats: Processing statistics
    """
    
    # Validate file
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400, 
            detail="Only PDF files accepted. Please upload a .pdf file."
        )
    
    # Validate summary length
    if not 3 <= summary_sentences <= 10:
        raise HTTPException(
            status_code=400,
            detail="summary_sentences must be between 3 and 10"
        )
    
    # Check file size (50MB limit)
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 50MB."
        )
    
    job_id = str(uuid.uuid4())
    path = os.path.join(UPLOAD_DIR, f"{job_id}.pdf")
    
    try:
        # Save file
        print(f"Processing job {job_id}: {file.filename}")
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process
        result = process_pdf_and_summarize(path, summary_sentences=summary_sentences)
        
        # Validate output
        if not result.get("summary") or len(result["summary"]) < 50:
            raise ValueError("Generated summary is too short or empty")
        
        # Clean up
        if os.path.exists(path):
            os.remove(path)
        
        print(f"✅ Job {job_id} completed successfully")
        
        return {
            "job_id": job_id,
            "filename": file.filename,
            "summary": result["summary"],
            "keyphrases": result["keyphrases"],
            "stats": {
                "num_sentences": result["num_sentences"],
                "num_chunks": result["num_chunks"],
                "summary_length": summary_sentences,
                "file_size_kb": round(file_size / 1024, 2)
            }
        }
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(path):
            os.remove(path)
        
        error_msg = str(e)
        error_trace = traceback.format_exc()
        
        print(f"❌ Error in job {job_id}: {error_msg}")
        print(error_trace)
        
        # Return helpful error messages
        if "PDF" in error_msg or "extract" in error_msg.lower():
            detail = "Failed to extract text from PDF. The file may be corrupted or image-based."
        elif "memory" in error_msg.lower() or "CUDA" in error_msg:
            detail = "Insufficient memory to process this document. Try a smaller file."
        else:
            detail = f"Processing error: {error_msg}"
        
        raise HTTPException(status_code=500, detail=detail)

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Academic PDF Summarization API...")
    print("📍 Server will be available at: http://127.0.0.1:8000")
    print("📖 API docs at: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)

