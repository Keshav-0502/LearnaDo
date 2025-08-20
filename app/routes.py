"""
API routes for LearnADo application.
All endpoints for file upload, processing, and retrieval.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from app.utils import transcribe_audio, query_pdf, process_image, query_audio
from pathlib import Path
import shutil

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

@router.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Upload an audio file and get transcription.
    """
    try:
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Run Whisper transcription
        text = transcribe_audio(str(file_path))
        return {"filename": file.filename, "transcription": text}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- PDF Q&A ---
@router.post("/pdf-query")
async def pdf_query(file: UploadFile = File(...), question: str = Form(...)):
    """
    Upload a PDF and ask a question about it.
    """
    try:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        answer = query_pdf(str(file_path), question)
        return {"filename": file.filename, "question": question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- IMAGE Q&A ---
@router.post("/image-query")
async def image_query(
    file: UploadFile = File(...),
    query: str = Form("Describe this image")
):
    """
    Upload an image → OCR with Tesseract → send text+image to Gemini.
    """
    try:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        response = process_image(str(file_path), query)
        return {"filename": file.filename, "query": query, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- AUDIO Q&A ---
@router.post("/audio-query")
async def audio_query(file: UploadFile = File(...), question: str = Form(...)):
    """
    Upload audio → Whisper transcription → ask Gemini about it.
    """
    try:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        answer = query_audio(str(file_path), question)
        return {"filename": file.filename, "question": question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# TODO: Import local modules
# from app.schemas import *
# from app.services import *
# from app.database import get_db

# Create router
# router = APIRouter()

# TODO: File Upload Endpoints
# - POST /upload: Upload a file (PDF, image, or audio)
# - GET /uploads: List all uploaded files
# - GET /uploads/{upload_id}: Get specific upload details
# - DELETE /uploads/{upload_id}: Delete an upload

# TODO: Processing Endpoints
# - POST /process/{upload_id}: Start processing a file
# - GET /process/{upload_id}/status: Get processing status
# - GET /process/{upload_id}/result: Get processing results

# TODO: File Type Specific Endpoints
# - POST /process/pdf/{upload_id}: Process PDF specifically
# - POST /process/image/{upload_id}: Process image specifically
# - POST /process/audio/{upload_id}: Process audio specifically

# TODO: Utility Endpoints
# - GET /health: Health check
# - GET /files/{file_id}/download: Download processed file
# - GET /logs/{upload_id}: Get processing logs

# TODO: Error handling
# - 400: Bad request (invalid file type, missing parameters)
# - 404: File not found
# - 500: Processing error
# - 413: File too large
