"""
Utility functions for LearnADo application.
Helper functions for OCR, STT, translation, and common operations.
"""

# TODO: Import required libraries
# import os
# import logging
# from typing import Optional, Dict, Any, List
# from pathlib import Path
# import hashlib
# import mimetypes

import io
import pymupdf
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import whisper
import torch
from pathlib import Path

load_dotenv()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF. Falls back to OCR if page is image-only.
    """
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:
        page_text = page.get_text()
        if page_text.strip():
            text += page_text
        else:
            # Fallback to OCR
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text += pytesseract.image_to_string(img)
    return text

def process_image(file_path: str, user_query: str) -> str:
    """
    Extract text with Tesseract, send both extracted text and image to Gemini.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    try:
        # OCR with Tesseract
        extracted_text = pytesseract.image_to_string(Image.open(file_path))

        # Send both to Gemini (multimodal input: text + image)
        response = llm.invoke([
            {"role": "user", "content": [
                {"type": "text", "text": f"User query: {user_query}\n\nExtracted text (OCR): {extracted_text}"},
                {"type": "image_url", "image_url": f"file://{file_path}"}
            ]}
        ])
        return response.content
    except Exception as e:
        raise RuntimeError(f"Error processing image: {e}")

def query_audio(file_path: str, question: str) -> str:
    """
    Transcribe audio with Whisper, then send transcription + question to Gemini.
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        transcription = model.transcribe(file_path)["text"]
        response = llm.invoke(
            f"User question: {question}\n\nTranscribed audio:\n{transcription}"
        )
        return response.content
    except Exception as e:
        raise RuntimeError(f"Error processing audio: {e}")

def query_pdf(pdf_path: str, question: str) -> str:
    """
    Ask a question about the contents of a PDF.
    """
    content = extract_text_from_pdf(pdf_path)
    res = llm.invoke(question + "\n\n" + content)
    return res.content

# Choose device (CUDA if available, else CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model once globally for efficiency
# You can change model size: "tiny", "small", "base", "medium", "large"
model = whisper.load_model("base", device=DEVICE)

def transcribe_audio(file_path: str) -> str:
    """
    Transcribe audio file to text using Whisper.
    
    Args:
        file_path (str): Path to the audio file
    
    Returns:
        str: Transcribed text
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    result = model.transcribe(file_path)
    return result["text"]



# TODO: Translation Utilities
# - setup_translation_model(): Configure translation model
# - detect_language(): Detect text language
# - validate_translation(): Validate translation quality
# - format_translation_output(): Format translation results

# TODO: File Utilities
# - get_file_extension(): Get file extension
# - validate_file_size(): Check file size limits
# - generate_unique_filename(): Create unique filenames
# - get_mime_type(): Detect file MIME type

# TODO: Text Processing Utilities
# - clean_text(): Clean and normalize text
# - extract_keywords(): Extract important keywords
# - split_text_into_chunks(): Split large texts into manageable chunks
# - merge_text_chunks(): Merge processed text chunks

# TODO: Logging Utilities
# - setup_logging(): Configure application logging
# - log_processing_step(): Log processing steps
# - log_error(): Log errors with context
# - create_processing_report(): Generate processing reports

# TODO: Configuration Utilities
# - load_config(): Load application configuration
# - validate_config(): Validate configuration settings
# - get_model_paths(): Get paths to AI models
# - setup_model_cache(): Setup model caching
