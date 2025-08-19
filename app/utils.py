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

# TODO: OCR Utilities
# - setup_tesseract(): Configure Tesseract OCR
# - preprocess_image_for_ocr(): Prepare image for better OCR results
# - post_process_ocr_text(): Clean and format OCR output
# - validate_ocr_result(): Validate OCR accuracy

import whisper
import torch
from pathlib import Path

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
