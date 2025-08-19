# LearnADo

AI-powered learning assistant for document processing and analysis.

## 🎯 Phase 1: Setup & Preprocessing Layer

**Goal**: Convert all types of user input (text, PDF, image, audio) into machine-readable text for further AI processing.

### Current Focus
This project is currently in **Phase 1** development, focusing on building a robust preprocessing layer that can handle multiple input formats and convert them into clean, machine-readable text.

## 🚀 Features (Phase 1)

### Input Processing Capabilities
- **Text Input** → Direct text processing and validation
- **PDF Processing** → Text extraction using pdfplumber and PyMuPDF
- **Image OCR** → Text extraction from images using pytesseract
- **Audio Transcription** → Speech-to-Text conversion using OpenAI Whisper
- **Language Translation** → Multi-language support using HuggingFace MarianMT

### Core Functionality
- **Unified API** → Single endpoint to handle all input types
- **Text Extraction** → Clean, structured text output from any input
- **Language Detection** → Automatic language identification
- **Translation Pipeline** → Convert content between languages
- **File Management** → Secure upload, processing, and storage

## 🏗️ Project Structure

```
LearnADo/
├── main.py                  # FastAPI entrypoint
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
│
├── app/
│   ├── __init__.py         # Package initialization
│   ├── database.py         # Database setup and configuration
│   ├── models.py           # SQLAlchemy ORM models
│   ├── schemas.py          # Pydantic request/response schemas
│   ├── routes.py           # API endpoints and routing
│   ├── services.py         # Preprocessing logic (PDF, image, audio, translation)
│   └── utils.py            # Helper functions (OCR, STT, translation)
│
├── data/
│   ├── uploads/            # Uploaded raw files
│   └── processed/          # Extracted text outputs
│
└── tests/
    └── test_app.py         # Unit tests
```

## 🛠️ Technology Stack (Phase 1)

### Backend Framework
- **FastAPI** - Modern, fast web framework for APIs
- **SQLAlchemy** - Database ORM and integration
- **Pydantic** - Data validation and serialization

### Text Extraction & Processing
- **pdfplumber** - PDF text extraction
- **PyMuPDF** - Advanced PDF processing
- **pytesseract** - OCR for image text extraction
- **Pillow** - Image processing and manipulation

### Speech-to-Text
- **OpenAI Whisper** - High-quality audio transcription
- **HuggingFace Whisper** - Alternative STT implementation

### Translation & NLP
- **HuggingFace Transformers** - MarianMT for translation
- **Sentence Transformers** - Text embeddings and similarity
- **LangChain** - AI/ML pipeline orchestration

### Database & Storage
- **SQLite** - Lightweight database for development
- **File System** - Local file storage for uploads and outputs

## 📋 API Endpoints (Phase 1)

### File Upload & Processing
- `POST /api/v1/upload` - Upload files (text, PDF, image, audio)
- `POST /api/v1/process/{upload_id}` - Start text extraction
- `GET /api/v1/process/{upload_id}/status` - Check processing status
- `GET /api/v1/process/{upload_id}/result` - Get extracted text

### File Management
- `GET /api/v1/uploads` - List all uploaded files
- `GET /api/v1/uploads/{id}` - Get specific upload details
- `DELETE /api/v1/uploads/{id}` - Delete an upload

### Translation
- `POST /api/v1/translate` - Translate extracted text
- `GET /api/v1/languages` - Get supported languages

### Utilities
- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

## 🚀 Setup Instructions

### Prerequisites

- Python 3.12+
- uv (Python package manager)
- Tesseract OCR (for image processing)
- FFmpeg (for audio processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd LearnADo
   ```

2. **Create virtual environment**
   ```bash
   uv venv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Install system dependencies**
   ```bash
   # For Tesseract OCR (Ubuntu/Debian)
   sudo apt-get install tesseract-ocr
   
   # For FFmpeg (Ubuntu/Debian)
   sudo apt-get install ffmpeg
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000`

## 📊 Phase 1 Milestones

### ✅ Completed
- [x] Project environment setup (virtualenv, FastAPI backend)
- [x] Database schema design
- [x] Project structure and file organization
- [x] Dependencies installation and configuration

### 🔄 In Progress
- [ ] Text input processing implementation
- [ ] PDF processing with pdfplumber/PyMuPDF
- [ ] Image OCR with pytesseract
- [ ] Audio STT with Whisper
- [ ] Translation with MarianMT

### 📋 Upcoming
- [ ] API endpoint implementation
- [ ] Error handling and validation
- [ ] File upload and storage
- [ ] Processing pipeline integration
- [ ] Unit tests and documentation

## 🎯 Deliverables (Phase 1)

### Backend APIs
- **Unified Upload API** - Accept text, image, PDF, audio files
- **Text Extraction API** - Return clean, extracted text from any input
- **Translation API** - Convert text between supported languages
- **Status API** - Monitor processing progress

### Processing Capabilities
- **Text Input** → Direct processing and validation
- **PDF Files** → Text extraction with layout preservation
- **Images** → OCR text extraction with confidence scoring
- **Audio Files** → High-quality transcription with timestamps
- **Multi-language** → Automatic language detection and translation

## 🔧 Development

### Running Tests
```bash
pytest tests/
```

### Code Style
The project follows PEP 8 standards. Use a linter like `flake8` or `black` for code formatting.

### API Documentation
Once running, visit `http://localhost:8000/docs` for interactive API documentation.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

[Add your license here]

## 🆘 Support

For support and questions, please open an issue on GitHub.

## 🔮 Future Phases

### Phase 2: AI Analysis Layer
- Text summarization and key point extraction
- Content classification and tagging
- Question-answering capabilities
- Knowledge graph generation

### Phase 3: Learning Enhancement
- Personalized learning recommendations
- Interactive study materials
- Progress tracking and analytics
- Collaborative learning features
