#  QA RAG Project

Ask questions about a PDF using **text or voice**.  
This project uses a **Retrieval-Augmented Generation (RAG)** pipeline to extract, index, and query document content, with optional OCR and speech support.

Built to run **locally** using a fully open-source stack (no paid APIs).

---

## Features

- 📄 Ask questions about PDFs
- 🔍 RAG-based document retrieval
- 🧠 Local / open-source models
- 🗣️ Voice input (speech-to-text)
- 🖼️ OCR for scanned PDFs

---

## Architecture (High Level)

1. **PDF ingestion**
   - Text extraction (Poppler)
   - OCR for scanned pages (Tesseract)
2. **Chunking & embeddings**
3. **Vector search (ChromaDB)**
4. **LLM answer generation**
5. **Optional voice input**
   - Audio decoding (FFmpeg)
   - Speech-to-text model

---

## System Dependencies (REQUIRED)

These must be installed **before running the app locally**.

| Tool | Purpose |
|----|----|
| **Tesseract OCR** | OCR for scanned PDFs |
| **Poppler** | PDF text extraction |
| **FFmpeg** | Audio processing |
| **Git LFS** | Download large ML models |

---

### Ubuntu / Debian

```bash
sudo apt update && sudo apt install -y \
  tesseract-ocr \
  poppler-utils \
  ffmpeg \
  git-lfs

```

### Install on Ubuntu / Debian
```
brew install tesseract poppler ffmpeg git-lfs
```

## Python Setup
```
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Python Dependencies

Install Python requirements:

```
pip install -r requirements.txt
```

## Model Downloads (Pre-Download Required)

Models are NOT auto-downloaded.
Place your LLM model manually inside the models/ directory.
Example (GGUF model)
```
models/
└── mistral-7b-instruct.Q4_K_M.gguf
```
The app will fail fast if the model is missing.

## Run the App
```
python app.py
```

Open in browser:
```
http://127.0.0.1:7860
```

## Project Structure
```
.
├── app.py
├── requirements.txt
├── README.md
├── models/
│   └── mistral-7b-instruct.Q4_K_M.gguf
├── data/
│   ├── chroma/
│   └── audio/
```


## Notes

- First run is slow due to model downloads (embeddings, Whisper, TTS)
- OCR is slow for large scanned PDFs
- GPU is optional; CPU-only works
- ChromaDB persists automatically on disk

## License

MIT License.
Do whatever you want.

## Acknowledgements

Built using:
- Open-source LLMs (GGUF / llama.cpp)
- Sentence Transformers
- ChromaDB
- Whisper & Coqui TTS
- Unix tools that refuse to die