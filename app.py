import gradio as gr                                             # For deployment
import pdfplumber                                               # For machine readable pdf extraction
import pytesseract                                              # For scanned pdf extraction (OCR)
from pdf2image import convert_from_path                         # For converting scanned pdf to images for OCR
import shutil                                                   # For OCR paths in Windows
import re                                                       # For text cleaning 
from sentence_transformers import SentenceTransformer           # For Embedding
import chromadb                                                 # Vector Database
from faster_whisper import WhisperModel                         # For Speech -> Text
import numpy as np                                              # For cacheing
from TTS.api import TTS                                         # For Text -> Speech
from llama_cpp import Llama                                     # For base model
from collections import deque                                   # For cache dequeing
import uuid                                                     # For naming database collection
import os                                                       # For validating model path
import threading                                                # For Serializing LLM calls
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

# Setting all the paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# OCR Failsafe
if shutil.which("tesseract") is None:
    raise RuntimeError(
        "Tesseract not found. Install it and add to PATH:\n"
        "Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
        "Linux: sudo apt install tesseract-ocr"
    )

# Loading all models needed and setting variables
embedding = SentenceTransformer("all-mpnet-base-v2", device="cuda" if CUDA_AVAILABLE else "cpu", cache_folder=os.path.join(MODEL_DIR, "embeddings"))
BASE_MODEL = os.path.join(MODEL_DIR, "mistral-7b-instruct.Q4_K_M.gguf")
if not os.path.exists(BASE_MODEL):
    raise FileNotFoundError(f"Missing model file:\n{BASE_MODEL}")
llm = Llama(model_path=BASE_MODEL, n_ctx=4096, n_gpu_layers=-1 if CUDA_AVAILABLE else 0, n_threads=os.cpu_count() or 4, n_batch=512 if CUDA_AVAILABLE else 128)
llm_lock = threading.Lock()
ENABLE_TTS = True
ENABLE_WHISPER = True
whisper = WhisperModel("base", device="cuda" if CUDA_AVAILABLE else "cpu", compute_type="float16" if CUDA_AVAILABLE else "int8") if ENABLE_WHISPER else None
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", gpu=CUDA_AVAILABLE) if ENABLE_TTS else None
SIMILARITY_THRESHOLD = 0.35
CACHE_THRESHOLD = 0.2
MAX_CACHE_SIZE = 750
ERROR_MESSAGE = (
    "Sorry, I couldn't find an answer to the question you asked. "
    "Please ensure the audio you uploaded is clear (if you spoke the question), "
    "or make sure your question is related to the uploaded document."
)   
cache_q = deque(maxlen=MAX_CACHE_SIZE)
cache_a = deque(maxlen=MAX_CACHE_SIZE)
local_client = chromadb.Client(chromadb.config.Settings(persist_directory=CHROMA_DIR, anonymized_telemetry=False))
collection = local_client.get_or_create_collection(name="pdf_qa_collection")

def normalize(vec):
    return vec / (np.linalg.norm(vec) + 1e-8)                   # Function for cosine similarity normalizing 


# Extracting text from PDF
def extract_text(pdf_file) -> str:
    text_chunks = []
    with pdfplumber.open(pdf_file.name) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text and text.strip():
                text_chunks.append(text)
    extracted_text = "\n".join(text_chunks)
    if not extracted_text.strip():
        ocr_text = ocr_pdf(pdf_file.name)
        if ocr_text.strip():
            return ocr_text
    return extracted_text


# For OCR extraction if pdf reading does not work
def ocr_pdf(pdf_path: str) -> str:
    try:
        images = convert_from_path(pdf_path, dpi=300, grayscale=True)
        ocr_text = []

        for img in images:
            text = pytesseract.image_to_string(img)
            if text.strip():
                ocr_text.append(text)

        return "\n".join(ocr_text)
    except Exception as e:
        return ""


# Cleaning texts
def clean_text(text):
    text = text.encode("utf-8", "ignore").decode()                                  # Normalize unicode quotes, dashes, etc.
    text = re.sub(r'(?<![.!?])\n(?!\n)', ' ', text)                                 # Join lines that were split in the middle of sentences.
    text = re.sub(r'\n{2,}', '\n\n', text)                                          # Keep paragraph breaks.
    text = re.sub(r'Page\s*\d+(\s*of\s*\d+)?', '', text, flags=re.IGNORECASE)       # Remove page numbers like "Page 3","3 | Page", etc.
    text = re.sub(r'\b\d+\s*\|\s*Page\b', '', text, flags=re.IGNORECASE)            # - Same as before.
    text = re.sub(r'\n?\s*\d+\s*\n', '\n', text)                                    # Remove standalone numbers (mostly page numbers).
    text = text.lower()                                                             # Make all text lowercased.
    text = re.sub(r'[^\w\s.,;:!?()\-\n/@:%+]', '', text)                            # Keep useful Special Characters and Remove useless ones.
    cleaned_text = re.sub(r'\s+', ' ', text)                                        # Normalize spaces.

    return cleaned_text



# Chunking texts
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def vector_space(chunks, collection):
    existing = collection.get()
    if existing and existing.get("ids"):
        collection.delete(ids=existing["ids"])
    embeddings = embedding.encode(chunks)
    embeddings = [normalize(e).tolist() for e in embeddings]
    collection.add(documents=chunks, embeddings=embeddings, ids=[str(i) for i in range(len(chunks))])
    return collection


# Process pdf
def process_pdf(pdf_file):
    if pdf_file is None:
        return "Please upload a PDF first."
    text = extract_text(pdf_file)
    cleaned_text = clean_text(text)
    if not cleaned_text.strip():
        return "No readable text found in PDF."
    chunks = chunk_text(cleaned_text)
    vector_space(chunks, collection)
    cache_q.clear()
    cache_a.clear()
    return "Document processed successfully"


# Speech -> Text
def transcribe(audio_path):
    try:
        if audio_path is None:
            return ""
        if whisper is None:
            return "Speech disabled"
        segments, info = whisper.transcribe(audio_path, vad_filter=True)
        if info.duration > 30:
            return "Audio too long. Please keep it under 30 seconds"
        text = " ".join(segment.text for segment in segments)
        return text
    except:
        return "Failed to transcribe audio"


# Retrieving chunks
def retrieving_chunks(query, collection):
    query_embedding = normalize(embedding.encode(query)).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3, include=['documents', 'distances'])
    docs = results["documents"][0]
    distances = results["distances"][0]
    if not docs or all(d > SIMILARITY_THRESHOLD for d in distances):
        return None
    return docs


# Text -> Speech
def speak(text):
    if not ENABLE_TTS:
        return None
    output_path = os.path.join(AUDIO_DIR, f"answer_{uuid.uuid4()}.wav")
    tts.tts_to_file(text=text, file_path=output_path)
    return output_path


# Cache adding and lookup
def add_to_cache(query, answer):
    try:
        emb = normalize(embedding.encode(query))
        cache_q.append(emb)
        cache_a.append(answer)
    except Exception as e:
        return ""

def check_cache(query):
    if not cache_q:
        return None
    query_emb = normalize(embedding.encode(query))
    question_matrix = np.vstack(cache_q)

    sims = question_matrix @ query_emb

    best_idx = np.argmax(sims)
    if sims[best_idx] > (1 - CACHE_THRESHOLD):
        return cache_a[best_idx]
    return None


# Running model
def run_model(query, context_chunks, max_context_tokens=3000):
    context = ""
    token_count = 0

    for chunk in context_chunks:
        tokens = int(len(chunk) / 4)
        if token_count + tokens > max_context_tokens:
            break
        context += chunk + "\n"
        token_count += tokens

    prompt = f"""
You are a document question-answering assistant.

Rules:
- Answer ONLY using the provided context.
- If the answer is not present, say: "I do not know based on the document."
- Be concise and factual.

Context:
{context}

Question:
{query}

Answer:
"""
    with llm_lock:
        output = llm(
            prompt,
            max_tokens=300,
            temperature=0.3,
            top_p=0.9,
            stop=["</s>", "Question:", "Context:"]
        )

    return output["choices"][0]["text"].strip()


# Answering User Queries
def answer_question(query):
    if not query or not query.strip():
        return "Please enter a question."
    if collection.count() == 0:
        return "Upload a document first."
    cached = check_cache(query)
    if cached:
        return cached
    chunks = retrieving_chunks(query, collection)
    if chunks is None:
        return ERROR_MESSAGE
    answer = run_model(query, chunks)
    if ERROR_MESSAGE not in answer and "i do not know" not in answer.lower():
        add_to_cache(query, answer)
    return answer


# Running web app
with gr.Blocks() as demo:
    gr.Markdown("# Document Question Answering System")

    pdf_input = gr.File(label="Upload a PDF", file_types=[".pdf"])
    process_btn = gr.Button("Process Document")
    status = gr.Textbox(label="Status")

    question_text = gr.Textbox(label="Ask a question")
    question_audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Ask via voice")
    
    answer_box = gr.Textbox(label="Answer", lines=8)
    speak_btn = gr.Button("🔊")
    audio_output = gr.Audio(label="Answer Audio")

    # Document processing
    process_btn.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=status
    )

    # Voice → text
    transcribe_btn = gr.Button("Transcribe Audio")
    transcribe_btn.click(
        fn=transcribe,
        inputs=question_audio,
        outputs=question_text
    )

    # Text question → answer
    question_text.submit(
        fn=answer_question,
        inputs=question_text,
        outputs=answer_box
    )

    # Text → speech
    speak_btn.click(
        fn=speak,
        inputs=answer_box,
        outputs=audio_output
    )

demo.queue()
demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
