from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal, List, Dict, Any
from openai import AsyncOpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
from mangum import Mangum
from conversation_jordan import router as conversation_jordan_router
from conversation_alex import (
    router as conversation_alex_router,
    configure_conversation_alex,
)
from logging_routes_conversation import (
    router as logging_conversation_router,
)
import uvicorn
import openai
import os
import re
import json
import base64
import numpy as np
import faiss
import pickle
import glob
import io
import soundfile as sf

load_dotenv()

useCORS = True

# Endpoints allowed to access this server
origins = ["https://main.d355vauwiio7nq.amplifyapp.com", "https://idea.d355vauwiio7nq.amplifyapp.com", "https://clinical-trial-conversation.d3mhus154b7dn6.amplifyapp.com", "http://localhost:5173", "https://ufl.qualtrics.com"]

# UF base URL for using LLM's w liteLLM + litellm api key
base_url = "https://api.ai.it.ufl.edu/v1"
RASHI_LITELLM_KEY = os.getenv('RASHI_LITELLM_KEY')

# Function to build a local RAG (From UF AI Agents Workshop)
# ---- Choose an embedding model available on your Navigator proxy ----
EMBED_MODEL = "nomic-embed-text-v1.5"  # change if your proxy uses a different name or a different model
UF_LOCAL_MODEL = 'gpt-oss-120b'

client_rag = openai.OpenAI(
    api_key=RASHI_LITELLM_KEY,
    base_url=base_url
)

# Regular LiteLLM client for conversational responses (async)
client_chat = AsyncOpenAI(
    api_key= RASHI_LITELLM_KEY,
    base_url= base_url # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
)

def clean_pdf_text(text: str) -> str:
    # Fix weird PDF extraction characters
    text = text.replace("\x00", "f")

    # Existing cleanup
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def read_pdf_pages(path: str) -> List[Dict]:
    reader = PdfReader(path)
    pages = []

    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = clean_pdf_text(text)

        if text.strip():
            pages.append({
                "page_number": page_index + 1,
                "text": text,
            })

    return pages

def chunk_pages(
    pages: List[Dict],
    chunk_size: int = 1200,
    overlap: int = 200,
) -> List[Dict]:
    chunks = []

    for page in pages:
        page_number = page["page_number"]
        text = re.sub(r"\s+", " ", page["text"]).strip()

        start = 0

        while start < len(text):
            proposed_end = min(len(text), start + chunk_size)
            end = proposed_end

            # Prefer ending at a sentence boundary, unless this is already
            # the final chunk on the page.
            if proposed_end < len(text):
                sentence_end = text.rfind(".", start, proposed_end)

                if sentence_end != -1 and sentence_end > start + 300:
                    end = sentence_end + 1

            chunk = text[start:end].strip()

            if len(chunk.split()) >= 20:
                chunks.append({
                    "text": chunk,
                    "page_number": page_number,
                    "char_start": start,
                    "char_end": end,
                })

            # We reached the end of the page. Do not create overlapping
            # one-character-shifted copies of the final text.
            if end >= len(text):
                break

            next_start = max(0, end - overlap)

            # Safety check to guarantee meaningful forward movement.
            if next_start <= start:
                next_start = end

            start = next_start

    return chunks

def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client_rag.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs.extend([d.embedding for d in resp.data])
    return np.array(vecs, dtype=np.float32)

def load_source_manifests(docs_folder="./docs") -> Dict[str, Dict]:
    manifest_by_file = {}

    json_paths = glob.glob(os.path.join(docs_folder, "*_sources.json"))

    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        for entry in entries:
            manifest_by_file[entry["file"]] = entry

    return manifest_by_file

class LocalRAG:
    def __init__(self):
        self.index: Any = None
        self.texts: List[str] = []
        self.meta: List[Dict] = []

    def build_from_pdfs(self, pdf_paths: List[str]):
        all_chunks = []
        all_meta = []

        source_manifest = load_source_manifests("./docs")

        for path in pdf_paths:
            source_label = os.path.basename(os.path.dirname(path))
            file_name = os.path.basename(path)

            info = source_manifest.get(file_name)

            if info is None:
                print(f"⚠️ No manifest entry found for {file_name}")
                info = {}

            print(f"📄 Processing {file_name} from source: {info.get('source', source_label)}")

            pages = read_pdf_pages(path)
            chunks = chunk_pages(pages)
            print(
                f"*** {file_name}: {len(chunks)} chunks "
                f"across {len(pages)} pages"
            )

            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk["text"])
                all_meta.append({
                    "source": info.get("source", source_label),
                    "file": file_name,
                    "title": info.get("title", file_name),
                    "type": info.get("type", "unknown"),
                    "url": info.get("url", ""),
                    "chunk_id": j,
                    "page_number": chunk["page_number"],
                    "char_start": chunk["char_start"],
                    "char_end": chunk["char_end"],
                })

        word_counts = [len(chunk.split()) for chunk in all_chunks]

        print("*** TOTAL CHUNKS:", len(word_counts))
        print("*** MIN WORDS:", min(word_counts))
        print("*** MAX WORDS:", max(word_counts))
        print("*** AVG WORDS:", sum(word_counts) / len(word_counts))

        print("*** TOTAL RAG CHUNKS:", len(all_chunks))
        print(
            "*** CHUNK WORD COUNTS:",
            min(len(chunk.split()) for chunk in all_chunks),
            max(len(chunk.split()) for chunk in all_chunks),
        )

        if not all_chunks:
            raise ValueError("No chunks were created. Check your PDF paths or PDF text extraction.")
        emb = embed_texts(all_chunks)

        dim = emb.shape[1]
        index: Any = faiss.IndexFlatIP(dim)

        faiss.normalize_L2(emb)
        index.add(emb)

        self.index = index
        self.texts = all_chunks
        self.meta = all_meta

    def retrieve(self, query: str, k: int = 6) -> List[Dict]:
        q = embed_texts([query])
        q = np.array(q, dtype=np.float32)          # ensure correct dtype
        if q.ndim == 1:
            q = np.expand_dims(q, axis=0)           # ensure shape is (1, dim)
        faiss.normalize_L2(q)

        if self.index is None:
            raise RuntimeError("RAG index has not been loaded")

        scores, ids = self.index.search(q, k)

        out = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            out.append({
                "text": self.texts[idx],
                "score": float(score),
                "meta": self.meta[idx],
            })
        return out
    
    def save(self, folder="rag_storage"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        # Save the FAISS index
        faiss.write_index(self.index, os.path.join(folder, "index.faiss"))
        # Save the texts and metadata
        with open(os.path.join(folder, "data.pkl"), "wb") as f:
            pickle.dump({"texts": self.texts, "meta": self.meta}, f)

    def load(self, folder="rag_storage"):
        # Load the FAISS index
        self.index = faiss.read_index(os.path.join(folder, "index.faiss"))
        # Load the texts and metadata
        with open(os.path.join(folder, "data.pkl"), "rb") as f:
            data = pickle.load(f)
            self.texts = data["texts"]
            self.meta = data["meta"]

print("💭 CHECKING IF LOCAL RAG ...")

rag = LocalRAG()
STORAGE_DIR = "rag_storage"

if os.path.exists(STORAGE_DIR):
    print("✅ FOUND LOCAL PERSISTED RAG -- LOADING!")
    rag.load(STORAGE_DIR)
else:
    print("🛠️ NO LOCAL RAG FOUND — BUILDING FROM PDFS...")
    pdf_paths = glob.glob("./docs/**/*.pdf", recursive=True)
    rag.build_from_pdfs(pdf_paths)
    rag.save(STORAGE_DIR)
    print("📝 BUILT AND SAVED LOCAL RAG!")

# --------------------------------------------------------------------------
# FASTAPI APP
# --------------------------------------------------------------------------

app = FastAPI()

app.include_router(conversation_jordan_router)
app.include_router(conversation_alex_router)
app.include_router(logging_conversation_router)

handler = Mangum(app)

if useCORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,  # or specify your Amplify URL e.g. ["https://yourapp.amplifyapp.com"]
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ─────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────

from typing import Literal

class TTSRequest(BaseModel):
    text: str
    character: str

class QueryPreprocess(BaseModel):
    route: Literal[
        "clinical_trials_education",
        "personal_medical_advice",
        "trial_recommendation_or_eligibility",
        "political_or_policy",
        "unrelated",
    ]
    search_query: str

def normalize_word(word: str):
    return re.sub(r"[^a-z0-9']", "", word.lower())

def prepare_text_for_speech(text: str) -> str:
    text = text.strip()

    # Normalize Unicode punctuation.
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", ", ")
        .replace("…", "...")
        .replace("\u00a0", " ")
    )

    # Make common source names easier to speak and transcribe.
    replacements = {
        r"\bNCI\b": "National Cancer Institute",
        r"\bNIH\b": "National Institutes of Health",
        r"\bFDA\b": "F D A",
        r"\bIRB\b": "I R B",
        r"ClinicalTrials\.gov": "Clinical Trials dot gov",
        r"&": " and ",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Remove formatting that should not be spoken.
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[*_#`•]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def create_fallback_timestamps(
    text: str,
    duration: float,
) -> list[dict]:
    words = [
        word
        for word in text.split()
        if normalize_word(word)
    ]

    if not words or duration <= 0:
        return []

    step = duration / len(words)

    return [
        {
            "word": word,
            "start": index * step,
            "end": min((index + 1) * step, duration),
        }
        for index, word in enumerate(words)
    ]

@app.post("/tts")
async def tts(request: TTSRequest):
    spoken_text = prepare_text_for_speech(request.text)

    if not spoken_text:
        raise HTTPException(
            status_code=400,
            detail="No valid text to synthesize",
        )

    character = request.character

    character_voice = "af_heart"
    voice_speed = 1.2

    if character == "companion":
        character_voice = "am_echo"
        voice_speed = 1.2

    print(
        "[TTS INPUT]",
        {
            "original_text": request.text,
            "spoken_text": spoken_text,
            "character": character,
        },
    )

    # Generate the audio.
    res = await client_chat.audio.speech.create(
        model="kokoro",
        voice=character_voice,
        input=spoken_text,
        speed=voice_speed,
    )

    audio_bytes = res.content

    # Decode audio so we know its real duration.
    audio_buffer = io.BytesIO(audio_bytes)
    pcm, samplerate = sf.read(
        audio_buffer,
        dtype="int16",
    )

    if samplerate is None or len(pcm) == 0:
        raise HTTPException(
            status_code=500,
            detail="TTS returned empty audio",
        )

    duration = len(pcm) / samplerate

    timestamps = create_fallback_timestamps(
        spoken_text,
        duration,
    )

    return {
        "audio": base64.b64encode(audio_bytes).decode("utf-8"),
        "timestamps": timestamps,
    }
    
async def preprocess_question(question, history):
    print("***IN PREPROCESS QUESTION")

    prompt = f"""
        You are helping prepare search queries for a clinical trials education assistant.

        Your tasks:
        1. Classify the user's latest message into ONE route:
        - clinical_trials_education
        - personal_medical_advice
        - trial_recommendation_or_eligibility
        - political_or_policy
        - unrelated

        2. If the route is clinical_trials_education,
        rewrite the user's latest message into a concise
        standalone search query.

        Rules:
        - Preserve the user's meaning exactly.
        - Stay as close as possible to the user's original wording.
        - Only rewrite to resolve pronouns or references from conversation history.
        - Do not narrow or broaden the question.
        - Do not introduce medical terminology or assumptions that were not in the user's message.
        - Do not optimize for experts; optimize for retrieving patient education documents.

        Recent conversation:
        {history}

        Latest message:
        {question}
        """

    response = await client_chat.beta.chat.completions.parse(
        model=UF_LOCAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=QueryPreprocess,
    )

    print("***PREPROCESS RETURNED", response.choices[0].message.parsed)

    parsed = response.choices[0].message.parsed

    if parsed is None:
        raise RuntimeError("Failed to parse question preprocessing")

    return parsed

def clean_alex_answer(text: str) -> str:
    text = text.strip()

    # Convert clinical trial phase Roman numerals to regular numbers.
    phase_replacements = {
        r"\bPhase\s+IV\b": "Phase 4",
        r"\bPhase\s+III\b": "Phase 3",
        r"\bPhase\s+II\b": "Phase 2",
        r"\bPhase\s+I\b": "Phase 1",
    }

    for pattern, replacement in phase_replacements.items():
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    # Remove bullet or numbered-list markers.
    text = re.sub(
        r"(?:^|\n)\s*(?:[-*•▪◦]+|\d+[.)])\s*",
        " ",
        text,
    )

    # Turn all remaining line breaks into one conversational paragraph.
    text = re.sub(r"\s+", " ", text)

    return text.strip()

configure_conversation_alex(
    rag_instance=rag,
    chat_client=client_chat,
    model=UF_LOCAL_MODEL,
    preprocess_func=preprocess_question,
    clean_alex_answer_func=clean_alex_answer,
)

@app.get("/")
async def root():
    return {"message": "Welcome to Rashi's FastAPI server!"}

if __name__ == "__main__":
    # Run the FastAPI application on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)