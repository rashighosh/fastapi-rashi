from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Literal
import uvicorn
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
import base64
import os, re, json, math
from typing import List, Dict, Tuple
from pypdf import PdfReader
import numpy as np
import faiss
import openai
from contextlib import asynccontextmanager
import pickle
import glob

load_dotenv()

# Endpoints allowed to access this server
origins = ["https://main.d1qbymvh7dh0n4.amplifyapp.com", "http://localhost:5173"]

# UF base URL for using LLM's w liteLLM + litellm api key
base_url = "https://api.ai.it.ufl.edu"
RASHI_LITELLM_KEY = os.getenv('RASHI_LITELLM_KEY')

# Function to build a local RAG (From UF AI Agents Workshop)
# ---- Choose an embedding model available on your Navigator proxy ----
EMBED_MODEL = "nomic-embed-text-v1.5"  # change if your proxy uses a different name or a different model

client_rag = openai.OpenAI(
    api_key=RASHI_LITELLM_KEY,
    base_url=base_url
)

def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        t = p.extract_text() or ""
        # Remove extra whitespace and fix broken line breaks
        t = re.sub(r'(\w)-\n(\w)', r'\1\2', t) # Fix hyphenated words at line breaks
        t = re.sub(r'(?<!\n)\n(?!\n)', ' ', t) # Replace single newlines with spaces
        pages.append(t)
    return "\n".join(pages)

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    # simple whitespace chunker (good enough to start)
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = max(end - overlap, start + 1)
    return chunks

def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client_rag.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs.extend([d.embedding for d in resp.data])
    return np.array(vecs, dtype=np.float32)

class LocalRAG:
    def __init__(self):
        self.index = None
        self.texts: List[str] = []
        self.meta: List[Dict] = []

    def build_from_pdfs(self, pdf_paths: List[str]):
        all_chunks = []
        all_meta = []

        for path in pdf_paths:
            # Extract folder name (e.g., 'nih') and filename
            # os.path.dirname(path) gets './docs/nih'
            # os.path.basename(...) of that gets 'nih'
            source_label = os.path.basename(os.path.dirname(path))
            file_name = os.path.basename(path)
            
            print(f"📄 Processing {file_name} from source: {source_label}")
            
            raw = read_pdf_text(path)
            chunks = chunk_text(raw)
            
            for j, c in enumerate(chunks):
                all_chunks.append(c)
                all_meta.append({
                    "source": source_label,  # This will be 'nih', 'nci', etc.
                    "file": file_name,
                    "chunk_id": j
                })

        # ... keep your embedding and FAISS logic exactly as it was ...
        emb = embed_texts(all_chunks)
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(emb)
        index.add(emb)

        self.index = index
        self.texts = all_chunks
        self.meta = all_meta

    def retrieve(self, query: str, k: int = 6) -> List[Dict]:
        q = embed_texts([query])
        faiss.normalize_L2(q)
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

print("AB TO BUILD LOCAL RAG")

rag = LocalRAG()

# 1. Define the lifespan logic
@asynccontextmanager
async def lifespan(app: FastAPI):
    STORAGE_DIR = "rag_storage"
    
    if os.path.exists(STORAGE_DIR):
        print("🚀 LOADING PERSISTED RAG...")
        rag.load(STORAGE_DIR)
    else:
        print("🏗️ NO RAG FOUND. BUILDING...")
        
        # This pattern matches anything in docs/SUBFOLDER/*.pdf
        # recursive=True ensures we catch docs/nih/folder2/file.pdf too
        pdf_files = glob.glob("./docs/**/*.pdf", recursive=True)
        
        if not pdf_files:
            print("⚠️ WARNING: No PDFs found in ./docs/ subfolders!")
        else:
            rag.build_from_pdfs(pdf_files)
            rag.save(STORAGE_DIR)
        
    print("✅ RAG READY FOR QUERIES")
    yield
    print("🛑 SHUTTING DOWN...")

# 2. Pass the lifespan to the FastAPI app
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or specify your Amplify URL e.g. ["https://yourapp.amplifyapp.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base Models for structuring responses from LLM
class ChatRequest(BaseModel):
    thread_id: str | None = None
    message: str

class TTSRequest(BaseModel):
    text: str

class PrecheckResponse(BaseModel):
    user_message: str | None = None
    gesture: Literal["ready", "thinking", "thumbsup", "shrug"]
    label: Literal["ready", "vague", "good", "thoughtful"]
    tip: str | None = None                 # tooltip shown while typing
    suggestions: list[str] | None = None   # AI-generated suggestion chips

class SourceSnippet(BaseModel):
    source: str       # e.g., "NIH"
    file: str         # e.g., "guidelines.pdf"
    content: str      # The specific sentence or paragraph used
    why_this_snippet_addresses_the_question: str    # Why this specific bit matters

class RAGResponse(BaseModel):
    answer: str       # The high-level combined synthesis
    citations: List[SourceSnippet] # List of specific snippets used
    confidence: float # 0.0 to 1.0

# Regular LiteLLM client for conversational responses (async)
client_chat = AsyncOpenAI(
    api_key= RASHI_LITELLM_KEY,
    base_url= base_url # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
)

@app.post("/simple-chat")
async def simple_chat(request: ChatRequest):
    print("IN SIMPLE CHAT", request)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "You are a friendly virtual assistant helping users with clinical trial questions. Respond in maximum 300 characters."},
        {"role": "user", "content": request.message}
    ]
    try:
        # call the async LLM client
        response = await client_chat.chat.completions.create(model='gpt-4o-mini', messages=messages, temperature=0)
        # response might be a string or a dict; check your client
        print("RESPONSE IS", response.choices[0].message.content)
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

@app.post("/precheck", response_model=PrecheckResponse)
async def precheck(request: ChatRequest):

    system_prompt = """
        You are evaluating whether a user's message is ready to be processed by a clinical trials assistant. You act like a friendly, conversational helper named Milo.

        Assess the user's message and respond ONLY in valid JSON matching this exact structure:
        {
        "user_message": null,
        "label": one of: "ready", "vague", "good", "thoughtful",
        "gesture": one of: "ready", "thinking", "thumbsup", "shrug",
        "tip": a short friendly nudge shown as a tooltip while the user types using warm, kind words, or null,
        "suggestions": a list of 2-3 short suggested questions asked from the user's perspective, or null. suggestions must be phrased as questions the USER would ask Dr. Alex, not questions Dr. Alex would ask the user. Suggestions should be based on what the user is most likely trying to ask, completing or expanding on their partial thought. If the message is incomplete or cut off, suggest the most common complete versions of that question related to clinical trials. For example, if the user types "what is the cl", suggest questions like "What is the clinical trial process?", "What is the clinical trial eligibility criteria?", "What is the clinical trial consent process?" For example: "What are common side effects?" not "What side effects are you concerned about?"
        }

        Use this guide to pick the right values:

        - Message is empty or just a few characters → label: "ready", gesture: "ready", tip: null, suggestions: null
        - Message is small talk or a greeting or not relevant to clinical trials → label: "ready", gesture: "ready", tip: tell the user Dr. Alex can only respond to questions about clinical trials, suggestions: 2-3 questions to ask about clinical trials
        - Message is incomplete → label: "vague", gesture: "shrug", tip: tell the user the message looks incomplete and ask if they meant any of your suggestions, suggestions: 2-3 complete questions they might mean that is about clinical trials
        - Message is too short or a fragment → label: "vague", gesture: "shrug", tip: tell the user the question is too short or could use some or detail and to consider your suggestions, suggestions: 2-3 complete example questions they might mean
        - Message is vague or unfocused → label: "vague", gesture: "shrug", tip: let the user know the question may be too vague to get a good answer and to consider your suggestions, suggestions: 2-3 example questions of what they might mean
        - Message is a clear, decent question → label: "good", gesture: "thumbsup", tip: tell the user the question is good and why it's important to ask, suggestions: 1-2 related follow-up questions
        - Message is detailed and well-formed → label: "good", gesture: "thumbsup", tip: tell the user the question is good and why it's important to ask, suggestions: 1-2 related follow-up questions
        - Message touches on sensitive topics (risk, safety, fear, harm) → label: "thoughtful", gesture: "thinking", tip: let the user know it's good they're considering this and why it's good to consider, suggestions: 1-2 related gentle follow-up example questions

        Suggestions should be short (under 7 words), phrased as questions, and specific to clinical trials.
        Respond ONLY with valid JSON. No preamble, no explanation, no markdown.
    """

    response = await client_chat.beta.chat.completions.parse(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message}
        ],
        response_format=PrecheckResponse  # your Pydantic model
    )

    result = response.choices[0].message.parsed

    if result is None:
        raise HTTPException(status_code=500, detail="Failed to parse response")

    return PrecheckResponse(
        user_message=request.message,
        tip=result.tip,
        suggestions=result.suggestions,
        label=result.label,
        gesture=result.gesture
    )

# TTS endpoint ...
@app.post("/tts")
async def tts(request: TTSRequest):
    # Step 1: get audio
    response = await client_chat.audio.speech.create(
        model="kokoro",
        voice="af_heart",
        input=request.text,
        speed=1.0
    )
    audio_bytes = response.content

    # Step 2: get word timestamps from Whisper
    transcript = await client_chat.audio.transcriptions.create(
        model="whisper-large-v3",
        file=("audio.mp3", audio_bytes, "audio/mpeg"),
        response_format="verbose_json",
        timestamp_granularities=["word"]
    )

    # Step 3: return both
    return {
        "audio": base64.b64encode(audio_bytes).decode("utf-8"),
        "timestamps": transcript.words
    }
    
@app.post("/rag-chat")
async def rag_chat(request: ChatRequest):
    question = request.message
    
    # 1. Get RAG results
    results = rag.retrieve(question, k=5)
    
    # 2. Format the "Raw Material" for the LLM
    # We include IDs so the LLM can easily distinguish chunks
    context_list = []
    for i, res in enumerate(results):
        m = res['meta']
        context_list.append(
            f"ID: {i}\nSOURCE: {m['source']}\nFILE: {m['file']}\nCONTENT: {res['text']}"
        )
    context_str = "\n\n---\n\n".join(context_list)

    # 3. The System Prompt (Milo's older brother, the Researcher)
    system_prompt = """
        You are a clinical trials data synthesizer. 

        RULES FOR CITATIONS:
        1. MANDATORY: You must provide at least 2-3 citations from DIFFERENT sources if available.
        2. VERBATIM REQUIREMENT: The 'content' must be a substantial block of text (3-4 full sentences). Note: The source text might have strange line breaks due to PDF formatting; ignore these and provide the full logical sentences.
        3. NO HEADERS: Do not cite section titles or questions. Cite the actual data/findings/policy text.
        4. SYNTHESIS: In your 'answer', explicitly mention the sources in a conversational way. e.g., "While the FDA focuses on X, the NIH documentation emphasizes Y." Or, "The FDA says X" and "NIH also mentions Y".
        5. Keep your 'answer' to 150 words or less.

        GOAL: 
        Provide a detailed answer. If the user asks 'who runs trials', find the specific paragraphs listing sponsors, investigators, or institutions.
    """

    # 4. Call the LLM with .parse()
    response = await client_chat.beta.chat.completions.parse(
        model='gpt-4o-mini', # Mini is great at this; use 4o for very complex logic
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXT:\n{context_str}\n\nQUESTION: {question}"}
        ],
        response_format=RAGResponse
    )

    return response.choices[0].message.parsed

# --- How to run it ---
# import asyncio
# answer = asyncio.run(ask_knowledge_base("What are NCI's latest lung cancer findings?", rag))
# print(answer)

@app.get("/")
async def root():
    return {"message": "Welcome to Rashi's FastAPI server!"}

if __name__ == "__main__":
    # Run the FastAPI application on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)