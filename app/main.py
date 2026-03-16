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
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import openai
from contextlib import asynccontextmanager
import pickle
import glob
import io, wave, struct
import soundfile as sf
from mangum import Mangum

load_dotenv()

useCORS = True

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
        q = np.array(q, dtype=np.float32)          # ensure correct dtype
        if q.ndim == 1:
            q = np.expand_dims(q, axis=0)           # ensure shape is (1, dim)
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

# ─────────────────────────────────────────────
# QUESTION BANK
# ─────────────────────────────────────────────

QUESTION_BANK = [
    "What are clinical trials?",
    "What is informed consent?",
    "What is an IRB?",
    "What is a placebo?",
    "Will I have side effects on a clinical trial?",
    "What is standard treatment?",
    "Will I have to receive my care at a different clinic if I am on a clinical trial?",
    "Is there a clinical trial for everyone?",
    "Where can I find information about clinical trials?",
    "Will my own doctor know what happens to me when I am on a clinical trial?",
    "Will taking part in a clinical trial help me?",
    "Who pays for the cost of a clinical trial?",
    "Should I ask my doctor about clinical trials?",
    "Are clinical trials only used as a last resort?",
    "Are there ways to deal with transportation and financial issues?",
    "What is randomization?",
    "Is it safe to try new treatments that haven't been around for long?",
    "What will pharmaceutical or drug companies gain from a clinical trial?",
    "Can I trust the medical establishment?",
    "How would clinical trials affect my family?",
    "Will I get good care if I take part in a clinical trial?",
    "How long do I need to stay in a clinical trial?",
    "Are clinical trials appropriate for cancer patients?",
    "How is my privacy protected on a clinical trial?",
    "Will a clinical trial take up a lot of my time?",
    "Will I be able to handle being in a clinical trial?",
    "What will my doctor gain from this clinical trial research?",
    "Is taking part in a clinical trial voluntary?",
]

# STORAGE_DIR = "rag_storage"
# QUESTION_BANK_EMBEDDINGS_PATH = os.path.join(STORAGE_DIR, "question_bank_embeddings.npy")

# def get_embedding(text: str) -> list[float]:
#     response = client_rag.embeddings.create(
#         model="text-embedding-3-small",
#         input=text
#     )
#     return response.data[0].embedding

# def load_or_build_bank_embeddings() -> np.ndarray:
#     if os.path.exists(QUESTION_BANK_EMBEDDINGS_PATH):
#         print("✅ FOUND LOCAL PERSISTED QUESTION BANK EMBEDDINGS - LOADING!")
#         return np.load(QUESTION_BANK_EMBEDDINGS_PATH)
#     else:
#         print("🛠️ NO LOCAL EMBEDDINGS FOUND - BUILDING QUESTION BANK EMBEDDINGS...")
#         embeddings = np.array([get_embedding(q) for q in QUESTION_BANK])
#         np.save(QUESTION_BANK_EMBEDDINGS_PATH, embeddings)
#         print("📝 BUILT AND SAVED LOCAL EMBEDDINGS!")
#         return embeddings

# bank_embeddings = load_or_build_bank_embeddings()

# 2. Pass the lifespan to the FastAPI app
app = FastAPI()

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

class ChatRequest(BaseModel):
    thread_id: str | None = None
    message: str

class TTSRequest(BaseModel):
    text: str
    character: str

class PrecheckResponse(BaseModel):
    user_message: str | None = None
    gesture: Literal[ "thinking", "thumbsup", "shrug"]
    label: Literal["ready", "vague", "good", "thoughtful", "unknown"]
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

class SimilarQuestionsRequest(BaseModel):
    message: str
    top_n: int = 3

class SimilarQuestion(BaseModel):
    question: str
    score: float

class SimilarQuestionsResponse(BaseModel):
    similar_questions: list[SimilarQuestion]

# Regular LiteLLM client for conversational responses (async)
client_chat = AsyncOpenAI(
    api_key= RASHI_LITELLM_KEY,
    base_url= base_url # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
)

@app.get("/debug")
async def debug():
    import os
    storage_exists = os.path.exists("rag_storage")
    task_contents = os.listdir("/var/task")
    rag_contents = os.listdir("/var/task/rag_storage") if storage_exists else "FOLDER MISSING"
    return {
        "rag_index_loaded": rag.index is not None,
        "storage_exists": storage_exists,
        "var_task_contents": task_contents,
        "rag_storage_contents": rag_contents,
        "cwd": os.getcwd()
    }

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
    
async def run_precheck(message: str) -> PrecheckResponse:
    # 1. Get similar questions from bank first
    # user_embedding = np.array(get_embedding(request.message))
    # scores = cosine_similarity([user_embedding], bank_embeddings)[0]
    # top_indices = np.argsort(scores)[::-1][:3]
    # similar = [QUESTION_BANK[i] for i in top_indices]
    # top_score = round(float(scores[top_indices[0]]), 4)  # 👈 best match score
    # print("SIMILAR TOP SCORE ARE", top_score)
    # print("SIMILAR ARE", similar)
    system_prompt = f"""
        You are evaluating whether a user's message is ready to be processed by a clinical trials assistant. You act like a friendly, conversational helper named Milo.

        Here is the complete list of questions Dr. Alex can answer well:
        {QUESTION_BANK}

        When providing suggestions, ONLY use questions EXACTLY as they appear in the list above.

        Assess the user's message and respond ONLY in valid JSON matching this exact structure:
        {{
        "user_message": null,
        "label": one of: "ready", "vague", "good", "thoughtful, "unknown",
        "gesture": one of: "thinking", "thumbsup", "shrug",
        "in_scope": true or false,
        "tip": a short friendly nudge. If suggestions are provided, briefly explain in one sentence why they're relevant to what the user typed. Keep it warm and under 20 words,
        "suggestions": a list of 2-3 short suggested questions asked from the user's perspective, or null. suggestions must be phrased as questions the USER would ask Dr. Alex, not questions Dr. Alex would ask the user. Suggestions should be based on what the user is most likely trying to ask, completing or expanding on their partial thought. If the message is incomplete or cut off, suggest the most common complete versions of that question related to clinical trials. Suggestions must not be about finding or enrolling in a specific clinical trial — stick to general educational questions about how clinical trials work.
        }}

        Use this guide to pick the right values:
        - Message is empty or just a few characters (not a full word) → label: "ready", gesture: "thumbsup", tip: null, suggestions: null, in_scope: true
        - Message is small talk, a greeting, not relevant to clinical trials, or asking about a specific clinical trial → label: "unknown", gesture: "shrug", tip: tell the user Dr. Alex can only respond to questions about clinical trials, suggestions: 2-3 questions to ask about clinical trials, in_scope: false
        - Message is incomplete → label: "vague", gesture: "shrug", tip: briefly explain why the suggestions relate to what the user typed, then ask if any match what they meant. For example: "Incomplete questions about X often mean Y or Z — did you mean one of these?", suggestions: 2-3 complete questions they might mean that is about clinical trials, in_scope: true
        - Message is too short or a fragment → label: "vague", gesture: "shrug", tip: briefly explain why the suggestions relate to what the user typed, then nudge them to expand. For example: "X often comes up around Y and Z — do any of these match?", suggestions: 2-3 complete example questions they might mean, in_scope: true
        - Message is vague or unfocused → label: "vague", gesture: "shrug", tip: briefly explain why the suggestions relate to what the user typed, then let them know a bit more detail would help. For example: "X can mean a few things — here are the most common ones!", suggestions: 2-3 example questions of what they might mean, in_scope: true
        - Message is a clear, decent question → label: "good", gesture: "thumbsup", tip: tell the user the question is good and why it's important to ask, suggestions: 1-2 related follow-up questions, in_scope: true
        - Message is detailed and well-formed → label: "good", gesture: "thumbsup", tip: tell the user the question is good and why it's important to ask, suggestions: 1-2 related follow-up questions, in_scope: true
        - Message touches on sensitive topics (risk, safety, fear, harm) → label: "thoughtful", gesture: "thinking", tip: let the user know it's good they're considering this and why it's good to consider, suggestions: 1-2 related gentle follow-up example questions, in_scope: true
        - Message is a clear, decent question → label: "good", gesture: "thumbsup", tip: tell the user the question is good and why it's important to ask, suggestions: null, in_scope: true
        - Message is detailed and well-formed → label: "good", gesture: "thumbsup", tip: tell the user the question is good and why it's important to ask, suggestions: null, in_scope: true
        
        Suggestions should be short (under 7 words), phrased as questions, and specific to clinical trials.
        Respond ONLY with valid JSON. No preamble, no explanation, no markdown.
    """

    response = await client_chat.beta.chat.completions.parse(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ],
        response_format=PrecheckResponse
    )

    result = response.choices[0].message.parsed

    if result is None:
        raise HTTPException(status_code=500, detail="Failed to parse response")

    return PrecheckResponse(
        user_message=message,
        tip=result.tip,
        suggestions=result.suggestions,
        label=result.label,
        gesture=result.gesture
    )


# existing endpoint unchanged
@app.post("/precheck", response_model=PrecheckResponse)
async def precheck(request: ChatRequest):
    return await run_precheck(request.message)


# landing endpoint now calls run_precheck directly
@app.post("/landing-example")
async def landing_example(request: ChatRequest):
    precheck = await run_precheck(request.message)

    suggestions_str = json.dumps(precheck.suggestions) if precheck.suggestions else "none"

    system_prompt_example = f"""
        You are Jordan, a warm and approachable virtual companion helping a user
        navigate a clinical trial information tool. Your personality is friendly,
        casual, and non-clinical — like a knowledgeable friend, not a doctor.

        The user was asked: "What's one thing you've wondered about clinical trials?
        Don't worry about getting it perfect — just type whatever comes to mind."

        A precheck system has already analyzed their message and produced this:
        - Label: "{precheck.label}"
        - Tip: "{precheck.tip}"
        - Suggestions: {suggestions_str}

        Use this to craft a response with the following structure:
        1. Briefly acknowledge their message — if label is "good" or "thoughtful",
        affirm them warmly and naturally (e.g. "that's one of the most common
        things people wonder about"). If "vague" or "unknown", be warm and
        reassuring that it's a great starting point.
        2. Before introducing the suggestions, add one short framing sentence
        that explains why you're suggesting them — vary it based on label:
        - If label is "good" or "thoughtful": use phrasing like
            "To help you dig deeper into that..." or
            "To help you explore that further..."
        - If label is "vague" or "unknown" or user has nothing:
            use phrasing like "To help get you started..." or
            "To give you a jumping off point..."
        Then introduce the suggestions naturally woven into a sentence using
        phrasings like "I might suggest questions like X or Y" or
        "some questions worth exploring might be X or Y".
        Do not say "in the actual tool" or "in the demo".
        3. In one sentence, describe your role using this exact framing:
        "I'll be suggesting questions like these during the interaction to
        support you during your information search."
        4. Close with one sentence handing off to Dr. Alex:
        "Now, click the button below to meet Doctor Alex, who will be actually answering your questions."

        Keep it conversational and brief — 3 to 5 sentences max. Do not use
        clinical jargon. Do not answer their question yourself. Do not ask any
        follow-up questions. Do not break the fourth wall by referencing the
        tool, demo, or onboarding.
    """
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt_example},
        {"role": "user", "content": request.message}
    ]

    try:
        response = await client_chat.chat.completions.create(
            model='gpt-4o-mini', messages=messages, temperature=0
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

# Replace AudioSegment usage with this helper:
def decode_mp3_to_pcm(mp3_bytes: bytes):
    """Use soundfile + numpy — no ffmpeg needed"""
    buf = io.BytesIO(mp3_bytes)
    data, samplerate = sf.read(buf, dtype='int16')
    return data, samplerate

def encode_pcm_to_mp3(pcm_data, samplerate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, pcm_data, samplerate, format='mp3')
    buf.seek(0)
    return buf.read()

# TTS endpoint ...
@app.post("/tts")
async def tts(request: TTSRequest):
    sentences = re.split(r'(?<=[.!?]) +', request.text)
    character = request.character
    characterVoice = "af_heart"
    voiceSpeed = 1.0
    if character == "companion":
        characterVoice = "am_echo"
        voiceSpeed = 1.2
    
    all_words = []
    all_pcm = []
    samplerate = None
    time_offset = 0.0

    for sentence in sentences:
        if not sentence.strip():
            continue

        res = await client_chat.audio.speech.create(
            model="kokoro", voice=characterVoice, input=sentence, speed=voiceSpeed
        )

        # Decode MP3 → PCM (soundfile uses libsndfile, no ffmpeg)
        buf = io.BytesIO(res.content)
        pcm, sr = sf.read(buf, dtype='int16')
        if samplerate is None:
            samplerate = sr
        all_pcm.append(pcm)

        duration = len(pcm) / sr

        # Whisper this sentence
        whisper_buf = io.BytesIO(res.content)
        whisper_buf.name = "audio.mp3"
        transcript = await client_chat.audio.transcriptions.create(
            model="whisper-large-v3",
            file=whisper_buf,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            prompt=sentence
        )
        for segment in transcript.model_dump().get("segments", []):
            for word in segment.get("words", []):
                all_words.append({
                    "word": word["word"],
                    "start": word["start"] + time_offset,
                    "end": word["end"] + time_offset,
                })

        time_offset += duration

    # Stitch PCM arrays
    combined_pcm = np.concatenate(all_pcm, axis=0)

    # Encode back to MP3
    out = io.BytesIO()
    sf.write(out, combined_pcm, samplerate, format='mp3')
    combined_audio = out.getvalue()

    all_words = [w for w in all_words if w["end"] - w["start"] > 0.01]

    return {
        "audio": base64.b64encode(combined_audio).decode("utf-8"),
        "timestamps": all_words
    }
    
@app.post("/rag-chat")
async def rag_chat(request: ChatRequest):
    print("IN RAG CHAT")
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
        1. MANDATORY: Provide 2-3 citations from DIFFERENT sources if available.
        2. VERBATIM REQUIREMENT: The 'content' must be 2-3 full, logical sentences. 
        3. NO HEADERS: Cite actual findings/policy text only.
        4. SYNTHESIS: Mention sources conversationally (e.g., "According to the FDA...").
        5. TTS OPTIMIZATION (CRITICAL): Keep your 'answer' under 80 words. 
        - Use simple sentence structures.
        - Avoid long lists of names or dates that can trip up the voice.

        GOAL: 
        Provide a detailed answer. If the user asks 'who runs trials', find the specific paragraphs listing sponsors, investigators, or institutions.
    """
    print("IN RAG CHAT AB TO CALL LLM")
    # 4. Call the LLM with .parse()
    response = await client_chat.beta.chat.completions.parse(
        model='gpt-4o-mini', # Mini is great at this; use 4o for very complex logic
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXT:\n{context_str}\n\nQUESTION: {question}"}
        ],
        response_format=RAGResponse
    )
    print("RESPONSE IS", response.choices[0].message.parsed)

    return response.choices[0].message.parsed

@app.post("/similar-questions")
async def similar_questions(request: SimilarQuestionsRequest):
    user_embedding = np.array(get_embedding(request.message))
    scores = cosine_similarity([user_embedding], bank_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:request.top_n]
    return SimilarQuestionsResponse(
        similar_questions=[
            SimilarQuestion(question=QUESTION_BANK[i], score=round(float(scores[i]), 4))
            for i in top_indices
        ]
    )

@app.get("/")
async def root():
    return {"message": "Welcome to Rashi's FastAPI server!"}

if __name__ == "__main__":
    # Run the FastAPI application on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)