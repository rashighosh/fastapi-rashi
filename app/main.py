from fastapi import FastAPI, HTTPException, Query
from logging_routes import router as log_router
from logging_routes_pilot import router as log_router_pilot
from qualtrics import (
    get_presurvey_row_from_qualtrics,
    score_presurvey_row,
    generate_goals_from_scores,
    generate_more_goals_from_scores,
)
from pydantic import BaseModel, Field
from typing import Literal
import uvicorn
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from dotenv import load_dotenv
import os
import random
from collections import defaultdict
import pandas as pd
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
origins = ["https://main.d355vauwiio7nq.amplifyapp.com", "http://localhost:5173"]

# UF base URL for using LLM's w liteLLM + litellm api key
base_url = "https://api.ai.it.ufl.edu/v1"
RASHI_LITELLM_KEY = os.getenv('RASHI_LITELLM_KEY')

generated_personas = []
conversation_history = defaultdict(list)

# Function to build a local RAG (From UF AI Agents Workshop)
# ---- Choose an embedding model available on your Navigator proxy ----
EMBED_MODEL = "nomic-embed-text-v1.5"  # change if your proxy uses a different name or a different model
UF_LOCAL_MODEL = 'gpt-oss-120b'

client_rag = openai.OpenAI(
    api_key=RASHI_LITELLM_KEY,
    base_url=base_url
)

def getUser(id: str):
    print("IN GET USER", id)
    user_folder = "themes"
    # 1. Define your file paths
    users = os.path.join(user_folder, "users.csv")
    df_users = pd.read_csv(users)
    # Assuming your DataFrame is named df_users

    rename_dict = {
        "eHEALS_2_PRE": "Knows_what_health_resources_are_available_on_the_Internet_OutOf100:",
        "eHEALS_3_PRE": "Knows_where_to_find_helpful_health_resources_on_the_Internet_OutOf100",
        "eHEALS_5_PRE": "Knows_how_to_find_helpful_health_resources_on_the_Internet_OutOf100",
        "eHEALS_6_PRE": "Knows_how_to_use_the_Internet_to_answer_questions_about_health_OutOf100",
        "eHEALS_10_PRE": "Knows_how_to_use_health_information_I_find_on_the_Internet_to_help_me_OutOf100",
        "eHEALS_7_PRE": "Have_the_skills_needed_to_evaluate_health_resources_I_find_on_the_Internet_OutOf100",
        "eHEALS_8_PRE": "Can_tell_high_quality_health_resources_from_low_quality_on_the_Internet_OutOf100",
        "eHEALS_9_PRE": "Feel_confident_using_health_information_on_the_Internet_to_make_health_decisions_OutOf100",
        "eHEALS_Average_PRE": "eHealthLiteracy_OutOf100",
        "HL_Category_PRE": "GeneralHealthLiteracy"
    }

    # Reassign back to the dataframe to save the changes
    df_users = df_users.rename(columns=rename_dict)

    matched_user = df_users[df_users['id'] == id]
    
    # Check if a matching user was found
    if not matched_user.empty:
        # .iloc[0] grabs the first match, .to_dict() converts it
        print(matched_user.iloc[0].to_dict())
        return matched_user.iloc[0].to_dict()
    
def clean_pdf_text(text: str) -> str:
    # Fix weird PDF extraction characters
    text = text.replace("\x00", "f")

    # Existing cleanup
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = []

    for p in reader.pages:
        t = p.extract_text() or ""
        t = clean_pdf_text(t)
        pages.append(t)

    return "\n".join(pages)

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0

    while start < len(text):
        end = min(len(text), start + chunk_size)

        # try to end at sentence boundary
        sentence_end = text.rfind(".", start, end)
        if sentence_end != -1 and sentence_end > start + 300:
            end = sentence_end + 1

        chunk = text[start:end].strip()

        if len(chunk.split()) >= 20:
            chunks.append(chunk)

        start = max(end - overlap, start + 1)

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
        self.index = None
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

            raw = read_pdf_text(path)
            chunks = chunk_text(raw)

            for j, c in enumerate(chunks):
                all_chunks.append(c)
                all_meta.append({
                    "source": info.get("source", source_label),
                    "file": file_name,
                    "title": info.get("title", file_name),
                    "type": info.get("type", "unknown"),
                    "url": info.get("url", ""),
                    "chunk_id": j
                })

        if not all_chunks:
            raise ValueError("No chunks were created. Check your PDF paths or PDF text extraction.")
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

QUESTION_BANK_LITERATURE = [
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

QUESTION_BANK = [
  "What are clinical trials?",
  "What types of participants are typically involved in clinical trials?",
  "How do I know if I am eligible to participate in a clinical trial?",
  "What are some common risks associated with clinical trials?",
  "How do clinical trials ensure participant safety?",
  "What is informed consent in the context of clinical trials?",
  "What are the costs associated with participating in a clinical trial?",
  "Who typically pays for clinical trials?",
  "How does randomization work in clinical trials?",
  "What is the role of a principal investigator in a clinical trial?",
  "What is the difference between treatment trials and prevention trials?",
  "What are the common phases of clinical trials?",
  "What happens if the treatment being studied turns out to be harmful?",
  "How have virtual clinical trials changed the landscape of participation?",
  "What measures are being taken to enhance diversity in clinical trials?",
  "What is the significance of using placebos in clinical trials?",
  "How do clinical trials affect the treatment options available for cancer patients?",
  "What is telemedicine's role in modern clinical trials?",
  "What kind of feedback can participants provide to researchers during trials?",
  "How can the inclusion of older adults in clinical trials be improved?",
  "What efforts are made to recruit participants from minority communities in clinical trials?",
  "How does the study design impact patient recruitment in clinical trials?",
  "What are the implications if a clinical trial terminates early?",
  "Why might a study exclude participants with certain health conditions?",
  "What is the importance of trial protocols in clinical research?",
  "What role do community advocacy groups play in clinical trials?",
  "How are clinical trial sites chosen for new studies?",
  "What challenges do clinical trials face in recruiting participants?",
  "How can technology improve patient participation in clinical trials?",
  "What considerations are there regarding travel for clinical trial participants?",
  "How is data collected during clinical trials?",
  "What kind of support might be available for participants who incur costs related to trials?",
  "What happens to participants after the clinical trial ends?",
  "Why is informed consent important for clinical trial participants?",
  "What unique barriers do racial and ethnic minorities face in clinical trials?",
  "How do clinical trials contribute to cancer research and treatment advancements?",
  "What does 'enrolling by invitation' mean in clinical trials?",
  "How has the COVID-19 pandemic affected clinical trial operations?"
]

# 2. Pass the lifespan to the FastAPI app
app = FastAPI()
app.include_router(log_router)
app.include_router(log_router_pilot, prefix="/pilot")

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

class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequestHistory(BaseModel):
    message: str
    history: list[ChatTurn] = []

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

class LandingExample(BaseModel):
    response: str | None = None
    suggestions: list[str] | None = None   # AI-generated suggestion chips
    
class SourceSnippetOld(BaseModel):
    source: str       # e.g., "NIH"
    file: str         # e.g., "guidelines.pdf"
    chunk_id: int | None = None
    score: float | None = None
    content: str      # The specific sentence or paragraph used
    why_this_snippet_addresses_the_question: str    # Why this specific bit matters

class RAGResponseOld(BaseModel):
    answer: str       # The high-level combined synthesis
    citations: List[SourceSnippetOld] # List of specific snippets used
    confidence: float # 0.0 to 1.0

class SourceExplanation(BaseModel):
    id: int
    relevance_explanation: str

class RAGResponse(BaseModel):
    answer: str
    source_explanations: List[SourceExplanation]
    confidence: float

class SimilarQuestionsRequest(BaseModel):
    message: str
    top_n: int = 3

class SimilarQuestion(BaseModel):
    question: str
    score: float

class SimilarQuestionsResponse(BaseModel):
    similar_questions: list[SimilarQuestion]

class UserInfo(BaseModel):
    userId: str

class GoalNote(BaseModel):
    id: str | None = None
    text: str

class GoalItem(BaseModel):
    id: str
    title: str
    notes: List[GoalNote] | None = None
    addressed: bool | None = False

class GoalEvalRequest(BaseModel):
    user_message: str
    alex_answer: str
    goals: List[GoalItem]
    condition: str | None = None
    previous_suggestions: List[str] = Field(default_factory=list)

class GoalEvalMatch(BaseModel):
    goal_id: str
    user_question_relevant: bool
    alex_answered_question: bool
    already_addressed: bool = False
    jordan_message: str
    note_to_add: str | None = None

class GoalEvalResponse(BaseModel):
    matches: List[GoalEvalMatch]
    suggested_goal_question: str | None = None
    no_match_jordan_message: str | None = None
    all_goals_covered_message: str | None = None
    next_step_message: str | None = None

class SourceExplanation(BaseModel):
    id: int
    relevance_explanation: str

class RAGResponse(BaseModel):
    answer: str
    source_explanations: list[SourceExplanation]
    confidence: float
    talking_points: list[str] = Field(default_factory=list)

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
        {"role": "system", "content": "You are a friendly virtual assistant helping users with clinical trial questions. Respond in maximum 80 characters."},
        {"role": "user", "content": request.message}
    ]
    try:
        # call the async LLM client
        response = await client_chat.chat.completions.create(model='mistral-small-3.1', messages=messages, temperature=0)
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
        You act like a friendly, conversational helper named Jordan. You are evaluating whether a user's message is ready to be answered by Doctor Alex, a clinical trials assistant. 
        Your task is to provide feedback on the user's current input. You are helping someone put casual, half‑formed thoughts into everyday questions, not rewriting them into professional language. You're thinking alongside the user.
        Here is a list of questions that Doctor Alex can answer:
        {QUESTION_BANK + QUESTION_BANK_LITERATURE}

        When providing suggestions, use the questions from the list above as a guide to suggest relevant questions. Do NOT use the exact questions. Only use these questions as a guide to suggest relevant questions (no higher than 5th grade reading level).
        Prefer language and questions like "what happens if", do I have to", "can I change my mind", "how does this work". Avoid language like "ensure", "determine", "assess", "protocol", "criteria", "participants"

        Assess the user's message and respond ONLY in valid JSON matching this exact structure:
        {{
        "user_message": null,
        "label": one of: "ready", "vague", "good", "thoughtful, "unknown",
        "gesture": one of: "thinking", "thumbsup", "shrug",
        "in_scope": true or false,
        "tip": If suggestions are provided, briefly explain your thinking in one sentence why they're relevant to what the user typed using very casual, everyday, plain language to demonstrate informal low-authority conversational alignment; this explanation should be super conversational, like you're talking to someone. Keep it warm and under 20 words,
        "suggestions": a list of 2-3 short suggested questions asked from the user's perspective, or null. suggestions must be phrased as questions the USER would ask Doctor Alex, not questions Doctor Alex would ask the user. Suggestions should be based on what the user is most likely trying to ask and phrased very casually, informally, using everyday words, completing or expanding on their partial thought. Each question should be phrased in spoken, casual language, like a person would say out loud while thinking. Suggestions must not be about finding or enrolling in a specific clinical trial — stick to general educational questions about how clinical trials work. Use very casual language to demonstrate informal, low‑authority conversational alignment. The readability of each question should be no more than a 5th grade reading level (IMPORTANT). Use casual, spoken‑language questions, like something a person would say out loud while thinking — not formal or institutional phrasing. 
        }}

        Use this guide to pick the right values:
        - Message is empty → label: "ready", gesture: "thumbsup", tip: null, suggestions: null, in_scope: true
        - Message is a few characters with no meaningful words (such as "ddd) → label: "unknown", gesture: "shrug", tip: tell the user Doctor Alex can only respond to questions about clinical trials, suggestions: 2-3 questions to ask about clinical trials, in_scope: false
        - Message is small talk, a greeting, not relevant to clinical trials, or asking about a specific clinical trial (such as "i see" or "okay" or "where is the trial located") → label: "unknown", gesture: "shrug", tip: tell the user Doctor Alex can only respond to questions about clinical trials, suggestions: 2-3 questions to ask about clinical trials, in_scope: false
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
        model=UF_LOCAL_MODEL,
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
        You use very casual, everyday, plain language when you communicate to demonstrate informal low-authority conversational alignment. Every sentence should be like a spoken conversation.

        The user was asked: "What's one thing you've wondered about clinical trials?
        Don't worry about getting it perfect — just type whatever comes to mind."

        A precheck system has already analyzed their message and produced this:
        - Label: "{precheck.label}"
        - Tip: "{precheck.tip}"
        - Suggestions: {suggestions_str}

        Use this to craft a response with the following structure:
        1. In one brief sentence, acknowledge their message — if label is "good" or "thoughtful",
        affirm them warmly and naturally. If "vague" or "unknown", be warm and
        reassuring. Use casual, everyday, plain language to demonstrate informal low-authority conversational alignment. Phrase it as if you are having a spoken, casual conversation.
        2. Before introducing the suggestions, add one short framing sentence
        that explains why you're suggesting them based on the label.
        Again, use casual, everyday, plain language to demonstrate informal low-authority conversational alignment; phrase it as if you are having a spoken, casual conversation.
        Then introduce the suggestions naturally woven into a sentence using
        phrasings like "I might suggest questions like X or Y" or
        "some questions worth exploring might be X or Y".
        Do not say "in the actual tool" or "in the demo". Use casual, everyday, plain language to demonstrate informal low-authority conversational alignment; phrase it as if you are having a spoken, casual conversation.
        3. In one sentence, describe your role as suggesting questions like these during the interaction to
        support the user during their information search. Use casual, everyday, plain language to demonstrate informal low-authority conversational alignment; phrase it as if you are having a spoken, casual conversation.
        4. Close with one sentence handing off to Doctor Alex (eg, starting with "Now, or Next,"), letting the user know they are going to now meet Doctor Alex who will later be answering their questions. Use casual, everyday, plain language to demonstrate informal low-authority conversational alignment; phrase it as if you are having a spoken, casual conversation.

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
        response = await client_chat.beta.chat.completions.parse(
            model=UF_LOCAL_MODEL, messages=messages, temperature=0, response_format=LandingExample
        )
        result = response.choices[0].message.parsed
        return {"reply": result}
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

def normalize_word(word: str):
    return re.sub(r"[^a-z0-9']", "", word.lower())


@app.post("/tts")
async def tts(request: TTSRequest):
    sentences = [request.text.strip()]
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
        sentence = sentence.strip()
        if not sentence:
            continue

        res = await client_chat.audio.speech.create(
            model="kokoro",
            voice=characterVoice,
            input=sentence,
            speed=voiceSpeed,
        )

        buf = io.BytesIO(res.content)
        pcm, sr = sf.read(buf, dtype="int16")

        if samplerate is None:
            samplerate = sr

        all_pcm.append(pcm)

        duration = len(pcm) / sr

        expected_words = [
            normalize_word(w)
            for w in sentence.split()
            if normalize_word(w)
        ]
        expected_set = set(expected_words)

        whisper_buf = io.BytesIO(res.content)
        whisper_buf.name = "audio.mp3"

        transcript = await client_chat.audio.transcriptions.create(
            model="whisper-large-v3",
            file=whisper_buf,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            prompt=sentence,
            temperature=0,
        )

        for segment in transcript.model_dump().get("segments", []):
            for word in segment.get("words", []):
                clean_word = normalize_word(word.get("word", ""))
                start = word.get("start")
                end = word.get("end")

                if start is None or end is None:
                    continue

                word_duration = end - start

                # Drop Whisper hallucinations like "thank you for watching"
                if clean_word not in expected_set:
                    continue

                # Drop weird long fake words/timestamps
                if word_duration <= 0.01 or word_duration > 1.2:
                    continue

                all_words.append({
                    "word": word["word"],
                    "start": start + time_offset,
                    "end": end + time_offset,
                })

        time_offset += duration

    if not all_pcm:
        raise HTTPException(status_code=400, detail="No valid text to synthesize")

    combined_pcm = np.concatenate(all_pcm, axis=0)

    out = io.BytesIO()
    sf.write(out, combined_pcm, samplerate, format="mp3")
    combined_audio = out.getvalue()

    return {
        "audio": base64.b64encode(combined_audio).decode("utf-8"),
        "timestamps": all_words,
    }
    
def readable_source_name(source):
    source_map = {
        "NCI": "the National Cancer Institute",
        "FDA": "the FDA",
        "NIH": "the NIH",
        "ClinicalTrials.gov": "ClinicalTrials.gov",
    }
    return source_map.get(source, source)


def starts_with_source_cue(answer: str):
    lowered = answer.strip().lower()
    return lowered.startswith((
        "according to",
        "based on",
        "the fda",
        "the national cancer institute",
        "clinicaltrials.gov",
        "from what i found",
    ))


def add_spoken_source_cue(answer: str, sources: list, confidence: float):
    if not answer or not sources:
        return answer

    if confidence is not None and confidence < 0.35:
        return answer

    lowered = answer.lower()
    no_answer_phrases = [
        "i do not have enough information",
        "i don't have enough information",
        "the context does not contain",
        "i could not find",
        "i can’t find",
        "i can't find",
    ]

    if any(phrase in lowered for phrase in no_answer_phrases):
        return answer

    if starts_with_source_cue(answer):
        return answer

    source_names = []
    for source in sources[:2]:
        name = readable_source_name(source.get("source", ""))
        if name and name not in source_names:
            source_names.append(name)

    if not source_names:
        return answer

    if len(source_names) == 1:
        templates = [
            "According to {s1}, {answer}",
            "Based on what I found from {s1}, {answer}",
            "{s1} explains that {answer}",
            "From {s1}, the key idea is this: {answer}",
        ]
    else:
        templates = [
            "According to {s1} and {s2}, {answer}",
            "Based on what I found from {s1} and {s2}, {answer}",
            "{s1} and {s2} explain that {answer}",
            "From {s1} and {s2}, the key idea is this: {answer}",
        ]

    template = random.choice(templates)

    # Lowercase first character so it reads smoothly after the cue.
    cleaned_answer = answer.strip()
    cleaned_answer = cleaned_answer[0].lower() + cleaned_answer[1:]

    return template.format(
        s1=source_names[0],
        s2=source_names[1] if len(source_names) > 1 else "",
        answer=cleaned_answer,
    )
    
@app.post("/rag-chat")
async def rag_chat(request: ChatRequestHistory):
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
            f"""ID: {i}
        SOURCE: {m.get('source', '')}
        TITLE: {m.get('title', m.get('file', ''))}
        TYPE: {m.get('type', '')}
        FILE: {m.get('file', '')}
        URL: {m.get('url', '')}
        CONTENT: {res['text']}"""
        )
    context_str = "\n\n---\n\n".join(context_list)

    # 3. The System Prompt (Milo's older brother, the Researcher)
    system_prompt = """
    You are a clinical trials educator.

    Use the conversation history only to understand what the user is referring to.
    Do not use conversation history as evidence or as a source of facts.

    Answer the user's current question using ONLY the provided context.
    If the context does not contain the answer, clearly say you do not have enough information to answer.

    Write for someone with limited health literacy:
    - Aim for about a 5th–6th grade reading level.
    - Use simple, everyday words.
    - Keep sentences short.
    - Use active voice.
    - Prefer words with one or two syllables when possible.
    - Avoid medical jargon whenever possible.
    - If you must use a medical term, explain it in simple language.
    - Be friendly, clear, and reassuring.
    - Focus on the most important information.
    - Do not overwhelm the user with unnecessary details.
    - Speak directly to the user in a conversational tone.

    Also return talking_points:
    - Max 3 bullets.
    - Each bullet should be 4–9 words.
    - Use plain language.
    - Put them in the same order as the answer.
    - Do not include source names or citations.
    - If there is not enough information, return an empty list.

    Do not include citations, references, source names, or parenthetical citations in the answer.
    The application will handle displaying and mentioning sources separately.

    Keep the answer under 90 words.

    Return:
    1. answer
    2. source_explanations for the 2–3 most helpful source IDs
    3. confidence (0.0–1.0)
    4. talking_points

    For each source_explanation:
    - Use the exact source ID from the provided context.
    - Briefly explain, in simple language, why that source helped answer the question.
    - Only include sources that directly support the answer.

    Do not invent source IDs.
    """
    
    chat_messages = [
        {"role": "system", "content": system_prompt},
    ]

    # Add recent conversation history
    for turn in request.history[-20:]:
        chat_messages.append({
            "role": turn.role,
            "content": turn.content,
        })

    # Add current question + retrieved context
    chat_messages.append({
        "role": "user",
        "content": f"""
    CONTEXT:
    {context_str}

    CURRENT QUESTION:
    {question}
    """
    })

    print("IN RAG CHAT AB TO CALL LLM")
    
    # 4. Call the LLM with .parse()
    response = await client_chat.beta.chat.completions.parse(
        model=UF_LOCAL_MODEL,
        messages=chat_messages,
        response_format=RAGResponse
    )
    print("RESPONSE IS", response.choices[0].message.parsed)

    parsed = response.choices[0].message.parsed

    explanation_by_id = {
        item.id: item.relevance_explanation
        for item in parsed.source_explanations
    }

    sources = []

    for i, res in enumerate(results):
        if i not in explanation_by_id:
            continue

        meta = res["meta"]

        sources.append({
            "id": i,
            "source": meta.get("source"),
            "title": meta.get("title", meta.get("file")),
            "type": meta.get("type"),
            "url": meta.get("url", ""),
            "file": meta.get("file"),
            "chunk_id": meta.get("chunk_id"),
            "score": res["score"],
            "content": res["text"][:500],
            "relevance_explanation": explanation_by_id[i],
        })

    answer_with_source_cue = add_spoken_source_cue(
        parsed.answer,
        sources,
        parsed.confidence,
    )

    return {
        "answer": answer_with_source_cue,
        "sources": sources,
        "confidence": parsed.confidence,
        "talking_points": parsed.talking_points or [],
    }

@app.post("/evaluate-goal-progress", response_model=GoalEvalResponse)
async def evaluate_goal_progress(request: GoalEvalRequest):
    goals_json = json.dumps(
        [goal.model_dump() for goal in request.goals],
        ensure_ascii=False
    )

    system_prompt = """
    You are Jordan, a warm, casual companion helping a user track their learning goals while they talk with Dr. Alex about clinical trials.

    Evaluate the user's latest question and Dr. Alex's answer against ALL goals.

    Goal evaluation:
    - Evaluate each goal independently.
    - Only return a match if the user's latest question is meaningfully related to that goal.
    - If no goals match, return matches as an empty list.

    Suggested question:
    - Generate one suggested_goal_question when it would help.
    - Prefer an uncovered goal if one remains.
    - Otherwise, suggest a broader question about clinical trials.
    - Write at about a 4th–5th grade reading level.
    - Use short, everyday words.
    - Write in the user's voice using first-person wording when natural, like "Can I...", "Will I...", "How do I...", or "What happens if I..."
    - Ask about a practical detail, concern, decision, or next step.
    - Do not simply turn the goal title into a question.
    - Do not imply the user is already in a trial, choosing a trial, eligible for a trial, or receiving trial care.
    - Avoid: "this trial", "the trial", "my trial", "this study", "for me", "would I qualify", and "what if treatment doesn't work for me".
    - Return null if there is no useful new question.

    Note rules:
    - For note_to_add, write one short note beginning with "Learned that..."
    - Example: "Learned that people can leave a clinical trial at any time."
    - Example: "Learned that some trials help compare standard treatments."
    - Do not mention "the user", "Dr. Alex", or the conversation.
    - Keep it to one or two short sentences.
    - Return null if there is nothing useful to save.

    Response logic:
    - If no goals match and all goals are already covered, set all_goals_covered_message.
    - If no goals match and some goals remain, set no_match_jordan_message.
    - If one or more goals match and Alex answered well, set next_step_message and note_to_add.
    - If Alex did not answer well, suggest a follow-up question if useful.
    - If all goals are covered, set all_goals_covered_message and optionally suggest one broader question.

    Jordan message:
    - Mention the matched goal naturally.
    - If Alex answered the question, say a note was saved.
    - If Alex did not answer it well, say a follow-up may help.
    - Keep it under 20 words.
    - Be casual, encouraging, and simple.

    Repeated suggestion rules:
    - Do not suggest a question that appears in PREVIOUS JORDAN SUGGESTIONS.
    - Treat slightly reworded versions as repeats.
    - Do not repeat the user's latest question.
    - If Dr. Alex said there was not enough information, suggest a different question related to that goal.
    - If no useful alternative exists, return null.

    Return:
    - matches: a list of matched goals.
    - suggested_goal_question: one useful next question or null.
    - no_match_jordan_message: only populate when no goals match.
    - next_step_message: only populate when one or more goals matched and Alex answered well.
    - all_goals_covered_message: only populate when all goals appear covered.

    Each match should include only:
    - goal_id
    - user_question_relevant
    - alex_answered_question
    - jordan_message
    - note_to_add
    """

    response = await client_chat.beta.chat.completions.parse(
        model=UF_LOCAL_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
        USER MESSAGE:
        {request.user_message}

        DR. ALEX ANSWER:
        {request.alex_answer}

        PREVIOUS JORDAN SUGGESTIONS:
        {json.dumps(request.previous_suggestions, ensure_ascii=False)}

        CONDITION:
        {request.condition}

        GOALS:
        {goals_json}
        """
            }
        ],
        temperature=0,
        response_format=GoalEvalResponse,
    )

    parsed = response.choices[0].message.parsed

    if parsed is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to parse goal evaluation"
        )

    # Lookup goals by id
    goal_lookup = {goal.id: goal for goal in request.goals}

    # Populate already_addressed
    for match in parsed.matches:
        goal = goal_lookup.get(match.goal_id)
        if goal:
            match.already_addressed = (
                goal.addressed or bool(goal.notes)
            )

    return parsed

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

class PersonaMessage(BaseModel):
    response: str = Field(description="Response in first person language.")
    wordCount: int = Field(description="Number of words in response")


@app.post("/chat-with-personas")
async def chat_with_personas(request: ChatRequest):
    user_id = request.thread_id  # or session/conversation id
    history = conversation_history[user_id]

    shared_messages = history + [{"role": "user", "content": request.message}]

    personas = [generated_personas[0], generated_personas[1]]
    random.shuffle(personas)

    def build_msgs(persona, extra_context=""):
        system_text = (
            "Respond in first person based on your persona. "
            "Be concise, and use the conversation history. Keep your response to maximum 100 words. "
            + json.dumps(persona)
        )
        if extra_context:
            system_text += "\nOther persona just said the following; make sure your response is very different: " + extra_context
        return [
            {"role": "system", "content": system_text},
            *shared_messages
        ]

    response_a = await client_chat.beta.chat.completions.parse(
        model="mistral-small-3.1",
        messages=build_msgs(personas[0]),
        temperature=0,
        response_format=PersonaMessage,
    )

    reply_a = response_a.choices[0].message.parsed

    response_b = await client_chat.beta.chat.completions.parse(
        model="mistral-small-3.1",
        messages=build_msgs(personas[1], extra_context=str(reply_a)),
        temperature=0,
        response_format=PersonaMessage,
    )

    reply_b = response_b.choices[0].message.parsed

    conversation_history[user_id].extend([
        {"role": "user", "content": request.message},
        {"role": "assistant", "content": str(reply_a)},
        {"role": "assistant", "content": str(reply_b)},
    ])
    print("**conversation history", conversation_history)

    return {"reply1": reply_a, "reply2": reply_b}

@app.post("/chat-with-personas-old")
async def chat_with_personas_old(request: ChatRequest):
    print("IN SIMPLE CHAT", request)
    global generated_personas
    print("persona1", generated_personas[0])
    print("persona2", generated_personas[1])
    messages1: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "Respond based on your persona in first person (maximum 50 words):" + json.dumps(generated_personas[0])},
        {"role": "user", "content": request.message}
    ]
    messages2: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "Respond based on your persona in first person (maximum 50 words):" + json.dumps(generated_personas[1])},
        {"role": "user", "content": request.message}
    ]
    try:
        # call the async LLM client
        response1 = await client_chat.beta.chat.completions.parse(model='mistral-small-3.1', messages=messages1, temperature=0, response_format=PersonaMessage)
        response2 = await client_chat.beta.chat.completions.parse(model='mistral-small-3.1', messages=messages2, temperature=0, response_format=PersonaMessage)
        # response might be a string or a dict; check your client
        print("RESPONSE 1 IS", response1.choices[0].message.parsed)
        print("RESPONSE 2 IS", response2.choices[0].message.parsed)
        return {"reply1": response1.choices[0].message.parsed, "reply2": response2.choices[0].message.parsed}
    except Exception as e:
        return {"error": str(e)}


class Persona(BaseModel):
    name_and_surname: str = Field(description="A realistic name and surname.")
    country: str = Field(description="The country of the persona.")
    career_level_and_discipline: str = Field(description="Career level and discipline.")
    
    # Goals and Frustrations
    main_goal: str = Field(description="Main goal chosen from the Goals list.")
    main_frustration: str = Field(description="Main frustration chosen from the Frustrations list.")
    goal_quote: str = Field(description="A quote taken from the Goal Quotes list representing the main goal.")
    
    # Narrative & Traits
    narrative_background: str = Field(description="A narrative background of the persona (max 250 words).")
    behaviors: List[str] = Field(description="List of 3 key behaviors chosen from the Behaviors list (approx 20 words each).")
    personality_traits: List[str] = Field(description="List of 3 main personality traits chosen from the Personality Traits list (approx 20 words each).")
    
    # Additional Details
    additional_goals: List[str] = Field(description="2 additional goals from the Goals list (max 20 words each).")
    additional_frustrations: List[str] = Field(description="2 additional frustrations from the Frustrations list (max 20 words each).")
    connection_to_user: str = Field(description="Brief statement why/how this persona is a good match based on the user's message (max 250 words).")

# This is the wrapper that forces the output to be a clean array (list) of 2 personas
class PersonaResponse(BaseModel):
    personas: List[Persona] = Field(description="A list containing exactly 2 detailed personas.")

@app.post("/personas")
async def personas(request: UserInfo):
    print("IN PERSONAS", request)
    global generated_personas
    themes_folder = "themes"

    # 1. Define your file paths
    behaviors = os.path.join(themes_folder, "Behaviors.csv")
    frustrations = os.path.join(themes_folder, "Frustrations.csv")
    goals = os.path.join(themes_folder, "Goals.csv")
    goals_quotes = os.path.join(themes_folder, "Goals_Quotes.csv")
    personality_traits = os.path.join(themes_folder, "PersonalityTraits.csv")

    # Capture the user data from your CSV lookup function
    user_data = getUser(request.userId)

    # 2. Read the contents of each file
    with open(behaviors, "r", encoding="utf-8") as f:
        behaviors_data = f.read()

    with open(frustrations, "r", encoding="utf-8") as f:
        frustrations_data = f.read()

    with open(goals, "r", encoding="utf-8") as f:
        goals_data = f.read()

    with open(goals_quotes, "r", encoding="utf-8") as f:
        goals_quotes_data = f.read()

    with open(personality_traits, "r", encoding="utf-8") as f:
        personality_data = f.read()

    # 3. Cleaned-up Prompt (No formatting rules needed anymore!)
    persona_generation_prompt = f"""
    Using the lists provided below, generate exactly 2 detailed and comprehensive personas for individuals who are learning about clinical trials.
    These individuals should have similar views on the information provided in the user's survey responses, but their personality and behaviors should be very different from each other.

    List of frustrations: 
    {frustrations_data}

    List of Goals: 
    {goals_data}

    List of Behaviors: 
    {behaviors_data}

    List of Personality Traits: 
    {personality_data}

    List of Goal Quotes:
    {goals_quotes_data}
    """

    # Pass the actual data from the user row into the prompt if possible, 
    # instead of just their ID string, so the LLM can match views effectively!
    user_content = f"User Survey Data: {user_data}" if user_data else f"User ID: {request.userId}"

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": persona_generation_prompt},
        {"role": "user", "content": user_content}
    ]
    
    try:
        # --- THE CHANGES START HERE ---
        
        # 1 & 2: Use .beta.chat.completions.parse and pass the response_format
        response = await client_chat.beta.chat.completions.parse(
            model='gpt-oss-120b', 
            messages=messages, 
            temperature=0,
            response_format=PersonaResponse  # Instructs OpenAI to structure data to this model
        )
        
        # 3. Access the strongly typed, validated Pydantic object
        parsed_response = response.choices[0].message.parsed
        
        print("PARSED RESPONSE OBJECT:", parsed_response)
        
        # Turn the object back into a clean list of dictionaries (an array) for your API reply
        personas_array = [persona.model_dump() for persona in parsed_response.personas]
        generated_personas = personas_array
        return {"reply": personas_array}
        
        # --- THE CHANGES END HERE ---

    except Exception as e:
        return {"error": str(e)}
    
@app.post("/personas-old")
async def personas_old(request: UserInfo):
    print("IN PERSONAS", request)
    themes_folder = "themes"

    # 1. Define your file paths
    behaviors = os.path.join(themes_folder, "Behaviors.csv")
    frustrations = os.path.join(themes_folder, "Frustrations.csv")
    goals = os.path.join(themes_folder, "Goals.csv")
    goals_quotes = os.path.join(themes_folder, "Goals_Quotes.csv")
    personality_traits = os.path.join(themes_folder, "PersonalityTraits.csv")

    getUser(request.userId)

    # 2. Read the contents of each file
    with open(behaviors, "r", encoding="utf-8") as f:
        behaviors_data = f.read()

    with open(frustrations, "r", encoding="utf-8") as f:
        frustrations_data = f.read()

    with open(goals, "r", encoding="utf-8") as f:
        goals_data = f.read()

    with open(goals_quotes, "r", encoding="utf-8") as f:
        goals_quotes_data = f.read()

    with open(personality_traits, "r", encoding="utf-8") as f:
        personality_data = f.read()

    # 3. Insert the data into the prompt using an f-string
    # (Note the 'f' before the triple quotes and the variables wrapped in {})
    persona_generation_prompt = f"""
    Using the lists provided below, generate 2 detailed and comprehensive personas for individuals who are learning about clinical trials that would have similar views on the information in the user's message, which is a user's survey responses. Each persona should be structured as follows:

    # "Name and surname": make up a realistic name and surname
    ## "Country": make up a the country of the persona
    ## "Career Level and Discipline": make up a career level and discipline 
    ## "Goal & Frustration": tell what is the persona main goal (choose 1 from the Goals list) and what is the persona main frustration (choose 1 from the Frustrations list). Include a quote taken from the quotes in the Goal Quotes list representing the main goal 
    ### "Narrative": include also a narrative background of the persona (max 250 words).

    In the Narrative include also the following two aspects: 
    ## "Behavior": identify the persona's key behaviours (choose 3 from the Behaviors list, 20 words each), 
    ## "Personality": identify the main persona personality traits (choose 3 from the Personality Traits list, 20 words each).

    # Additional goals: Identify with bullet points 2 additional goals from the Goals list ( max 20 words each) and 2 additional frustrations from the Frustrations list (max 20 words each) of the persona.
    # Connection to user: Brief statement why/how this persona is a good match based on the user's message (maximum 250 words)

    Return the 2 personas as 2 objects in a JSON.

    List of frustrations: 
    {frustrations_data}

    List of Goals: 
    {goals_data}

    List of Behaviors: 
    {behaviors_data}

    List of Personality Traits: 
    {personality_data}

    List of Goal Quotes:
    {goals_quotes_data}
    """

    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": persona_generation_prompt},
        {"role": "user", "content": request.userId}
    ]
    try:
        # call the async LLM client
        response = await client_chat.chat.completions.create(model=UF_LOCAL_MODEL, messages=messages, temperature=0)
        # response might be a string or a dict; check your client
        print("RESPONSE IS", response.choices[0].message.content)
        return {"reply": response.choices[0].message.content}
        # return {"reply": "check console"}
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/generate-initial-goals")
async def generate_initial_goals(request: dict):
    response_id = request.get("response_id")

    if not response_id:
        raise HTTPException(status_code=400, detail="Missing response_id")

    presurvey_row = await get_presurvey_row_from_qualtrics(response_id)

    scores = score_presurvey_row(presurvey_row)

    goals = await generate_goals_from_scores(scores)

    return goals

@app.post("/suggest-more-goals")
async def suggest_more_goals(request: dict):
    response_id = request.get("response_id")
    existing_goals = request.get("existing_goals", [])

    if not response_id:
        raise HTTPException(status_code=400, detail="Missing response_id")

    presurvey_row = await get_presurvey_row_from_qualtrics(response_id)

    scores = score_presurvey_row(presurvey_row)

    goals = await generate_more_goals_from_scores(
        scores=scores,
        existing_goals=existing_goals,
    )

    return goals

@app.get("/")
async def root():
    return {"message": "Welcome to Rashi's FastAPI server!"}

if __name__ == "__main__":
    # Run the FastAPI application on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)