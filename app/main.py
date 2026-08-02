from fastapi import FastAPI, HTTPException, Query
from logging_routes import router as log_router
from logging_routes_pilot import router as log_router_pilot
from qualtrics import (
    get_presurvey_row_from_qualtrics,
    score_presurvey_row,
    generate_goals_from_scores,
    generate_more_goals_from_scores,
)
from pydantic import BaseModel, Field, field_validator
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
from enum import Enum

load_dotenv()

useCORS = True

# Endpoints allowed to access this server
origins = ["https://main.d355vauwiio7nq.amplifyapp.com", "https://idea.d355vauwiio7nq.amplifyapp.com", "http://localhost:5173", "https://ufl.qualtrics.com"]

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

# Regular LiteLLM client for conversational responses (async)
client_chat = AsyncOpenAI(
    api_key= RASHI_LITELLM_KEY,
    base_url= base_url # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
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

def read_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = []

    for p in reader.pages:
        t = p.extract_text() or ""
        t = clean_pdf_text(t)
        pages.append(t)

    return "\n".join(pages)

def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0

    while start < len(text):
        proposed_end = min(len(text), start + chunk_size)
        end = proposed_end

        if proposed_end < len(text):
            sentence_end = text.rfind(".", start, proposed_end)

            if sentence_end != -1 and sentence_end > start + 300:
                end = sentence_end + 1

        chunk = text[start:end].strip()

        if len(chunk.split()) >= 20:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = max(0, end - overlap)

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

def build_resource_cards(results, max_cards=3):
    cards = []
    seen_files = set()

    for i, res in enumerate(results):
        meta = res["meta"]
        file_key = meta.get("file") or meta.get("title") or f"source-{i}"

        if file_key in seen_files:
            continue

        seen_files.add(file_key)

        cards.append({
            "id": i,
            "source": meta.get("source", "Trusted source"),
            "title": meta.get("title") or meta.get("file") or "Trusted resource",
            "file": meta.get("file"),
            "type": meta.get("type", "unknown"),
            "url": meta.get("url", ""),
            "page_number": meta.get("page_number"),
            "chunk_id": meta.get("chunk_id"),
            "excerpt": res["text"][:500],
            "score": res["score"],
        })

        if len(cards) == max_cards:
            break

    return cards

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

class JordanThemeDetail(BaseModel):
    id: str
    text: str
    source_question: str = ""
    source_answer: str = ""

    @field_validator(
        "source_question",
        "source_answer",
        mode="before",
    )
    @classmethod
    def replace_null_with_empty_string(cls, value):
        return value if isinstance(value, str) else ""

class JordanTheme(BaseModel):
    id: str
    label: str
    summary: str
    details: List[JordanThemeDetail] = Field(default_factory=list)


class JordanConnection(BaseModel):
    # Theme the newest idea was added to.
    theme_id: str

    # Earlier detail that the newest idea meaningfully connects to.
    earlier_detail_id: str

    # Short text for internal logging or an optional compact UI cue.
    label: str

    # User-facing explanation of the relationship.
    text: str

    # Natural reminder of what the user asked earlier.
    earlier_question_reference: str


class JordanConversationModelData(BaseModel):
    themes: List[JordanTheme] = Field(default_factory=list)
    latestConnection: JordanConnection | None = None


class JordanTurnUpdateRequest(BaseModel):
    user_question: str
    alex_answer: str
    history: List[ChatTurn] = Field(default_factory=list)
    current_model: JordanConversationModelData

    previous_guidance_types: List[
        Literal[
            "make_more_specific",
            "different_perspective",
            "related_idea",
        ]
    ] = Field(default_factory=list)

    previous_guidance_messages: List[str] = Field(default_factory=list)


class JordanTurnUpdateResponse(BaseModel):
    # The complete updated theme collection replaces frontend memory.
    themes: List[JordanTheme]

    # Present only when the latest idea meaningfully builds on an earlier one.
    latest_connection: JordanConnection | None = None

    guidance_type: Literal[
        "make_more_specific",
        "different_perspective",
        "related_idea",
    ]

    # The only text Jordan speaks and the primary text shown to the user.
    jordan_message: str

class QueryPreprocess(BaseModel):
    route: Literal[
        "clinical_trials_education",
        "personal_medical_advice",
        "trial_recommendation_or_eligibility",
        "political_or_policy",
        "unrelated",
    ]
    search_query: str

class AlexAnswerScope(str, Enum):
    GENERAL_ANSWER = "general_answer"
    VARIES_BY_TRIAL = "varies_by_trial"
    PERSONALIZED_DECISION = "personalized_decision"
    INSUFFICIENT_CONTEXT = "insufficient_context"


class RAGResponseV2(BaseModel):
    answer: str

    # Reuse your existing SourceExplanation model:
    # class SourceExplanation(BaseModel):
    #     id: int
    #     relevance_explanation: str
    source_explanations: List[SourceExplanation] = Field(
        default_factory=list
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    talking_points: List[str] = Field(
        default_factory=list,
        max_length=3,
    )

    answer_scope: AlexAnswerScope


class JordanDecisionTurnUpdateRequest(BaseModel):
    user_question: str
    alex_answer: str
    alex_answer_scope: AlexAnswerScope

    # Your existing history model is ChatTurn.
    history: List[ChatTurn] = Field(
        default_factory=list
    )

    # Your existing workspace model has this exact name.
    current_model: JordanConversationModelData

    previous_guidance_types: List[
        Literal[
            "make_more_specific",
            "different_perspective",
            "related_idea",
        ]
    ] = Field(default_factory=list)

    previous_guidance_messages: List[str] = Field(
        default_factory=list
    )

def next_jordan_theme_id(themes: List[JordanTheme]) -> str:
    highest_number = 0

    for theme in themes:
        match = re.fullmatch(r"theme-(\d+)", theme.id)

        if match:
            highest_number = max(
                highest_number,
                int(match.group(1)),
            )

    return f"theme-{highest_number + 1}"


def next_jordan_detail_id(themes: List[JordanTheme]) -> str:
    highest_number = 0

    for theme in themes:
        for detail in theme.details:
            match = re.fullmatch(r"detail-(\d+)", detail.id)

            if match:
                highest_number = max(
                    highest_number,
                    int(match.group(1)),
                )

    return f"detail-{highest_number + 1}"

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

def sanitize_timestamps(timestamps: list[dict], duration: float) -> list[dict]:
    cleaned = []
    last_end = 0.0

    for entry in timestamps:
        start, end = entry["start"], entry["end"]

        # Reject anything beyond the actual audio length
        if start > duration or end > duration + 0.1:
            continue

        # Reject anything that goes backward or overlaps badly with the last accepted word
        if start < last_end - 0.05:
            continue

        # Reject exact-duplicate consecutive words with near-zero gap (hallucination pattern)
        if cleaned and normalize_word(entry["word"]) == normalize_word(cleaned[-1]["word"]):
            if start - cleaned[-1]["end"] < 0.05:
                continue

        cleaned.append(entry)
        last_end = end

    if len(cleaned) < 0.6 * len(timestamps):
        return []

    return cleaned

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

    # # Send the generated audio to Whisper.
    # whisper_buffer = io.BytesIO(audio_bytes)
    # whisper_buffer.name = "audio.mp3"

    # transcript = await client_chat.audio.transcriptions.create(
    #     model="whisper-large-v3",
    #     file=whisper_buffer,
    #     response_format="verbose_json",
    #     timestamp_granularities=["word"],
    #     prompt=spoken_text,
    #     temperature=0,
    # )

    # timestamps = []

    # transcript_data = transcript.model_dump()

    # for segment in transcript_data.get("segments", []):
    #     for word_data in segment.get("words", []):
    #         word = word_data.get("word", "").strip()
    #         clean_word = normalize_word(word)

    #         start = word_data.get("start")
    #         end = word_data.get("end")

    #         if not clean_word:
    #             continue

    #         if start is None or end is None:
    #             continue

    #         word_duration = end - start

    #         # Reject only clearly invalid timestamps.
    #         if word_duration <= 0.01:
    #             continue

    #         if word_duration > 2.0:
    #             continue

    #         timestamps.append({
    #             "word": word,
    #             "start": float(start),
    #             "end": float(end),
    #         })

    # expected_word_count = len([
    #     word
    #     for word in spoken_text.split()
    #     if normalize_word(word)
    # ])

    # timestamps = sanitize_timestamps(timestamps, duration)

    # timestamp_coverage = (
    #     len(timestamps) / expected_word_count
    #     if expected_word_count > 0
    #     else 0
    # )

    # print(
    #     "[TTS TIMESTAMP DEBUG]",
    #     {
    #         "audio_duration": round(duration, 2),
    #         "expected_words": expected_word_count,
    #         "timestamp_words": len(timestamps),
    #         "coverage": round(timestamp_coverage, 2),
    #         "whisper_text": transcript_data.get("text", ""),
    #     },
    # )

    # # Always use the original spoken text for subtitle words.
    # # Whisper can mishear numbers such as Phase 1, Phase 2, and Phase 3.
    # timestamps = create_fallback_timestamps(
    #     spoken_text,
    #     duration,
    # )

    # return {
    #     "audio": base64.b64encode(
    #         audio_bytes,
    #     ).decode("utf-8"),
    #     "timestamps": timestamps,
    # }

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
    recent_history = history[-4:] if history else []

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
        {recent_history}

        Latest message:
        {question}
        """

    response = await client_chat.beta.chat.completions.parse(
        model=UF_LOCAL_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=QueryPreprocess,
    )

    print("***PREPROCESS RETURNED", response.choices[0].message.parsed)

    return response.choices[0].message.parsed

class RagDebugRequest(BaseModel):
    question: str
    k: int = 10


@app.post("/debug-rag")
async def debug_rag(request: RagDebugRequest):
    # Generate the same search query used by Alex.
    preprocess = await preprocess_question(
        request.question,
        history=[],
    )

    results_original = rag.retrieve(
        request.question,
        k=request.k,
    )

    results_rewritten = rag.retrieve(
        preprocess.search_query,
        k=request.k,
    )

    return {
        "original_question": request.question,
        "search_query": preprocess.search_query,

        "original_results": [
            {
                "rank": index + 1,
                "score": result["score"],
                "title": result["meta"].get("title"),
                "page_number": result["meta"].get("page_number"),
                "text": result["text"],
            }
            for index, result in enumerate(results_original)
        ],

        "rewritten_results": [
            {
                "rank": index + 1,
                "score": result["score"],
                "title": result["meta"].get("title"),
                "page_number": result["meta"].get("page_number"),
                "text": result["text"],
            }
            for index, result in enumerate(results_rewritten)
        ],
    }

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

@app.post("/rag-chat")
async def rag_chat(request: ChatRequestHistory):
    print("***IN RAG CHAT")
    question = request.message

    preprocess = await preprocess_question(question, request.history)
    print("***THIS IS PREPROCESS IN RAG CHAT", preprocess)
    print(type(preprocess))
    print(preprocess.route)
    print(preprocess.search_query)

    if preprocess.route != "clinical_trials_education":
        print("PREPROCESS ROUTE IS NOT CLINICAL TRIAL EDUCATION RELATED", preprocess)

        # route_responses = {
        #     "personal_medical_advice": {
        #         "answer": (
        #             "I can’t give medical advice or tell you what treatment is best for you. "
        #             "That depends on your health and should be discussed with your cancer care team. "
        #             "Jordan can help you save this as a question to ask your doctor."
        #         ),
        #         "talking_points": [
        #             "Ask your care team",
        #             "Save this question",
        #             "Discuss your options",
        #         ],
        #         "sources": [
        #             {
        #                 "id": 0,
        #                 "source": "NCI",
        #                 "title": "Questions to Ask Your Doctor About Treatment",
        #                 "type": "support_resource",
        #                 "url": "https://www.cancer.gov/about-cancer/coping/questions",
        #                 "file": None,
        #                 "chunk_id": None,
        #                 "score": 1.0,
        #                 "content": "A guide to help patients prepare questions for their cancer care team.",
        #                 "relevance_explanation": "This can help you prepare questions for your doctor instead of making treatment decisions here.",
        #             }
        #         ],
        #     },
        #     "trial_recommendation_or_eligibility": {
        #         "answer": (
        #             "I can’t tell you if a specific clinical trial is right for you or if you qualify. "
        #             "Eligibility depends on many personal health details. "
        #             "Jordan can help you prepare questions about what doctors look at."
        #         ),
        #         "talking_points": [
        #             "Eligibility depends on health",
        #             "Ask what doctors check",
        #             "Prepare trial questions",
        #         ],
        #         "sources": [
        #             {
        #                 "id": 0,
        #                 "source": "NCI",
        #                 "title": "Questions to Ask Your Doctor About Treatment",
        #                 "type": "support_resource",
        #                 "url": "https://www.cancer.gov/about-cancer/coping/questions",
        #                 "file": None,
        #                 "chunk_id": None,
        #                 "score": 1.0,
        #                 "content": "A guide to help patients prepare questions for their cancer care team.",
        #                 "relevance_explanation": "This can help you prepare questions for your doctor instead of making treatment decisions here.",
        #             }
        #         ],
        #     },
        #     "political_or_policy": {
        #         "answer": (
        #             "I’m here to help explain clinical trials, not discuss political opinions or policy debates. "
        #             "Jordan can help you learn about participant protections, like informed consent, IRBs, and safety rules."
        #         ),
        #         "talking_points": [
        #             "Stay focused on trials",
        #             "Learn participant protections",
        #             "Ask about safety rules",
        #         ],
        #         "sources": [
        #             {
        #                 "id": 0,
        #                 "source": "HHS",
        #                 "title": "About Research Participation",
        #                 "type": "support_resource",
        #                 "url": "https://www.hhs.gov/ohrp/education-and-outreach/about-research-participation/index.html",
        #                 "file": None,
        #                 "chunk_id": None,
        #                 "score": 1.0,
        #                 "content": "Information about research participation, rights, consent, and protections.",
        #                 "relevance_explanation": "This helps redirect policy questions toward participant rights and safety protections.",
        #             }
        #         ],
        #     },
        #     "unrelated": {
        #         "answer": (
        #             "I’m here to help with questions about clinical trials. "
        #             "Jordan can help bring us back to your goals and suggest a question you may want to ask next."
        #         ),
        #         "talking_points": [
        #             "Return to your goals",
        #             "Ask about trials",
        #             "Jordan can suggest questions",
        #         ],
        #     },
        # }

        # fallback = route_responses.get(preprocess.route, route_responses["unrelated"])

        # return {
        #     "answer": fallback["answer"],
        #     "sources": [],
        #     "confidence": 1.0,
        #     "talking_points": fallback["talking_points"],
        # }
    
    
    # 1. Get RAG results
    results = rag.retrieve(preprocess.search_query, k=8)
    
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

    If the context does not contain the answer:
    - Clearly say the sources you have access to doesn't have enough information to answer.
    - Do NOT say "the sources you gave"; always refer to the sources as "the sources I have" or something along those lines

    Write using plain language:
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

    Answer format:
    - Write the answer as one conversational paragraph.
    - Do not use bullet points, numbered lists, headings, or line breaks.
    - Do not begin sentences with symbols such as "-", "*", or "•".
    - The answer should sound natural when spoken aloud.

    Numbers and trial phases:
    - Write clinical trial phases using regular numbers: Phase 1, Phase 2, Phase 3, and Phase 4.
    - Never write phase numbers as Roman numerals such as Phase I, Phase II, Phase III, or Phase IV.

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

    CURRENT USER'S ORIGINAL QUESTION:
    {question}

    Search query used for retrieval:
    {preprocess.search_query}
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

    sources = build_resource_cards(results)

    return {
        "answer": clean_alex_answer(parsed.answer),
        "sources": sources,
        "confidence": parsed.confidence,
        "talking_points": parsed.talking_points or [],
    }

@app.post("/rag-chat-v2")
async def rag_chat_v2(
    request: ChatRequestHistory,
):
    print("*** IN RAG CHAT V2")

    question = request.message

    preprocess = await preprocess_question(
        question,
        request.history,
    )

    print("*** RAG V2 PREPROCESS:", preprocess)
    print("*** RAG V2 ROUTE:", preprocess.route)
    print("*** RAG V2 SEARCH QUERY:", preprocess.search_query)

    # Retrieve educational source material.
    results = rag.retrieve(
        preprocess.search_query,
        k=8,
    )

    context_list = []

    for i, res in enumerate(results):
        meta = res["meta"]

        context_list.append(
            f"""
ID: {i}
SOURCE: {meta.get("source", "")}
TITLE: {meta.get("title", meta.get("file", ""))}
TYPE: {meta.get("type", "")}
FILE: {meta.get("file", "")}
URL: {meta.get("url", "")}
CONTENT: {res["text"]}
""".strip()
        )

    context_str = "\n\n---\n\n".join(context_list)

    system_prompt = """
You are Alex, a clinical trials educator.

Use conversation history only to understand what the user means.
Use ONLY the provided context as factual evidence.

Answer the user's current question in plain, conversational language.
When the concern behind the question is clear, respond to that concern while staying within the provided facts.

If there is no single general answer or the context cannot answer the question exactly:
- Do not stop at saying the sources lack the answer.
- Briefly explain why the answer varies or remains unknown.
- Give any useful general information supported by the context.
- Name 2–3 things that would need to be checked for a specific trial.

Do not ask for personal health information.
Do not recommend a trial, judge eligibility, choose a treatment, or give medical advice.

Choose one answer_scope:
- general_answer: the context directly answers the general question;
- varies_by_trial: the general idea is known, but details depend on the specific trial;
- personalized_decision: the user asks what is best or appropriate for them;
- insufficient_context: the context provides almost no useful information.

Use simple words and short sentences.
Explain medical terms in plain language.
Be friendly, direct, and reassuring.

Write one conversational paragraph under 90 words.
Do not use headings, lists, citations, source names, or line breaks.
Write phases as Phase 1, Phase 2, Phase 3, and Phase 4.

Return:
1. answer
2. source_explanations
3. confidence
4. talking_points
5. answer_scope

For talking_points:
- Return at most 3.
- Each should be 4–9 words.
- Use plain language.
- Keep them in the same order as the answer.
- Do not include citations or source names.
- Return an empty list only when no useful supported information can be given.

For each source_explanation:
- Use the exact source ID in the id field.
- Explain why it helped in relevance_explanation.
- Include only sources that directly support the answer.

Do not invent source IDs.
"""

    chat_messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    for turn in request.history[-20:]:
        chat_messages.append(
            {
                "role": turn.role,
                "content": turn.content,
            }
        )

    chat_messages.append(
        {
            "role": "user",
            "content": f"""
CONTEXT:
{context_str}

CURRENT USER QUESTION:
{question}

PREPROCESS ROUTE:
{preprocess.route}

SEARCH QUERY:
{preprocess.search_query}
""".strip(),
        }
    )

    try:
        response = await client_chat.beta.chat.completions.parse(
            model=UF_LOCAL_MODEL,
            messages=chat_messages,
            temperature=0,
            response_format=RAGResponseV2,
        )

        parsed = response.choices[0].message.parsed

        if parsed is None:
            raw_content = response.choices[0].message.content

            print(
                "*** RAG CHAT V2 PARSE FAILED:",
                raw_content,
            )

            raise HTTPException(
                status_code=500,
                detail="Failed to parse Alex response",
            )

        print("*** RAG CHAT V2 RESPONSE:", parsed)

        sources = build_resource_cards(results)

        return {
            "answer": clean_alex_answer(parsed.answer),
            "sources": sources,
            "confidence": parsed.confidence,
            "talking_points": parsed.talking_points or [],
            "answer_scope": parsed.answer_scope.value,
        }

    except HTTPException:
        raise

    except Exception as error:
        print(
            "*** RAG CHAT V2 ERROR:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail="Could not generate Alex response",
        )

def validate_jordan_turn_update_v2(
    *,
    parsed: JordanTurnUpdateResponse,
    request: JordanDecisionTurnUpdateRequest,
    new_theme_id: str,
    new_detail_id: str,
) -> None:
    existing_theme_ids = {theme.id for theme in request.current_model.themes}
    existing_theme_lookup = {theme.id: theme for theme in request.current_model.themes}

    existing_detail_ids = {
        detail.id
        for theme in request.current_model.themes
        for detail in theme.details
    }
    existing_detail_lookup = {
        detail.id: (theme.id, detail)
        for theme in request.current_model.themes
        for detail in theme.details
    }

    returned_theme_ids = {theme.id for theme in parsed.themes}

    # --- Repair: re-attach any existing themes the model dropped ---
    missing_theme_ids = existing_theme_ids - returned_theme_ids
    for missing_id in missing_theme_ids:
        print("*** JORDAN V2 REPAIR: re-adding dropped theme", missing_id)
        parsed.themes.append(existing_theme_lookup[missing_id])

    # --- Repair: re-attach any existing details the model dropped ---
    returned_detail_ids = {
        detail.id for theme in parsed.themes for detail in theme.details
    }
    missing_detail_ids = existing_detail_ids - returned_detail_ids
    for missing_id in missing_detail_ids:
        print("*** JORDAN V2 REPAIR: re-adding dropped detail", missing_id)
        theme_id, detail = existing_detail_lookup[missing_id]
        target_theme = next(
            (t for t in parsed.themes if t.id == theme_id), None
        )
        if target_theme is not None:
            target_theme.details.append(detail)
        else:
            # Theme itself vanished and wasn't in existing_theme_ids repair
            # (shouldn't normally happen) — fall back to original theme.
            parsed.themes.append(existing_theme_lookup.get(theme_id) or None)

    # --- Fix placeholder / empty labels instead of failing the turn ---
    for theme in parsed.themes:
        if not theme.label or theme.label.strip().lower() in {
            "new theme", "untitled", "theme"
        }:
            # Derive a fallback label from the first detail's text.
            fallback_source = theme.details[0].text if theme.details else theme.summary
            fallback_label = (fallback_source or "General concern")[:40].strip()
            print(
                "*** JORDAN V2 REPAIR: replacing placeholder label",
                {"theme_id": theme.id, "was": theme.label, "now": fallback_label},
            )
            theme.label = fallback_label

        if not theme.summary or not theme.summary.strip():
            theme.summary = (
                theme.details[0].text[:100] if theme.details else "Ideas related to this concern."
            )

    # --- Enforce the new detail was actually created; try to recover ---
    all_returned_details = [d for t in parsed.themes for d in t.details]
    new_details = [d for d in all_returned_details if d.id == new_detail_id]

    if len(new_details) == 0:
        # Model may have used a wrong/duplicate id for the "new" detail.
        # Find a detail whose id isn't in existing_detail_ids at all —
        # treat it as the intended new detail and relabel it.
        unknown_details = [
            d for d in all_returned_details if d.id not in existing_detail_ids
        ]
        if len(unknown_details) == 1:
            print(
                "*** JORDAN V2 REPAIR: relabeling stray new detail id",
                {"was": unknown_details[0].id, "now": new_detail_id},
            )
            unknown_details[0].id = new_detail_id
            new_details = unknown_details
        else:
            # Truly nothing new — synthesize a minimal detail rather than 500ing.
            print("*** JORDAN V2 REPAIR: no new detail found, synthesizing fallback")
            fallback_theme = parsed.themes[0] if parsed.themes else None
            fallback_detail = JordanThemeDetail(
                id=new_detail_id,
                text="Alex shared new information relevant to this concern.",
            )
            if fallback_theme is not None:
                fallback_theme.details.append(fallback_detail)
            else:
                parsed.themes.append(
                    JordanTheme(
                        id=new_theme_id,
                        label="New consideration",
                        summary="A new idea from Alex's latest answer.",
                        details=[fallback_detail],
                    )
                )
            new_details = [fallback_detail]

    elif len(new_details) > 1:
        # Duplicate new-detail ids — keep the first, rename the rest.
        print("*** JORDAN V2 REPAIR: duplicate new detail ids, deduping")
        for extra in new_details[1:]:
            extra.id = f"{new_detail_id}-dup-{id(extra)}"
        new_details = new_details[:1]

    new_detail = new_details[0]
    new_detail.source_question = request.user_question
    new_detail.source_answer = request.alex_answer

    # --- Cap at 3 themes, keeping the one with the new detail ---
    if len(parsed.themes) > 3:
        theme_with_new_detail = next(
            (t for t in parsed.themes if any(d.id == new_detail.id for d in t.details)),
            None,
        )
        remaining = [t for t in parsed.themes if t is not theme_with_new_detail]
        parsed.themes = (
            ([theme_with_new_detail] if theme_with_new_detail else []) + remaining
        )[:3]

    # --- Drop any stray IDs that aren't recognized (instead of raising) ---
    allowed_theme_ids = existing_theme_ids | {new_theme_id} | {t.id for t in parsed.themes}
    allowed_detail_ids = existing_detail_ids | {new_detail_id}

    for theme in parsed.themes:
        theme.details = [
            d for d in theme.details
            if d.id in allowed_detail_ids or d.id == new_detail.id
        ]

    # --- Validate the optional connection (unchanged, already lenient) ---
    if parsed.latest_connection is None:
        return

    connection = parsed.latest_connection
    returned_theme_ids = {theme.id for theme in parsed.themes}

    if connection.theme_id not in returned_theme_ids:
        parsed.latest_connection = None
        return

    if connection.earlier_detail_id not in existing_detail_ids:
        parsed.latest_connection = None

def _infer_concern_label(theme: JordanTheme) -> str:
    """
    Best-effort fallback label when the model fails to supply one.
    Frames the theme as the user's concern (grounded in what they asked),
    oriented toward deciding — not a raw topic, not fear language.
    """
    if not theme.details:
        return "A question I'm working through"

    # Prefer grounding in the user's actual question if we have it.
    source_question = (theme.details[0].source_question or "").strip().lower()
    text = source_question or theme.details[0].text.strip().lower()

    if any(word in text for word in ["cost", "pay", "insurance", "afford", "financ"]):
        return "Whether I can afford this"
    if any(word in text for word in ["risk", "side effect", "harm", "safe", "danger"]):
        return "How this could affect my health"
    if any(word in text for word in ["consent", "withdraw", "leave", "quit", "voluntary", "change my mind"]):
        return "How much say I'd have"
    if any(word in text for word in ["eligib", "qualify", "criteria"]):
        return "Whether this applies to me"
    if any(word in text for word in ["time", "visit", "schedule", "travel", "appointment"]):
        return "How much time this takes"
    if any(word in text for word in ["care", "doctor", "team", "own physician"]):
        return "How this affects my current care"
    if any(word in text for word in ["privacy", "data", "record", "information"]):
        return "How my information is handled"

    return "A question I'm working through"

def _infer_concern_summary(theme: JordanTheme) -> str:
    if not theme.details:
        return "Tracking something the user is trying to work out."
    source_question = theme.details[0].source_question or theme.details[0].text
    return f"Working out: {source_question[:90]}"

@app.post(
    "/jordan-turn-update-v2",
    response_model=JordanTurnUpdateResponse,
)
async def jordan_turn_update_v2(
    request: JordanDecisionTurnUpdateRequest,
):
    print("*** IN JORDAN TURN UPDATE V2")

    current_themes = request.current_model.themes
    recent_history = request.history[-8:]

    new_theme_id = next_jordan_theme_id(
        current_themes
    )

    new_detail_id = next_jordan_detail_id(
        current_themes
    )

    system_prompt = """
        You are Jordan, a warm guide helping someone make sense of what they learn from Alex about clinical trial participation.

        Each turn:
        1. add one lasting idea from Alex's answer to the workspace;
        2. connect it to an earlier idea only when that helps explain a larger participation concern;
        3. write one short message showing how the information affects the user's larger decision.

        Return the COMPLETE updated themes collection.

        THEME LABELS AND SUMMARIES
        - Themes are the user's concerns — things they're trying to figure out in order to decide whether/how to participate — grounded in what the user has actually asked about, not textbook topic headings.
        - A theme should read like a concern the user is holding, not a dictionary entry for a term Alex used.
        - Base the theme on the user's own question(s), not just on whatever fact Alex happened to state.
        - label: 2–6 words, phrased as the user's concern.
        - summary: one sentence, at most 22 words, describing what the user is trying to work out on this concern in order to decide.
        - Never write generic placeholders like "New Theme," "Untitled," or "Theme 1."
        - When adding a detail to an EXISTING theme, keep its label/summary unless the new detail shows the concern is broader or different than first captured — then refine the wording, but keep it framed as the user's concern.
        - When creating a NEW theme, base the label/summary on what the user was actually asking about, filtered through what that means for their decision — not just the raw content of Alex's answer.
        
        WORKSPACE
        - Keep at most 3 themes.
        - If 3 themes already exist, you MUST place the new detail into one of those existing themes.
        - Never remove an existing theme or detail.
        - Return every existing theme and detail unchanged, plus exactly one new detail.
        - Themes should represent concerns someone may weigh when considering participation, not textbook categories.
        - Add exactly one new detail on every turn.
        - Preserve all existing theme and detail IDs.
        - Use only the provided new IDs for new content.
        - Add to an existing theme when the idea fits.
        - Create a new theme only when needed.
        - Choose the idea from Alex's answer that is most useful for evaluating participation.
        - The new detail must contain only information Alex stated.
        - Leave source_question and source_answer empty.
        - The new detail's text must be no more than 25 words.

        CONNECTION
        Create latest_connection only when the new idea:
        - clarifies another part of the same concern;
        - shows a tradeoff or dependency;
        - separates ideas that could be confused;
        - or distinguishes general information from what depends on a specific trial.

        Do not connect ideas only because they share a topic.
        Return null when there is no meaningful connection.

        MESSAGE
        Write no more than 25 words, ideally 1 short sentence.

        Perform one useful sensemaking move:
        - break a broad concern into smaller decision questions;
        - separate two meanings or concerns;
        - connect multiple questions to one larger concern;
        - or distinguish what is generally known from what must be checked for a specific trial.

        Do not merely repeat Alex's answer.
        Do not always suggest another topic.
        Suggest one next direction only when it clearly helps with the larger concern.

        You may use the user's question to understand their concern, but not as factual evidence.
        Use only Alex's answer and the existing workspace details as factual sources.

        Do not:
        - give medical advice;
        - recommend a trial or treatment;
        - judge eligibility;
        - ask for personal health information;
        - add outside facts;
        - claim the user believes something;
        - mention IDs, notes, stored data, turns, or the system.

        GUIDANCE TYPE
        Choose the best fit:
        - make_more_specific
        - different_perspective
        - related_idea
        """

    user_content = f"""
        NEW THEME ID:
        Use only if a new theme is needed.
        {new_theme_id}

        NEW DETAIL ID:
        Use for exactly one new detail.
        {new_detail_id}

        USER'S LATEST QUESTION:
        {request.user_question}

        ALEX'S LATEST ANSWER:
        {request.alex_answer}

        ALEX ANSWER SCOPE:
        {request.alex_answer_scope.value}

        CURRENT THEMES:
        {json.dumps(
            [
                theme.model_dump()
                for theme in current_themes
            ],
            ensure_ascii=False,
        )}

        RECENT HISTORY:
        {json.dumps(
            [
                turn.model_dump()
                for turn in recent_history
            ],
            ensure_ascii=False,
        )}

        PREVIOUS GUIDANCE TYPES:
        {json.dumps(
            request.previous_guidance_types[-5:],
            ensure_ascii=False,
        )}

        PREVIOUS GUIDANCE MESSAGES:
        {json.dumps(
            request.previous_guidance_messages[-5:],
            ensure_ascii=False,
        )}
        """.strip()

    try:
        response = await client_chat.beta.chat.completions.parse(
            model=UF_LOCAL_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            temperature=0,
            response_format=JordanTurnUpdateResponse,
        )

        parsed = response.choices[0].message.parsed

        if parsed is None:
            raw_content = response.choices[0].message.content

            print(
                "*** JORDAN V2 PARSE FAILED:",
                raw_content,
            )

            raise HTTPException(
                status_code=500,
                detail="Failed to parse Jordan turn update",
            )

        try:
            validate_jordan_turn_update_v2(
                parsed=parsed,
                request=request,
                new_theme_id=new_theme_id,
                new_detail_id=new_detail_id,
            )
        except HTTPException as validation_error:
            # Only truly unrecoverable cases reach here now.
            print("*** JORDAN V2 VALIDATION FAILED:", validation_error.detail)
            raise

            print(
                "*** EXPECTED NEW IDS:",
                {
                    "theme_id": new_theme_id,
                    "detail_id": new_detail_id,
                },
            )

            print(
                "*** CURRENT THEMES:",
                json.dumps(
                    [
                        theme.model_dump()
                        for theme in current_themes
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
            )

            print(
                "*** RETURNED THEMES:",
                json.dumps(
                    [
                        theme.model_dump()
                        for theme in parsed.themes
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
            )

            raise

        
        print(
            "*** JORDAN V2 WORKSPACE COUNTS:",
            {
                "previous_theme_count": len(current_themes),
                "returned_theme_count": len(parsed.themes),
                "previous_detail_count": sum(
                    len(theme.details)
                    for theme in current_themes
                ),
                "returned_detail_count": sum(
                    len(theme.details)
                    for theme in parsed.themes
                ),
                "new_detail_id": new_detail_id,
            },
        )

        return parsed

    except HTTPException:
        raise

    except Exception as error:
        print(
            "*** JORDAN TURN UPDATE V2 ERROR:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail="Could not update Jordan",
        )

@app.post(
    "/jordan-turn-update",
    response_model=JordanTurnUpdateResponse,
)
async def jordan_turn_update(
    request: JordanTurnUpdateRequest,
):
    current_themes = request.current_model.themes
    recent_history = request.history[-8:]

    new_theme_id = next_jordan_theme_id(current_themes)
    new_detail_id = next_jordan_detail_id(current_themes)

    system_prompt = """
    You are Jordan, a warm, thoughtful guide helping someone organize and connect what they learn about clinical trial participation while talking with Alex.
    Each turn:
    1. organize the newest idea into themes,
    2. identify a meaningful connection to an earlier idea only when one exists,
    3. write one short conversational message that helps the user continue exploring.
    Return the COMPLETE updated themes collection.

    ------------------------------------------------------------------
    STEP 1 — ORGANIZE THE NEW IDEA INTO THEMES
    ------------------------------------------------------------------
    Themes are broad concepts that organize related ideas.
    - Keep at most 5 themes.
    - If 5 themes already exist, add the new idea to one of them.
    - You may rename or merge themes, but never create a 6th theme.
    - Preserve existing theme and detail IDs.
    - Only use the provided NEW THEME ID and NEW DETAIL ID for anything new.
    Each theme contains:
    - id
    - label (2–5 words)
    - summary (≤22 words)
    - details
    For the newest turn:
    - Identify the single most useful idea from Alex's answer.
    - Add exactly one new detail.
    - Place it into an existing theme when it fits; otherwise create a new theme.
    Details should:
    - capture one lasting idea, not summarize Alex's whole answer;
    - use simple conceptual language;
    - contain only information Alex stated;
    - leave source_question and source_answer empty.

    ------------------------------------------------------------------
    STEP 2 — IDENTIFY A REAL CONNECTION
    ------------------------------------------------------------------
    Create latest_connection only when the newest detail meaningfully builds on an earlier detail.
    Good connections:
    - explain or expand an earlier idea,
    - describe another part of the same process,
    - compare two ideas,
    - show a dependency or tradeoff.
    Do not create a connection simply because two details share a topic.
    If no meaningful connection exists, return null.
    A connection includes:
    - theme_id
    - earlier_detail_id
    - label (2–5 words)
    - text (maximum 25 words)
    - earlier_question_reference (maximum 10 words)
    Never mention IDs, notes, memory, turns, or the system.

    ------------------------------------------------------------------
    STEP 3 — WRITE JORDAN'S MESSAGE
    ------------------------------------------------------------------
    Maximum 35 words. Ideally 2 short sentences.
    Jordan's message should:
    - briefly organize the newest idea;
    - mention a meaningful connection when one exists;
    - otherwise simply highlight the new idea;
    - end by suggesting one specific direction to explore next.
    The exploration direction must come directly from Alex's answer.
    - The exploration direction MUST point to something Alex did NOT already explain or resolve in that same answer. For example, if Alex mentions "sponsor" you could suggest expanding on who sponsors typically are.
    - If Alex mentioned several equally important ideas, do not select one to focus on. Instead, briefly name the options and invite the user to ask for more details about whichever one interests them. 
    Never:
    - introduce concepts Alex did not mention;
    - answer the suggested direction;
    - use outside knowledge;
    - attribute Alex's explanation to the user ("Alex..." not "you...");
    - write a complete question.
    If this is the first topic:
    - do not mention connections;
    - briefly highlight the main idea Alex introduced;
    - point the user toward something that Alex briefly mentioned without fully explaining
    - if Alex mentioned several equally important ideas, do not select one to focus on. Instead, briefly name the options and invite the user to ask for more details about whichever one interests them. 
    If Alex says the available sources do not contain the answer:
    - briefly acknowledge what remains unknown;
    - do not suggest another version of the same unanswered question;
    - If Alex's response included something it could speak to (e.g. "my sources say X, but I couldn't find Y"), point the user toward asking more about that X.
    - Otherwise, look back at the conversation so far and suggest revisiting something Alex touched on but didn't fully explain.

    ------------------------------------------------------------------
    GUIDANCE TYPE
    ------------------------------------------------------------------
    Choose the guidance_type that best matches Jordan's message:
    - make_more_specific
    - different_perspective
    - related_idea
    Prefer variety when multiple types fit equally well.

    ------------------------------------------------------------------
    CONTENT BOUNDARIES
    ------------------------------------------------------------------
    - Use only Alex's latest answer and the stored themes as factual sources.
    - Stay within general clinical trial education.
    - Do not recommend specific trials, websites, enrollment, eligibility, talking to any person, or medical advice.
    - When pointing the user toward a topic to explore further, phrase it as something to ask about (e.g. "you might want to ask about travel or lodging assistance"), not as an action directed at a person.
    """

    user_content = f"""
    NEW THEME ID, USE ONLY IF A NEW THEME IS REQUIRED:
    {new_theme_id}

    NEW DETAIL ID, USE FOR THE ONE NEW DETAIL:
    {new_detail_id}

    USER'S LATEST QUESTION:
    {request.user_question}

    DR. ALEX'S LATEST ANSWER:
    {request.alex_answer}

    CURRENT THEMES:
    {json.dumps(
        [theme.model_dump() for theme in current_themes],
        ensure_ascii=False,
    )}

    RECENT HISTORY:
    {json.dumps(
        [turn.model_dump() for turn in recent_history],
        ensure_ascii=False,
    )}

    PREVIOUS GUIDANCE TYPES:
    {json.dumps(
        request.previous_guidance_types[-5:],
        ensure_ascii=False,
    )}

    PREVIOUS GUIDANCE MESSAGES:
    {json.dumps(
        request.previous_guidance_messages[-5:],
        ensure_ascii=False,
    )}
    """

    try:
        response = await client_chat.beta.chat.completions.parse(
            model=UF_LOCAL_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            temperature=0,
            response_format=JordanTurnUpdateResponse,
        )

        parsed = response.choices[0].message.parsed

        if parsed is None:
            raw_content = response.choices[0].message.content

            print(
                "*** JORDAN TURN UPDATE PARSE FAILED:",
                raw_content,
            )

            raise HTTPException(
                status_code=500,
                detail="Failed to parse Jordan turn update",
            )

        existing_theme_ids = {
            theme.id
            for theme in request.current_model.themes
        }

        existing_detail_ids = {
            detail.id
            for theme in request.current_model.themes
            for detail in theme.details
        }

        all_returned_details = [
            detail
            for theme in parsed.themes
            for detail in theme.details
        ]

        new_details = [
            detail
            for detail in all_returned_details
            if detail.id == new_detail_id
        ]

        if len(new_details) != 1:
            raise HTTPException(
                status_code=500,
                detail="Jordan must create exactly one new detail",
            )

        # The server controls the original question and answer.
        new_detail = new_details[0]
        new_detail.source_question = request.user_question
        new_detail.source_answer = request.alex_answer

        # Reject unknown IDs instead of silently accepting invented structure.
        allowed_theme_ids = existing_theme_ids | {new_theme_id}
        allowed_detail_ids = existing_detail_ids | {new_detail_id}

        for theme in parsed.themes:
            if theme.id not in allowed_theme_ids:
                raise HTTPException(
                    status_code=500,
                    detail=f"Jordan returned an invalid theme ID: {theme.id}",
                )

            for detail in theme.details:
                if detail.id not in allowed_detail_ids:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Jordan returned an invalid detail ID: {detail.id}",
                    )

        # Keep the workspace compact.
        # Keep the workspace compact without dropping the newest detail.
        if len(parsed.themes) > 3:
            theme_with_new_detail = next(
                (
                    theme
                    for theme in parsed.themes
                    if any(detail.id == new_detail_id for detail in theme.details)
                ),
                None,
            )

            remaining_themes = [
                theme
                for theme in parsed.themes
                if theme is not theme_with_new_detail
            ]

            parsed.themes = (
                ([theme_with_new_detail] if theme_with_new_detail else [])
                + remaining_themes
            )[:3]

        if parsed.latest_connection:
            connection = parsed.latest_connection

            valid_theme_ids = {
                theme.id
                for theme in parsed.themes
            }

            if connection.theme_id not in valid_theme_ids:
                parsed.latest_connection = None

            elif connection.earlier_detail_id not in existing_detail_ids:
                # It must point backward, never to the newest detail.
                parsed.latest_connection = None

        return parsed

    except HTTPException:
        raise

    except Exception as error:
        print(
            "*** JORDAN TURN UPDATE ERROR:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail="Could not update Jordan",
        )

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

from adaptive_chat import (
    create_adaptive_router,
    RagResponseModel,
)

adaptive_router = create_adaptive_router(
    client_chat=client_chat,
    model_name=UF_LOCAL_MODEL,
    rag=rag,
    preprocess_question=preprocess_question,
    rag_response_model=RagResponseModel,
    build_resource_cards=build_resource_cards,
    clean_alex_answer=clean_alex_answer,
)

app.include_router(adaptive_router)

@app.get("/")
async def root():
    return {"message": "Welcome to Rashi's FastAPI server!"}

if __name__ == "__main__":
    # Run the FastAPI application on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)