import pyodbc
import os
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv


router = APIRouter()

load_dotenv()

table_name = os.getenv('DB_TABLE')

def get_conn():
    conn_str = (
        f"DRIVER={{ODBC Driver 18 for SQL Server}};"
        f"SERVER={os.getenv('DB_SERVER')};"
        f"DATABASE={os.getenv('DB_DATABASE')};"
        f"UID={os.getenv('DB_USER')};"
        f"PWD={os.getenv('DB_PASSWORD')};"
        "Encrypt=yes;TrustServerCertificate=yes;"
    )
    return pyodbc.connect(conn_str)

class SessionLog(BaseModel):
    participant_id: str
    condition: int
    start_time: str

class LandingQuestionLog(BaseModel):
    participant_id: str
    landing_question: str  # full transcript JSON string

class LandingPrecheckLog(BaseModel):
    participant_id: str
    landing_precheck: str  # full events JSON string you're replacing each time

class TranscriptLog(BaseModel):
    participant_id: str
    transcript: str  # full transcript JSON string

class CompletionLog(BaseModel):
    participant_id: str
    end_time: str

@router.post("/log-session")
def log_session(body: SessionLog):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT TOP 1 participant_id FROM {table_name} WHERE participant_id = ?",
            body.participant_id
        )
        if cursor.fetchone():
            return {"message": "already exists"}
        cursor.execute(
            f"INSERT INTO {table_name} (participant_id, condition, start_time) VALUES (?, ?, ?)",
            body.participant_id, body.condition, body.start_time
        )
        conn.commit()
    return {"message": "session logged"}

@router.post("/log-landing-question")
def long_landing(body: LandingQuestionLog):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE {table_name} SET landing_question = ? WHERE participant_id = ?",
            body.landing_question, body.participant_id
        )
        conn.commit()
    return {"message": "events updated"}

@router.post("/log-landing-precheck")
def log_events(body: LandingPrecheckLog):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE {table_name} SET landing_precheck = ? WHERE participant_id = ?",
            body.landing_precheck, body.participant_id
        )
        conn.commit()
    return {"message": "events updated"}

@router.post("/log-main-interaction")
def log_transcript(body: TranscriptLog):
    print("in log main interaction")
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE {table_name} SET main_interaction = ? WHERE participant_id = ?",
            body.transcript, body.participant_id
        )
        conn.commit()
    print("updated main_interaction with transcript")
    return {"message": "transcript updated"}

@router.post("/log-completion")
def log_completion(body: CompletionLog):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE {table_name} SET end_time = ? WHERE participant_id = ?",
            body.end_time, body.participant_id
        )
        conn.commit()
    return {"message": "completion logged"}