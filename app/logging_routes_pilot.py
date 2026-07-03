import os
import pyodbc
from fastapi import APIRouter
from pydantic import BaseModel
from dotenv import load_dotenv

router = APIRouter()
load_dotenv()

table_name = os.getenv("DB_TABLE_PILOT")


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
    condition: str
    proactivity: str | None = None
    start_time: str


class GoalSettingLog(BaseModel):
    participant_id: str
    goals: str
    timestamp: str | None = None


class MainInteractionLog(BaseModel):
    participant_id: str
    transcript: str


class CompletionLog(BaseModel):
    participant_id: str
    end_time: str


@router.post("/log-session")
def log_session(body: SessionLog):
    with get_conn() as conn:
        cursor = conn.cursor()

        cursor.execute(
            f"SELECT TOP 1 participant_id FROM {table_name} WHERE participant_id = ?",
            body.participant_id,
        )

        if cursor.fetchone():
            cursor.execute(
                f"""
                UPDATE {table_name}
                SET condition = ?, proactivity = ?, start_time = ?
                WHERE participant_id = ?
                """,
                body.condition,
                body.proactivity,
                body.start_time,
                body.participant_id,
            )
        else:
            cursor.execute(
                f"""
                INSERT INTO {table_name}
                (participant_id, condition, proactivity, start_time)
                VALUES (?, ?, ?, ?)
                """,
                body.participant_id,
                body.condition,
                body.proactivity,
                body.start_time,
            )

        conn.commit()

    return {"message": "session logged"}


@router.post("/log-goal-setting")
def log_goal_setting(body: GoalSettingLog):
    with get_conn() as conn:
        cursor = conn.cursor()

        cursor.execute(
            f"""
            UPDATE {table_name}
            SET goal_setting = ?, goal_setting_time = ?
            WHERE participant_id = ?
            """,
            body.goals,
            body.timestamp,
            body.participant_id,
        )

        conn.commit()

    return {"message": "goal setting logged"}


@router.post("/log-main-interaction")
def log_main_interaction(body: MainInteractionLog):
    with get_conn() as conn:
        cursor = conn.cursor()

        cursor.execute(
            f"""
            UPDATE {table_name}
            SET main_interaction = ?
            WHERE participant_id = ?
            """,
            body.transcript,
            body.participant_id,
        )

        conn.commit()

    return {"message": "main interaction logged"}


@router.post("/log-completion")
def log_completion(body: CompletionLog):
    with get_conn() as conn:
        cursor = conn.cursor()

        cursor.execute(
            f"""
            UPDATE {table_name}
            SET end_time = ?
            WHERE participant_id = ?
            """,
            body.end_time,
            body.participant_id,
        )

        conn.commit()

    return {"message": "completion logged"}