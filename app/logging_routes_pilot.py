import os
import pyodbc
from fastapi import APIRouter
from fastapi import Query
import json
from datetime import datetime
from typing import Literal
from fastapi import HTTPException
from dotenv import load_dotenv
from datetime import datetime
from pydantic import BaseModel, Field

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
    condition: int
    condition_name: str
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

class NotesReviewLog(BaseModel):
    participant_id: str
    selected_notes: list = []
    selected_alex_resources: list = []
    all_notes: list = []
    all_alex_resources: list = []

class InteractionCountIncrement(BaseModel):
    participant_id: str
    field: str
    amount: int = 1

class FinishButtonLog(BaseModel):
    participant_id: str
    appeared_at: datetime

class IntroPartLog(BaseModel):
    participant_id: str
    intro_part: str

class IntroFinishedLog(BaseModel):
    participant_id: str

class ConversationTurnIncrement(BaseModel):
    participant_id: str

class SummaryUrlLog(BaseModel):
    participant_id: str = Field(min_length=1)
    summary_url: str = Field(min_length=1)


class SummaryRequestLog(BaseModel):
    participant_id: str = Field(min_length=1)
    clicked_print_summary: bool

@router.post("/log-session")
def log_session(body: SessionLog):
    with get_conn() as conn:
        cursor = conn.cursor()

        cursor.execute(
            f"""
            SELECT TOP 1 participant_id
            FROM {table_name}
            WHERE participant_id = ?
            """,
            body.participant_id,
        )

        if cursor.fetchone():
            cursor.execute(
                f"""
                UPDATE {table_name}
                SET condition = ?,
                    condition_name = ?,
                    start_time = ?
                WHERE participant_id = ?
                """,
                body.condition,
                body.condition_name,
                body.start_time,
                body.participant_id,
            )
        else:
            cursor.execute(
                f"""
                INSERT INTO {table_name}
                    (
                        participant_id,
                        condition,
                        condition_name,
                        start_time
                    )
                VALUES (?, ?, ?, ?)
                """,
                body.participant_id,
                body.condition,
                body.condition_name,
                body.start_time,
            )

        conn.commit()

    return {
        "message": "session logged",
        "condition": body.condition,
        "condition_name": body.condition_name,
    }

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

@router.post("/save-notes-review")
def save_notes_review(body: NotesReviewLog):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            UPDATE {table_name}
            SET
                notes_review_selected_notes = ?,
                notes_review_selected_alex_resources = ?,
                notes_review_all_notes = ?,
                notes_review_all_alex_resources = ?,
                notes_review_completed_at = ?
            WHERE participant_id = ?
            """,
            json.dumps(body.selected_notes),
            json.dumps(body.selected_alex_resources),
            json.dumps(body.all_notes),
            json.dumps(body.all_alex_resources),
            datetime.utcnow(),
            body.participant_id,
        )
        conn.commit()

    return {"message": "notes review saved"}

@router.get("/notes-review")
def get_notes_review(participant_id: str = Query(...)):
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT
                notes_review_selected_notes,
                notes_review_selected_alex_resources
            FROM {table_name}
            WHERE participant_id = ?
            """,
            participant_id,
        )

        row = cursor.fetchone()

    if row is None:
        return {
            "selected_notes": [],
            "selected_alex_resources": [],
        }

    return {
        "selected_notes": json.loads(row[0] or "[]"),
        "selected_alex_resources": json.loads(row[1] or "[]"),
    }

@router.post("/increment-interaction-count")
def increment_interaction_count(body: InteractionCountIncrement):
    allowed_fields = {
        "source_open_count",
        "source_save_count",
        "sensemaking_click_count",
        "workspace_add_click_count",
        "workspace_edit_click_count",
        "workspace_delete_click_count",
    }

    if body.field not in allowed_fields:
        raise HTTPException(
            status_code=400,
            detail="Invalid interaction count field",
        )

    with get_conn() as conn:
        cursor = conn.cursor()

        cursor.execute(
            f"""
            UPDATE {table_name}
            SET {body.field} =
                CASE
                    WHEN COALESCE({body.field}, 0) + ? < 0 THEN 0
                    ELSE COALESCE({body.field}, 0) + ?
                END
            WHERE participant_id = ?
            """,
            body.amount,
            body.amount,
            body.participant_id,
        )

        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail="Participant not found",
            )

        conn.commit()

    return {
        "message": "count updated",
        "field": body.field,
        "amount": body.amount,
    }

@router.post("/log-finish-button-appeared")
def log_finish_button_appeared(body: FinishButtonLog):
    with get_conn() as conn:
        cursor = conn.cursor()

        cursor.execute(
            f"""
            UPDATE {table_name}
            SET finish_button_appeared_at =
                COALESCE(finish_button_appeared_at, ?)
            WHERE participant_id = ?
            """,
            body.appeared_at,
            body.participant_id,
        )

        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Participant {body.participant_id} was not found.",
            )

        conn.commit()

    return {
        "ok": True,
        "participant_id": body.participant_id,
        "finish_button_appeared_at": body.appeared_at,
    }

@router.post("/log-intro-part")
def log_intro_part(body: IntroPartLog):
    try:
        with get_conn() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                UPDATE {table_name}
                SET intro_part = ?
                WHERE participant_id = ?
                """,
                body.intro_part,
                body.participant_id,
            )

            if cursor.rowcount == 0:
                raise HTTPException(
                    status_code=404,
                    detail="No participant row was found.",
                )

            conn.commit()

        return {
            "success": True,
            "message": "Intro transcript saved.",
            "participant_id": body.participant_id,
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("[log-intro-part] ERROR:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail=f"Could not save intro transcript: {str(exc)}",
        )


@router.post("/log-intro-finished")
def log_intro_finished(body: IntroFinishedLog):
    try:
        with get_conn() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                UPDATE {table_name}
                SET intro_finished = SYSDATETIME()
                WHERE participant_id = ?
                """,
                body.participant_id,
            )

            if cursor.rowcount == 0:
                raise HTTPException(
                    status_code=404,
                    detail="No participant row was found.",
                )

            conn.commit()

        return {
            "success": True,
            "message": "Intro completion time saved.",
            "participant_id": body.participant_id,
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("[log-intro-finished] ERROR:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail=f"Could not save intro completion time: {str(exc)}",
        )
   
@router.post("/increment-conversation-turns")
def increment_conversation_turns(body: ConversationTurnIncrement):
    try:
        with get_conn() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                UPDATE {table_name}
                SET num_conversation_turns =
                    COALESCE(num_conversation_turns, 0) + 1
                WHERE participant_id = ?
                """,
                body.participant_id,
            )

            if cursor.rowcount == 0:
                raise HTTPException(
                    status_code=404,
                    detail="No participant row was found.",
                )

            cursor.execute(
                f"""
                SELECT num_conversation_turns
                FROM {table_name}
                WHERE participant_id = ?
                """,
                body.participant_id,
            )

            row = cursor.fetchone()
            updated_count = row[0] if row else None

            conn.commit()

        return {
            "success": True,
            "message": "Conversation turn count incremented.",
            "participant_id": body.participant_id,
            "num_conversation_turns": updated_count,
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("[increment-conversation-turns] ERROR:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail=f"Could not increment conversation turn count: {str(exc)}",
        )
    
@router.post("/log-summary-url")
def log_summary_url(body: SummaryUrlLog):
    with get_conn() as conn:
        cursor = conn.cursor()

        cursor.execute(
            f"""
            UPDATE {table_name}
            SET summary_url = ?
            WHERE participant_id = ?
            """,
            body.summary_url,
            body.participant_id,
        )

        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail="Participant session not found",
            )

        conn.commit()

    return {
        "ok": True,
        "participant_id": body.participant_id,
        "summary_url": body.summary_url,
    }

@router.post("/log-summary-request")
def log_summary_request(body: SummaryRequestLog):
    with get_conn() as conn:
        cursor = conn.cursor()

        cursor.execute(
            f"""
            UPDATE {table_name}
            SET clicked_print_summary = ?
            WHERE participant_id = ?
            """,
            int(body.clicked_print_summary),
            body.participant_id,
        )

        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail="Participant session not found",
            )

        conn.commit()

    return {
        "ok": True,
        "participant_id": body.participant_id,
        "clicked_print_summary": body.clicked_print_summary,
    }