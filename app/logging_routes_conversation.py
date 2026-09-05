import os
import json
import pyodbc

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Router stuff
router = APIRouter(
    prefix="/logs",
    tags=["logs"],
)

table_name = os.getenv("DB_TABLE_CONVERSATION")


# ---------------------------------------------------------------------------
# Eastern Time
# ---------------------------------------------------------------------------

EASTERN_TIME_SQL = """
CAST(
    SYSUTCDATETIME() AT TIME ZONE 'UTC'
                     AT TIME ZONE 'Eastern Standard Time'
    AS datetime2
)
"""


# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ConversationEnteredLog(BaseModel):
    participant_id: str = Field(min_length=1)
    c: int
    condition_name: str

class SelectedTopicsLog(BaseModel):
    participant_id: str = Field(min_length=1)
    selected_topics: list[str]

class ConversationStartedLog(BaseModel):
    participant_id: str = Field(min_length=1)

class ConversationStateLog(BaseModel):
    participant_id: str = Field(min_length=1)
    state: dict

class TopicCoveredLog(BaseModel):
    participant_id: str = Field(min_length=1)
    topic_number: int

class ConversationTranscriptLog(BaseModel):
    participant_id: str = Field(min_length=1)
    transcript: str

class ConversationFinishedLog(BaseModel):
    participant_id: str = Field(min_length=1)

class IntroFinishedLog(BaseModel):
    participant_id: str = Field(min_length=1)

class SelectedResourcesLog(BaseModel):
    participant_id: str = Field(min_length=1)
    selected_resources: list[str]

class WebsiteFinishedLog(BaseModel):
    participant_id: str = Field(min_length=1)

# ---------------------------------------------------------------------------
# Entered conversation activity
# ---------------------------------------------------------------------------

@router.post("/log-conversation-entered")
def log_conversation_entered(body: ConversationEnteredLog):
    print("Logging conversation entered...")
    try:
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

            existing = cursor.fetchone()

            if existing:
                # Preserve the participant's FIRST entry time
                cursor.execute(
                    f"""
                    UPDATE {table_name}
                    SET
                        entered_at = COALESCE(
                            entered_at,
                            {EASTERN_TIME_SQL}
                        ),
                        c = ?,
                        condition_name = ?
                    WHERE participant_id = ?
                    """,
                    body.c,
                    body.condition_name,
                    body.participant_id,
                )

            else:
                cursor.execute(
                    f"""
                    INSERT INTO {table_name}
                    (
                        participant_id,
                        c,
                        condition_name,
                        entered_at
                    )
                    VALUES (?, ?, ?, {EASTERN_TIME_SQL})
                    """,
                    body.participant_id,
                    body.c,
                    body.condition_name,
                )

            conn.commit()

        return {
            "success": True,
            "message": "Conversation entry logged.",
            "participant_id": body.participant_id,
            "c": body.c,
            "condition_name": body.condition_name,
        }

    except Exception as exc:
        print("[log-conversation-entered] ERROR:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail=f"Could not log conversation entry: {str(exc)}",
        )

# ---------------------------------------------------------------------------
# Selected topics
# ---------------------------------------------------------------------------

@router.post("/log-selected-topics")
def log_selected_topics(body: SelectedTopicsLog):
    try:
        if len(body.selected_topics) != 3:
            raise HTTPException(
                status_code=400,
                detail="Exactly 3 topics must be selected.",
            )

        with get_conn() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                UPDATE {table_name}
                SET
                    selected_topic_1 = ?,
                    selected_topic_2 = ?,
                    selected_topic_3 = ?,
                    topics_selected_at = {EASTERN_TIME_SQL}
                WHERE participant_id = ?
                """,
                body.selected_topics[0],
                body.selected_topics[1],
                body.selected_topics[2],
                body.participant_id,
            )

            if cursor.rowcount == 0:
                raise HTTPException(
                    status_code=404,
                    detail="Participant conversation row was not found.",
                )

            conn.commit()

        return {
            "success": True,
            "message": "Selected topics logged.",
            "participant_id": body.participant_id,
            "selected_topics": body.selected_topics,
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("[log-selected-topics] ERROR:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail=f"Could not save selected topics: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Conversation started
# ---------------------------------------------------------------------------

@router.post("/log-conversation-started")
def log_conversation_started(body: ConversationStartedLog):
    try:
        with get_conn() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                UPDATE {table_name}
                SET conversation_started_at = COALESCE(
                    conversation_started_at,
                    {EASTERN_TIME_SQL}
                )
                WHERE participant_id = ?
                """,
                body.participant_id,
            )

            if cursor.rowcount == 0:
                raise HTTPException(
                    status_code=404,
                    detail="Participant conversation row was not found.",
                )

            conn.commit()

        return {
            "success": True,
            "message": "Conversation start logged.",
            "participant_id": body.participant_id,
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("[log-conversation-started] ERROR:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail=f"Could not log conversation start: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------

@router.post("/save-conversation-state")
def save_conversation_state(body: ConversationStateLog):
    try:
        with get_conn() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                UPDATE {table_name}
                SET conversation_state = ?
                WHERE participant_id = ?
                """,
                json.dumps(body.state),
                body.participant_id,
            )

            if cursor.rowcount == 0:
                raise HTTPException(
                    status_code=404,
                    detail="Participant conversation row was not found.",
                )

            conn.commit()

        return {
            "success": True,
            "message": "Conversation state saved.",
            "participant_id": body.participant_id,
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("[save-conversation-state] ERROR:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail=f"Could not save conversation state: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Topic completed / covered
# ---------------------------------------------------------------------------

@router.post("/log-topic-covered")
def log_topic_covered(body: TopicCoveredLog):
    try:
        allowed_columns = {
            1: "selected_topic_1_covered_at",
            2: "selected_topic_2_covered_at",
            3: "selected_topic_3_covered_at",
        }

        column = allowed_columns.get(body.topic_number)

        if column is None:
            raise HTTPException(
                status_code=400,
                detail="topic_number must be 1, 2, or 3.",
            )

        with get_conn() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                UPDATE {table_name}
                SET {column} = COALESCE(
                    {column},
                    {EASTERN_TIME_SQL}
                )
                WHERE participant_id = ?
                """,
                body.participant_id,
            )

            if cursor.rowcount == 0:
                raise HTTPException(
                    status_code=404,
                    detail="Participant conversation row was not found.",
                )

            conn.commit()

        return {
            "success": True,
            "message": f"Topic {body.topic_number} completion logged.",
            "participant_id": body.participant_id,
            "topic_number": body.topic_number,
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("[log-topic-covered] ERROR:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail=f"Could not log topic completion: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Conversation transcript
# ---------------------------------------------------------------------------

@router.post("/log-conversation-transcript")
def log_conversation_transcript(body: ConversationTranscriptLog):
    try:
        with get_conn() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                UPDATE {table_name}
                SET conversation_transcript = ?
                WHERE participant_id = ?
                """,
                body.transcript,
                body.participant_id,
            )

            if cursor.rowcount == 0:
                raise HTTPException(
                    status_code=404,
                    detail="Participant conversation row was not found.",
                )

            conn.commit()

        return {
            "success": True,
            "message": "Conversation transcript saved.",
            "participant_id": body.participant_id,
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("[log-conversation-transcript] ERROR:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail=f"Could not save conversation transcript: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Conversation finished
# ---------------------------------------------------------------------------

@router.post("/log-conversation-finished")
def log_conversation_finished(body: ConversationFinishedLog):
    try:
        with get_conn() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                UPDATE {table_name}
                SET finished_conversation_at = COALESCE(
                    finished_conversation_at,
                    {EASTERN_TIME_SQL}
                )
                WHERE participant_id = ?
                """,
                body.participant_id,
            )

            if cursor.rowcount == 0:
                raise HTTPException(
                    status_code=404,
                    detail="Participant conversation row was not found.",
                )

            conn.commit()

        return {
            "success": True,
            "message": "Conversation completion logged.",
            "participant_id": body.participant_id,
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("[log-conversation-finished] ERROR:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail=f"Could not log conversation completion: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Get saved conversation data
# ---------------------------------------------------------------------------

@router.get("/conversation-data/{participant_id}")
def get_conversation_data(participant_id: str):
    try:
        with get_conn() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                SELECT
                    entered_at,
                    participant_id,
                    topics_selected_at,
                    selected_topic_1,
                    selected_topic_2,
                    selected_topic_3,
                    conversation_started_at,
                    conversation_state,
                    selected_topic_1_covered_at,
                    selected_topic_2_covered_at,
                    selected_topic_3_covered_at,
                    conversation_transcript,
                    finished_conversation_at,
                    selected_resources,
                    resources_selected_at,
                    finished_at
                FROM {table_name}
                WHERE participant_id = ?
                """,
                participant_id,
            )

            row = cursor.fetchone()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail="Participant conversation row was not found.",
            )

        conversation_state = None
        selected_resources = None

        if row[7]:
            try:
                conversation_state = json.loads(row[7])
            except json.JSONDecodeError:
                conversation_state = None

        if row[13]:
            try:
                selected_resources = json.loads(row[13])
            except json.JSONDecodeError:
                selected_resources = None

        return {
            "entered_at": row[0],
            "participant_id": row[1],
            "topics_selected_at": row[2],
            "selected_topics": [
                row[3],
                row[4],
                row[5],
            ],
            "conversation_started_at": row[6],
            "conversation_state": conversation_state,
            "selected_topic_1_covered_at": row[8],
            "selected_topic_2_covered_at": row[9],
            "selected_topic_3_covered_at": row[10],
            "conversation_transcript": row[11],
            "finished_conversation_at": row[12],
            "selected_resources": selected_resources,
            "resources_selected_at": row[14],
            "finished_at": row[15],
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("[get-conversation-data] ERROR:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail=f"Could not load conversation data: {str(exc)}",
        )

# ---------------------------------------------------------------------------
# Intro finished
# ---------------------------------------------------------------------------

@router.post("/log-intro-finished")
def log_intro_finished(body: IntroFinishedLog):
    try:
        with get_conn() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                UPDATE {table_name}
                SET intro_finished_at = COALESCE(
                    intro_finished_at,
                    {EASTERN_TIME_SQL}
                )
                WHERE participant_id = ?
                """,
                body.participant_id,
            )

            if cursor.rowcount == 0:
                raise HTTPException(
                    status_code=404,
                    detail="Participant conversation row was not found.",
                )

            conn.commit()

        return {
            "success": True,
            "message": "Intro completion logged.",
            "participant_id": body.participant_id,
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("[log-intro-finished] ERROR:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail=f"Could not log intro completion: {str(exc)}",
        )

# ---------------------------------------------------------------------------
# Selected next-step resources
# ---------------------------------------------------------------------------

@router.post("/log-selected-resources")
def log_selected_resources(body: SelectedResourcesLog):
    try:
        with get_conn() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                UPDATE {table_name}
                SET
                    selected_resources = ?,
                    resources_selected_at = {EASTERN_TIME_SQL}
                WHERE participant_id = ?
                """,
                json.dumps(body.selected_resources),
                body.participant_id,
            )

            if cursor.rowcount == 0:
                raise HTTPException(
                    status_code=404,
                    detail="Participant conversation row was not found.",
                )

            conn.commit()

        return {
            "success": True,
            "message": "Selected resources logged.",
            "participant_id": body.participant_id,
            "selected_resources": body.selected_resources,
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("[log-selected-resources] ERROR:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail=f"Could not save selected resources: {str(exc)}",
        )

# ---------------------------------------------------------------------------
# Website finished
# ---------------------------------------------------------------------------

@router.post("/log-finished")
def log_finished(body: WebsiteFinishedLog):
    try:
        with get_conn() as conn:
            cursor = conn.cursor()

            cursor.execute(
                f"""
                UPDATE {table_name}
                SET finished_at = COALESCE(
                    finished_at,
                    {EASTERN_TIME_SQL}
                )
                WHERE participant_id = ?
                """,
                body.participant_id,
            )

            if cursor.rowcount == 0:
                raise HTTPException(
                    status_code=404,
                    detail="Participant conversation row was not found.",
                )

            conn.commit()

        return {
            "success": True,
            "message": "Website completion logged.",
            "participant_id": body.participant_id,
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("[log-finished] ERROR:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail=f"Could not log website completion: {str(exc)}",
        )