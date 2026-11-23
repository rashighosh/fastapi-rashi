from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import os
import re
import json
from pathlib import Path

router = APIRouter(prefix="/study", tags=["Study Conversation"])

base_url = "https://api.ai.it.ufl.edu"
RASHI_LITELLM_KEY = os.getenv('RASHI_LITELLM_KEY')
UF_LOCAL_MODEL = 'gpt-oss-20b'

client = AsyncOpenAI(
    api_key=RASHI_LITELLM_KEY,
    base_url=base_url,
)

MODEL = UF_LOCAL_MODEL


STUDY_TOPICS = [
    {
        "id": "cost",
        "name": "Costs of Participating",
    },
    {
        "id": "support",
        "name": "Getting Help with Participation",
    },
    {
        "id": "logistics",
        "name": "Transportation and Financial Support",
    },
    {
        "id": "doctor",
        "name": "Keeping My Doctor Informed",
    },
    {
        "id": "side_effects",
        "name": "Side Effects",
    },
    {
        "id": "quality",
        "name": "Quality of Care",
    },
]

SESSION_FILE = Path(__file__).resolve().parent / "study_sessions.json"

def load_study_sessions():
    if SESSION_FILE.exists():
        with open(SESSION_FILE, "r") as f:
            return json.load(f)
    return {}


study_sessions = load_study_sessions()

def save_study_sessions():
    with open(SESSION_FILE, "w") as f:
        json.dump(study_sessions, f, indent=2)

class ConversationMessage(BaseModel):
    from_: str = Field(alias="from")
    text: str

class OpeningMessageRequest(BaseModel):
    participant_id: str
    condition: int
    message: str
    history: list[ConversationMessage] = Field(default_factory=list)

class TopicPreparationRequest(BaseModel):
    participant_id: str
    condition: int
    history: list[ConversationMessage] = Field(default_factory=list)

class ConfirmTopicsRequest(BaseModel):
    participant_id: str
    condition: int
    selected_topics: list[str] = Field(min_length=3, max_length=3)

class TopicOpeningRequest(BaseModel):
    participant_id: str

@router.post("/opening-message")
async def opening_message(request: OpeningMessageRequest):

    if request.condition not in (2, 3):
        raise HTTPException(
            status_code=400,
            detail="Opening needs elicitation is only available for conditions 2 and 3."
        )

    speaker = "Alex" if request.condition == 2 else "Jordan"


    system_prompt = """
    You are friendly, casual virtual companion helping
    someone explore what matters to them about clinical trials.

    Briefly summarize the participant's main questions or concerns
    in your own words, without adding assumptions.

    Use VERY casual, informal spoken language.
    Keep it to 1-2 short sentences.

    Do not ask questions or provide factual information.
    """

    # Only include what the participant actually shared.
    # This matches the simpler approach used by Alex.

    user_messages = [
        item.text
        for item in request.history
        if item.from_.lower() in ("user", "participant")
    ]
    
    completion = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "\n".join(user_messages) or request.message,
            },
        ],
        temperature=0.5,
    )

    response_text = (completion.choices[0].message.content or "").strip()

    return {
        "speaker": speaker,
        "response": response_text,
    }



@router.post("/prepare-topics")
async def prepare_topics(request: TopicPreparationRequest):

    if request.condition not in (2, 3):
        raise HTTPException(
            status_code=400,
            detail="Topic preparation is only available for conditions 2 and 3."
        )

    user_messages = [
        item.text
        for item in request.history
        if item.from_.lower() in ("user", "participant")
    ]

    topic_options = "\n".join(
        f"{topic['id']}: {topic['name']}"
        for topic in STUDY_TOPICS
    )

    numbered_user_messages = "\n".join(
        f"{i}: {message}"
        for i, message in enumerate(user_messages)
    )
    
    
    system_prompt = f"""
    You are a professional clinical trials educator.

    Based on the participant's responses, select the three
    most relevant educational topics.

    AVAILABLE TOPICS:
    {topic_options}

    For each selected topic:
    - Choose based on the participant's actual concerns or needs.
    - Identify the number of the most relevant participant message.
    - Use NONE if there is no clear connection.

    Also write a brief, professional recommendation that:
    - Explicitly names all three topics using their exact names.
    - Briefly explains how they relate to the participant's concerns.
    - Does not invent concerns the participant hasn't expressed.

    Return exactly four lines:

    topic_id | brief explanation | message number or NONE
    topic_id | brief explanation | message number or NONE
    topic_id | brief explanation | message number or NONE
    SUMMARY | brief recommendation

    Do not include any additional text.
    """


    completion = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": numbered_user_messages or "No concerns provided.",
            },
        ],
        temperature=0,
    )

    
    result = (completion.choices[0].message.content or "").strip()

    print("\n========== TOPIC MAPPING DEBUG ==========")
    print("ORIGINAL USER MESSAGES:")
    print(repr(user_messages))
    print("\nRAW ALEX RESPONSE:")
    print(repr(result))
    print("=========================================\n")

    spoken_summary = ""

    for line in result.splitlines():
        if line.startswith("SUMMARY |"):
            spoken_summary = line.split("|", 1)[1].strip()
            break

    valid_topics = {topic["id"]: topic for topic in STUDY_TOPICS}
    selected_topics = []

    
    for line in result.splitlines():
        parts = line.split("|", 2)

        if len(parts) != 3:
            continue

        topic_id = parts[0].strip()
        explanation = parts[1].strip()
        message_index = parts[2].strip()

        # Retrieve the original participant message by its index.
        # Never rely on the LLM to reproduce the participant's words.
        if message_index.isdigit():
            index = int(message_index)
            excerpt = user_messages[index] if index < len(user_messages) else None
        else:
            excerpt = None

        if (
            topic_id in valid_topics
            and topic_id not in [topic["id"] for topic in selected_topics]
            and explanation
        ):
            selected_topics.append({
                **valid_topics[topic_id],
                "explanation": explanation,
                "user_excerpt": excerpt,
            })

        if len(selected_topics) == 3:
            break


    # Fallback if fewer than three valid topics were returned
    for topic in STUDY_TOPICS:
        if len(selected_topics) == 3:
            break

        if topic["id"] not in [item["id"] for item in selected_topics]:
            selected_topics.append({
                **topic,
                "explanation": "This is another area you may find useful to explore.",
                "user_excerpt": None,
            })

    study_sessions[request.participant_id] = {
        "condition": request.condition,
        "opening_responses": user_messages,
        "prepared_topics": selected_topics,
    }
    save_study_sessions()
 
    return {
        "status": "ready",
        "opening_responses": user_messages,
        "selected_topics": selected_topics,
        "spoken_summary": spoken_summary,
    }

@router.post("/confirm-topics")
async def confirm_topics(request: ConfirmTopicsRequest):

    valid_topics = {
        topic["id"]: topic
        for topic in STUDY_TOPICS
    }

    # Confirm exactly three different, valid topics.
    if len(set(request.selected_topics)) != 3:
        raise HTTPException(
            status_code=400,
            detail="Please select three different topics."
        )

    if any(
        topic_id not in valid_topics
        for topic_id in request.selected_topics
    ):
        raise HTTPException(
            status_code=400,
            detail="One or more selected topics are invalid."
        )

    # Retrieve Alex's original recommendations and mappings.
    session = study_sessions.get(request.participant_id, {})
    prepared_topics = {
        topic["id"]: topic
        for topic in session.get("prepared_topics", [])
    }

    # Preserve the participant's selections and original mappings.
    confirmed_topics = []

    for topic_id in request.selected_topics:
        original = prepared_topics.get(topic_id)

        confirmed_topics.append({
            **valid_topics[topic_id],
            "user_excerpt": original.get("user_excerpt") if original else None,
            "user_selected": original is None,
        })

    # Initialize the conversation while preserving opening data.
    study_sessions[request.participant_id] = {
        **session,
        "condition": request.condition,
        "topics": confirmed_topics,
        "current_topic_index": 0,
        "phase": "topics",
        "conversation_complete": False,
    }
    save_study_sessions()

    return {
        "status": "ready",
        "phase": "topics",
        "current_topic_index": 0,
        "conversation_complete": False,
        "topics": confirmed_topics,
    }


@router.post("/topic-opening")
async def topic_opening(request: TopicOpeningRequest):

    session = study_sessions.get(request.participant_id)

    if not session or not session.get("topics"):
        raise HTTPException(
            status_code=404,
            detail="No confirmed topics found for this participant."
        )

    topic_index = session["current_topic_index"]
    topic = session["topics"][topic_index]

    topic_name = topic["name"]
    user_excerpt = topic.get("user_excerpt")

    # Initialize the shared participant profile from opening elicitation.
    # Preserve any responses collected during earlier topics.
    profile = session.setdefault("user_profile", {
        "opening_responses": session.get("opening_responses", []),
        "topic_responses": {},
    })

    system_prompt = """
    You are a friendly, casual virtual companion.

    Your role is to help someone explore their personal concerns,
    circumstances, needs, and preferences when considering clinical trials.

    Ask one short, natural question about the CURRENT TOPIC.

    Use relevant things the participant shared, but stay focused
    on the current topic. Do not combine unrelated concerns.

    Explore their perspective without assuming they want to
    participate in a trial.

    Do not provide factual information or medical advice.
    Use VERY casual, information language.
    """

    opening_responses = session.get("opening_responses", [])

    user_prompt = f"""
    CURRENT TOPIC:
    {topic_name}

    PARTICIPANT'S EARLIER RESPONSES:
    {profile["opening_responses"]}
    {profile["topic_responses"]}

    MESSAGE ASSOCIATED WITH THIS TOPIC:
    {user_excerpt or 'None'}

    Ask one very casual question exploring the participant's
    perspective on the current topic.
    """




    completion = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
    )

    jordan_reply = (
        completion.choices[0].message.content or ""
    ).strip()

    return {
        "speaker": "Jordan",
        "topic_id": topic["id"],
        "topic_name": topic_name,
        "user_excerpt": user_excerpt,
        "jordan_reply": jordan_reply,
    }


async def generate_study_search_query(topic_name: str, user_profile: dict):
    
    system_prompt = """
    Generate one search query for a clinical trial education
    vector database.

    Use the current topic and any directly relevant concerns,
    needs, or circumstances the participant expressed.

    Focus on the current topic. Do not invent information.

    Return only the search query.
    """


    completion = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""
                CURRENT TOPIC: {topic_name}

                PARTICIPANT CONTEXT:
                {json.dumps(user_profile)}
                """,
            },
        ],
        temperature=0,
    )

    return (
        completion.choices[0].message.content or topic_name
    ).strip()


@router.post("/alex-topic-intro")
async def alex_topic_intro(request: TopicOpeningRequest):
    from conversation_alex import (
        generate_topic_factual_content,
        generate_alex_topic_intro,
    )

    session = study_sessions.get(request.participant_id)

    if not session or not session.get("topics"):
        raise HTTPException(
            status_code=404,
            detail="No confirmed topics found for this participant."
        )

    topic_index = session["current_topic_index"]
    topic = session["topics"][topic_index]

    # Information the participant shared during opening elicitation
    opening_responses = session.get("opening_responses", [])

    # Retrieve the participant's shared context.
    user_profile = session.get("user_profile", {
        "opening_responses": session.get("opening_responses", []),
        "topic_responses": {},
    })

    # Generate a retrieval query informed by the participant's needs.
    search_query = await generate_study_search_query(
        topic_name=topic["name"],
        user_profile=user_profile,
    )

    print("[STUDY ALEX RETRIEVAL QUERY]", search_query, flush=True)

    # Retrieve relevant factual information from the existing vector store.
    factual_content = await generate_topic_factual_content(
        f"{topic['name']}: {search_query}"
    )

    # Generate a brief introduction using the retrieved evidence
    # and the participant's earlier stated needs
    introduction = await generate_alex_topic_intro(
        current_topic=topic["name"],
        factual_content=factual_content,
        prior_summaries=[
            *session.get("user_profile", {}).get(
                "opening_responses",
                session.get("opening_responses", [])
            ),
            *[
                response
                for responses in session.get("user_profile", {}).get(
                    "topic_responses", {}
                ).values()
                for response in responses
            ],
        ],
        topic_position="first" if topic_index == 0 else "middle",
        condition=session["condition"],
    )

    return {
        "speaker": "Alex",
        "topic_id": topic["id"],
        "topic_name": topic["name"],
        "alex_reply": introduction.reply,
    }


class JordanTopicTurnRequest(BaseModel):
    participant_id: str
    message: str


@router.post("/jordan-topic-turn")
async def jordan_topic_turn(request: JordanTopicTurnRequest):
    session = study_sessions.get(request.participant_id)

    if not session or not session.get("topics"):
        raise HTTPException(
            status_code=404,
            detail="No active study conversation found."
        )

    topic_index = session["current_topic_index"]
    topic = session["topics"][topic_index]
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    # Initialize the profile using the opening elicitation.
    profile = session.setdefault("user_profile", {
        "opening_responses": session.get("opening_responses", []),
        "topic_responses": {}
    })

    # Preserve the participant's actual words under this topic.
    topic_responses = profile["topic_responses"].setdefault(
        topic["id"], []
    )

    previous_responses = list(topic_responses)
    topic_responses.append(message)
    save_study_sessions()

    system_prompt = """
    You are a friendly, casual virtual companion.

    Your role is to help someone explore their personal concerns,
    circumstances, needs, and preferences when considering clinical trials.

    Respond naturally to what the participant shared, and ask
    one meaningful follow-up question when appropriate.

    Stay focused on the CURRENT TOPIC and build on what you've
    already learned without repeating questions.

    Use VERY casual, informal spoken language. Keep responses brief.

    Do not assume the participant wants to join a trial.
    Do not provide factual information or medical advice.
    """

    user_prompt = f"""
    CURRENT TOPIC:
    {topic["name"]}

    PARTICIPANT'S OPENING RESPONSES:
    {profile["opening_responses"]}

    PARTICIPANT'S RESPONSES ACROSS TOPICS:
    {profile["topic_responses"]}

    LATEST PARTICIPANT MESSAGE:
    {message}

    Respond very casually and continue exploring the participant's
    perspective on the current topic.
    """


    completion = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
    )

    jordan_reply = (
        completion.choices[0].message.content or ""
    ).strip()

    return {
        "speaker": "Jordan",
        "topic_id": topic["id"],
        "jordan_reply": jordan_reply,
        "user_profile": profile,
    }


class AlexTopicTurnRequest(BaseModel):
    participant_id: str
    message: str
    history: list[ConversationMessage] = Field(default_factory=list)


@router.post("/alex-topic-turn")
async def alex_topic_turn(request: AlexTopicTurnRequest):
    from conversation_alex import (
        ConversationAlexRequest,
        ConversationMessage as AlexConversationMessage,
        conversation_alex,
    )

    session = study_sessions.get(request.participant_id)

    if not session or not session.get("topics"):
        raise HTTPException(
            status_code=404,
            detail="No active study conversation found."
        )

    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    topic = session["topics"][session["current_topic_index"]]

    # Retrieve participant-expressed context from Jordan's conversations.
    profile = session.get("user_profile", {
        "opening_responses": session.get("opening_responses", []),
        "topic_responses": {},
    })

    participant_context = [
        *profile.get("opening_responses", []),
        *[
            response
            for responses in profile.get("topic_responses", {}).values()
            for response in responses
        ],
    ]

    # Reuse Alex's existing vector-store question-answering pipeline.
    alex_request = ConversationAlexRequest(
        message=message,
        history=[
            AlexConversationMessage.model_validate(
                item.model_dump(by_alias=True)
            )
            for item in request.history
        ],
        earlier_memory=f"Current discussion topic: {topic['name']}",
        prior_topic_summaries=participant_context,
        condition=session.get("condition", 3),
    )

    result = await conversation_alex(alex_request)

    return {
        "speaker": "Alex",
        "topic_id": topic["id"],
        "topic_name": topic["name"],
        "alex_reply": result["answer"],
    }


@router.post("/advance-topic")
async def advance_study_topic(request: TopicOpeningRequest):
    session = study_sessions.get(request.participant_id)

    if not session or not session.get("topics"):
        raise HTTPException(
            status_code=404,
            detail="No confirmed topics found for this participant."
        )

    if session.get("conversation_complete"):
        raise HTTPException(
            status_code=400,
            detail="The conversation is already complete."
        )

    previous_index = session["current_topic_index"]
    next_index = previous_index + 1

    # Participant has completed all three topics.
    if next_index >= len(session["topics"]):
        session["conversation_complete"] = True
        session["phase"] = "complete"
        save_study_sessions()

        return {
            "conversation_complete": True,
            "current_topic_index": previous_index,
            "completed_topic_number": previous_index + 1,
            "phase": "complete",
            "topics": session["topics"],
        }

    # Advance to the next confirmed topic.
    session["current_topic_index"] = next_index
    save_study_sessions()

    return {
        "conversation_complete": False,
        "current_topic_index": next_index,
        "completed_topic_number": previous_index + 1,
        "phase": "topics",
        "topics": session["topics"],
    }
