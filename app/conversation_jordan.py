from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import json
import re
from typing import Any
import asyncio

load_dotenv()

# Router stuff
router = APIRouter(
    prefix="/jordan",
    tags=["jordan"],
)

from conversation_alex import (
    analyze_alex_support,
    generate_alex_topic_intro,
    generate_topic_factual_content,
)

from conversation_tracker import (
    classify_user_needs,
    generate_jordan_capture,
)

from logging_routes_conversation import (
    load_conversation_state_from_db,
    save_conversation_state_to_db,
)

# Set up LLM stuff
BASE_URL = "https://api.ai.it.ufl.edu/v1"
RASHI_LITELLM_KEY = os.getenv("RASHI_LITELLM_KEY")
UF_LOCAL_MODEL = "gpt-oss-20b"

client_chat = AsyncOpenAI(
    api_key=RASHI_LITELLM_KEY,
    base_url=BASE_URL,
    timeout=45.0
)

# Condition info
CONDITION_SINGLE_INFO = 1
CONDITION_SINGLE_COMBINED = 2
CONDITION_MULTIPLE = 3

INTERACTION_BOUNDARY = """
INTERACTION BOUNDARY:
This interaction provides conversational support about clinical-trial participation.

You cannot find specific trials:
- Do not search for, identify, match, shortlist, rank, or recommend specific
  clinical trials for the user.
- Do not claim that you can find or select trials for the user.
- Do not ask for medical, eligibility, or location details to determine which
  trials might fit the user.
- If the user mentions a specific trial, do not evaluate whether it is appropriate
  for them or help determine whether they are eligible. Keep the conversation at
  a general educational level.

You cannot perform external actions:
- You cannot take actions outside this conversation.
- Do not claim or imply that you created, changed, scheduled, sent, submitted,
  uploaded, saved, ordered, contacted, or configured anything.
- Do not act as a clinic, research team, patient portal, email service, or other
  external system.
- If the user asks you to perform an external action, briefly explain that you
cannot do so.

You cannot request private or security-sensitive information:
- Do not ask for passwords, PINs, authentication codes, account details, contact
  information, identifying information, medical records, or other private or
  security-sensitive information.
- If the user shares private or security-sensitive information, do not repeat or
  use it. Briefly tell them not to share private information here, then continue
  without referring to its contents.
"""

def get_conversation_speaker(state):
    if state.get("condition") == CONDITION_SINGLE_COMBINED:
        return "alex"

    return "jordan"

# Creates a new conversation state for a user
# Includes: current topic index, if conversation is complete, and topic list
def create_conversation_state(selected_topics, condition):
    topics: list[dict[str, Any]] = []

    for i, topic in enumerate(selected_topics):
        topics.append({
            "topic": topic,
            "status": "active" if i == 0 else "upcoming",
            "summary": None,
            "rolling_summary": None,
            "summarized_until": 0,
            "personally_situated_questions": [],
        })

    return {
        "current_topic_index": 0,
        "conversation_complete": False,
        "phase": "topics",
        "topic_history_start": 0,
        "wrapup_history_start": None,
        "wrapup_rolling_summary": None,
        "wrapup_summarized_until": 0,
        "topics": topics,
        "condition": condition,
    }

# Given a current user state, get the topic they're on
def get_current_topic(state):
    index = state["current_topic_index"]
    return state["topics"][index]

def save_personally_situated_question(state, question, need_type):
    current_topic = get_current_topic(state)

    questions = current_topic.setdefault(
        "personally_situated_questions", []
    )

    question = question.strip()

    if not question:
        return

    # Avoid duplicate questions within the same topic
    if any(
        q["question"].lower() == question.lower()
        for q in questions
    ):
        return

    questions.append({
        "question": question,
        "type": need_type,
    })

# Given a user, create new state or grab their existing state
def get_state(participant_id):
    state = load_conversation_state_from_db(participant_id)

    if state is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation topics have not been initialized."
        )

    return state

def persist_state(participant_id, state):
    save_conversation_state_to_db(
        participant_id,
        state,
    )

# Marks current state as 'complete' and advances to the next topic
def advance_topic(state):
    current_topic = get_current_topic(state)

    current_topic["status"] = "completed"

    # If this was the last topic, the conversation is done
    if state["current_topic_index"] == len(state["topics"]) - 1:
        state["conversation_complete"] = True
        state["phase"] = "wrapup"
        return None

    state["current_topic_index"] += 1

    next_topic = get_current_topic(state)
    next_topic["status"] = "active"

    return next_topic

def get_completed_topic_summaries(state):
    summaries = []

    for topic in state["topics"]:
        if topic["status"] != "completed":
            continue

        if topic.get("summary") is None:
            continue

        summaries.append({
            "topic": topic["topic"],
            "summary": topic["summary"],
        })

    return summaries

def get_unsummarized_chunk(
    conversation_history,
    history_start,
    summarized_until,
    chunk_size=10,
):
    chunk_start = history_start + summarized_until
    chunk_end = chunk_start + chunk_size

    # Only summarize when a full new chunk of 10 messages exists
    # before the recent raw context window.
    if len(conversation_history) < chunk_end + chunk_size:
        return []

    return conversation_history[chunk_start:chunk_end]

# Pydantic Models for structured LLM responses
class ConversationStartRequest(BaseModel):
    participant_id: str
    conversation_history: list = Field(default_factory=list)
    topic_history_start: int = 0
    condition: int = CONDITION_MULTIPLE

class ConversationTurnRequest(BaseModel):
    participant_id: str
    user_message: str
    conversation_history: list = Field(default_factory=list)
    condition: int = CONDITION_MULTIPLE

class JordanTurnResult(BaseModel):
    reply: str

class JordanAfterAlexResult(BaseModel):
    reply: str

class JordanAfterAlexRequest(BaseModel):
    participant_id: str
    conversation_history: list = Field(default_factory=list)
    earlier_memory: str | None = None

class PrepareNextTopicRequest(BaseModel):
    participant_id: str
    conversation_history: list = Field(default_factory=list)
    condition: int = CONDITION_MULTIPLE

class AdvanceTopicRequest(BaseModel):
    participant_id: str
    conversation_history: list = Field(default_factory=list)

class TopicSummaryResult(BaseModel):
    user_perspective: str
    concerns: list[str]
    preferences_or_priorities: list[str]

class TopicSelectionRequest(BaseModel):
    participant_id: str
    selected_topics: list[str]
    condition: int = CONDITION_MULTIPLE

class SuggestResourcesRequest(BaseModel):
    participant_id: str

class SuggestedResourcesResult(BaseModel):
    resource_ids: list[str]
    explanation: str

APPROVED_RESOURCES = [
    {
        "id": "featured-trials",
        "title": "See examples of real clinical trials",
        "description": "Explore featured clinical trials from the National Cancer Institute.",
    },
    {
        "id": "personal-stories",
        "title": "Read personal stories",
        "description": "Hear about experiences from people who have participated in cancer research.",
    },
    {
        "id": "sample-consent",
        "title": "See a sample consent form",
        "description": "See the kind of information participants may receive before deciding whether to join a study.",
    },
    {
        "id": "questions-to-ask",
        "title": "See questions you could ask",
        "description": "Review questions you may want to ask a doctor or research team about a clinical trial.",
    },
    {
        "id": "details-checklist",
        "title": "Get ready to look for a trial",
        "description": "See what information about your cancer may be useful when looking for clinical trials.",
    },
    {
        "id": "finding-guide",
        "title": "Learn how to find clinical trials",
        "description": "Read a step-by-step guide to finding clinical trials from the National Cancer Institute.",
    },
    {
        "id": "nci-specialist",
        "title": "Talk with an NCI specialist",
        "description": "Get help and cancer information from a trained National Cancer Institute specialist.",
    },
    {
        "id": "trial-search",
        "title": "Search for clinical trials",
        "description": "Search the National Cancer Institute database for cancer clinical trials.",
    },
]


async def generate_resource_suggestions(state):
    # Gather the participant's completed topic summaries and saved questions.
    participant_context = []

    for topic in state["topics"]:
        participant_context.append({
            "topic": topic["topic"],
            "summary": topic.get("summary"),
            "personally_situated_questions": topic.get(
                "personally_situated_questions", []
            ),
        })

    system_prompt = f"""
    You are selecting educational resources for someone who has just
    completed a conversation about clinical trial participation.

    PARTICIPANT CONTEXT:
    {json.dumps(participant_context)}

    APPROVED RESOURCES:
    {json.dumps(APPROVED_RESOURCES)}

    YOUR TASK:
    Select 2 or 3 resources that would be most useful for this participant
    based on the topics they discussed and the needs they expressed.

    SELECTION GUIDELINES:
    - Prioritize the participant's explicitly expressed questions,
      concerns, preferences, and priorities.
    - Consider their personally situated questions when relevant.
    - Use the topic summaries to understand their information needs.
    - Do not assume that selecting a topic means the participant has
      a particular concern or preference.
    - Do not invent information about the participant.
    - Do not infer medical conditions, clinical trial eligibility,
      or readiness to participate.
    - If few personal needs were expressed, select broadly useful
      educational resources relevant to the topics discussed.
    - Return only resource IDs from APPROVED RESOURCES.
    - Do not recommend the same resource twice.
    - Do not generate URLs or invent additional resources.
    - Do not search for, match, evaluate, or recommend specific trials.

    EXPLANATION:
    - Write one brief, conversational explanation introducing
      the suggested resources.
    - Explain the relevance to what the participant discussed,
      without overstating what you know about them.
    - Do not imply that these resources are medically appropriate
      or that the participant should join a clinical trial.
    - Do not mention individual resource titles; the interface
      will display them separately.
    - Use 35 words or fewer.
    """

    response = await client_chat.beta.chat.completions.parse(
        model=UF_LOCAL_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
        ],
        temperature=0,
        response_format=SuggestedResourcesResult,
    )

    parsed = response.choices[0].message.parsed

    if parsed is None:
        raise RuntimeError("Failed to parse resource suggestions")

    # Only accept IDs from the approved resource catalog.
    allowed_ids = {
        resource["id"] for resource in APPROVED_RESOURCES
    }

    valid_ids = []

    for resource_id in parsed.resource_ids:
        if resource_id in allowed_ids and resource_id not in valid_ids:
            valid_ids.append(resource_id)

    if not 2 <= len(valid_ids) <= 3:
        raise ValueError(
            "Model did not return 2-3 distinct approved resources"
        )

    return {
        "resource_ids": valid_ids,
        "explanation": parsed.explanation.strip(),
    }

def clean_character_reply(text: str) -> str:
    return re.sub(
        r"^\s*(?:Alex|Jordan)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

def is_simple_acknowledgement(user_message: str) -> bool:
    text = user_message.strip().lower()
    text = re.sub(r"[.!?]+$", "", text).strip()

    acknowledgements = {
        "okay",
        "ok",
        "got it",
        "i got it",
        "makes sense",
        "that makes sense",
        "understood",
        "i understand",
        "thanks",
        "thank you",
        "sounds good",
        "alright",
        "all right",
    }

    return text in acknowledgements

def extract_json_object(text: str) -> dict:
    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            f"No valid JSON object found in model response: {text}"
        )

    return json.loads(text[start:end + 1])

async def generate_jordan_wrapup_intro(
    state,
    conversation_history,
    speaker="jordan",
):
    topics_covered = [
        topic["topic"]
        for topic in state["topics"]
    ]

    if speaker == "alex":
        identity = (
            "You are Alex, a virtual character helping a user talk through their "
            "thoughts about clinical trial participation."
        )
        factual_support = (
            "Make clear that factual information can be provided when helpful."
        )
    else:
        identity = (
            "You are Jordan, a virtual companion helping a user talk through their "
            "thoughts about clinical trial participation."
        )
        factual_support = (
            "Make clear that Alex can provide factual information when helpful."
        )

    system_prompt = f"""
    {identity}

    The user has now finished discussing all three of their selected topics.

    TOPICS THEY COVERED:
    {json.dumps(topics_covered)}

    YOUR TASK:
    - Briefly acknowledge that they have finished their selected topics.
    - Naturally mention the three topics they discussed.
    - Do not list them mechanically or use bullets.
    - Then open the floor for anything else they would like to talk through
      about clinical trial participation.
    - They may have another question, concern, belief, or thought they want
      to explore.
    - {factual_support}
    - Let the user know that if they are finished with the conversation, they can
    use the Finish button in the top-right corner of their screen.
    - Do not provide factual clinical trial information in this response.

    {INTERACTION_BOUNDARY}

    RESPONSE STYLE:
    - Keep your response conversational and concise.
    - Keep your response to 50 words or less.
    """

    history_messages = []

    for message in conversation_history:
        sender = message.get("from")
        text = message.get("text")

        if not text:
            continue

        if sender == "user":
            history_messages.append({
                "role": "user",
                "content": text
            })
        else:
            history_messages.append({
                "role": "assistant",
                "content": f"{sender}: {text}"
            })

    response = await client_chat.beta.chat.completions.parse(
        model=UF_LOCAL_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *history_messages,
        ],
        temperature=0,
        response_format=JordanTurnResult,
    )

    parsed = response.choices[0].message.parsed

    if parsed is None:
        raise RuntimeError("Failed to parse Jordan wrap-up intro")

    parsed.reply = clean_character_reply(parsed.reply)
    return parsed

# Jordan's LLM response
async def generate_jordan_response(
    state,
    user_message,
    conversation_history,
    earlier_memory=None,
    speaker="jordan",
):
    if state.get("phase") == "wrapup":
        conversation_context = f"""
        The user has finished discussing their three selected topics.
        You are now in an open-ended wrap-up discussion about clinical trial participation.

        EARLIER CONVERSATION MEMORY:
        {earlier_memory or "None yet."}
        """

        resolution_instruction = (
            "If the user indicates that their question was answered or the information "
            "was helpful and raises no new information need, briefly acknowledge "
            "their response without asking another question."
        )

    else:
        current_topic = get_current_topic(state)
        prior_topic_summaries = get_completed_topic_summaries(state)

        conversation_context = f"""
        CURRENT TOPIC:
        {current_topic["topic"]}

        PRIOR TOPIC SUMMARIES:
        {json.dumps(prior_topic_summaries)}

        EARLIER CONVERSATION MEMORY FOR THIS TOPIC:
        {earlier_memory or "None yet."}
        """

        resolution_instruction = (
            "If the user indicates that their question was answered or the information "
            "was helpful and raises no new information need, briefly acknowledge "
            "their response without asking another question."
        )
    history_messages = []

    for message in conversation_history:
        sender = message.get("from")
        text = message.get("text")

        if not text:
            continue

        if sender == "user":
            history_messages.append({
                "role": "user",
                "content": text
            })
        else:
            history_messages.append({
                "role": "assistant",
                "content": f"{sender}: {text}"
            })

    if speaker == "alex":
        identity = (
            "You are Alex, a virtual character having an ongoing conversation with a user "
            "about clinical trial participation."
        )
        factual_request = (
            "Switch to your separate factual response to provide the factual information."
        )
        wrapup_role = (
            "During the open-ended wrap-up discussion, refer to the support you can provide "
            "without referring to another virtual character."
        )
    else:
        identity = (
            "You are Jordan, a virtual companion having an ongoing conversation with a user "
            "about clinical trial participation."
        )
        factual_request = (
            "Request Alex to provide the factual information."
        )
        wrapup_role = (
            "During the open-ended wrap-up discussion, speak on behalf of both Jordan and "
            "Alex when referring to the support available in the conversation. Use collective "
            "language rather than presenting Jordan as the only character helping the user."
        )

    # Jordan continues exploring the user's perspective
    system_prompt = f"""
    {identity}

    CONVERSATION CONTEXT:
    {conversation_context}

    YOUR GOAL:
    Your role is to elicit the user's thoughts, beliefs, concerns, priorities, and questions 
    to help identify what information about clinical trial participation would be most useful to the user.

    CONVERSATION USE:
    - Continue naturally from the recent conversation and do not repeat questions already answered.
    - Use EARLIER CONVERSATION MEMORY to remember relevant things no longer in the recent history.
    - Prefer the recent conversation if it adds to or changes something from the memory.
    - Use PRIOR TOPIC SUMMARIES only when they are relevant to the current conversation.

    As the conversation develops:
    - If the user expresses a broad reaction or concern, ask one focused question
    about what they want to understand.
    - If a factual question or clear information need emerges, stop clarifying.
    {factual_request}
    - Do not keep narrowing a concern or preference once it is clear.
    - Do not ask how information changes the user's comfort, confidence,
    willingness to participate, or treatment decision.
    - {resolution_instruction}
    - Stay with the current topic unless the user raises another clinical-trial topic.
    - {wrapup_role}
    - Do not provide factual clinical-trial information in this response.

    {INTERACTION_BOUNDARY}

    RESPONSE STYLE:
    - Keep your reply conversational and concise.
    - Ask one question only when clarification is needed; otherwise, briefly acknowledge the user without asking a question.
    - Keep your response to 35 words or less.
    """

    response = await client_chat.beta.chat.completions.parse(
        model=UF_LOCAL_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *history_messages,
            {"role": "user", "content": user_message},
        ],
        temperature=0,
        response_format=JordanTurnResult,
    )

    parsed = response.choices[0].message.parsed

    if parsed is None:
        raise RuntimeError("Failed to parse Jordan response")

    parsed.reply = clean_character_reply(parsed.reply)
    return parsed

async def generate_jordan_after_alex(
    state,
    conversation_history,
    earlier_memory=None,
    speaker="jordan",
    after_topic_intro=False,
):
    if after_topic_intro:
        response_guidance = """
        Information about a new topic was just introduced.
        Invite the user to share their initial thoughts, beliefs, concerns, priorities,
        or questions in reaction to the information that was introduced without assuming how they feel. 
        Do not ask a generic question that could have been asked without hearing the introduction.
        """
    else:
        response_guidance = """
        A factual information need was just addressed.
        Return the conversation to the user by inviting them to share
        their reaction to the information Alex provided.
        Avoid repeating the follow-up approach used recently.
        """

    if state.get("phase") == "wrapup":
        conversation_context = f"""
        The user has finished discussing their three selected topics.
        You are now in an open-ended wrap-up discussion about clinical trial participation.

        EARLIER CONVERSATION MEMORY:
        {earlier_memory or "None yet."}
        """
    else:
        current_topic = get_current_topic(state)
        prior_topic_summaries = get_completed_topic_summaries(state)

        conversation_context = f"""
        CURRENT TOPIC:
        {current_topic["topic"]}

        PRIOR TOPIC SUMMARIES:
        {json.dumps(prior_topic_summaries)}

        EARLIER CONVERSATION MEMORY FOR THIS TOPIC:
        {earlier_memory or "None yet."}
        """

    if speaker == "alex":
        identity = (
            "You are Alex, a virtual character having an ongoing conversation with a user "
            "about clinical trial participation."
        )
        role_transition = """
        - Use the history to understand why you just provided factual information and
        what the conversation was about before that response.
        - Continue naturally from the factual information you just provided.
        """
    else:
        identity = (
            "You are Jordan, a virtual companion having an ongoing conversation with a user "
            "about clinical trial participation."
        )
        role_transition = """
        - Use the history to understand why Alex just spoke and what the conversation
        was about before Alex responded.
        - Briefly thank Alex for providing the information, then continue with the user. 
        Make this feel like a natural handoff between the two characters,
        rather than simply reacting to Alex's information as though you provided it.
        """

    system_prompt = f"""
    {identity}

    CONVERSATION CONTEXT:
    {conversation_context}

    YOUR ROLE:
    - Continue naturally from the conversation history.
    {role_transition}
    - Use EARLIER CONVERSATION MEMORY when relevant to avoid repeating or losing earlier context.
    - Use PRIOR TOPIC SUMMARIES only when they are relevant to the current conversation.
    {response_guidance}
    - Do not ask how the information changes the user's comfort, confidence,
    willingness to participate, or treatment decision.
    - Do not repeat or re-explain the factual information.
    - Do not interpret what the factual information means for the user.
    - Do not provide additional factual clinical trial information in this response.
    - Ask at most one main question.

    {INTERACTION_BOUNDARY}

    RESPONSE STYLE:
    - Keep your reply conversational and concise.
    - Sound like you are continuing the existing conversation.
    - Keep your response to 35 words or less.
    """

    history_messages = []

    for message in conversation_history:
        sender = message.get("from")
        text = message.get("text")

        if not text:
            continue

        if sender == "user":
            history_messages.append({
                "role": "user",
                "content": text
            })
        else:
            history_messages.append({
                "role": "assistant",
                "content": f"{sender}: {text}"
            })

    response = await client_chat.beta.chat.completions.parse(
        model=UF_LOCAL_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *history_messages,
        ],
        temperature=0,
        response_format=JordanAfterAlexResult,
    )

    parsed = response.choices[0].message.parsed

    if parsed is None:
        raise RuntimeError("Failed to parse Jordan response")

    parsed.reply = clean_character_reply(parsed.reply)
    return parsed

async def generate_topic_summary(
    topic_name,
    conversation_history,
    earlier_memory=None,
):
    system_prompt = f"""
    Summarize what was learned about the user while discussing this clinical
    trial topic:

    TOPIC:
    {topic_name}

    EARLIER CONVERSATION MEMORY:
    {earlier_memory or "None yet."}

    Use the earlier memory together with the recent conversation to produce the final topic summary.

    Capture only information the user actually expressed.

    USER PERSPECTIVE:
    Briefly summarize the user's overall thoughts, beliefs, or feelings about
    the topic.

    CONCERNS:
    List any concerns, uncertainties, or questions the user expressed.

    PREFERENCES OR PRIORITIES:
    List any preferences, priorities, values, or personal considerations the
    user expressed.

    Do not infer information the user did not express.
    Keep the summary concise.
    """

    history_messages = []

    for message in conversation_history:
        sender = message.get("from")
        text = message.get("text")

        if not text:
            continue

        if sender == "user":
            history_messages.append({
                "role": "user",
                "content": text,
            })
        else:
            history_messages.append({
                "role": "assistant",
                "content": f"{sender}: {text}",
            })

    response = await client_chat.beta.chat.completions.parse(
        model=UF_LOCAL_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *history_messages,
        ],
        temperature=0,
        response_format=TopicSummaryResult,
    )

    parsed = response.choices[0].message.parsed

    if parsed is None:
        raise RuntimeError("Failed to parse topic summary response")

    return parsed

async def update_rolling_summary(
    existing_summary,
    conversation_chunk,
):
    history_messages = []

    for message in conversation_chunk:
        sender = message.get("from")
        text = message.get("text")

        if not text:
            continue

        if sender == "user":
            history_messages.append({
                "role": "user",
                "content": text,
            })
        else:
            history_messages.append({
                "role": "assistant",
                "content": f"{sender}: {text}",
            })

    system_prompt = f"""
    You are maintaining a concise memory of an ongoing conversation
    about clinical trial participation.

    EXISTING MEMORY:
    {existing_summary or "None yet."}

    Update the memory using the new conversation chunk.

    Focus on information that may help continue the conversation later:
    - the user's thoughts, beliefs, concerns, preferences, and priorities,
    - factual questions or information needs they expressed,
    - relevant reactions to information Alex provided,
    - anything already resolved or clarified that should not be asked again.

    Do not add clinical trial facts on your own.
    Do not invent anything about the user.
    Do not repeat conversational filler.

    Return a concise paragraph.
    """

    response = await client_chat.chat.completions.create(
        model=UF_LOCAL_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *history_messages,
        ],
        temperature=0,
    )

    return (
        response.choices[0].message.content or ""
    ).strip()

async def finish_pending_summary(
    participant_id,
    state,
    current_topic,
    summary_task,
    chunk_size,
):
    try:
        updated_summary = await summary_task

        current_topic["rolling_summary"] = updated_summary
        current_topic["summarized_until"] += chunk_size

        persist_state(
            participant_id,
            state,
        )

    except Exception as error:
        print("*** ROLLING SUMMARY ERROR:", repr(error))

# ENDPOINTS 

@router.post("/suggest-resources")
async def suggest_resources(request: SuggestResourcesRequest):
    state = get_state(request.participant_id)

    # Recommendations are available only after all topics are complete.
    if not state.get("conversation_complete"):
        raise HTTPException(
            status_code=400,
            detail="Complete all three topics before requesting resources.",
        )

    # Only conversational conditions receive tailored suggestions.
    if state["condition"] not in (
        CONDITION_SINGLE_COMBINED,
        CONDITION_MULTIPLE,
    ):
        raise HTTPException(
            status_code=403,
            detail="Resource suggestions are not available for this condition.",
        )

    try:
        suggestions = await generate_resource_suggestions(state)
    except Exception as error:
        print("*** RESOURCE SUGGESTION ERROR:", repr(error))
        raise HTTPException(
            status_code=503,
            detail="Could not generate resource suggestions.",
        ) from error

    return suggestions

@router.post("/turn")
async def conversation_turn(request: ConversationTurnRequest):

    state = get_state(request.participant_id)

    speaker = get_conversation_speaker(state)

    # WRAP-UP PHASE:
    # The selected topics are complete, but Jordan and Alex can continue
    # discussing anything else the user wants to explore.
    if state.get("phase") == "wrapup":

        shared_memory = state.get("wrapup_rolling_summary")

        unsummarized_start = (
            state["wrapup_history_start"]
            + state["wrapup_summarized_until"]
        )

        shared_history = request.conversation_history[
            unsummarized_start:
        ]

        unsummarized_chunk = get_unsummarized_chunk(
            request.conversation_history,
            state["wrapup_history_start"],
            state["wrapup_summarized_until"],
        )

        wrapup_summary_finish_task = None
        summary_task = None

        if unsummarized_chunk:
            summary_task = asyncio.create_task(
                update_rolling_summary(
                    shared_memory,
                    unsummarized_chunk,
                )
            )

        if summary_task:
            async def finish_wrapup_summary():
                try:
                    updated_summary = await summary_task

                    state["wrapup_rolling_summary"] = updated_summary
                    state["wrapup_summarized_until"] += len(
                        unsummarized_chunk
                    )

                    persist_state(
                        request.participant_id,
                        state,
                    )

                except Exception as error:
                    print(
                        "*** WRAPUP ROLLING SUMMARY ERROR:",
                        repr(error),
                    )

            wrapup_summary_finish_task = asyncio.create_task(
                finish_wrapup_summary()
            )

        if state["condition"] == CONDITION_SINGLE_INFO:
            if wrapup_summary_finish_task:
                await wrapup_summary_finish_task

            return {
                "jordan_reply": None,
                "alex_reply": None,
                "alex_info_needed": True,
                "alex_reasoning": "",
                "shared_memory": shared_memory,
                "shared_history": shared_history,
                "prior_topic_summaries": get_completed_topic_summaries(state),
                "topic_done": False,
                "topic_advanced": False,
                "conversation_complete": state["conversation_complete"],
                "current_topic_index": state["current_topic_index"],
                "phase": state["phase"],
                "topics": state["topics"],
            }

        alex_support_result = await analyze_alex_support(
            "Open-ended wrap-up discussion about clinical trial participation",
            request.user_message,
            shared_history,
            earlier_memory=shared_memory,
            condition=state["condition"],
        )

        if alex_support_result.alex_info_needed:
            if wrapup_summary_finish_task:
                await wrapup_summary_finish_task

            return {
                "jordan_reply": None,
                "alex_info_needed": True,
                "alex_reasoning": alex_support_result.reasoning,

                "shared_memory": shared_memory,
                "shared_history": shared_history,

                "prior_topic_summaries": get_completed_topic_summaries(state),

                "topic_done": False,
                "topic_advanced": False,
                "conversation_complete": True,
                "current_topic_index": state["current_topic_index"],
                "phase": state["phase"],
                "topics": state["topics"],
            }

        jordan_result = await generate_jordan_response(
            state,
            request.user_message,
            shared_history,
            earlier_memory=shared_memory,
            speaker=speaker,
        )

        if wrapup_summary_finish_task:
            await wrapup_summary_finish_task

        return {
            "jordan_reply": jordan_result.reply,
            "alex_info_needed": False,
            "alex_reasoning": alex_support_result.reasoning,
            "topic_done": False,
            "topic_advanced": False,
            "conversation_complete": True,
            "current_topic_index": state["current_topic_index"],
            "phase": state["phase"],
            "topics": state["topics"],
        }
    
    current_topic = get_current_topic(state)

    # Freeze the rolling memory that existed at the START of this turn.
    shared_memory = current_topic.get("rolling_summary")

    # Everything after summarized_until has NOT been represented
    # in rolling memory yet, so keep all of it raw.
    unsummarized_start = (
        state["topic_history_start"]
        + current_topic["summarized_until"]
    )

    shared_history = request.conversation_history[
        unsummarized_start:
    ]

    # Check whether the oldest 10 unsummarized messages
    # are now ready to be folded into memory.
    unsummarized_chunk = get_unsummarized_chunk(
        request.conversation_history,
        state["topic_history_start"],
        current_topic["summarized_until"],
    )

    summary_finish_task = None
    summary_task = None

    if unsummarized_chunk:
        summary_task = asyncio.create_task(
            update_rolling_summary(
                shared_memory,
                unsummarized_chunk,
            )
        )

    if summary_task:
        summary_finish_task = asyncio.create_task(
            finish_pending_summary(
                request.participant_id,
                state,
                current_topic,
                summary_task,
                len(unsummarized_chunk),
            )
        )


    # Start personal-need classification concurrently with Alex routing.
    classification_task = None

    if state["condition"] != CONDITION_SINGLE_INFO:
        classification_task = asyncio.create_task(
            classify_user_needs(
                user_message=request.user_message,
                conversation_history=request.conversation_history,
            )
        )

    # Determine whether Alex should provide information.
    if state["condition"] == CONDITION_SINGLE_INFO:
        alex_info_needed = True
        alex_reasoning = ""
    else:
        alex_support_result = await analyze_alex_support(
            current_topic["topic"],
            request.user_message,
            shared_history,
            earlier_memory=shared_memory,
            condition=state["condition"],
        )

        alex_info_needed = alex_support_result.alex_info_needed
        alex_reasoning = alex_support_result.reasoning

    
    # Detect and save personally situated information needs.
    if classification_task is not None:
        try:
            classification = await classification_task

            # Remember whether this turn needs Jordan to capture a question.
            # Determine whether this turn contains a need worth capturing.
            need_types = [need.type for need in classification.needs]

            if "personally_situated" in need_types:
                state["pending_capture_type"] = "personally_situated"
            elif "personal_application" in need_types:
                state["pending_capture_type"] = "personal_application"
            else:
                state["pending_capture_type"] = None

            state["pending_capture"] = (
                state["pending_capture_type"] is not None
            )

            if state["pending_capture"]:
                state["pending_capture_user_message"] = request.user_message
            else:
                state.pop("pending_capture_user_message", None)
                state.pop("pending_capture_type", None)

            persist_state(request.participant_id, state)

        except Exception as error:
            print(
                "*** PERSONALLY SITUATED TRACKING ERROR:",
                repr(error),
            )
            state.pop("pending_capture", None)
            state.pop("pending_capture_user_message", None)
            state.pop("pending_capture_type", None)

            persist_state(request.participant_id, state)

    # In the information-only condition, a simple acknowledgement
    # should not trigger another unsolicited factual response.
    if (
        state["condition"] == CONDITION_SINGLE_INFO
        and is_simple_acknowledgement(request.user_message)
    ):
        if summary_finish_task:
            await summary_finish_task

        return {
            "alex_reply": (
                "You're welcome! Feel free to ask any other questions about this topic."
            ),
            "jordan_reply": None,
            "alex_info_needed": False,
            "alex_reasoning": "",
            "topic_done": False,
            "topic_advanced": False,
            "conversation_complete": state["conversation_complete"],
            "current_topic_index": state["current_topic_index"],
            "phase": state["phase"],
            "topics": state["topics"],
        }

    # 3. Decide whether Alex should respond.
    if alex_info_needed:
        if summary_finish_task:
            await summary_finish_task

        return {
            "jordan_reply": None,
            "alex_info_needed": True,
            "alex_reasoning": alex_reasoning,

            "shared_memory": shared_memory,
            "shared_history": shared_history,

            "prior_topic_summaries": get_completed_topic_summaries(state),

            "topic_done": False,
            "topic_advanced": False,
            "conversation_complete": state["conversation_complete"],
            "current_topic_index": state["current_topic_index"],
            "phase": state["phase"],
            "topics": state["topics"],
        }

    # Capture personally situated questions when Alex does not respond.
    if (
        state["condition"] in (CONDITION_SINGLE_COMBINED, CONDITION_MULTIPLE)
        and state.get("pending_capture")
    ):
        capture = await generate_jordan_capture(
            user_message=state["pending_capture_user_message"],
            alex_response="",
            conversation_history=request.conversation_history,
            speaker=speaker,
            need_type=state["pending_capture_type"],
        )

        save_personally_situated_question(
            state,
            capture.question,
            state["pending_capture_type"],
        )

        saved_type = state["pending_capture_type"]
        state.pop("pending_capture", None)
        state.pop("pending_capture_user_message", None)
        state.pop("pending_capture_type", None)

        if summary_finish_task:
            await summary_finish_task

        persist_state(request.participant_id, state)

        return {
            "jordan_reply": capture.response,
            "alex_info_needed": False,
            "alex_reasoning": alex_reasoning,
            "topic_done": False,
            "topic_advanced": False,
            "conversation_complete": state["conversation_complete"],
            "current_topic_index": state["current_topic_index"],
            "phase": state["phase"],
            "topics": state["topics"],
            "question_saved": capture.question,
            "question_type": saved_type,
        }
    
    # 4. Otherwise, Jordan continues helping the user clarify
    # their perspective or information need.
    jordan_result = await generate_jordan_response(
        state,
        request.user_message,
        shared_history,
        earlier_memory=shared_memory,
        speaker=speaker,
    )

    if summary_finish_task:
        await summary_finish_task

    return {
        "jordan_reply": jordan_result.reply,
        "alex_info_needed": False,
        "alex_reasoning": alex_reasoning,
        "topic_done": False,
        "topic_advanced": False,
        "conversation_complete": state["conversation_complete"],
        "current_topic_index": state["current_topic_index"],
        "phase": state["phase"],
        "topics": state["topics"],
    }

@router.post("/advance-topic")
async def advance_topic_from_button(request: AdvanceTopicRequest):
    state = get_state(request.participant_id)

    if state["conversation_complete"]:
        raise HTTPException(
            status_code=400,
            detail="All topics have already been completed.",
        )

    current_topic = get_current_topic(state)

    # Combine previously summarized memory with remaining messages
    shared_memory = current_topic.get("rolling_summary")

    unsummarized_start = (
        state["topic_history_start"]
        + current_topic["summarized_until"]
    )

    final_topic_history = request.conversation_history[unsummarized_start:]

    # Generate and save the topic summary
    topic_summary = await generate_topic_summary(
        current_topic["topic"],
        final_topic_history,
        earlier_memory=shared_memory,
    )

    current_topic["summary"] = topic_summary.model_dump()

    # Record which topic was completed
    completed_topic_index = state["current_topic_index"]
    completed_topic_number = completed_topic_index + 1

    # Clear any pending question capture
    state.pop("pending_capture", None)
    state.pop("pending_capture_user_message", None)
    state.pop("pending_capture_type", None)

    # Advance to the next topic or wrap-up
    advance_topic(state)

    # Set the conversation history boundary for the next phase
    if state["phase"] == "wrapup":
        state["wrapup_history_start"] = len(request.conversation_history)
    else:
        state["topic_history_start"] = len(request.conversation_history)

    
    # Static closing after the third topic
    alex_reply = None
    jordan_reply = None

    if state["conversation_complete"]:
        alex_reply = (
            "That brings us to the end of your three topics! "
            "I hope our conversation has helped you better understand "
            "clinical trial participation."
        )

        if state["condition"] != CONDITION_SINGLE_INFO:
            jordan_reply = (
                "Before you go, would you like to see some suggested "
                "resources to explore next, based on what you shared "
                "during our conversation?"
            )

    # Save the updated state
    persist_state(request.participant_id, state)

    return {
        "topic_advanced": True,
        "completed_topic_number": completed_topic_number,
        "current_topic_index": state["current_topic_index"],
        "conversation_complete": state["conversation_complete"],
        "phase": state["phase"],
        "topics": state["topics"],
        "prepare_next_topic": not state["conversation_complete"],
        "alex_reply": alex_reply,
        "jordan_reply": jordan_reply,
    }

@router.post("/prepare-next-topic")
async def prepare_next_topic(request: PrepareNextTopicRequest):

    state = get_state(request.participant_id)

    speaker = get_conversation_speaker(state)

    current_topic = get_current_topic(state)

    factual_content = await generate_topic_factual_content(
        current_topic["topic"]
    )

    topic_position = (
        "last"
        if state["current_topic_index"] == len(state["topics"]) - 1
        else "middle"
    )

    alex_result = await generate_alex_topic_intro(
        current_topic=current_topic["topic"],
        factual_content=factual_content,
        prior_summaries=get_completed_topic_summaries(state),
        topic_position=topic_position,
        condition=state["condition"],
    )

    jordan_history = [
        {
            "from": "Alex",
            "text": alex_result.reply,
        }
    ]

    if state["condition"] == CONDITION_SINGLE_INFO:
        jordan_reply = None
    else:
        jordan_result = await generate_jordan_after_alex(
            state,
            jordan_history,
            speaker=speaker,
            after_topic_intro=True,
        )
        jordan_reply = jordan_result.reply

    return {
        "alex_reply": alex_result.reply,
        "jordan_reply": jordan_reply,
    }

@router.post("/after-alex")
async def conversation_after_alex(request: JordanAfterAlexRequest):

    state = get_state(request.participant_id)

    speaker = get_conversation_speaker(state)

    if (
        state["condition"] in (CONDITION_SINGLE_COMBINED, CONDITION_MULTIPLE)
        and state.get("pending_capture")
    ):
        alex_messages = [
            message["text"]
            for message in request.conversation_history
            if message.get("from", "").lower() == "alex"
            and message.get("text")
        ]

    
        capture = await generate_jordan_capture(
            user_message=state["pending_capture_user_message"],
            alex_response=alex_messages[-1] if alex_messages else "",
            conversation_history=request.conversation_history,
            speaker=speaker,
            need_type=state["pending_capture_type"],
        )

        save_personally_situated_question(
            state,
            capture.question,
            state["pending_capture_type"],
        )

        saved_type = state["pending_capture_type"]
        state.pop("pending_capture", None)
        state.pop("pending_capture_user_message", None)
        state.pop("pending_capture_type", None)

        persist_state(request.participant_id, state)

        return {
            "jordan_reply": capture.response,
            "question_saved": capture.question,
            "question_type": saved_type,
        }

    jordan_result = await generate_jordan_after_alex(
        state,
        request.conversation_history,
        earlier_memory=request.earlier_memory,
        speaker=speaker,
        after_topic_intro=False,
    )

    return {
        "jordan_reply": jordan_result.reply
    }

@router.get("/state/{participant_id}")
async def get_conversation_state(participant_id: str):

    state = get_state(participant_id)

    return {
        "current_topic_index": state["current_topic_index"],
        "conversation_complete": state["conversation_complete"],
        "phase": state["phase"],
        "topics": state["topics"],
    }

@router.post("/start")
async def conversation_start(request: ConversationStartRequest):

    state = get_state(request.participant_id)

    speaker = get_conversation_speaker(state)

    current_topic = get_current_topic(state)

    # 1. Use factual content generated when topics were selected
    factual_content = await generate_topic_factual_content(
        current_topic["topic"]
    )

    topic_position = "first"

   # 1. Generate factual content for Topic 1
    alex_result = await generate_alex_topic_intro(
        current_topic=current_topic["topic"],
        factual_content=factual_content,
        prior_summaries=[],
        topic_position=topic_position,
        condition=state["condition"],
    )

    # 3. Give Jordan Alex's opening as context
    jordan_history = [
        *request.conversation_history,
        {
            "from": "Alex",
            "text": alex_result.reply,
        }
    ]

    # Topic 1 starts here
    state["topic_history_start"] = request.topic_history_start

    persist_state(
        request.participant_id,
        state,
    )

    # 4. Jordan continues after Alex's topic introduction
    if state["condition"] == CONDITION_SINGLE_INFO:
        jordan_reply = None
    else:
        jordan_result = await generate_jordan_after_alex(
            state,
            jordan_history,
            speaker=speaker,
            after_topic_intro=True,
        )
        jordan_reply = jordan_result.reply

    return {
        "alex_reply": alex_result.reply,
        "jordan_reply": jordan_reply,
        "current_topic_index": state["current_topic_index"],
        "conversation_complete": state["conversation_complete"],
        "phase": state["phase"],
        "topics": state["topics"],
    }

@router.post("/topics")
async def save_selected_topics(request: TopicSelectionRequest):

    if len(request.selected_topics) != 3:
        raise HTTPException(
            status_code=400,
            detail="Exactly 3 topics must be selected."
        )

    state = create_conversation_state(
        request.selected_topics,
        request.condition,
    )

    persist_state(
        request.participant_id,
        state,
    )

    print(
        "*** SAVED TOPICS:",
        request.participant_id,
        request.selected_topics
    )

    return {
        "participant_id": request.participant_id,
        "selected_topics": request.selected_topics,
        "state": state
    }