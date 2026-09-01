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
    prefix="/conversation",
    tags=["conversation"],
)

from conversation_alex import (
    analyze_alex_support,
    generate_alex_topic_intro,
    generate_topic_factual_content,
)

# Set up LLM stuff
BASE_URL = "https://api.ai.it.ufl.edu/v1"
RASHI_LITELLM_KEY = os.getenv("RASHI_LITELLM_KEY")
UF_LOCAL_MODEL = "gpt-oss-120b"

client_chat = AsyncOpenAI(
    api_key=RASHI_LITELLM_KEY,
    base_url=BASE_URL,
    timeout=45.0
)

# This holds every single participant's conversation state
# Will clear on server restart
conversation_states = {}
pending_summary_tasks = {}

# Creates a new conversation state for a user
# Includes: current topic index, if conversation is complete, and topic list
def create_conversation_state(selected_topics):
    topics: list[dict[str, Any]] = []

    for i, topic in enumerate(selected_topics):
        topics.append({
            "topic": topic,
            "status": "active" if i == 0 else "upcoming",
            "summary": None,
            "rolling_summary": None,
            "summarized_until": 0,
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
    }

# Given a current user state, get the topic they're on
def get_current_topic(state):
    index = state["current_topic_index"]
    return state["topics"][index]

# Given a user, create new state or grab their existing state
def get_state(participant_id):
    state = conversation_states.get(participant_id)

    if state is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation topics have not been initialized."
        )

    return state

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

class ConversationTurnRequest(BaseModel):
    participant_id: str
    user_message: str
    conversation_history: list = Field(default_factory=list)

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

class TopicCompletionResult(BaseModel):
    topic_done: bool
    needs_confirmation: bool
    reasoning: str

class TopicSummaryResult(BaseModel):
    user_perspective: str
    concerns: list[str]
    preferences_or_priorities: list[str]

class TopicSelectionRequest(BaseModel):
    participant_id: str
    selected_topics: list[str]

def clean_jordan_reply(text: str) -> str:
    return re.sub(
        r"^\s*Jordan\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

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
):
    topics_covered = [
        topic["topic"]
        for topic in state["topics"]
    ]

    system_prompt = f"""
    You are Jordan, a virtual companion helping a user talk through their
    thoughts about clinical trial participation.

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
    - Make clear that Alex can provide factual information when helpful.
    - Do not provide factual clinical trial information yourself.

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
        response_format=JordanTurnResult,
    )

    parsed = response.choices[0].message.parsed

    if parsed is None:
        raise RuntimeError("Failed to parse Jordan wrap-up intro")

    parsed.reply = clean_jordan_reply(parsed.reply)
    return parsed

# Jordan's LLM response
async def generate_jordan_response(
    state,
    user_message,
    conversation_history,
    earlier_memory=None,
):
    if state.get("phase") == "wrapup":
        conversation_context = f"""
        The user has finished discussing their three selected topics.
        You are now in an open-ended wrap-up discussion about clinical trial participation.

        EARLIER CONVERSATION MEMORY:
        {state["wrapup_rolling_summary"] or "None yet."}
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

    # Jordan continues exploring the user's perspective
    system_prompt = f"""
    You are Jordan, a virtual companion having an ongoing conversation with a user
    about clinical trial participation.

    CONVERSATION CONTEXT:
    {conversation_context}

    Another virtual character, Alex, provides factual clinical trial information.

    YOUR GOAL:
    Your role is to help the user talk through and make sense of their thoughts,
    beliefs, concerns, priorities, and questions so that, when useful, Alex can
    provide information that is relevant to what matters to the user.

    CONVERSATION USE:
    - Continue naturally from the recent conversation and do not repeat questions already answered.
    - Use EARLIER CONVERSATION MEMORY to remember relevant things no longer in the recent history.
    - Prefer the recent conversation if it adds to or changes something from the memory.
    - Use PRIOR TOPIC SUMMARIES only when they are relevant to the current conversation.

    As the conversation develops:
    - Encourage the user to elaborate when their concern, belief, preference, or
    priority is unclear.
    - Follow the thread of what the user is saying rather than starting a new line
    of questioning each turn.
    - Help move from a broad reaction toward the underlying concern, priority, or
    information need.
    - If a clear factual question or information need emerges, do not keep probing
    unnecessarily. Request Alex to provide the factual information.
    - If the discussion is becoming increasingly narrow, personal, hypothetical,
    or unrelated to what information would actually be useful, stop probing
    further and redirect toward what the user would want or need to know.
    - If there is a CURRENT TOPIC, keep the conversation focused on that topic
    unless the user's comment is necessary to understand their concern.
    - If you are in the open-ended wrap-up discussion, the user may explore any
    clinical trial participation topic that matters to them.
    - Do not provide factual clinical trial information yourself.
    - Ask at only one main question at a time.

    RESPONSE STYLE:
    - Keep your reply conversational and concise.
    - Focus on helping identify what information would be useful to the user.
    - Keep your response to 35 words or less.
    """

    response = await client_chat.beta.chat.completions.parse(
        model=UF_LOCAL_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *history_messages,
            {"role": "user", "content": user_message},
        ],
        response_format=JordanTurnResult,
    )

    parsed = response.choices[0].message.parsed

    if parsed is None:
        raise RuntimeError("Failed to parse Jordan response")

    parsed.reply = clean_jordan_reply(parsed.reply)
    return parsed

async def generate_jordan_after_alex(
    state,
    conversation_history,
    earlier_memory=None,
):
    if state.get("phase") == "wrapup":
        conversation_context = f"""
        The user has finished discussing their three selected topics.
        You are now in an open-ended wrap-up discussion about clinical trial participation.

        EARLIER CONVERSATION MEMORY:
        {state["wrapup_rolling_summary"] or "None yet."}
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

    system_prompt = f"""
    You are Jordan, a virtual companion having an ongoing conversation with a user
    about clinical trial participation.

    CONVERSATION CONTEXT:
    {conversation_context}

    Another virtual character, Alex, has just spoken.

    YOUR ROLE:
    - Continue naturally from the conversation history.
    - Use the history to understand why Alex just spoke and what the conversation
    was about before Alex responded.
    - Use EARLIER CONVERSATION MEMORY when relevant to avoid repeating or losing earlier context.
    - Use PRIOR TOPIC SUMMARIES only when they are relevant to the current conversation.
    - Bring the conversation back to the user's own thoughts, beliefs, concerns,
    priorities, or questions.
    - If Alex just introduced a new topic, invite the user to share their initial
    perspective on what Alex shared about that topic.
    - If Alex just provided factual information in response to something the user
    wanted or needed to know, invite the user to react to or process that
    information and continue the existing thread.
    - Do not repeat or re-explain Alex's information.
    - Do not interpret what Alex's information means for the user.
    - Do not provide factual clinical trial information yourself.
    - Ask at most one main question.

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
        response_format=JordanAfterAlexResult,
    )

    parsed = response.choices[0].message.parsed

    if parsed is None:
        raise RuntimeError("Failed to parse Jordan response")

    parsed.reply = clean_jordan_reply(parsed.reply)
    return parsed

async def analyze_topic_completion(
    state,
    user_message,
    conversation_history,
    earlier_memory=None,
):
    current_topic = get_current_topic(state)
    rolling_summary = earlier_memory

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

    system_prompt = f"""
    You are determining whether the user has finished discussing the current topic.

    CURRENT TOPIC:
    {current_topic["topic"]}

    EARLIER CONVERSATION MEMORY:
    {rolling_summary or "None yet."}

    Use the earlier conversation memory only when it helps determine whether the user has already finished or resolved this topic.

    Set topic_done to true when the user indicates they are finished with the
    current topic or ready to move on.

    If the user explicitly asks to move on, continue, or confirms that they are
    ready after Jordan asks, set needs_confirmation to false.

    If the user appears finished but has not clearly asked or confirmed that they
    want to move on, set needs_confirmation to true.

    If the user is still expressing a thought, concern, question, belief,
    preference, or other perspective, set topic_done to false and
    needs_confirmation to false.


    When uncertain, keep topic_done false.

    Set reasoning to no more than 20 words.
    """

    response = await client_chat.beta.chat.completions.parse(
        model=UF_LOCAL_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *history_messages,
            {"role": "user", "content": user_message},
        ],
        response_format=TopicCompletionResult,
    )

    parsed = response.choices[0].message.parsed

    if parsed is None:
        raise RuntimeError("Failed to parse topic completion response")

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
    current_topic,
    summary_task,
    chunk_size,
):
    try:
        updated_summary = await summary_task

        current_topic["rolling_summary"] = updated_summary
        current_topic["summarized_until"] += chunk_size

    except Exception as error:
        print("*** ROLLING SUMMARY ERROR:", repr(error))

    finally:
        pending_summary_tasks.pop(participant_id, None)

# ENDPOINTS 

@router.post("/turn")
async def conversation_turn(request: ConversationTurnRequest):

    state = get_state(request.participant_id)

    # WRAP-UP PHASE:
    # The selected topics are complete, but Jordan and Alex can continue
    # discussing anything else the user wants to explore.
    if state.get("phase") == "wrapup":
        unsummarized_chunk = get_unsummarized_chunk(
            request.conversation_history,
            state["wrapup_history_start"],
            state["wrapup_summarized_until"],
        )

        wrapup_summary_task = None

        if unsummarized_chunk:
            wrapup_summary_task = asyncio.create_task(
                update_rolling_summary(
                    state["wrapup_rolling_summary"],
                    unsummarized_chunk,
                )
            )

        wrapup_history = request.conversation_history[
            state["wrapup_history_start"]:
        ][-10:]

        wrapup_routing_history = (
            unsummarized_chunk + wrapup_history
            if unsummarized_chunk
            else wrapup_history
        )

        if wrapup_summary_task:
            alex_support_result, updated_wrapup_summary = await asyncio.gather(
                analyze_alex_support(
                    "Open-ended wrap-up discussion about clinical trial participation",
                    request.user_message,
                    wrapup_routing_history,
                    earlier_memory=state.get("wrapup_rolling_summary"),
                ),
                wrapup_summary_task,
            )
        else:
            alex_support_result = await analyze_alex_support(
                "Open-ended wrap-up discussion about clinical trial participation",
                request.user_message,
                wrapup_routing_history,
                earlier_memory=state.get("wrapup_rolling_summary"),
            )

        if wrapup_summary_task:
            state["wrapup_rolling_summary"] = updated_wrapup_summary
            state["wrapup_summarized_until"] += len(unsummarized_chunk)

        # If the user's information need is already clear,
        # Alex can answer directly.
        if alex_support_result.alex_info_needed:
            return {
                "jordan_reply": None,
                "alex_info_needed": True,
                "alex_reasoning": alex_support_result.reasoning,
                "earlier_memory": state.get("wrapup_rolling_summary"),
                "prior_topic_summaries": get_completed_topic_summaries(state),
                "topic_done": False,
                "topic_advanced": False,
                "conversation_complete": True,
                "current_topic_index": state["current_topic_index"],
                "phase": state["phase"],
                "topics": state["topics"],
            }

        # Otherwise Jordan continues helping the user explore/clarify.
        jordan_result = await generate_jordan_response(
            state,
            request.user_message,
            wrapup_history,
        )

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

    summary_task = None

    if unsummarized_chunk:
        summary_task = asyncio.create_task(
            update_rolling_summary(
                shared_memory,
                unsummarized_chunk,
            )
        )

    if summary_task:
        pending_summary_tasks[request.participant_id] = asyncio.create_task(
            finish_pending_summary(
                request.participant_id,
                current_topic,
                summary_task,
                len(unsummarized_chunk),
            )
        )

    print("\n===== CURRENT SHARED CONTEXT =====")
    print("EARLIER MEMORY:")
    print(shared_memory)

    print("\nRECENT HISTORY:")
    print(json.dumps(shared_history, indent=2))

    print("\nPRIOR TOPIC SUMMARIES:")
    print(json.dumps(get_completed_topic_summaries(state), indent=2))

    print("==================================\n")

    # 1. Run the two routing decisions at the same time.
    # Both use the exact same frozen shared context for this turn.
    topic_completion_result, alex_support_result = await asyncio.gather(
        analyze_topic_completion(
            state,
            request.user_message,
            shared_history,
            earlier_memory=shared_memory,
        ),
        analyze_alex_support(
            current_topic["topic"],
            request.user_message,
            shared_history,
            earlier_memory=shared_memory,
        ),
    )

    # 2. If they seem finished but did not explicitly ask to move on,
    # have Jordan confirm first
    if (
        topic_completion_result.topic_done
        and topic_completion_result.needs_confirmation
    ):
        current_topic = get_current_topic(state)

        jordan_reply = (
            "It sounds like you may be good with this topic for now. "
            "Are you ready to move on?"
        )

        return {
            "jordan_reply": jordan_reply,
            "alex_reply": None,
            "alex_info_needed": False,
            "alex_reasoning": "",
            "topic_done": False,
            "topic_advanced": False,
            "conversation_complete": state["conversation_complete"],
            "current_topic_index": state["current_topic_index"],
            "phase": state["phase"],
            "topics": state["topics"],
        }

    # 2. If they are finished, automatically advance
    if topic_completion_result.topic_done:
        completed_topic_index = state["current_topic_index"]
        completed_topic_number = completed_topic_index + 1

        current_topic = get_current_topic(state)

        transition_history = [
            *request.conversation_history,
            {
                "from": "user",
                "text": request.user_message
            }
        ]

        remaining_topic_history = transition_history[
            state["topic_history_start"] + current_topic["summarized_until"]:
        ]

        topic_summary = await generate_topic_summary(
            current_topic["topic"],
            remaining_topic_history,
            earlier_memory=current_topic.get("rolling_summary"),
        )

        current_topic["summary"] = topic_summary.model_dump()

        completed_topic = current_topic.copy()

        advance_topic(state)

        if state["phase"] == "wrapup":
            state["wrapup_history_start"] = len(transition_history)

        # If there is another topic, return now so the frontend can
        # show "Preparing next topic" while the next intro generates.
        if not state["conversation_complete"]:
            state["topic_history_start"] = len(transition_history)

            return {
                "alex_reply": None,
                "jordan_reply": None,
                "alex_info_needed": False,
                "alex_reasoning": "",
                "topic_done": True,
                "topic_advanced": True,
                "prepare_next_topic": True,
                "completed_topic_index": completed_topic_index,
                "completed_topic_number": completed_topic_number,
                "completed_topic": completed_topic["topic"],
                "conversation_complete": state["conversation_complete"],
                "current_topic_index": state["current_topic_index"],
                "phase": state["phase"],
                "topics": state["topics"],
            }

        wrapup_intro_history = transition_history[-10:]

        jordan_result = await generate_jordan_wrapup_intro(
            state,
            wrapup_intro_history,
        )

        return {
            "alex_reply": None,
            "jordan_reply": jordan_result.reply,
            "alex_info_needed": False,
            "alex_reasoning": "",
            "topic_done": True,
            "topic_advanced": True,
            "completed_topic_index": completed_topic_index,
            "completed_topic_number": completed_topic_number,
            "completed_topic": completed_topic["topic"],
            "conversation_complete": state["conversation_complete"],
            "current_topic_index": state["current_topic_index"],
            "phase": state["phase"],
            "topics": state["topics"],
        }

    # 3. If the information need is already clear,
    # Alex can respond directly without a Jordan handoff.
    if alex_support_result.alex_info_needed:
        return {
            "jordan_reply": None,
            "alex_info_needed": True,
            "alex_reasoning": alex_support_result.reasoning,

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

    # 4. Otherwise, Jordan continues helping the user clarify
    # their perspective or information need.
    jordan_result = await generate_jordan_response(
        state,
        request.user_message,
        shared_history,
        earlier_memory=shared_memory,
    )

    return {
        "jordan_reply": jordan_result.reply,
        "alex_info_needed": False,
        "alex_reasoning": alex_support_result.reasoning,
        "topic_done": False,
        "topic_advanced": False,
        "conversation_complete": state["conversation_complete"],
        "current_topic_index": state["current_topic_index"],
        "phase": state["phase"],
        "topics": state["topics"],
    }

@router.post("/prepare-next-topic")
async def prepare_next_topic(request: PrepareNextTopicRequest):

    state = get_state(request.participant_id)
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
    )

    jordan_history = [
        {
            "from": "Alex",
            "text": alex_result.reply,
        }
    ]

    jordan_result = await generate_jordan_after_alex(
        state,
        jordan_history,
    )

    return {
        "alex_reply": alex_result.reply,
        "jordan_reply": jordan_result.reply,
    }

@router.post("/after-alex")
async def conversation_after_alex(request: JordanAfterAlexRequest):

    state = get_state(request.participant_id)

    jordan_result = await generate_jordan_after_alex(
        state,
        request.conversation_history,
        earlier_memory=request.earlier_memory,
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

    # 4. Jordan continues after Alex's topic introduction
    jordan_result = await generate_jordan_after_alex(
        state,
        jordan_history,
    )

    return {
        "alex_reply": alex_result.reply,
        "jordan_reply": jordan_result.reply,
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

    state = create_conversation_state(request.selected_topics)

    conversation_states[request.participant_id] = state

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