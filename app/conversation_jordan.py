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

# Condition info
CONDITION_SINGLE_INFO = 1
CONDITION_SINGLE_COMBINED = 2
CONDITION_MULTIPLE = 3

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
    condition: int = CONDITION_MULTIPLE

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
    speaker="jordan",
):
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
    Your role is to help the user talk through and make sense of their thoughts,
    beliefs, concerns, priorities, and questions so that, when useful, information can be provided separately.

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
    unnecessarily. {factual_request}
    - If the discussion is becoming increasingly narrow, personal, hypothetical,
    or unrelated to what information would actually be useful, stop probing
    further and redirect toward what the user would want or need to know.
    - If there is a CURRENT TOPIC, keep the conversation focused on that topic
    unless the user's comment is necessary to understand their concern.
    - If you are in the open-ended wrap-up discussion, the user may explore any
    clinical trial participation topic that matters to them.
    - {wrapup_role}
    - Do not provide factual clinical trial information in this response.
    - Ask at only one main question at a time.

    RESPONSE STYLE:
    - Keep your reply conversational and concise.
    - Focus on understanding the user's underlying perspective, concern,
    belief, preference, or priority in order to identify what information would be useful to the user.
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
    speaker="jordan",
):
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
        factual_transition = (
            "You just provided factual clinical trial information."
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
        factual_transition = (
            "Another virtual character, Alex, has just spoken."
        )
        role_transition = """
        - Use the history to understand why Alex just spoke and what the conversation
        was about before Alex responded.
        - Briefly acknowledge Alex for providing the information before continuing with
        the user. Make this feel like a natural handoff between the two characters,
        rather than simply reacting to Alex's information as though you provided it.
        """

    system_prompt = f"""
    {identity}

    CONVERSATION CONTEXT:
    {conversation_context}

    {factual_transition}

    YOUR ROLE:
    - Continue naturally from the conversation history.
    {role_transition}
    - Use EARLIER CONVERSATION MEMORY when relevant to avoid repeating or losing earlier context.
    - Use PRIOR TOPIC SUMMARIES only when they are relevant to the current conversation.
    - Bring the conversation back to the user's own thoughts, beliefs, concerns,
    priorities, or questions.
    - If a new topic was just introduced, invite the user to share their initial
    perspective on what was shared about that topic.
    - If factual information was just provided in response to something the user
    wanted or needed to know, invite the user to react to or process that
    information and continue the existing thread.
    - Do not repeat or re-explain the factual information.
    - Do not interpret what the factual information means for the user.
    - Do not provide additional factual clinical trial information in this response.
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

    Set topic_done to true only when the user clearly indicates they are finished
    with the current topic or ready to move on.

    If the user explicitly asks to move on, continue, says they are done, says they
    are good for now, or confirms they are ready after being asked, set
    topic_done to true.

    Do not infer that the user is finished simply because:
    - they express agreement or reassurance,
    - they make a concluding-sounding statement,
    - one concern appears resolved,
    - or they do not ask a question.

    If the user is still expressing a thought, concern, question, belief,
    preference, assumption, or other perspective, set topic_done to false.

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

    speaker = get_conversation_speaker(state)

    pending_task = pending_summary_tasks.get(request.participant_id)

    if pending_task:
        await pending_task

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

                except Exception as error:
                    print(
                        "*** WRAPUP ROLLING SUMMARY ERROR:",
                        repr(error),
                    )

                finally:
                    pending_summary_tasks.pop(
                        request.participant_id,
                        None,
                    )

            pending_summary_tasks[
                request.participant_id
            ] = asyncio.create_task(
                finish_wrapup_summary()
            )

        if state["condition"] == CONDITION_SINGLE_INFO:
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

    # 1. Run the routing decisions.
    # c=1 only needs topic completion.
    # c=2/c=3 run topic completion and Alex-support routing concurrently.
    if state["condition"] == CONDITION_SINGLE_INFO:
        topic_completion_result = await analyze_topic_completion(
            state,
            request.user_message,
            shared_history,
            earlier_memory=shared_memory,
        )

        alex_info_needed = True
        alex_reasoning = ""

    else:
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
                condition=state["condition"],
            ),
        )

        alex_info_needed = alex_support_result.alex_info_needed
        alex_reasoning = alex_support_result.reasoning

    # 2. If they seem finished but did not explicitly ask to move on,
    # confirm before advancing.
    if (
        topic_completion_result.topic_done
        and topic_completion_result.needs_confirmation
    ):
        current_topic = get_current_topic(state)

        confirmation_reply = (
            "It sounds like you may be good with this topic for now. "
            "Are you ready to move on?"
        )

        return {
            "jordan_reply": (
                None
                if state["condition"] == CONDITION_SINGLE_INFO
                else confirmation_reply
            ),
            "alex_reply": (
                confirmation_reply
                if state["condition"] == CONDITION_SINGLE_INFO
                else None
            ),
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

        pending_task = pending_summary_tasks.pop(
            request.participant_id,
            None,
        )

        if pending_task and not pending_task.done():
            pending_task.cancel()

            try:
                await pending_task
            except asyncio.CancelledError:
                pass

        transition_history = [
            *request.conversation_history,
            {
                "from": "user",
                "text": request.user_message,
            }
        ]

        final_topic_history = [
            *shared_history,
            {
                "from": "user",
                "text": request.user_message,
            }
        ]

        topic_summary = await generate_topic_summary(
            current_topic["topic"],
            final_topic_history,
            earlier_memory=shared_memory,
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

        if state["condition"] == CONDITION_SINGLE_INFO:
            alex_reply = (
                "We've finished your three selected topics. "
                "You can ask me any other questions you have about clinical trial participation, "
                "or use the Finish button when you're done."
            )
            jordan_reply = None
        else:
            jordan_result = await generate_jordan_wrapup_intro(
                state,
                wrapup_intro_history,
                speaker=speaker,
            )
            alex_reply = None
            jordan_reply = jordan_result.reply

        return {
            "alex_reply": alex_reply,
            "jordan_reply": jordan_reply,
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

    # 3. Decide whether Alex should respond.
    if alex_info_needed:
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
    
    # 4. Otherwise, Jordan continues helping the user clarify
    # their perspective or information need.
    jordan_result = await generate_jordan_response(
        state,
        request.user_message,
        shared_history,
        earlier_memory=shared_memory,
        speaker=speaker,
    )

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

    jordan_result = await generate_jordan_after_alex(
        state,
        request.conversation_history,
        earlier_memory=request.earlier_memory,
        speaker=speaker,
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

    # 4. Jordan continues after Alex's topic introduction
    if state["condition"] == CONDITION_SINGLE_INFO:
        jordan_reply = None
    else:
        jordan_result = await generate_jordan_after_alex(
            state,
            jordan_history,
            speaker=speaker,
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