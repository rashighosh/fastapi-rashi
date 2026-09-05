from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import json
import re
from typing import Any

# Router stuff
router = APIRouter(
    prefix="/alex",
    tags=["alex"],
)

# --------------------------------------------------------------------------
# Dependencies supplied by main.py
# --------------------------------------------------------------------------

rag: Any = None
client_chat: Any = None
UF_LOCAL_MODEL: Any = None

preprocess_question: Any = None
clean_alex_answer: Any = None

def configure_conversation_alex(
    *,
    rag_instance,
    chat_client,
    model,
    preprocess_func,
    clean_alex_answer_func,
):
    global rag
    global client_chat
    global UF_LOCAL_MODEL
    global preprocess_question
    global clean_alex_answer

    rag = rag_instance
    client_chat = chat_client
    UF_LOCAL_MODEL = model

    preprocess_question = preprocess_func
    clean_alex_answer = clean_alex_answer_func

def clean_character_reply(text: str) -> str:
    return re.sub(
        r"^\s*(?:Alex|Jordan)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

# Conditions
CONDITION_SINGLE_INFO = 1
CONDITION_SINGLE_COMBINED = 2
CONDITION_MULTIPLE = 3

TRIAL_MATCHING_BOUNDARY = """
SCOPE BOUNDARY:
- Do not search for, identify, match, shortlist, rank, or recommend specific clinical trials.
- Do not claim that you can find trials for the user.
- Do not collect or request personal medical, location, or eligibility details for the purpose of matching the user to trials.
- If the user asks how to find trials, you may explain general resources and people who can help.
- If the user asks you to find trials for them, stay at that general educational level.
"""

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class ConversationMessage(BaseModel):
    sender: str = Field(alias="from")
    text: str

class ConversationAlexRequest(BaseModel):
    message: str
    history: list[ConversationMessage] = []
    earlier_memory: str | None = None
    prior_topic_summaries: list = Field(default_factory=list)
    condition: int = CONDITION_MULTIPLE

class AlexResponse(BaseModel):
    answer: str

class AlexSupportResult(BaseModel):
    alex_info_needed: bool
    reasoning: str


class AlexTopicIntroResult(BaseModel):
    reply: str

async def analyze_alex_support(
    current_topic,
    user_message,
    conversation_history,
    earlier_memory=None,
    condition=CONDITION_MULTIPLE,
):
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

    if condition == CONDITION_SINGLE_COMBINED:
        clarification_instruction = (
            "Otherwise, continue clarifying what information would help in the separate conversational response."
        )
    else:
        clarification_instruction = (
            "Otherwise, let Jordan continue clarifying what information would help."
        )
            
    system_prompt = f"""
    You are deciding whether the user currently needs NEW factual clinical trial
    information.

    CURRENT TOPIC:
    {current_topic}

    EARLIER CONVERSATION MEMORY:
    {earlier_memory or "None yet."}

    USER MESSAGE:
    {user_message}

    Use the conversation history and earlier memory to understand what the user
    means and what has already been discussed.

    Set alex_info_needed to true when:
    - the user asks a factual question,
    - the user clearly needs factual clarification,
    - the user expresses a clear factual assumption that is shaping their
    perspective, concern, or decision,
    - or their concern has already been clarified and new factual information is
    needed to address it.

    Set alex_info_needed to false when:
    - the user is mainly expressing an opinion, reaction, feeling, preference,
    hesitation, or concern,
    - their underlying reason is still unclear and should be explored first,
    - the relevant factual information has already been provided,
    - or Alex would mainly repeat information already given.

    Do not involve the factual response simply because factual information could
    be relevant. Ask whether the user currently needs NEW factual information for
    the conversation to move forward.

    {clarification_instruction}

    Set reasoning to a brief explanation no more than 25 words.
    """

    try:
        response = await client_chat.beta.chat.completions.parse(
            model=UF_LOCAL_MODEL,
            messages=[
            {"role": "system", "content": system_prompt},
            *history_messages,
            {"role": "user", "content": user_message},
        ],
            temperature=0,
            response_format=AlexSupportResult,
        )

        parsed = response.choices[0].message.parsed

        if parsed is not None:
            return parsed

    except Exception as error:
        print("*** ALEX SUPPORT STRUCTURED PARSE ERROR:", repr(error))

    fallback_prompt = system_prompt + """

    Return ONLY valid JSON in exactly this format:

    {
        "alex_info_needed": true,
        "reasoning": "brief explanation"
    }

    alex_info_needed must be true or false.
    Do not include any text before or after the JSON.
    """

    response = await client_chat.chat.completions.create(
        model=UF_LOCAL_MODEL,
        messages=[
            {"role": "system", "content": fallback_prompt},
            *history_messages,
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )

    raw_content = response.choices[0].message.content or ""

    match = re.search(r"\{.*\}", raw_content, flags=re.DOTALL)

    if not match:
        raise HTTPException(
            status_code=500,
            detail="Could not analyze Alex support",
        )

    return AlexSupportResult.model_validate(
        json.loads(match.group(0))
    )

async def generate_topic_factual_content(topic: str):
    print("*** GENERATING TOPIC FACTUAL CONTENT:", topic)

    results = rag.retrieve(
        topic,
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

    system_prompt = f"""
    You are preparing factual material for a clinical trials educator.

    TOPIC:
    {topic}

    Use ONLY the provided source context.

    Extract the main factual information needed to briefly introduce this topic.

    Do not:
    - personalize the information,
    - give medical advice,
    - recommend a clinical trial,
    - discuss whether a specific person should participate,
    - add information that is not supported by the context.

    Keep the factual content concise and focused on the TOPIC.
    """

    response = await client_chat.chat.completions.create(
        model=UF_LOCAL_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"""
                SOURCE CONTEXT:
                {context_str}
                """.strip(),
            },
        ],
        temperature=0,
    )

    return (
        response.choices[0].message.content or ""
    ).strip()

async def generate_alex_topic_intro(
    current_topic: str,
    factual_content: str,
    prior_summaries: list,
    topic_position: str,
    condition: int = CONDITION_MULTIPLE,
):
    if topic_position == "first":
        topic_position_guidance = (
            "This is the FIRST topic in the conversation. Start by introducing the CURRENT TOPIC as the first thing you will discuss in a natural way."
        )
    elif topic_position == "last":
        topic_position_guidance = (
            "This is the FINAL topic in the conversation. Use a natural transition to start this final topic."
        )
    else:
        topic_position_guidance = (
            "This is a NEW topic after the user has already discussed an earlier topic. Use a natural transition to switch to this topic."
        )

    print("***IN GENERATE ALEX TOPIC INTRO, TOPIC POSITION IS", topic_position)
    print(topic_position_guidance)

    if condition == CONDITION_SINGLE_INFO:
        conversation_transition = (
            "You are briefly introducing the current topic to the user."
        )
        followup_instruction = (
            "Do not ask the user a question."
        )
    elif condition == CONDITION_SINGLE_COMBINED:
        conversation_transition = (
            "You are briefly introducing the current topic before continuing the "
            "conversation with the user about their own thoughts, beliefs, concerns, and priorities."
        )
        followup_instruction = (
            "Do not ask the user a question. A separate conversational response will follow."
        )
    else:
        conversation_transition = (
            "You are briefly introducing the current topic before Jordan talks with the "
            "user about their own thoughts, beliefs, concerns, and priorities."
        )
        followup_instruction = (
            "Do not ask the user a question. Jordan will continue the conversation."
        )

    system_prompt = f"""
    You are Alex, a clinical trials educator.

    {conversation_transition}

    CURRENT TOPIC:
    {current_topic}

    FACTUAL CONTENT:
    {factual_content}

    WHAT WE HAVE LEARNED ABOUT THE USER FROM EARLIER TOPICS:
    {json.dumps(prior_summaries)}

    TOPIC POSITION:
    {topic_position_guidance}

    YOUR TASK:
    - Introduce the CURRENT TOPIC naturally based on the TOPIC POSITION guidance above.
    - Then give a brief factual introduction to the CURRENT TOPIC.
    - Use only the FACTUAL CONTENT above for factual claims.
    - Cover the main information needed to understand the topic.
    - If something from the earlier topic summaries is relevant,
      explicitly mention it and connect the explanation to it.
    - Do not force personalization when the earlier summary information is not relevant.
    - Do not introduce facts based on the summaries.
    - {followup_instruction}

    {TRIAL_MATCHING_BOUNDARY}

    RESPONSE STYLE:
    - Sound like you are introducing a topic, not answering a question.
    - Keep your response to 75 words or less.
    """

    response = await client_chat.beta.chat.completions.parse(
        model=UF_LOCAL_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
        ],
        temperature=0,
        response_format=AlexTopicIntroResult,
    )

    parsed = response.choices[0].message.parsed

    if parsed is None:
        raise RuntimeError("Failed to parse Alex topic intro")

    parsed.reply = clean_character_reply(parsed.reply)
    return parsed

@router.post("/conversation-alex")
async def conversation_alex(request: ConversationAlexRequest):
    print("*** IN CONVERSATION ALEX")

    question = request.message
    earlier_memory = request.earlier_memory
    prior_topic_summaries = request.prior_topic_summaries

    history = []

    for message in request.history:
        if message.sender == "user":
            history.append({
                "role": "user",
                "content": message.text,
            })
        else:
            history.append({
                "role": "assistant",
                "content": f"{message.sender}: {message.text}",
            })

    if preprocess_question is None:
        raise RuntimeError("conversation_alex has not been configured")

    preprocess_history = history.copy()

    if earlier_memory:
        preprocess_history.insert(
            0,
            {
                "role": "assistant",
                "content": f"Earlier conversation memory: {earlier_memory}",
            },
        )

    if prior_topic_summaries:
        preprocess_history.insert(
            0,
            {
                "role": "assistant",
                "content": f"Prior topic summaries: {json.dumps(prior_topic_summaries)}",
            },
        )

    preprocess = await preprocess_question(
        question,
        preprocess_history,
    )

    # Retrieve relevant trusted-source information.
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

    if request.condition in (
        CONDITION_SINGLE_INFO,
        CONDITION_SINGLE_COMBINED,
    ):
        entry_instruction = (
            "Naturally connect the factual response to what the user just expressed, "
            "then provide the relevant factual information."
        )
    else:
        entry_instruction = (
            "When entering the conversation, briefly signal that you are stepping in to provide "
            "relevant factual information. Naturally connect your response to what the user just "
            "expressed, then give the factual explanation."
        )

    system_prompt = f"""
    You are Alex, a clinical trials educator.

    You are providing factual information that supports an ongoing
    conversation about clinical trial participation.

    Use conversation history only to understand what the user is referring to.
    Use EARLIER CONVERSATION MEMORY when relevant to understand context that may no longer appear in the recent history.
    Treat that memory only as conversation context, not as factual evidence.
    Use PRIOR TOPIC SUMMARIES only when relevant to understand the user's context.
    Treat them as conversation memory, not factual evidence.
    Use ONLY the provided context as factual evidence.

    Respond directly to the factual question, uncertainty, assumption,
    or misunderstanding expressed by the user.

    Use the conversation history and earlier memory to avoid repeating factual
    information already provided.

    Focus on NEW information that directly addresses the user's current need.
    Do not restate the same facts simply because they are still relevant.

    If part of the answer was already explained, briefly acknowledge that and
    provide only what is new or necessary.

    Do not ask the user a follow-up question.
    Do not give personal medical advice.
    Do not recommend a clinical trial.
    Do not judge whether the user would be eligible for a clinical trial.

    If there is no single general answer:
    - briefly explain why it varies,
    - provide any useful information supported by the context,
    - and explain what may depend on the specific trial.

    Use simple words and short sentences.
    Explain medical terms in plain language.
    Be friendly, direct, and conversational.

    {entry_instruction}

    The transition should make your reason for speaking clear without sounding
    formulaic, repetitive, or like a separate announcement.

    {TRIAL_MATCHING_BOUNDARY}

    Write one conversational paragraph under 75 words.
    Do not use headings, lists, citations, source names, or line breaks.
    """

    chat_messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    for turn in history:
        chat_messages.append(turn)

    chat_messages.append(
        {
            "role": "user",
            "content": f"""
            CONTEXT:
            {context_str}

            EARLIER CONVERSATION MEMORY:
            {earlier_memory or "None yet."}

            PRIOR TOPIC SUMMARIES:
            {json.dumps(prior_topic_summaries)}

            CURRENT USER MESSAGE:
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
            response_format=AlexResponse,
        )

        parsed = response.choices[0].message.parsed

        if parsed is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to parse Alex response",
            )

        return {
            "answer": clean_alex_answer(parsed.answer),
        }

    except HTTPException:
        raise

    except Exception as error:
        print(
            "*** CONVERSATION ALEX ERROR:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail="Could not generate Alex response",
        )