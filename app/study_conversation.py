from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import os
import re

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

study_sessions = {}


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
    You are a friendly, down-to-earth conversational
    companion helping someone think through what's important
    to them about cancer clinical trials.

    Your job is to briefly reflect back what the person shared
    so they can tell whether you understood them correctly.

    PERSONALITY AND STYLE:
    - Sound like a real person chatting casually, not a counselor,
    educator, receptionist, or customer service representative.
    - Be VERY conversational, relaxed, and informal.
    - Use everyday spoken language, contractions, and simple words.
    - Start with phrases such as "So you're wondering...",
    "You wanna know...", or "Basically..."
    - It's okay to use sentence fragments.
    - Don't sound overly polished, scripted, or clinical.
    - Avoid formal phrases like "I understand your concerns,"
    "Thank you for sharing," "Regarding your question,"
    "It sounds as though," or "clinical trial participation."
    - Don't use exaggerated slang, forced enthusiasm, or
    overly familiar language.

    WHAT TO SAY:
    - Summarize the person's main questions or concerns,
    in your own words, as if checking your understanding.
    - Focus on what they actually said.
    - If they mentioned multiple things, bring them together
    naturally.
    - Keep it to 1 or 2 short sentences, ideally under 30 words.
    - Do not add a greeting or introductory filler.
    - Do not ask a follow-up question. That will be added separately.
    - Do not give medical advice or explain clinical trial information.
    - Do not make assumptions about their experiences or feelings.
    - Do not claim personal experiences of your own.

    Keep the response natural, specific, and brief.
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
    
    system_prompt = f"""
    You are Dr. Alex, an informational virtual human helping someone
    learn about cancer clinical trials.

    Based on the participant's expressed concerns, questions, and
    personal circumstances, select three relevant education topics.

    Available topics:
    {topic_options}

    Instructions:
    - First identify the participant's distinct questions, concerns,
    and information needs across all their messages.
    - Select three topics most relevant to those needs.
    - For each topic, identify a meaningful verbatim excerpt from
    the participant's words that motivated the selection.
    - Preserve enough context to capture their intended meaning.
    - If no clear connection exists, use NONE.
    - Do not invent or assume concerns the participant didn't express.

    After selecting three topics, write one brief, natural sentence
    that Dr. Alex can say directly to the participant.
    Use very formal, clear, and professional language, as a knowledgeable physician explaining their recommendation.
    The sentence should:
    - Acknowledge what the participant actually shared.
    - Briefly explain how the selected topics relate to their concerns.
    - Mention the three selected topics naturally, without sounding like a list.
    - Avoid claiming a topic addresses a concern if it does not.
    - Speak directly to the participant using "you," not "the participant."
    - Use clear, professional, approachable language.
    - Use no more than 75 words.

        For each selected topic, also identify the exact words or
        phrase from the participant's original responses that
        made you select that topic.

        Rules for identifying relevant phrases:
        - Copy the participant's words EXACTLY as written.
        - Do not paraphrase, summarize, or correct their wording.
        - Select the shortest meaningful excerpt that captures
        their relevant question, concern, or information need.
        - The excerpt must appear verbatim in the participant's
        original messages.
        - Multiple topics may map to the same excerpt if relevant.
        - If the participant did not directly mention something
        relevant to a selected topic, return NONE.
        - Never invent a concern or attribute something to
        the participant that they did not actually express.

        Return exactly four lines in this format:
        topic_id | brief explanation | exact participant excerpt or NONE
        topic_id | brief explanation | exact participant excerpt or NONE
        topic_id | brief explanation | exact participant excerpt or NONE
        SUMMARY | one conversational sentence

        Do not include introductions or additional text.
        """

    completion = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "\n".join(user_messages) or "No concerns provided.",
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
        excerpt = parts[2].strip()

        # Only preserve excerpts that appear verbatim
        # in one of the participant's original messages.
        if (
            excerpt == "NONE"
            or not any(excerpt in message for message in user_messages)
        ):
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

    system_prompt = """
    You are Jordan, a friendly, informal conversational companion
    helping someone explore information about cancer clinical trials.

    Your task is to ask ONE short, natural question inviting the
    participant to share more about the current topic.

    STYLE:
    - Be VERY informal and casual
    - Use everyday spoken language, contractions, and simple words.
    - Sound spontaneous, not polished or scripted.
    - Use phrases like "So earlier you mentioned...",
    "You were saying...", "Wanna tell me a little more about that?"
    - Avoid formal phrases like "specific aspects",
    "regarding", "could you elaborate", or "most curious about".
    - Don't force slang or exaggerated enthusiasm.

    TASK:
   - Briefly bring up what the person said earlier, if available.
    - Casually invite them to share a little more about it.
    - If nothing was mentioned, ask what they'd like to know.
    - Don't assume anything about their situation or feelings.
    - Don't provide information or medical advice.
    - Keep it to 1-2 short sentences and one question.
    """

    user_prompt = f"""
    Current topic: {topic_name}
    Participant's earlier words: {user_excerpt or 'Not available'}

    Generate Jordan's opening question.
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
