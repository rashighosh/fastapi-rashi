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

@router.post("/opening-message")
async def opening_message(request: OpeningMessageRequest):

    if request.condition not in (2, 3):
        raise HTTPException(
            status_code=400,
            detail="Opening needs elicitation is only available for conditions 2 and 3."
        )

    speaker = "Alex" if request.condition == 2 else "Jordan"

 
    system_prompt = """
    You are a warm, approachable conversational companion helping
    someone think about cancer clinical trials.

    Respond like a friendly, down-to-earth person having a natural
    conversation, not a counselor or medical professional.
    Use everyday language and natural expressions.

    Briefly acknowledge and validate the concerns the participant has shared
    in one warm, casual, and natural sentence.

    Consider the full conversation history, not just the latest message.
    When multiple concerns have been raised, acknowledge them together
    without repeating previous responses or going into detail.

    Do not assume how the participant feels, make assumptions about their
    personal circumstances, or imply you have shared experiences.

    Do not ask questions, explain clinical trials, offer advice,
    or introduce new concerns.

    Use a warm, relaxed, and down-to-earth conversational style,
    like a friendly peer chatting with someone.
    Write the way someone would actually say this out loud,
    not the way they would write it in an email or report.
    """


    conversation_history = []

    for item in request.history:
        if item.from_.lower() in ("user", "participant"):
            role = "user"
        elif item.from_.lower() in ("alex", "jordan"):
            role = "assistant"
        else:
            continue

        conversation_history.append({
            "role": role,
            "content": item.text,
        })
    
    completion = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *conversation_history,
            {"role": "user", "content": request.message},
        ],
        temperature=0.5,
    )

    response_text = (completion.choices[0].message.content or "").strip()

    follow_up = (
        "Before we move along, is there anything else you'd like to share "
        "that would be important to you when talking about clinical trial participation? "
        "If not, you can click the Continue button below!"
    )

    final_text = f"{response_text} {follow_up}"

    return {
        "speaker": speaker,
        "response": final_text,
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
    - Select exactly three distinct topics from the available list.
    - Prioritize topics that relate to what the participant shared.
    - For each topic, briefly explain why it may be relevant,
    connecting it to something the participant actually mentioned.
    - If a topic was not directly mentioned, explain its potential
    usefulness without implying the participant raised that concern.
    - Do not invent personal details or assume how the participant feels.

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
    - Be no more than 45 words.

    Return exactly four lines in this format:
    topic_id | brief explanation
    topic_id | brief explanation
    topic_id | brief explanation
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

    spoken_summary = ""

    for line in result.splitlines():
        if line.startswith("SUMMARY |"):
            spoken_summary = line.split("|", 1)[1].strip()
            break

    valid_topics = {topic["id"]: topic for topic in STUDY_TOPICS}
    selected_topics = []

    for line in result.splitlines():
        parts = line.split("|", 1)

        if len(parts) != 2:
            continue

        topic_id = parts[0].strip()
        explanation = parts[1].strip()

        if (
            topic_id in valid_topics
            and topic_id not in [topic["id"] for topic in selected_topics]
            and explanation
        ):
            selected_topics.append({
                **valid_topics[topic_id],
                "explanation": explanation,
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
            })
 
    return {
        "status": "ready",
        "opening_responses": user_messages,
        "selected_topics": selected_topics,
        "spoken_summary": spoken_summary,
    }



