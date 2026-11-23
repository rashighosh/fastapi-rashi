from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import os
import re
import json
from pathlib import Path
from typing import Literal

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

class TopicRecommendation(BaseModel):
    topic_id: Literal[
        "cost",
        "support",
        "logistics",
        "doctor",
        "side_effects",
        "quality",
    ]
    explanation: str
    user_excerpt: str | None


class TopicSelectionResponse(BaseModel):
    topics: list[TopicRecommendation] = Field(min_length=3, max_length=3)
    spoken_summary: str

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
    - Use only the topic IDs from AVAILABLE TOPICS.
    - Provide a brief explanation of why the topic is relevant.
    - Identify the specific concern or need from the participant's
      response that motivated your selection.
    - Write a short, topic-specific user_excerpt capturing that concern.
    - Use the participant's own wording when possible.
    - Do not include unrelated concerns in the excerpt.
    - Do not invent or assume information the participant did not express.
    - If there is no clear connection, set user_excerpt to null.
    - Select three different topics.

    Also write a brief, professional spoken_summary that:
    - Explicitly names all three topics using their exact names.
    - Briefly explains how they relate to the participant's concerns.
    - Does not invent concerns the participant hasn't expressed.
    - Sounds like a clinician making a professional suggestion.
    - Directly addresses the participant.

    Return ONLY a valid JSON object with this structure:

    {{
      "topics": [
        {{
          "topic_id": "cost",
          "explanation": "Brief explanation",
          "user_excerpt": "Specific participant concern or null"
        }}
      ],
      "spoken_summary": "Brief professional recommendation"
    }}

    The topics array must contain exactly three objects.
    Use null (without quotes) when no relevant user excerpt exists.
    Do not include markdown, code fences, or additional text.
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

    
    # Parse and validate the LLM's JSON response.
    try:
        parsed = TopicSelectionResponse.model_validate_json(result)
    except Exception as e:
        print("Topic selection validation failed:", e)
        raise HTTPException(
            status_code=502,
            detail="Could not generate valid topic recommendations."
        )

    # Ensure all three recommended topics are different.
    topic_ids = [topic.topic_id for topic in parsed.topics]

    if len(set(topic_ids)) != 3:
        raise HTTPException(
            status_code=502,
            detail="The model returned duplicate topics."
        )

    # Convert validated recommendations into the existing frontend format.
    valid_topics = {topic["id"]: topic for topic in STUDY_TOPICS}

    selected_topics = [
        {
            **valid_topics[topic.topic_id],
            "explanation": topic.explanation,
            "user_excerpt": topic.user_excerpt,
        }
        for topic in parsed.topics
    ]

    spoken_summary = parsed.spoken_summary

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
            "explanation": original.get("explanation") if original else None,
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
    You are a friendly, casual virtual companion helping
    someone explore clinical trial participation.

    Your role is to help participants express their personal
    concerns, circumstances, needs, and preferences.

    Use very casual, natural spoken language, like a supportive peer.

    Do not assume the participant wants to join a trial.
    Do not provide factual information or medical advice.
    Do not invent personal circumstances or concerns.
    """

    
    user_prompt = f"""
    CURRENT TOPIC RECOMMENDATION:
    {json.dumps(topic, indent=2)}

    PARTICIPANT'S EARLIER RESPONSES:
    {json.dumps(profile["opening_responses"])}

    PARTICIPANT'S RESPONSES DURING PREVIOUS TOPICS:
    {json.dumps(profile["topic_responses"])}

    Your task is to begin a conversation about the current topic
    by inviting the participant to share a relevant personal experience.

    1. Briefly introduce the topic in one casual sentence.
       Reference something the participant previously mentioned
       about this topic when relevant.
    2. Ask ONE short, open-ended question inviting the participant
       to share a past or current experience from their healthcare,
       treatment, or caregiving journey that relates to this topic.
       Avoid a yes/no question so the user can share their experience.
       The experience does not need to involve clinical trials.
    3. If the participant has already shared a relevant experience,
       invite them to elaborate on it rather than asking for a
       different experience.

    The topic recommendation contains:
    - "name": The topic selected for discussion.
    - "explanation": Alex's interpretation of why this topic is relevant.
    - "user_excerpt": The specific participant concern Alex identified
      as relevant to this topic (which may be paraphrased).

    Use the user_excerpt to identify which concern to explore.
    Refer to the participant's original responses for their actual wording.
    Use Alex's explanation only as context for why the topic was selected.

    Do not assume the participant has had a particular experience.
    Do not require them to disclose personal information.
    Do not introduce unrelated concerns or specific clinical trial details.
    Do not repeat questions already answered.
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


async def generate_study_alex_response(
    topic_context: dict,
    factual_content: str,
):
    system_prompt = """
    You are a professional clinical trials educator.

    Your role is to provide clear, accurate information about
    clinical trial participation, using trusted educational sources.

    You are speaking after a virtual companion has helped the
    participant share relevant healthcare experiences, personal
    circumstances, priorities, or constraints.

    Your task is to use that context to identify and explain
    factual information that would be most relevant and useful
    to the participant.

    Use ONLY the provided FACTUAL CONTENT for factual claims.
    Treat the participant's responses as personal context,
    not as medical evidence.

    Focus on the participant's expressed concerns and needs.
    Do not simply repeat or summarize their experiences.
    Do not invent personal circumstances or information needs.
    Do not provide personalized medical advice.
    Do not recommend specific trials or assess eligibility.
    Do not assume the participant wants to join a trial.

    If the factual content does not answer a relevant concern,
    acknowledge the limitation rather than inventing an answer.

    Use clear, professional language, as if you are a clinician
    sharing credible information.
    Keep your response concise and conversational.
    Keep your response to 150 words or less.
    Do not ask follow-up questions.
    """

    user_prompt = f"""
    CURRENT TOPIC AND PARTICIPANT CONTEXT:
    {json.dumps(topic_context, indent=2)}

    RETRIEVED FACTUAL INFORMATION:
    {factual_content}

    Explain the factual information most relevant to the
    participant's expressed needs as if you were having a conversation.
    Do not use headers, bullet points, or lists.

    Emphasize & relate information to what they shared so it is clear
    to the user why this information is important for them to know.

    Do not cover unrelated details simply because they
    appear in the retrieved information.
    """

    completion = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    return (
        completion.choices[0].message.content or ""
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


    # Organize the participant's information for the current topic.
    topic_context = {
        "topic_name": topic["name"],
        "initial_concern": topic.get("user_excerpt"),
        "participant_responses": user_profile.get(
            "topic_responses", {}
        ).get(topic["id"], []),
    }


    # Generate a retrieval query using the current topic's
    # initial concern and Jordan-elicited participant responses.
    search_query = await generate_study_search_query(
        topic_name=topic["name"],
        user_profile=topic_context,
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
            json.dumps(topic_context)
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


@router.post("/alex-topic-response")
async def alex_topic_response(request: TopicOpeningRequest):
    from conversation_alex import generate_topic_factual_content

    session = study_sessions.get(request.participant_id)

    if not session or not session.get("topics"):
        raise HTTPException(
            status_code=404,
            detail="No active study conversation found."
        )

    topic = session["topics"][session["current_topic_index"]]

    profile = session.get("user_profile", {
        "opening_responses": session.get("opening_responses", []),
        "topic_responses": {},
    })

    # Organize the information relevant to this topic.
    topic_context = {
        "topic_name": topic["name"],
        "initial_concern": topic.get("user_excerpt"),
        "participant_responses": profile.get(
            "topic_responses", {}
        ).get(topic["id"], []),
    }

    # Generate a topic-specific retrieval query.
    search_query = await generate_study_search_query(
        topic_name=topic["name"],
        user_profile=topic_context,
    )

    # Retrieve factual information from trusted sources.
    factual_content = await generate_topic_factual_content(
        f"{topic['name']}: {search_query}"
    )

    # Generate Alex's personalized factual response.
    alex_reply = await generate_study_alex_response(
        topic_context=topic_context,
        factual_content=factual_content,
    )

    print("\n========== STUDY ALEX DEBUG ==========", flush=True)
    print("TOPIC CONTEXT:", json.dumps(topic_context, indent=2), flush=True)
    print("SEARCH QUERY:", search_query, flush=True)
    print("RETRIEVED FACTUAL CONTENT:", factual_content, flush=True)
    print("FINAL ALEX RESPONSE:", alex_reply, flush=True)
    print("======================================\n", flush=True)


    return {
        "speaker": "Alex",
        "topic_id": topic["id"],
        "topic_name": topic["name"],
        "alex_reply": alex_reply,
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


    # The participant has answered Jordan's second question.
    # Their response is already saved, so Alex can now take over.
    if len(topic_responses) >= 2:
        return {
            "speaker": "Jordan",
            "topic_id": topic["id"],
            "jordan_reply": (
                "Thanks for sharing all that! "
                "I think Alex has some information that could be "
                "helpful based on what you've shared. "
                "Let's hear what he has to say."
            ),
            "elicitation_complete": True,
            "user_profile": profile,
        }


    
    system_prompt = """
    You are a friendly, casual virtual companion helping
    someone explore what is important to them to know about clinical trial participation.

    Your role is to help participants express and clarify their
    information needs, hopes, expectations, and preferences.

    Use very casual, natural spoken language, like a supportive peer.
    Keep responses brief and conversational.

    Acknowledge what the participant shares without assuming
    how they feel or what they want.

    Do not assume the participant wants to join a clinical trial.
    Do not provide factual information or medical advice.
    Do not invent personal circumstances or concerns.
    """
    
    user_prompt = f"""
    CURRENT TOPIC:
    {json.dumps(topic, indent=2)}

    PARTICIPANT'S ORIGINAL RESPONSES TO THE OPENING QUESTION:
    {json.dumps(profile["opening_responses"])}

    PARTICIPANT'S RESPONSES DURING TOPIC DISCUSSIONS:
    {json.dumps(profile["topic_responses"])}

    LATEST PARTICIPANT MESSAGE:
    {message}

    The participant was invited to share a relevant experience
    from their healthcare, treatment, or caregiving journey.

    Your task is to briefly acknowledge what they shared and
    explore how their experience relates to what matters to them
    about the current topic.

    Choose ONE of the following approaches based on their response:

    1. PERSONAL CIRCUMSTANCES:
       Explore how their current situation relates to this topic.
    2. PRIORITIES:
       Explore what matters most to them when considering this topic.
    3. CONSTRAINTS:
       Explore any practical concerns or limitations that might
       influence what would work for them.

    Ask ONE short, natural follow-up question using the approach
    that would help you learn something meaningful that is not
    already clear from their responses.

    Do not systematically ask about all three dimensions.
    Do not repeat information or questions already addressed.

    Focus on understanding the participant's personal needs
    and preferences, not collecting detailed medical information.

    Do not introduce specific clinical trial details, hypothetical
    trial scenarios, or concerns the participant hasn't mentioned.
    Do not assume the participant wants to participate in a trial.
    Do not sound like you are trying to convince them a trial is the best option.
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
