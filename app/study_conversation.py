from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import os
import re
import json
from pathlib import Path
from openai.types.chat import ChatCompletionMessageParam
from typing import Literal, Any
from conversation_personas import (
    INFORMATIONAL_PERSONA,
    SUPPORTIVE_PERSONA,
)

router = APIRouter(prefix="/study", tags=["Study Conversation"])

base_url = "https://api.ai.it.ufl.edu"
RASHI_LITELLM_KEY = os.getenv('RASHI_LITELLM_KEY')
UF_LOCAL_MODEL = 'gpt-oss-20b'

client = AsyncOpenAI(
    api_key=RASHI_LITELLM_KEY,
    base_url=base_url,
)

MODEL = UF_LOCAL_MODEL

CONDITION_SINGLE_INFO = 1
CONDITION_SINGLE_COMBINED = 2
CONDITION_MULTIPLE = 3

rag: Any = None
preprocess_question: Any = None
clean_alex_answer: Any = None


def configure_study_conversation(
    *,
    rag_instance,
    preprocess_func,
    clean_alex_answer_func,
):
    global rag
    global preprocess_question
    global clean_alex_answer

    rag = rag_instance
    preprocess_question = preprocess_func
    clean_alex_answer = clean_alex_answer_func

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

def clean_character_reply(text: str) -> str:
    return re.sub(
        r"^\s*(?:Alex|Jordan)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

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

class ConversationAlexRequest(BaseModel):
    message: str
    history: list[ConversationMessage] = Field(default_factory=list)
    earlier_memory: str | None = None
    prior_topic_summaries: list = Field(default_factory=list)
    condition: int = CONDITION_MULTIPLE


class AlexResponse(BaseModel):
    answer: str

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

class StudyClosingRequest(BaseModel):
    participant_id: str
    history: list[ConversationMessage] = Field(default_factory=list)

class JordanClosingResponse(BaseModel):
    reflection: str
    discussion_questions: list[str] = Field(
        min_length=3,
        max_length=3
    )
    spoken_overview: str

class AlexResourceRecommendation(BaseModel):
    resource_id: Literal[
        "featured-trials",
        "personal-stories",
        "sample-consent",
        "questions-to-ask",
        "details-checklist",
        "finding-guide",
        "nci-specialist",
        "trial-search",
    ]
    explanation: str

class AlexClosingResponse(BaseModel):
    recommended_resources: list[AlexResourceRecommendation] = Field(
        min_length=3,
        max_length=3
    )
    spoken_overview: str

class NextStepsSelectedRequest(BaseModel):
    participant_id: str
    questions_available: bool
    questions_selected: bool
    resources_selected: bool

@router.post("/opening-message")
async def opening_message(request: OpeningMessageRequest):

    if request.condition not in (2, 3):
        raise HTTPException(
            status_code=400,
            detail="Opening needs elicitation is only available for conditions 2 and 3."
        )

    speaker = "Alex" if request.condition == 2 else "Jordan"

    system_prompt = SUPPORTIVE_PERSONA + """
    Your task is to briefly reflect back what the individual has shared
    so far about their healthcare experiences, priorities, needs,
    concerns, or questions.

    - Capture what matters to them in their own words when possible.
    - Preserve the personal context they have shared.
    - Do not turn their experiences into clinical trial questions.
    - Do not interpret, evaluate, or add assumptions.
    - Keep your response to 1-2 short sentences.
    - Do not ask questions or provide factual information.
    """

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
    
    system_prompt = INFORMATIONAL_PERSONA + f"""
    Your task is to identify three educational topics that would be
    most relevant to the individual's experiences, circumstances,
    priorities, needs, concerns, or questions.

    AVAILABLE TOPICS:
    {topic_options}

    For each selected topic:
    - Consider what the individual has shared about themselves,
    even if they have not explicitly asked about clinical trials.
    - Identify how their experiences or priorities relate to the topic.
    - Use only the topic IDs from AVAILABLE TOPICS.
    - Provide a brief explanation of why the topic is relevant.
    - Write a short, topic-specific user_excerpt capturing the
    relevant experience, priority, need, or concern.
    - Use the individual's own wording when possible.
    - Do not include unrelated information in the excerpt.
    - Do not invent or assume information they did not express.
    - If there is no clear connection, set user_excerpt to null.
    - Select three different topics.

    Also write a brief spoken_summary that:
    - Explicitly names all three topics using their exact names.
    - Explains how the topics relate to what the individual shared.
    - Uses professional, natural spoken language.
    - Directly addresses the individual.
    - Presents the topics as suggestions, not recommendations
    about what healthcare decisions they should make.
    - Does not invent personal needs or concerns.

    Return ONLY a valid JSON object with this structure:

    {{
    "topics": [
        {{
        "topic_id": "cost",
        "explanation": "Brief explanation",
        "user_excerpt": "Relevant individual perspective or null"
        }}
    ],
    "spoken_summary": "Brief professional suggestion"
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

    system_prompt = SUPPORTIVE_PERSONA + """
    Your task is to begin a conversation about the current topic
    by exploring the individual's own healthcare experiences.

    - Make the transition into the topic feel natural.
    - Build on something the individual has already shared when relevant.
    - Invite them to reflect on their experiences rather than
    expecting them to know what to ask about clinical trials.
    - Express curiosity without sounding like an interviewer.
    - Keep the conversation collaborative and nonjudgmental.
    - Do not assume they have had a particular experience.
    - Do not assume they want to join a clinical trial.
    - Do not provide factual information or medical advice.
    - Do not invent personal circumstances or concerns.
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

    # Provide a natural transition into each topic.
    topic_opening_messages = [
        "Alright, let's dive into the first topic!",
        "Okay, time to move on to our next topic.",
        "Alright, we're on the last topic!"
    ]

    jordan_reply = topic_opening_messages[topic_index] + " " + jordan_reply

    return {
        "speaker": "Jordan",
        "topic_id": topic["id"],
        "topic_name": topic_name,
        "user_excerpt": user_excerpt,
        "jordan_reply": jordan_reply,
    }


async def generate_topic_factual_content(topic: str):
    print("*** GENERATING TOPIC FACTUAL CONTENT:", topic)

    results = rag.retrieve(
        topic,
        k=5,
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

    response = await client.chat.completions.create(
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
    system_prompt = INFORMATIONAL_PERSONA + """
    Your task is to provide factual information about clinical trial
    participation that is relevant to the individual's healthcare
    experiences, circumstances, priorities, and preferences.

    - Use their personal context to determine which information
    would be most relevant to them.
    - Explain how the information relates to what they shared.
    - Address their broader needs and preferences rather than
    attempting to resolve a specific past healthcare experience.
    - Do not simply repeat or summarize their experiences.

    Use ONLY the provided FACTUAL CONTENT for factual claims.
    Treat the individual's responses as personal context,
    not as medical evidence.

    - Do not invent personal circumstances or information needs.
    - Do not provide personalized medical advice.
    - Do not recommend specific trials or assess eligibility.
    - Do not assume the individual wants to join a clinical trial.
    - Do not overstate potential benefits or minimize potential risks.
    - If the factual content does not address a relevant concern,
    acknowledge that limitation.

    Keep your response concise and conversational.
    Use accessible language while maintaining a professional tone.
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
    alex_reply = await generate_study_alex_response(
        topic_context=topic_context,
        factual_content=factual_content,
    )

    return {
        "speaker": "Alex",
        "topic_id": topic["id"],
        "topic_name": topic["name"],
        "alex_reply": alex_reply,
    }


@router.post("/alex-topic-response")
async def alex_topic_response(request: TopicOpeningRequest):
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

    # Provide topic-specific navigation instructions.
    topic_closing_messages = [
        "Alright, that's some of the information I thought would be most relevant "
        "to you based on what you shared. You're welcome to continue asking me questions about this topic, or "
        "whenever you are ready to move on, click the Next Topic button.",

        "Okay, those are the particular points I wanted to highlight given what you "
        "shared. Again, you can ask me anything else you'd like to know about this topic, "
        ", or you can move on to the next topic "
        "whenever you're ready.",

        "Alright, those are the key points I wanted to share with you "
        "on this topic. You're welcome to ask any additional questions about it "
        "before we move on to concluding our conversation."
    ]

    alex_reply += " " + topic_closing_messages[
        session["current_topic_index"]
    ]

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

        handoff_messages = [
            "Got it, thanks for talking through all that with me! "
            "Let's see what info Doctor Alex has about it.",

            "Alright, good to know! Now, let's hand it over to Doctor Alex again "
            "to see what info might be helpful for you to know.",

            "Okay, thanks for sharing that with me! "
            "Let's hand it over to Doctor Alex one last time "
            "to get some useful info."
        ]

        return {
            "speaker": "Jordan",
            "topic_id": topic["id"],
            "jordan_reply": handoff_messages[topic_index],
            "elicitation_complete": True,
            "user_profile": profile,
        }

    system_prompt = SUPPORTIVE_PERSONA + """
    Your task is to build on what the individual has shared
    and explore what matters to them about the current topic.

    - Briefly acknowledge what they shared in a natural,
    conversational way.
    - Show interest in their perspective without evaluating it.
    - Ask a follow-up question that naturally builds on what
    they just shared.
    - Explore what matters to them, rather than trying to
    identify or solve a problem for them.
    - Use the individual's specific experiences as a starting
    point for exploring their broader needs and preferences
    related to the current topic.
    - Focus on what matters to them more generally, rather than
    investigating the details of a particular past experience.
    - Do not ask about resolving or managing a specific past
    situation.
    - Do not force a connection to clinical trial participation.
    - Keep your response brief and conversational.
    - Do not assume how they feel or what they want.
    - Do not provide factual information or medical advice.
    - Do not invent personal circumstances or concerns.
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

async def generate_study_alex_answer(request: ConversationAlexRequest):
    print("*** IN CONVERSATION ALEX")

    question = request.message
    earlier_memory = request.earlier_memory
    prior_topic_summaries = request.prior_topic_summaries

    history: list[ChatCompletionMessageParam] = []

    for message in request.history:
        if message.from_ == "user":
            history.append({
                "role": "user",
                "content": message.text,
            })
        else:
            history.append({
                "role": "assistant",
                "content": f"{message.from_}: {message.text}",
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

    results = rag.retrieve(
        preprocess.search_query,
        k=5,
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

    system_prompt = INFORMATIONAL_PERSONA + f"""

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

    If there is no single general answer:
    - briefly explain why it varies,
    - provide any useful information supported by the context,
    - and explain what may depend on the specific trial.

    Use professional, precise, and accessible spoken language.
    Explain medical terms in plain language.
    Maintain a composed and informative tone.
    Avoid overly casual expressions and unnecessary emotional validation.
    Do not overstate potential benefits or minimize potential risks.

    {entry_instruction}

    The transition should make your reason for speaking clear without sounding
    formulaic, repetitive, or like a separate announcement.

    {INTERACTION_BOUNDARY}

    Write one conversational paragraph under 75 words.
    Do not use headings, lists, citations, source names, or line breaks.
    Do not use labels such as 'Alex:' or 'Jordan:'
    """

    chat_messages: list[ChatCompletionMessageParam] = [
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
        response = await client.beta.chat.completions.parse(
            model=MODEL,
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
            "answer": clean_character_reply(
                clean_alex_answer(parsed.answer)
            ),
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

@router.post("/alex-topic-turn")
async def alex_topic_turn(request: AlexTopicTurnRequest):

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
        history=request.history,
        earlier_memory=f"Current discussion topic: {topic['name']}",
        prior_topic_summaries=participant_context,
        condition=session.get("condition", 3),
    )

    result = await generate_study_alex_answer(alex_request)

    return {
        "speaker": "Alex",
        "topic_id": topic["id"],
        "topic_name": topic["name"],
        "alex_reply": result["answer"],
    }

@router.post("/jordan-closing")
async def jordan_closing(request: StudyClosingRequest):

    session = study_sessions.get(request.participant_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail="No active study conversation found."
        )

    system_prompt = SUPPORTIVE_PERSONA + """
    Your task is to conclude a conversation about clinical trial
    participation by helping the individual prepare for potential
    future conversations with people they trust.

    Generate three things:

    1. CLOSING REFLECTION
    - Briefly acknowledge the perspectives and experiences
      the individual shared throughout the conversation.
    - Reflect what matters to them personally.
    - Keep it warm, casual, and conversational.
    - Use no more than 2-3 short sentences.
    - Do not evaluate their decisions or assume how they feel.
    - Do not introduce new information or ask a follow-up question.

    2. DISCUSSION QUESTIONS
    - Generate 3 personalized questions that the individual
      could discuss with family, friends, or healthcare providers.
    - Ground each question in the individual's expressed needs,
      concerns, experiences, or priorities.
    - Help them explore what matters to them and what they
      may want to understand or consider further.
    - Write the questions from the individual's perspective,
      using first-person language where appropriate.
    - Keep each question concise and easy to understand.
    - Keep each question under 20 words.
    - Do not assume they want to participate in a clinical trial.
    - Do not provide medical advice or recommend a decision.
    - Do not invent personal circumstances.

    3. SPOKEN OVERVIEW
    - Briefly introduce the three discussion questions you generated.
    - Explain how they connect to what the individual shared.
    - Highlight the main ideas behind the questions without
      reading each question word for word.
    - Speak directly to the individual in a warm, casual,
      peer-like manner.
    - Use 2-3 short sentences, no more than 50 words total.
    - Do not ask a follow-up question or encourage trial participation.
    - This overview will be spoken after the individual chooses
      to view their questions.

    The reflection should not mention or introduce the questions.
    These will be introduced separately.
    """ + INTERACTION_BOUNDARY

    conversation_history = [
        {
            "speaker": message.from_,
            "text": message.text
        }
        for message in request.history
        if message.from_.lower() != "system"
    ]

    completion = await client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": json.dumps(conversation_history)
            }
        ],
        temperature=0.5,
        response_format=JordanClosingResponse,
    )

    result = completion.choices[0].message.parsed

    if result is None:
        raise HTTPException(
            status_code=500,
            detail="Could not generate closing reflection."
        )

    # Preserve the generated outputs in the study session.
    session["closing_reflection"] = result.reflection
    session["discussion_questions"] = result.discussion_questions
    session["jordan_material_overview"] = result.spoken_overview

    save_study_sessions()

    return {
        "speaker": "Jordan",
        "reflection": result.reflection,
        "discussion_questions": result.discussion_questions,
        "spoken_overview": result.spoken_overview,
    }

@router.post("/alex-closing")
async def alex_closing(request: StudyClosingRequest):

    session = study_sessions.get(request.participant_id)

    if not session:
        raise HTTPException(
            status_code=404,
            detail="No active study conversation found."
        )

    system_prompt = INFORMATIONAL_PERSONA + """
    Your task is to identify additional educational resources
    that may be relevant to the individual based on the entire
    conversation.

    AVAILABLE RESOURCES:

    - featured-trials: Examples of clinical trials.
    - personal-stories: Personal stories about clinical trial participation.
    - sample-consent: An example clinical trial informed consent document.
    - questions-to-ask: General questions to ask about clinical trials.
    - details-checklist: A checklist of clinical trial details to consider.
    - finding-guide: Guidance on how to find clinical trials.
    - nci-specialist: Information about contacting an NCI information specialist.
    - trial-search: A tool for independently exploring clinical trials.

    Select exactly three different resources.

    - Base your selections on the individual's expressed
      questions, concerns, experiences, priorities, and needs.
    - Prioritize resources that help them continue exploring
      information relevant to their circumstances.
    - Consider the entire conversation, not just the final topic.
    - Provide one brief sentence explaining why each resource
      may be relevant to the individual.
    - Do not invent personal circumstances or information needs.
    - Do not assume the individual wants to participate in a trial.
    - Do not recommend specific clinical trials.
    - Only select from the available resource IDs.

    SPOKEN OVERVIEW:
    - Briefly introduce all three selected resources.
    - Refer to each resource using natural, descriptive language.
    - NEVER say the internal resource IDs aloud.
    - For example, say "a checklist of important clinical trial
      details" instead of "details-checklist".
    - Explain why each resource may be useful based on what
      the individual shared during the conversation.
    - Use professional, natural spoken language.
    - Make the overview sound conversational rather than
      like a numbered list of recommendations.
    - Use 2-3 short sentences, no more than 60 words total.
    - Do not read URLs aloud.
    - Do not introduce information beyond the selected resources.
    - This overview will be spoken after the individual chooses
      to view their resources.
    """

    conversation_history = [
        {
            "speaker": message.from_,
            "text": message.text
        }
        for message in request.history
        if message.from_.lower() != "system"
    ]

    completion = await client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": json.dumps(conversation_history)
            }
        ],
        temperature=0,
        response_format=AlexClosingResponse,
    )

    result = completion.choices[0].message.parsed

    if result is None:
        raise HTTPException(
            status_code=500,
            detail="Could not generate resource recommendations."
        )

    recommendations = [
        resource.model_dump()
        for resource in result.recommended_resources
    ]

    resource_ids = [
        resource["resource_id"]
        for resource in recommendations
    ]

    if len(set(resource_ids)) != 3:
        raise HTTPException(
            status_code=502,
            detail="Duplicate resource recommendations."
        )

    # Preserve Alex's personalized recommendations.
    session["recommended_resources"] = recommendations
    session["alex_material_overview"] = result.spoken_overview

    save_study_sessions()

    return {
        "speaker": "Alex",
        "recommended_resources": recommendations,
        "spoken_overview": result.spoken_overview,
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
