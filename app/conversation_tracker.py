from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from dotenv import load_dotenv
from typing import Literal
import os

load_dotenv()

router = APIRouter(
    prefix="/user",
    tags=["user"],
)

BASE_URL = "https://api.ai.it.ufl.edu"
RASHI_LITELLM_KEY = os.getenv("RASHI_LITELLM_KEY")
UF_LOCAL_MODEL = "gpt-oss-20b"

client_chat = AsyncOpenAI(
    api_key=RASHI_LITELLM_KEY,
    base_url=BASE_URL,
    timeout=30.0,
    max_retries=0,
)


# ---------- Request ----------

class ConversationMessage(BaseModel):
    sender: str = Field(alias="from")
    text: str


class UserNeedRequest(BaseModel):
    user_message: str
    alex_response: str
    conversation_history: list[ConversationMessage] = Field(
        default_factory=list
    )


# ---------- Structured LLM output ----------

class ClassifiedNeed(BaseModel):
    text: str
    type: Literal[
        "general",
        "personal_application",
        "personally_situated",
    ]
    reasoning: str

class UserNeedClassification(BaseModel):
    needs: list[ClassifiedNeed]

class JordanCaptureResponse(BaseModel):
    response: str
    question: str

# ---------- Classifier logic function ----------

async def classify_user_needs(
    user_message: str,
    conversation_history: list[dict],
):
    recent_history = conversation_history[-4:]

    history_text = "\n".join(
        f'{message.get("from")}: {message.get("text")}'
        for message in recent_history
        if message.get("text")
    )

    system_prompt = """
    You classify information needs expressed by users who are learning about clinical trial participation.
    Do not answer the user.
    Do not give medical advice.
    Do not generate conversational responses.
    Your only task is to identify the distinct information need(s) in the user's latest message and classify each need as:
    - general
    - personal_application
    - personally_situated
    A single user message may contain more than one distinct information need. Separate them when they concern different issues or would require different answers.
    Use recent conversation context only when necessary to understand what the user is referring to.

    GENERAL
    Classify a need as GENERAL when the user is primarily seeking factual or procedural information about clinical trials that can be answered using general trusted-source information.
    The question may use words such as "I," "me," or "my" and still be GENERAL if no personal concern, preference, circumstance, or individualized judgment is central to the need.

    GENERAL Examples:
    - "What does randomization mean in a clinical trial?"
    - "Who usually pays for treatment and tests in a clinical trial?"
    - "Who will have access to my medical information?"
    - "Will participants know what phase of a trial they are in?"
    - "What happens if I decide I no longer want to participate?"

    PERSONAL_APPLICATION
    Classify a need as PERSONAL_APPLICATION when the user is connecting a clinical-trial issue to their own anticipated experience, concern, preference, priority, or decision, but the issue can still be meaningfully discussed using general information.
    The user is no longer only asking "How do clinical trials work?" They are beginning to ask "What might this mean for me?" without yet requiring individualized clinical or trial-specific judgment.

    Signals may include:
    - expressing a concern or worry
    - expressing a preference or priority
    - considering how participation may affect them
    - evaluating whether an aspect of participation feels acceptable
    - relating general information to their own anticipated experience

    PERSONAL_APPLICATION Examples:
    - "Do I get treated like a patient or like an experiment?"
    - "I'm worried a clinical trial might take too much time. What is the time commitment usually like?"
    - "Could participating in a trial be worse for me if I ended up in the control group?"
    - "It's important to me that my regular doctor stays involved. Would that usually happen?"
    - "If I want my participation to stay private, would my doctor's office still know?"

    PERSONALLY_SITUATED
    Classify a need as PERSONALLY_SITUATED when adequately resolving the user's actual question requires information, facts, or judgment about their specific circumstances that cannot be determined from general clinical-trial information alone.

    This includes questions that depend on:
    - the user's medical history, diagnoses, medications, age, or health status
    - a specific clinical trial or treatment
    - the user's actual healthcare provider or clinic
    - the user's location
    - the user's insurance or financial situation
    - another concrete personal circumstance
    - an individualized prediction, recommendation, or eligibility judgment

    The user does not need to provide all of this information explicitly. If the question itself asks for an individualized judgment that would require such information, classify it as PERSONALLY_SITUATED.

    PERSONALLY_SITUATED Examples:
    - "I am a cancer survivor and don't have a thyroid. Can I participate?"
    - "I take several medications. Would that prevent me from qualifying?"
    - "Should my age keep me from joining a trial?"
    - "What if my insurance doesn't cover transportation?"
    - "Is there a clinical trial at my doctor's office?"
    - "There's a trial two hours from my house. Would I have to travel there for every visit?"
    - "Would taking part in a clinical trial help me?"

    DECISION PRINCIPLE
    For each information need, determine:
    1. Is the user simply seeking general factual or procedural information?
    -> GENERAL
    2. Is the user connecting the issue to their own concern, preference, priority, anticipated experience, or decision, while general information is still sufficient to discuss the issue?
    -> PERSONAL_APPLICATION
    3. Does answering the user's actual question require individualized facts, trial-specific information, provider-specific information, or clinical judgment that general information cannot provide?
    -> PERSONALLY_SITUATED

    Classify according to the highest level of personalization actually required by the need.
    Do not classify based only on first-person pronouns.
    Do not infer personal circumstances the user has not stated.
    If the message contains no substantive clinical-trial information need, return an empty needs list.
    Keep each reasoning explanation concise.
    """

    user_prompt = f"""
    RECENT CONVERSATION:
    {history_text or "No previous conversation."}

    LATEST USER MESSAGE:
    {user_message}
    """

    try:
        print("About to call model")
        response = await client_chat.beta.chat.completions.parse(
            model=UF_LOCAL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format=UserNeedClassification,
        )

        parsed = response.choices[0].message.parsed
        print("Got response")

        if parsed is None:
            raise RuntimeError("No parsed classification returned")

        return parsed

    except Exception as error:
        print("*** USER NEED CLASSIFICATION ERROR:", repr(error))

        raise HTTPException(
            status_code=500,
            detail="Could not classify user message",
        )

# ---------- Decider function ----------

def decide_jordan_action(classification: UserNeedClassification):
    types = [need.type for need in classification.needs]

    if "personally_situated" in types:
        return "capture"

    if "personal_application" in types:
        return "elicit"

    return "none"

async def generate_jordan_elicit(
    user_message: str,
    alex_response: str,
    conversation_history: list[dict],
):
    recent_history = conversation_history[-4:]

    history_text = "\n".join(
        f'{message.get("from")}: {message.get("text")}'
        for message in recent_history
        if message.get("text")
    )  

    system_prompt = """
    You are Jordan, a conversational support character helping users connect
    Alex's general clinical-trial information to what may matter for them personally.
    You are NOT talking about a specific clinical trial or participation, but clinical trial participation generally.

    Alex has already responded to the user's question.

    Your job is NOT to repeat Alex's answer, provide additional factual information,
    or give medical advice.

    Your job is to:
    1. Briefly refer to what Alex just said by connecting one specific piece of information Alex just provided that
    is relevant to the user's concern back to what the user raised.
    2. Ask at most ONE short, neutral, open-ended question based on the user's original message that invites the user to share to how the information is relevant to their situation.

    The response should make it obvious that you understood BOTH:
    - what the user was concerned about, and
    - what Alex just explained.

    Do not give a generic reflection.
    Instead, anchor the follow-up in the content Alex provided.

    - Do not imply that the user is currently considering a specific trial, planning
    to join one, or about to speak with a provider or research team.
    - Do not tell the user that they need to contact or confirm something with a
    provider or study team.
    - Do not ask the user to share any personal or medical information.

    Keep your response to 75 words or less.
    """

    user_prompt = f"""
    RECENT CONVERSATION:
    {history_text or "No previous conversation."}

    LATEST USER MESSAGE:
    {user_message}

    ALEX'S RESPONSE:
    {alex_response}
    """

    try:
        response = await client_chat.chat.completions.create(
            model=UF_LOCAL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )

        return response.choices[0].message.content

    except Exception as error:
        print("*** JORDAN ELICIT ERROR:", repr(error))

        raise HTTPException(
            status_code=500,
            detail="Could not generate Jordan response",
        )

async def generate_jordan_capture(
    user_message: str,
    alex_response: str,
    conversation_history: list[dict],
    speaker: str = "jordan",
    need_type: str = "personally_situated",
):
    recent_history = conversation_history[-4:]

    history_text = "\n".join(
        f'{message.get("from")}: {message.get("text")}'
        for message in recent_history
        if message.get("text")
    )

    alex_context = (
        "Alex has already provided general clinical-trial information."
        if alex_response
        else "Alex has not provided a factual response on this turn."
    )

    if need_type == "personally_situated":
        capture_guidance = """
        The user's question requires individualized information.

        Briefly acknowledge that the question has been saved.
        Explain that it may be useful to bring up if the user ever
        speaks with a healthcare provider or looks into a specific
        clinical trial.

        Do not attempt to answer the individualized question.
        """
    else:
        capture_guidance = """
        The user expressed a personally relevant concern or question.

        Briefly acknowledge that it has been saved on their list.
        Explain that it may be useful to revisit when talking with
        a healthcare provider or looking into a specific clinical trial.

        Do not imply that individualized medical advice is required.
        """

    if speaker == "alex":
        speaker_instructions = f"""
        You are Alex, the only conversational character.

        {alex_context}

        Briefly acknowledge the user's question or concern and
        explain that you will keep it on their list.

        Speak in first person. Do not refer to Jordan or
        another character.
        """
    else:
        speaker_instructions = f"""
        You are Jordan, a conversational support character.

        {alex_context}

        If Alex provided a response on this turn, you may briefly
        refer to what Alex shared.

        Briefly acknowledge the user's question or concern and
        explain that you will keep it on their list.

        Do not provide additional factual or medical information.
        """

    system_prompt = f"""
    {speaker_instructions}

    You are NOT talking about a specific clinical trial or giving medical advice.

    {alex_context}

    The user's information need has been classified as: {need_type}
    {capture_guidance}

    Your task is to produce TWO things:

    1. RESPONSE
    A brief conversational response from the character described above that:
    - Say naturally that you will note it down / keep it on their list.
    - Do not imply that the user is currently considering a specific trial, planning
    to join one, or about to speak with a provider or research team.
    - Do not tell the user that they need to contact or confirm something with a
    provider or study team.
    - Do not ask the user to share any personal or medical information.
    - Do not state the entire question itself.
    - Do not introduce a new concern.
    - After acknowledging the saved question, do not invite the user
    to discuss it further in this conversation.
    - Do not say phrases like "let me know when you're ready to discuss
    it", "we can explore that together", or "feel free to ask me more".
    - Frame the saved question as something the user may find useful
    in a future conversation with a healthcare provider or when
    exploring a specific clinical trial.
    - Do not imply that the user must take these actions.
    - Keep the response to 35 words or fewer.

    Keep the response to 100 words or less.

    2. QUESTION
    Generate one question the user could naturally ask a healthcare provider
    or research team when discussing a specific clinical trial.
    - Preserve the user's actual concern without introducing new details.
    - Use simple, conversational language from the user's perspective.
    - Ask about trial-specific information, options, or support relevant
      to their concern, without assuming anything is available.
    - Phrase it as a direct question, not a task, reminder, or reflection.
    """

    user_prompt = f"""
    RECENT CONVERSATION:
    {history_text or "No previous conversation."}

    LATEST USER MESSAGE:
    {user_message}

    ALEX'S RESPONSE:
    {alex_response}
    """

    try:
        response = await client_chat.beta.chat.completions.parse(
            model=UF_LOCAL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format=JordanCaptureResponse,
        )

        parsed = response.choices[0].message.parsed

        if parsed is None:
            raise RuntimeError("No parsed Jordan capture response returned")

        return parsed

    except Exception as error:
        print("*** JORDAN CAPTURE ERROR:", repr(error))

        raise HTTPException(
            status_code=500,
            detail="Could not generate Jordan capture response",
        )

# ---------- Endpoint ----------

@router.post("/classify-user-need")
async def classify_user_need_endpoint(
    request: UserNeedRequest,
):
    print("Hit classify-user-need endpoint")

    history = [
        {
            "from": message.sender,
            "text": message.text,
        }
        for message in request.conversation_history
    ]

    classification = await classify_user_needs(
        user_message=request.user_message,
        conversation_history=history,
    )

    jordan_action = decide_jordan_action(classification)

    jordan_response = None
    question_to_save = None

    if jordan_action == "elicit":
        jordan_response = await generate_jordan_elicit(
            user_message=request.user_message,
            alex_response=request.alex_response,
            conversation_history=history,
        )

    elif jordan_action == "capture":
        capture_result = await generate_jordan_capture(
            user_message=request.user_message,
            alex_response=request.alex_response,
            conversation_history=history,
        )

        jordan_response = capture_result.response
        question_to_save = capture_result.question

    return {
        "classification": classification,
        "jordan_action": jordan_action,
        "jordan_response": jordan_response,
        "question_to_save": question_to_save,
    }