from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os


load_dotenv()

router = APIRouter(
    prefix="/jordan",
    tags=["jordan"],
)

BASE_URL = "https://api.ai.it.ufl.edu/v1"
RASHI_LITELLM_KEY = os.getenv("RASHI_LITELLM_KEY")
UF_LOCAL_MODEL = "gpt-oss-120b"

client_chat = AsyncOpenAI(
    api_key=RASHI_LITELLM_KEY,
    base_url=BASE_URL,
    timeout=45.0
)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class JordanTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class AdaptationProfile(BaseModel):
    user_information: list[str] = Field(default_factory=list)
 
class JordanAdaptationRequest(BaseModel):
    user_feedback: str
    jordan_invitation: str | None = None
    alex_answer: str | None = None
    user_question: str | None = None

    jordan_history: list[JordanTurn] = Field(default_factory=list)

    current_profile: AdaptationProfile = Field(
        default_factory=AdaptationProfile
    )

class JordanAdaptationResponse(BaseModel):
    reply: str
    updated_profile: AdaptationProfile
    profile_changed: bool
    needs_followup: bool = False
    # True only when the user indicates they are done talking with Jordan.
    side_conversation_done: bool = False

class JordanInvitationRequest(BaseModel):
    user_question: str
    alex_answer: str
    alex_answer_scope: str

    current_profile: AdaptationProfile = Field(
        default_factory=AdaptationProfile
    )

class JordanInvitationResponse(BaseModel):
    potential_adaptation_signal: bool = False
    signal: str | None = None
    question: str

# --------------------------------------------------------------------------
# Endpoint
# --------------------------------------------------------------------------

@router.post(
    "/adaptation",
    response_model=JordanAdaptationResponse,
)
async def jordan_adaptation(
    request: JordanAdaptationRequest,
):
    print("*** JORDAN ADAPTATION ***")
    print("*** USER FEEDBACK:", request.user_feedback)
    print("*** CURRENT PROFILE:", request.current_profile.model_dump())

    system_prompt = """
    You are Jordan, a friendly virtual companion who helps Alex understand how
    clinical trial information should be communicated or framed to better fit
    the user. The user is having a brief side conversation with you after interacting
    with Alex. Your voice is warm, conversational, and empathetic.

    Your goal is to help the user share information about themselves so Alex can
    use that information to better tailor the conversation to them.
    Engage and encourage the user to talk more about the current topic, but do not
    try to get the user to start a new topic. Use the 'JORDAN'S INVITATION TO THE USER'
    to see if there is already a topic the user is referring to.

    If the user reveals useful information about themselves that would be useful to add to
    their 'CURRENT USER INFORMATION' for Alex to use to adapt the conversation to, add the information
    to their profile and let them know you did that. Do not ask to add anything that is already part of the user profile. 
    When adding information about the user, the information should:
    - describe the user, not instruct Alex
    - represent something the user expressed or confirmed
    - be relevant to adapting communication or framing
    - be short and specific
    - be concrete enough that Alex can act on it without guessing further —
    describe what specifically works or doesn't work for the user, not just
    a general topic, feeling, or constraint
    - avoid merely recording the topic or information the user wants
    - avoid repeating information already represented elsewhere

    Set needs_followup=true only when clarification is needed for you to figure out what to add about the user.

    Set side_conversation_done=true only when the user indicates they are done
    talking with you, or if you have gotten the information you need.

    Keep your reply brief and conversational.

    Return the complete updated profile.
    """

    user_content = f"""
    USER'S FEEDBACK TO JORDAN:
    {request.user_feedback}

    JORDAN'S INVITATION TO THE USER:
    {request.jordan_invitation or "Not provided"}

    QUESTION THE USER HAD ASKED ALEX:
    {request.user_question or "Not provided"}

    ALEX'S ANSWER:
    {request.alex_answer or "Not provided"}

    RECENT SIDE CONVERSATION WITH JORDAN:
    {[turn.model_dump() for turn in request.jordan_history[-6:]]}

    CURRENT USER INFORMATION:
    {request.current_profile.model_dump_json()}
    """.strip()

    try:
        print("*** ABOUT TO CALL JORDAN LLM ***")
        response = await client_chat.beta.chat.completions.parse(
            model=UF_LOCAL_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            temperature=0,
            response_format=JordanAdaptationResponse,
        )
        print("*** JORDAN LLM RETURNED ***")

        parsed = response.choices[0].message.parsed

        if parsed is None:
            print(
                "*** JORDAN ADAPTATION PARSE FAILED:",
                response.choices[0].message.content,
            )

            raise HTTPException(
                status_code=500,
                detail="Failed to parse Jordan adaptation response",
            )

        # Compute this ourselves instead of trusting the model.
        parsed.profile_changed = (
            parsed.updated_profile.model_dump()
            != request.current_profile.model_dump()
        )

        print("*** JORDAN REPLY:", parsed.reply)
        print(
            "*** UPDATED PROFILE:",
            parsed.updated_profile.model_dump(),
        )
        print("*** PROFILE CHANGED:", parsed.profile_changed)

        return parsed

    except HTTPException:
        raise

    except Exception as error:
        print(
            "*** JORDAN ADAPTATION ERROR:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail="Could not process Jordan feedback",
        )

@router.post(
    "/invitation",
    response_model=JordanInvitationResponse,
)
async def jordan_invitation(
    request: JordanInvitationRequest,
):
    system_prompt = """
    You are Jordan, a friendly virtual companion who helps Alex better understand
    the user so Alex can communicate clinical trial information in a way that fits
    them. Alex has just answered the user's question.

    Your job is as follows:
    Determine whether the user explicitly revealed information about them
    that could be used to build a user profile, or hinted at information that could
    be useful to know about them. Make sure this information does not already
    exist in CONFIRMED USER INFORMATION.

    If the user hinted at a possible piece of useful information about themself:
    - set potential_adaptation_signal=true
    - set signal to the specific thing the user revealed or hinted at, under 15 words
    - mention the specific thing you noticed
    - Ask the user if they would like to share more about it in a way that hints at the underlying generic concern/barrier
    regarding deciding to consider clinical trial participation; the only goal is to give the user an opportunity to share more about themselves.

    If the user did not hint at a possible piece of useful information about themself:
    - set potential_adaptation_signal=false
    - set signal=null
    - briefly invite the user to share anything about themselves that might be useful for Alex to know ,
    thoughts on Alex's response, or how they like information explained that might help Alex communicate better with them

    Sound like a person briefly joining the conversation, not a survey or
    decision aid. Keep your response to a maximum of 15 words.
    """

    user_content = f"""
    USER'S QUESTION:
    {request.user_question}

    ALEX'S ANSWER:
    {request.alex_answer}

    ALEX'S ANSWER SCOPE:
    {request.alex_answer_scope}

    CONFIRMED USER INFORMATION:
    {request.current_profile.model_dump_json()}
    """.strip()

    try:
        response = await client_chat.beta.chat.completions.parse(
            model=UF_LOCAL_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            temperature=0,
            response_format=JordanInvitationResponse,
        )

        parsed = response.choices[0].message.parsed

        if parsed is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to parse Jordan invitation",
            )

        print("*** JORDAN INVITATION:", parsed)

        return parsed

    except HTTPException:
        raise

    except Exception as error:
        print("*** JORDAN INVITATION ERROR:", repr(error))

        raise HTTPException(
            status_code=500,
            detail="Could not generate Jordan invitation",
        )