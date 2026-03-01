from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Literal
import uvicorn
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
import base64


app = FastAPI()
load_dotenv()

# Endpoints allowed to access this server
origins = ["https://main.d1qbymvh7dh0n4.amplifyapp.com", "http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or specify your Amplify URL e.g. ["https://yourapp.amplifyapp.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base Models for structuring responses from LLM
class ChatRequest(BaseModel):
    thread_id: str | None = None
    message: str

class TTSRequest(BaseModel):
    text: str

class PrecheckResponse(BaseModel):
    user_message: str | None = None
    gesture: Literal["ready", "thinking", "thumbsup", "shrug"]
    label: Literal["ready", "vague", "good", "thoughtful"]
    tip: str | None = None                 # tooltip shown while typing
    suggestions: list[str] | None = None   # AI-generated suggestion chips

# Set up LiteLLM client
# Regular LiteLLM client for conversational responses (async)
RASHI_LITELLM_KEY = os.getenv('RASHI_LITELLM_KEY')
client_chat = AsyncOpenAI(
    api_key= RASHI_LITELLM_KEY,
    base_url="https://api.ai.it.ufl.edu" # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
)

@app.post("/simple-chat")
async def simple_chat(request: ChatRequest):
    print("IN SIMPLE CHAT", request)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "You are a friendly virtual assistant helping users with clinical trial questions. Respond in maximum 300 characters."},
        {"role": "user", "content": request.message}
    ]
    try:
        # call the async LLM client
        response = await client_chat.chat.completions.create(model='gpt-4o-mini', messages=messages, temperature=0)
        # response might be a string or a dict; check your client
        print("RESPONSE IS", response.choices[0].message.content)
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

@app.post("/precheck", response_model=PrecheckResponse)
async def precheck(request: ChatRequest):

    system_prompt = """
        You are evaluating whether a user's message is ready to be processed by a clinical trials assistant. You act like a friendly, conversational helper named Milo.

        Assess the user's message and respond ONLY in valid JSON matching this exact structure:
        {
        "user_message": null,
        "label": one of: "ready", "vague", "good", "thoughtful",
        "gesture": one of: "ready", "thinking", "thumbsup", "shrug",
        "tip": a short friendly nudge shown as a tooltip while the user types using warm, kind words, or null,
        "suggestions": a list of 2-3 short suggested questions asked from the user's perspective, or null. suggestions must be phrased as questions the USER would ask Dr. Alex, not questions Dr. Alex would ask the user. Suggestions should be based on what the user is most likely trying to ask, completing or expanding on their partial thought. If the message is incomplete or cut off, suggest the most common complete versions of that question related to clinical trials. For example, if the user types "what is the cl", suggest questions like "What is the clinical trial process?", "What is the clinical trial eligibility criteria?", "What is the clinical trial consent process?" For example: "What are common side effects?" not "What side effects are you concerned about?"
        }

        Use this guide to pick the right values:

        - Message is empty or just a few characters → label: "ready", gesture: "ready", tip: null, suggestions: null
        - Message is small talk or a greeting or not relevant to clinical trials → label: "ready", gesture: "ready", tip: tell the user Dr. Alex can only respond to questions about clinical trials, suggestions: 2-3 questions to ask about clinical trials
        - Message is incomplete → label: "vague", gesture: "shrug", tip: tell the user the message looks incomplete and ask if they meant any of your suggestions, suggestions: 2-3 complete questions they might mean that is about clinical trials
        - Message is too short or a fragment → label: "vague", gesture: "shrug", tip: tell the user the question is too short or could use some or detail and to consider your suggestions, suggestions: 2-3 complete example questions they might mean
        - Message is vague or unfocused → label: "vague", gesture: "shrug", tip: let the user know the question may be too vague to get a good answer and to consider your suggestions, suggestions: 2-3 example questions of what they might mean
        - Message is a clear, decent question → label: "good", gesture: "thumbsup", tip: tell the user the question is good and why it's important to ask, suggestions: 1-2 related follow-up questions
        - Message is detailed and well-formed → label: "good", gesture: "thumbsup", tip: tell the user the question is good and why it's important to ask, suggestions: 1-2 related follow-up questions
        - Message touches on sensitive topics (risk, safety, fear, harm) → label: "thoughtful", gesture: "thinking", tip: let the user know it's good they're considering this and why it's good to consider, suggestions: 1-2 related gentle follow-up example questions

        Suggestions should be short (under 7 words), phrased as questions, and specific to clinical trials.
        Respond ONLY with valid JSON. No preamble, no explanation, no markdown.
    """

    response = await client_chat.beta.chat.completions.parse(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message}
        ],
        response_format=PrecheckResponse  # your Pydantic model
    )

    result = response.choices[0].message.parsed

    if result is None:
        raise HTTPException(status_code=500, detail="Failed to parse response")

    return PrecheckResponse(
        user_message=request.message,
        tip=result.tip,
        suggestions=result.suggestions,
        label=result.label,
        gesture=result.gesture
    )

# TTS endpoint ...
@app.post("/tts")
async def tts(request: TTSRequest):
    # Step 1: get audio
    response = await client_chat.audio.speech.create(
        model="kokoro",
        voice="af_heart",
        input=request.text,
        speed=1.0
    )
    audio_bytes = response.content

    # Step 2: get word timestamps from Whisper
    transcript = await client_chat.audio.transcriptions.create(
        model="whisper-large-v3",
        file=("audio.mp3", audio_bytes, "audio/mpeg"),
        response_format="verbose_json",
        timestamp_granularities=["word"]
    )

    # Step 3: return both
    return {
        "audio": base64.b64encode(audio_bytes).decode("utf-8"),
        "timestamps": transcript.words
    }

@app.get("/")
async def root():
    return {"message": "Welcome to Rashi's FastAPI server!"}

if __name__ == "__main__":
    # Run the FastAPI application on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)