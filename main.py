from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
import uvicorn
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from dotenv import load_dotenv
import os

app = FastAPI()
load_dotenv()

# Base Models for structuring responses from LLM
class ChatRequest(BaseModel):
    thread_id: str | None = None
    message: str

class PrecheckResponse(BaseModel):
    user_message: str | None = None
    tip: str | None = None                 # tooltip shown while typing
    suggestions: list[str] | None = None   # AI-generated suggestion chips
    label: Literal["ready", "thinking", "vague", "good", "great", "thoughtful"]
    color: Literal["#94a3b8", "#f59e0b", "#10b981", "#6366f1", "#3b82f6"]
    emoji: Literal["smile", "think", "frown"]

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
    "user_message": empty string (will populate later),
    "label": one of: "ready", "thinking", "vague", "good", "great", "thoughtful",
    "color": one of: "#94a3b8", "#f59e0b", "#10b981", "#6366f1", "#3b82f6",
    "emoji": one of: "smile", "think", "frown",
    "tip": a short friendly nudge shown as a tooltip while the user types, or null,
    "suggestions": a list of 2-3 short suggested questions the user could ask instead or in addition, or null
    }

    Use this guide to pick the right values:

    - Message is empty or just a few characters → label: "ready", color: "#94a3b8", emoji: "smile", tip: null, suggestions: null
    - Message is too short or a fragment → label: "thinking", color: "#f59e0b", emoji: "think", tip: "Try adding more detail!", suggestions: 2-3 relevant example questions
    - Message is vague or unfocused → label: "vague", color: "#f59e0b", emoji: "think", tip: "Could you narrow it down a bit?", suggestions: 2-3 more specific versions of what they might mean
    - Message is a clear, decent question → label: "good", color: "#10b981", emoji: "smile", tip: "Great question!", suggestions: 1-2 related follow-up questions they might also want to ask
    - Message is detailed and well-formed → label: "great", color: "#6366f1", emoji: "smile", tip: "Excellent — very thoughtful!", suggestions: 1-2 related follow-up questions
    - Message touches on sensitive topics (risk, safety, fear, harm) → label: "thoughtful", color: "#3b82f6", emoji: "smile", tip: "It's great that you're thinking carefully about this.", suggestions: 1-2 related gentle follow-ups

    Suggestions should be short (under 10 words), phrased as questions, and specific to clinical trials.

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
        label=result.label,
        color=result.color,
        emoji=result.emoji,
        tip=result.tip,
        suggestions=result.suggestions
    )

@app.get("/")
async def root():
    return {"message": "Welcome to Rashi's FastAPI server!"}

if __name__ == "__main__":
    # Run the FastAPI application on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)