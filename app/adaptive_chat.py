# adaptive_chat.py
import json
import re
from pydantic import ValidationError
from typing import Any, Callable, Literal, Optional
from fastapi.responses import StreamingResponse
from fastapi import APIRouter
from pydantic import BaseModel, Field
from difflib import SequenceMatcher
import asyncio

# -------------------------------------------------------------------
# JORDAN MENTAL MODEL
# -------------------------------------------------------------------

ChangeType = Literal[
    "created",
    "expanded",
    "clarified",
    "revised",
]

class JordanMentalModelUpdate(BaseModel):
    mental_model: str = Field(
        description=(
            "The complete updated mental model representing the current "
            "overall understanding."
        )
    )

    highlighted_text: str = Field(
        description=(
            "Copy one exact continuous substring from mental_model. "
            "Do not generate or paraphrase it separately. Every word and "
            "punctuation mark must match mental_model exactly."
        )
    )

    change_type: ChangeType = Field(
        description=(
            "How the newest information affected the mental model."
        )
    )

    change_explanation: str = Field(
        description=(
            "A brief, conversational reflection explaining why this change "
            "matters when thinking about clinical trial participation."
        )
    )

class JordanUnresolvedUpdate(BaseModel):
    message: str = Field(
        description=(
            "Jordan's brief conversational response explaining what remains "
            "uncertain and why it may need trial-specific or outside information."
        )
    )
    knowledge_gaps: list[str] = Field(
        default_factory=list,
        description=(
            "The complete updated running list of unresolved questions. "
            "Each item must be one concise question."
        ),
    )

class JordanAfter(BaseModel):
    message: Optional[str] = None
    mental_model: str
    highlighted_text: Optional[str] = None
    change_type: Optional[ChangeType] = None
    knowledge_gaps: list[str] = Field(default_factory=list)

# -------------------------------------------------------------------
# SOURCE BASE MODELS
# -------------------------------------------------------------------

AnswerScope = Literal[
    "general_answer",
    "varies_by_trial",
    "personalized_decision",
    "insufficient_context",
]

class SourceExplanation(BaseModel):
    id: int
    relevance_explanation: str

class RagResponseModel(BaseModel):
    answer: str
    source_explanations: list[SourceExplanation] = Field(
        default_factory=list
    )
    confidence: str
    talking_points: list[str] = Field(default_factory=list)
    answer_scope: AnswerScope

# -------------------------------------------------------------------
# REQUEST / RESPONSE MODELS
# -------------------------------------------------------------------

class AdaptiveChatTurn(BaseModel):
    role: Literal["user", "alex", "jordan"]
    content: str


class AdaptiveChatRequest(BaseModel):
    message: str
    history: list[AdaptiveChatTurn] = Field(default_factory=list)
    mental_model: str | None = None
    knowledge_gaps: list[str] = Field(default_factory=list)

class RouteResult(BaseModel):
    route: Literal["fact_finding", "hypothesis_testing"]
    reason: str


class JordanFrame(BaseModel):
    message: str
    information_need: str

class AdaptiveChatResponse(BaseModel):
    route: Literal["fact_finding", "hypothesis_testing"]
    jordan_before: Optional[str] = None
    information_need: str
    search_query: str
    alex_answer: str
    jordan_after: Optional[str] = None
    mental_model: str

    highlighted_text: Optional[str] = None
    change_type: Optional[ChangeType] = None
    knowledge_gaps: list[str] = Field(default_factory=list)

    sources: list[Any] = Field(default_factory=list)
    talking_points: list[str] = Field(default_factory=list)
    answer_scope: AnswerScope

# -------------------------------------------------------------------
# RESPONSE MODELS
# -------------------------------------------------------------------

class AdaptiveRouteRequest(BaseModel):
    message: str
    history: list[AdaptiveChatTurn] = Field(default_factory=list)


class AdaptiveRouteResponse(BaseModel):
    route: Literal["fact_finding", "hypothesis_testing"]
    reason: str


class AdaptiveFrameRequest(BaseModel):
    message: str
    history: list[AdaptiveChatTurn] = Field(default_factory=list)


class AdaptiveFrameResponse(BaseModel):
    message: str
    information_need: str


class AdaptiveAlexRequest(BaseModel):
    original_message: str
    information_need: str
    history: list[AdaptiveChatTurn] = Field(default_factory=list)


class AdaptiveAlexResponse(BaseModel):
    search_query: str
    answer: str
    sources: list[Any] = Field(default_factory=list)
    confidence: str
    talking_points: list[str] = Field(default_factory=list)
    answer_scope: AnswerScope
    has_supported_information: bool


class AdaptiveJordanRequest(BaseModel):
    original_message: str
    alex_answer: str
    answer_scope: AnswerScope
    has_supported_information: bool
    history: list[AdaptiveChatTurn] = Field(default_factory=list)
    mental_model: str | None = None
    knowledge_gaps: list[str] = Field(default_factory=list)


class AdaptiveJordanResponse(BaseModel):
    message: Optional[str] = None
    mental_model: str
    highlighted_text: Optional[str] = None
    change_type: Optional[ChangeType] = None
    knowledge_gaps: list[str] = Field(default_factory=list)

# -------------------------------------------------------------------
# ROUTER FACTORY
#
# main.py will pass its existing RAG object and LLM client into here.
# -------------------------------------------------------------------

def extract_json_object(text: str) -> dict:
    """
    Extract the first complete-looking JSON object from an LLM response.

    Handles responses such as:
    Jordan: Here is the result.
    {"message": "...", "information_need": "..."}
    """
    text = text.strip()

    # Remove common Markdown code fences.
    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    # First try parsing the entire response.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Otherwise, extract from the first { through the last }.
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            f"No valid JSON object found in model response: {text}"
        )

    return json.loads(text[start:end + 1])

def append_history_for_llm(
    messages: list[dict],
    history: list[AdaptiveChatTurn],
    limit: int = 10,
) -> None:
    """
    Add stored user/Alex/Jordan turns to an OpenAI-style message list.

    OpenAI chat messages only use roles such as user and assistant.
    Alex and Jordan are therefore sent as assistant messages, but their
    identities remain explicit in the content.
    """

    for turn in history[-limit:]:
        if turn.role == "user":
            messages.append({
                "role": "user",
                "content": turn.content,
            })

        elif turn.role == "alex":
            messages.append({
                "role": "assistant",
                "content": f"[Alex]\n{turn.content}"
            })

        elif turn.role == "jordan":
            messages.append({
                "role": "assistant",
                "content": f"[Jordan]\n{turn.content}"
            })

def format_history(history: list[AdaptiveChatTurn]) -> str:
    if not history:
        return "(no previous conversation)"

    lines = []

    for turn in history:
        if turn.role == "user":
            speaker = "User"
        elif turn.role == "alex":
            speaker = "Alex"
        else:
            speaker = "Jordan"

        lines.append(f"{speaker}: {turn.content}")

    return "\n".join(lines)

def create_adaptive_router(
    *,
    client_chat: Any,
    model_name: str,
    rag: Any,
    preprocess_question: Callable,
    rag_response_model: type[BaseModel],
    build_resource_cards: Callable,
    clean_alex_answer: Callable,
) -> APIRouter:

    router = APIRouter(prefix="/adaptive", tags=["adaptive-chat"])

    # ---------------------------------------------------------------
    # 1. ROUTE: FACT FINDING OR HYPOTHESIS TESTING
    # ---------------------------------------------------------------

    async def route_turn(
        message: str,
        history: list[AdaptiveChatTurn],
   ) -> RouteResult:

        messages = [
            {
                "role": "system",
                "content": """
                    You classify the user's CURRENT reasoning move in a conversation about
                    clinical trials.

                    Choose exactly one:

                    FACT FINDING
                    The user is asking for a missing fact, definition, explanation, process,
                    cost, rule, or other piece of information, and does not express any belief, assumption, or conclusion at all.

                    HYPOTHESIS TESTING
                    Any part of the user's query expresses a belief, interpretation, conclusion,
                    comparison, expectation, prediction, or feared implication.

                    Classify the CURRENT message in light of the earlier conversation.

                    Conversation history contains three speakers.
                    User:
                    asks questions and reasons about clinical trials.
                    Alex:
                    provides factual information from trusted sources.
                    Jordan:
                    helps interpret, organize, and frame ideas but does not introduce new facts.

                    The current user message may be responding to:
                    - Alex's evidence,
                    - Jordan's interpretation,
                    - or both.

                    Determine whether the user's CURRENT reasoning move is:
                    1. requesting new factual evidence
                    or
                    2. evaluating, revising, or extending an interpretation.

                    Return:
                    - route
                    - one short reason describing the user's reasoning move
                    """,
            }
        ]

        append_history_for_llm(
            messages=messages,
            history=history,
        )

        messages.append({
            "role": "user",
            "content": f"""
                Classify only this current message:
                {message}
            """,
        })

        response = await client_chat.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            response_format=RouteResult,
            temperature=0,
        )

        parsed = response.choices[0].message.parsed

        if parsed is None:
            raise ValueError("Router returned no parsed response.")

        print("*** ROUTER REASON FOR DETERMINED ROUTE:", parsed.reason)

        return parsed

    # ---------------------------------------------------------------
    # 2. JORDAN FRAMES THE INFORMATION NEED
    #
    # Only called for hypothesis-testing turns.
    # ---------------------------------------------------------------

    async def frame_information_need(
        message: str,
        history: list[AdaptiveChatTurn],
    ) -> JordanFrame:

        messages = [
            {
                "role": "system",
                "content": """
                    You are Jordan.

                    The user is expressing an interpretation, belief, comparison, prediction,
                    or concern.

                    Using the full conversation, your job is to prepare the user for Alex's search.

                    Your message should briefly:
                    1. acknowledge the user is trying to determine, and
                    2. what information we are going to look for in order to address the user's concern.

                    The message should naturally lead into Alex's answer.

                    Do not simply acknowledge the user's concern.
                    Do not answer the question yourself.
                    Do not add new facts.

                    The information described in your message should match the
                    information_need you generate.

                    Return ONLY this JSON structure:
                    {
                    "message": "1 or 2 short sentences using plain language that explain what Alex is about to investigate and why that information will help answer the user's question."
                    "information_need": "one factual question that Alex can answer using available sources. Do not ask the user for additional information."
                    }

                    Do not write "Jordan:".
                    Do not include text before or after the JSON.
                    Do not use markdown.
                """,
            }
        ]

        append_history_for_llm(
            messages=messages,
            history=history,
        )

        messages.append({
            "role": "user",
            "content": message,
        })

        try:
            response = await client_chat.beta.chat.completions.parse(
                model=model_name,
                messages=messages,
                response_format=JordanFrame,
                temperature=0,
            )

            parsed = response.choices[0].message.parsed

            if parsed is None:
                raise ValueError("Jordan returned no parsed response.")

            return parsed

        except (ValidationError, ValueError) as error:
            print("*** JORDAN STRUCTURED PARSE FAILED:", error)
            print("*** RETRYING WITH MANUAL JSON EXTRACTION")

            response = await client_chat.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
            )

            raw_content = response.choices[0].message.content or ""

            print("*** RAW JORDAN FRAME:", raw_content)

            data = extract_json_object(raw_content)

            return JordanFrame.model_validate(data)

    # ---------------------------------------------------------------
    # 3. ALEX RETRIEVES AND EXPLAINS EVIDENCE
    # ---------------------------------------------------------------
    def merge_rag_results(
        *result_groups: list[Any],
        limit: int = 8,
    ) -> list[Any]:
        """
        Combine retrieval results, remove duplicate chunks,
        keep the highest score for each chunk, and return
        the best results overall.
        """
        best_by_chunk = {}

        for results in result_groups:
            for result in results:
                metadata = result.get("meta", {})

                chunk_key = (
                    metadata.get("file"),
                    metadata.get("page_number"),
                    metadata.get("chunk_id"),
                )

                existing = best_by_chunk.get(chunk_key)

                if (
                    existing is None
                    or result.get("score", 0) > existing.get("score", 0)
                ):
                    best_by_chunk[chunk_key] = result

        merged_results = sorted(
            best_by_chunk.values(),
            key=lambda result: result.get("score", 0),
            reverse=True,
        )

        return merged_results[:limit]

    async def prepare_alex_search(
        *,
        information_need: str,
        history: list[AdaptiveChatTurn],
    ) -> dict:
        preprocess = await preprocess_question(
            information_need,
            history,
        )

        original_query = information_need.strip()
        rewritten_query = preprocess.search_query.strip()

        print("*** ADAPTIVE ORIGINAL QUERY:", original_query)
        print("*** ADAPTIVE REWRITTEN QUERY:", rewritten_query)

        # Always search using the natural-language information need.
        original_results = rag.retrieve(
            original_query,
            k=8,
        )

        # Avoid running the identical search twice.
        if rewritten_query.casefold() == original_query.casefold():
            results = await asyncio.to_thread(
                rag.retrieve,
                original_query,
                8,
            )
            retrieval_mode = "original_only_same_query"

        else:
            original_results, rewritten_results = await asyncio.gather(
                asyncio.to_thread(
                    rag.retrieve,
                    original_query,
                    8,
                ),
                asyncio.to_thread(
                    rag.retrieve,
                    rewritten_query,
                    8,
                ),
            )

            results = merge_rag_results(
                original_results,
                rewritten_results,
                limit=8,
            )

            retrieval_mode = "original_and_rewritten"
        print("*** ADAPTIVE RETRIEVAL MODE:", retrieval_mode)

        for index, result in enumerate(results):
            metadata = result.get("meta", {})

            print(
                "*** MERGED RESULT:",
                {
                    "rank": index + 1,
                    "score": result.get("score"),
                    "title": metadata.get("title"),
                    "page": metadata.get("page_number"),
                    "chunk_id": metadata.get("chunk_id"),
                },
            )

        return {
            # Keep this field unchanged so the rest of your code still works.
            "search_query": rewritten_query,
            "results": results,
        }
    
    async def run_alex(
        *,
        original_message: str,
        information_need: str,
        search_query: str,
        results: list[Any],
        history: list[AdaptiveChatTurn],
    ) -> dict:

        context_parts = []

        for index, result in enumerate(results):
            metadata = result["meta"]

            context_parts.append(
                f"""
                    ID: {index}
                    SOURCE: {metadata.get("source", "")}
                    TITLE: {metadata.get("title", metadata.get("file", ""))}
                    TYPE: {metadata.get("type", "")}
                    FILE: {metadata.get("file", "")}
                    URL: {metadata.get("url", "")}
                    CONTENT: {result["text"]}
                """.strip()
            )

        context = "\n\n---\n\n".join(context_parts)

        system_prompt = """
            You are Alex, a clinical trials educator.

            Use conversation history only to understand what the user means.
            Use ONLY the provided context as factual evidence.

            Answer the user's current question in plain, conversational language.

            When answering:
            - Answer as completely as the evidence supports, but no further.
            - If the evidence provides useful general information, explain it before noting what varies by trial or by person.
            - If the evidence only partially answers the question, clearly explain what the evidence establishes and what remains unknown.
            - Do not present inferences, examples, definitions, or status labels as established facts unless the context explicitly states them.
            - Do not guess or add information that is not supported by the provided context.
            - Do not simply tell the user to ask the study team if the context contains useful general information.
            - If the context provides almost no useful information, clearly say so instead of guessing.

            Choose answer_scope:
            - general_answer: The context directly answers the question.
            - varies_by_trial: The context provides useful general information, but important details depend on the specific trial.
            - personalized_decision: The context provides useful general information, but the user's personal situation cannot be answered. Do not recommend, discourage, or judge whether participation is worth it. Instead, explain what factors people commonly consider when making that decision.
            - insufficient_context: The context provides little or no useful information.

            Use simple words and short sentences.
            Explain medical terms in plain language.
            Be friendly, direct, and reassuring.

            Write one conversational paragraph under 90 words.
            Do not use headings, lists, citations, source names, or line breaks.
            Write phases as Phase 1, Phase 2, Phase 3, and Phase 4.

            Return:
            1. answer
            2. source_explanations
            3. confidence
            4. talking_points
            5. answer_scope

            For talking_points:
            - Return at most 3.
            - Each should be 4–9 words.
            - Use plain language.
            - Follow the same order as the answer.
            - Do not include citations or source names.
            - Return an empty list only when answer_scope is insufficient_context.

            For each source_explanation:
            - Use the exact source ID.
            - Include only sources that directly support the answer.
            - Briefly explain how each source supports the answer.
            - Return 1–3 source_explanations unless answer_scope is insufficient_context.

            Do not invent source IDs.
        """

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

        append_history_for_llm(
            messages=messages,
            history=history,
        )

        messages.append({
            "role": "user",
            "content": f"""
                CONTEXT:
                {context}

                CURRENT USER'S ORIGINAL QUESTION:
                {original_message}

                INFORMATION NEED:
                {information_need}

                SEARCH QUERY:
                {search_query}
            """.strip(),
        })

        response = await client_chat.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            response_format=rag_response_model,
            temperature=0,
        )

        parsed = response.choices[0].message.parsed

        if parsed is None:
            raise ValueError("Alex returned no parsed response.")

        valid_source_explanations = [
            explanation
            for explanation in parsed.source_explanations
            if 0 <= explanation.id < len(results)
        ]

        if len(valid_source_explanations) != len(parsed.source_explanations):
            print(
                "*** DROPPED INVALID SOURCE EXPLANATIONS:",
                {
                    "returned": [
                        explanation.model_dump()
                        for explanation in parsed.source_explanations
                    ],
                    "valid": [
                        explanation.model_dump()
                        for explanation in valid_source_explanations
                    ],
                },
            )

        print("*** ALEX RESPONSE:", parsed)
        print("*** ALEX ANSWER SCOPE:", parsed.answer_scope)
        print("*** ALEX SOURCE EXPLANATIONS:", valid_source_explanations)

        sources = []

        for explanation in valid_source_explanations:
            result_id = explanation.id

            if not 0 <= result_id < len(results):
                continue

            source_cards = build_resource_cards([results[result_id]])

            if not source_cards:
                continue

            source = source_cards[0]

            sources.append({
                **source,
                # Preserve the original retrieval ID selected by Alex.
                "id": result_id,
                "relevance": explanation.relevance_explanation.strip(),
            })

        has_supported_information = (
            parsed.answer_scope != "insufficient_context"
        )

        print("*** FINAL ALEX SOURCES:", sources)

        print(
            "*** ALEX HAS SUPPORTED INFORMATION:",
            has_supported_information,
        )

        return {
            "answer": clean_alex_answer(parsed.answer),
            "search_query": search_query,
            "sources": sources,
            "confidence": parsed.confidence,
            "talking_points": parsed.talking_points or [],
            "answer_scope": parsed.answer_scope,
            "has_supported_information": has_supported_information,
        }
    # ---------------------------------------------------------------
    # 4. JORDAN INTERPRETS THE NEW EVIDENCE
    # ---------------------------------------------------------------

    def validate_mental_model_highlight(
        mental_model: str,
        highlighted_text: str,
    ) -> str | None:
        normalized_model = mental_model.strip()
        normalized_highlight = highlighted_text.strip()

        if not normalized_highlight:
            return None

        # Best case: exact substring.
        if normalized_highlight in normalized_model:
            return normalized_highlight

        print(
            "*** INVALID MENTAL MODEL HIGHLIGHT:",
            {
                "mental_model": normalized_model,
                "highlighted_text": normalized_highlight,
            },
        )

        model_words = normalized_model.split()
        highlight_words = normalized_highlight.split()

        if not model_words or not highlight_words:
            return None

        target_length = len(highlight_words)
        min_length = max(3, target_length - 3)
        max_length = min(len(model_words), target_length + 3)

        best_phrase = None
        best_score = 0.0

        for window_length in range(min_length, max_length + 1):
            for start in range(len(model_words) - window_length + 1):
                candidate = " ".join(
                    model_words[start : start + window_length]
                )

                score = SequenceMatcher(
                    None,
                    normalized_highlight.lower(),
                    candidate.lower(),
                ).ratio()

                if score > best_score:
                    best_score = score
                    best_phrase = candidate

        if best_phrase and best_score >= 0.72:
            print(
                "*** REPAIRED MENTAL MODEL HIGHLIGHT:",
                {
                    "requested": normalized_highlight,
                    "repaired": best_phrase,
                    "score": best_score,
                },
            )
            return best_phrase

        return None

    async def update_knowledge_gaps(
        *,
        original_message: str,
        alex_answer: str,
        answer_scope: AnswerScope,
        knowledge_gaps: list[str],
    ) -> JordanUnresolvedUpdate:
        response = await client_chat.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": """
                        You are Jordan. Alex could not provide enough supported
                        information to resolve the user's current question.

                        Your role is to keep track of what the conversation has and has not established.
                        Alex could not provide enough supported information to answer the user's current question.

                        Your job is to:
                        1. Tell the user you've noted that this question remains unanswered.
                        2. Explain that you'll keep it in mind as the conversation continues.
                        3. Maintain the conversation's current knowledge gaps. Preserve earlier unanswered questions unless Alex later provides enough supported information to resolve one.

                        Do not:
                        - speculate about the answer.
                        - interpret why the information matters.
                        - repeat Alex's response.
                        - suggest what the answer might depend on unless Alex explicitly said so.
                        - say you added something to an interface, workspace, note, or list.

                        Your response should reinforce that distinguishing between what is known and what is still unknown helps build an accurate understanding.
                        
                        For message:
                        - Write 1 or 2 short conversational sentences.
                        - Your message should use plain language.
                        - Tell the user you've noted that Alex couldn't answer this question with the available evidence.
                        - Explain you'll keep this unanswered question in mind as the conversation continues.
                        - Do not answer the question.
                        - Do not speculate.
                        - Do not explain why the information matters.
                        - Do not mention interfaces, notes, workspaces, or lists.

                        For knowledge_gaps:
                        - Return the complete updated list.
                        - Add one concise question representing the current unknown.
                        - Preserve earlier unresolved questions.
                        - Merge or replace duplicates.
                        - Keep at most 4 questions.
                        - Each item must be understandable without conversation history.

                        Return only JSON matching JordanUnresolvedUpdate.
                    """,
                },
                {
                    "role": "user",
                    "content": f"""
                        USER'S QUESTION:
                        {original_message}

                        ALEX'S RESPONSE:
                        {alex_answer}

                        ANSWER SCOPE:
                        {answer_scope}

                        CURRENT OPEN QUESTIONS:
                        {json.dumps(knowledge_gaps)}
                    """,
                },
            ],
            response_format=JordanUnresolvedUpdate,
            temperature=0,
        )

        update = response.choices[0].message.parsed

        if update is None:
            raise ValueError("Jordan returned no unresolved-question update.")

        return update

    async def integrate_answer(
        *,
        original_message: str,
        alex_answer: str,
        answer_scope: AnswerScope,
        has_supported_information: bool,
        history: list[AdaptiveChatTurn],
        mental_model: str | None,
        knowledge_gaps: list[str],
    ) -> JordanAfter:
        """
        Update one evolving mental model using Alex's newest answer.

        Jordan always performs the same operation:
        1. determine the current best overall understanding;
        2. explain how the newest information affected that understanding.
        """

        if not has_supported_information:
            try:
                unresolved_update = await update_knowledge_gaps(
                    original_message=original_message,
                    alex_answer=alex_answer,
                    answer_scope=answer_scope,
                    knowledge_gaps=knowledge_gaps,
                )

                return JordanAfter(
                    message=unresolved_update.message.strip(),
                    mental_model=mental_model or "",
                    highlighted_text=None,
                    change_type=None,
                    knowledge_gaps=unresolved_update.knowledge_gaps,
                )

            except Exception as error:
                print("*** JORDAN OPEN QUESTIONS UPDATE FAILED:", repr(error))

                return JordanAfter(
                    message=(
                        "I've noted that Alex couldn't find enough supported information to answer this question. We'll keep it in mind as we continue building your understanding."
                    ),
                    mental_model=mental_model or "",
                    highlighted_text=None,
                    change_type=None,
                    knowledge_gaps=knowledge_gaps,
                )

        messages = [
            {
            "role": "system",
            "content": """
                You are Jordan. Your job is to maintain one evolving understanding of what the user has learned about clinical trial participation.
                Alex provides the factual information. Your job is to integrate each new piece of information into the user's developing understanding.

                Do not simply summarize Alex's answer.
                Do not introduce new facts.
                Do not make recommendations.
                Focus on how the new information changes, adds to, clarifies, or revises the overall understanding.
                Explain how the new information fits into the understanding developed so far.
                Speak like a conversation partner helping the user connect ideas.

                Return:
                1. mental_model: The complete updated understanding after incorporating Alex's newest answer.
                2. highlighted_text: The exact continuous phrase within mental_model that represents the most
                meaningful new addition or revision. After writing mental_model, select highlighted_text only by copying and pasting one exact continuous substring from the completed mental_model.
                Do not rewrite, paraphrase, add conjunctions, or change punctuation.
                3. change_type: Choose exactly one:
                - created: There was no previous mental model.
                - expanded: The previous understanding is still valid, and a genuinely new consideration is now included.
                - clarified: The same consideration remains, but it is now more specific, nuanced, or easier to understand.
                - revised: Part of the earlier understanding is no longer the best framing and
                has been meaningfully changed or replaced.
                4. change_explanation: One brief, conversational explanation of how the newest information changed or built on the overall understanding.

                For the first answer:
                - Create the initial mental_model (under 25 words).
                - Set change_type to created.
                - Set highlighted_text to the entire mental_model because the whole
                understanding is new.

                For later answers:
                - Revise the existing mental_model.
                - The mental model should never be a list of facts or information
                - The mental model should always be a cohesive main big picture
                - Focus on the overall understanding rather than listing individual facts.
                - It's okay to replace an earlier takeaway if a better understanding emerges.
                - Don't simply restate Alex's latest answer.
                - Never introduce facts that Alex didn't mention.
                - highlighted_text should usually be a short phrase, not the entire mental_model.
                - Select the smallest phrase that clearly represents the meaningful conceptual
                change.

                For change_explanation:
                - Explain how the newest information affected the overall understanding.
                - Connect it to earlier understanding when appropriate.
                - If this is the first piece of information, explain what it establishes.
                - Be specific to Alex's answer.
                - Avoid generic statements that could apply to almost any response.

                Keep:
                - mental_model under 55 words.
                - change_explanation under 35 words.

                Return only JSON matching the JordanMentalModelUpdate schema.
            """,
            },
            {
                "role": "user",
                "content": f"""
                    CONVERSATION SO FAR:
                    {format_history(history)}

                    CURRENT USER QUESTION:
                    {original_message}

                    ALEX'S NEWEST ANSWER:
                    {alex_answer}

                    CURRENT MENTAL MODEL:
                    {mental_model or "(none yet — create the initial mental model)"}
                """,
            },
        ]

        try:
            response = await client_chat.beta.chat.completions.parse(
                model=model_name,
                messages=messages,
                response_format=JordanMentalModelUpdate,
                temperature=0,
            )

            update = response.choices[0].message.parsed

            if update is None:
                raise ValueError("Jordan returned no mental model update.")

            updated_mental_model = update.mental_model.strip()

            validated_highlight = validate_mental_model_highlight(
                mental_model=updated_mental_model,
                highlighted_text=update.highlighted_text,
            )

            print("*** PREVIOUS MENTAL MODEL:", mental_model)
            print("*** UPDATED MENTAL MODEL:", updated_mental_model)
            print("*** MENTAL MODEL CHANGE TYPE:", update.change_type)
            print("*** REQUESTED HIGHLIGHT:", update.highlighted_text)
            print("*** VALIDATED HIGHLIGHT:", validated_highlight)
            print("*** JORDAN CHANGE EXPLANATION:", update.change_explanation)

            return JordanAfter(
                message=update.change_explanation.strip(),
                mental_model=updated_mental_model,
                highlighted_text=validated_highlight,
                change_type=update.change_type,
                knowledge_gaps=knowledge_gaps,
            )

        except Exception as error:
            print(
                "*** JORDAN MENTAL MODEL UPDATE FAILED:",
                repr(error),
            )

            # Preserve the previous understanding instead of breaking the turn.
            fallback_model = (
                mental_model
                or alex_answer.strip()[:250]
            )

            return JordanAfter(
                message=(
                    "This gives us an initial understanding to build from."
                    if mental_model is None
                    else
                    "This adds another piece to the understanding we were building."
                ),
                mental_model=fallback_model,
                highlighted_text=None,
                change_type=None,
                knowledge_gaps=knowledge_gaps,
            )

    # ---------------------------------------------------------------
    # MAIN ORCHESTRATION ENDPOINTS
    # ---------------------------------------------------------------

    def stream_part(part: str, **data: Any) -> str:
        """Encode one newline-delimited JSON event for the browser."""
        return json.dumps({"part": part, **data}) + "\n"

    @router.post("/chat", response_model=AdaptiveChatResponse)
    async def adaptive_chat(
        request: AdaptiveChatRequest,
    ) -> AdaptiveChatResponse:
        """Non-streaming endpoint retained for compatibility."""

        print("\n*** BEGIN ADAPTIVE CHAT")
        print("*** USER MESSAGE:", request.message)

        route_result = await route_turn(
            message=request.message,
            history=request.history,
        )
        route = route_result.route

        print("*** ROUTE DETERMINED:", route)

        jordan_before = None

        if route == "hypothesis_testing":
            jordan_frame = await frame_information_need(
                message=request.message,
                history=request.history,
            )
            jordan_before = jordan_frame.message
            information_need = jordan_frame.information_need
        else:
            information_need = request.message

        print("*** INFORMATION NEED DETERMINED:", information_need)

        search = await prepare_alex_search(
            information_need=information_need,
            history=request.history,
        )

        alex_result = await run_alex(
            original_message=request.message,
            information_need=information_need,
            search_query=search["search_query"],
            results=search["results"],
            history=request.history,
        )

        jordan_result = await integrate_answer(
            original_message=request.message,
            alex_answer=alex_result["answer"],
            answer_scope=alex_result["answer_scope"],
            has_supported_information=alex_result[
                "has_supported_information"
            ],
            history=request.history,
            mental_model=request.mental_model,
            knowledge_gaps=request.knowledge_gaps,
        )

        return AdaptiveChatResponse(
            route=route,
            jordan_before=jordan_before,
            information_need=information_need,
            search_query=alex_result["search_query"],
            alex_answer=alex_result["answer"],
            jordan_after=jordan_result.message,
            mental_model=jordan_result.mental_model,
            highlighted_text=jordan_result.highlighted_text,
            change_type=jordan_result.change_type,
            knowledge_gaps=jordan_result.knowledge_gaps,
            sources=alex_result["sources"],
            talking_points=alex_result["talking_points"],
            answer_scope=alex_result["answer_scope"],
        )

    @router.post("/chat-stream")
    async def adaptive_chat_stream(
        request: AdaptiveChatRequest,
    ) -> StreamingResponse:
        """Send each completed processing stage to the browser immediately."""

        async def generate():
            try:
                print("\n*** ADAPTIVE CHAT")
                print("*** USER:", request.message)

                route_result = await route_turn(
                    message=request.message,
                    history=request.history,
                )
                route = route_result.route

                print("*** ROUTE:", route)

                yield stream_part(
                    "route",
                    route=route,
                    reason=route_result.reason,
                )

                jordan_before = None

                if route == "hypothesis_testing":
                    jordan_frame = await frame_information_need(
                        message=request.message,
                        history=request.history,
                    )
                    jordan_before = jordan_frame.message
                    information_need = jordan_frame.information_need

                    print("*** INFORMATION NEED:", information_need)

                    yield stream_part(
                        "information_need",
                        information_need=information_need,
                    )
                    yield stream_part(
                        "jordan_before",
                        message=jordan_before,
                    )
                else:
                    information_need = request.message

                    print("*** INFORMATION NEED:", information_need)

                    yield stream_part(
                        "information_need",
                        information_need=information_need,
                    )

                search = await prepare_alex_search(
                    information_need=information_need,
                    history=request.history,
                )

                yield stream_part(
                    "search_query",
                    search_query=search["search_query"],
                )

                alex_result = await run_alex(
                    original_message=request.message,
                    information_need=information_need,
                    search_query=search["search_query"],
                    results=search["results"],
                    history=request.history,
                )

                yield stream_part(
                    "alex",
                    message=alex_result["answer"],
                    sources=alex_result["sources"],
                    talking_points=alex_result["talking_points"],
                    confidence=alex_result["confidence"],
                    answer_scope=alex_result["answer_scope"],
                    has_supported_information=alex_result[
                        "has_supported_information"
                    ],
                )

                jordan_result = await integrate_answer(
                    original_message=request.message,
                    alex_answer=alex_result["answer"],
                    answer_scope=alex_result["answer_scope"],
                    has_supported_information=alex_result[
                        "has_supported_information"
                    ],
                    history=request.history,
                    mental_model=request.mental_model,
                    knowledge_gaps=request.knowledge_gaps,
                )

                if jordan_result:
                    yield stream_part(
                        "jordan_after",
                        route=route,
                        message=jordan_result.message,
                        mental_model=jordan_result.mental_model,
                        highlighted_text=jordan_result.highlighted_text,
                        change_type=jordan_result.change_type,
                        knowledge_gaps=jordan_result.knowledge_gaps
                    )

                yield stream_part("done")

            except Exception as error:
                print("*** ADAPTIVE STREAM ERROR:", error)
                yield stream_part("error", message=str(error))

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ---------------------------------------------------------------
    # SEPARATED CALLS
    # ---------------------------------------------------------------

    @router.post("/route", response_model=AdaptiveRouteResponse)
    async def adaptive_route(
        request: AdaptiveRouteRequest,
    ) -> AdaptiveRouteResponse:
        print("\n*** ADAPTIVE ROUTE")
        print("*** USER:", request.message)

        result = await route_turn(
            message=request.message,
            history=request.history,
        )

        return AdaptiveRouteResponse(
            route=result.route,
            reason=result.reason,
        )


    @router.post("/frame", response_model=AdaptiveFrameResponse)
    async def adaptive_frame(
        request: AdaptiveFrameRequest,
    ) -> AdaptiveFrameResponse:
        print("\n*** ADAPTIVE FRAME")
        print("*** USER:", request.message)

        result = await frame_information_need(
            message=request.message,
            history=request.history,
        )

        return AdaptiveFrameResponse(
            message=result.message,
            information_need=result.information_need,
        )


    @router.post("/alex", response_model=AdaptiveAlexResponse)
    async def adaptive_alex(
        request: AdaptiveAlexRequest,
    ) -> AdaptiveAlexResponse:
        print("\n*** ADAPTIVE ALEX")
        print("*** INFORMATION NEED:", request.information_need)

        search = await prepare_alex_search(
            information_need=request.information_need,
            history=request.history,
        )

        alex_result = await run_alex(
            original_message=request.original_message,
            information_need=request.information_need,
            search_query=search["search_query"],
            results=search["results"],
            history=request.history,
        )

        return AdaptiveAlexResponse(
            search_query=alex_result["search_query"],
            answer=alex_result["answer"],
            sources=alex_result["sources"],
            confidence=alex_result["confidence"],
            talking_points=alex_result["talking_points"],
            answer_scope=alex_result["answer_scope"],
            has_supported_information=alex_result[
                "has_supported_information"
            ],
        )


    @router.post("/jordan", response_model=AdaptiveJordanResponse)
    async def adaptive_jordan(
        request: AdaptiveJordanRequest,
    ) -> AdaptiveJordanResponse:
        print("\n*** ADAPTIVE JORDAN")
        print("*** USER:", request.original_message)

        result = await integrate_answer(
            original_message=request.original_message,
            alex_answer=request.alex_answer,
            answer_scope=request.answer_scope,
            has_supported_information=request.has_supported_information,
            history=request.history,
            mental_model=request.mental_model,
            knowledge_gaps=request.knowledge_gaps,
        )

        return AdaptiveJordanResponse(
            message=result.message,
            mental_model=result.mental_model,
            highlighted_text=result.highlighted_text,
            change_type=result.change_type,
            knowledge_gaps=result.knowledge_gaps,
        )

    return router