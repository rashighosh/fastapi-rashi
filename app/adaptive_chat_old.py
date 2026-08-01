# adaptive_chat.py
import json
import re
from pydantic import ValidationError
from typing import Any, Callable, Literal, Optional
from fastapi.responses import StreamingResponse
from fastapi import APIRouter
from pydantic import BaseModel, Field

# -------------------------------------------------------------------
# THEME BASE MODELS
# -------------------------------------------------------------------

class ThemeDetail(BaseModel):
    id: str = Field(
        description="Stable string ID such as detail-1; never an integer."
    )
    text: str = Field(
        description="The detail content. Always use text, never summary."
    )
    from_question: Optional[str] = None
    from_answer: Optional[str] = None

class Theme(BaseModel):
    id: str = Field(
        description="Stable string ID such as theme-1; never an integer."
    )
    label: str = Field(
        description="A short 2-to-5-word theme label."
    )
    summary: str
    details: list[ThemeDetail] = Field(default_factory=list)

class JordanWorkspaceUpdate(BaseModel):
    themes: list[Theme] = Field(default_factory=list)
    running_summary: Optional[str] = None


class JordanNarration(BaseModel):
    message: str

class JordanAfter(BaseModel):
    # Kept unchanged so the frontend and endpoint response shapes do not change.
    message: Optional[str] = None
    themes: list[Theme] = Field(default_factory=list)
    running_summary: Optional[str] = None

# -------------------------------------------------------------------
# SOURCE BASE MODELS
# -------------------------------------------------------------------

AnswerScope = Literal[
    "general_answer",
    "varies_by_trial",
    "personalized_decision",
    "insufficient_context",
]

class EvidenceSnippet(BaseModel):
    source_id: int
    snippet: str
    relevance: str


class ValidatedEvidenceSnippet(EvidenceSnippet):
    source: str
    title: str
    url: str

class RagResponseModel(BaseModel):
    answer: str
    evidence_snippets: list[EvidenceSnippet] = Field(default_factory=list)
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

    themes: list[Theme] = Field(default_factory=list)
    running_summary: str | None = None

class RouteResult(BaseModel):
    route: Literal["fact_finding", "hypothesis_testing"]
    reason: str


class JordanFrame(BaseModel):
    message: str
    information_need: str

class AdaptiveChatResponse(BaseModel):
    route: Literal["fact_finding", "hypothesis_testing"]

    # What Jordan says before Alex during hypothesis testing.
    jordan_before: Optional[str] = None

    # The factual question sent into retrieval.
    information_need: str

    # Helpful for debugging whether retrieval searched the right thing.
    search_query: str

    alex_answer: str
    jordan_after: Optional[str] = None

    sources: list[Any] = Field(default_factory=list)
    evidence_snippets: list[ValidatedEvidenceSnippet] = Field(
        default_factory=list
    )
    talking_points: list[str] = Field(default_factory=list)
    themes: list[Theme] = Field(default_factory=list)
    running_summary: str | None = None
    answer_scope: AnswerScope

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
    limit: int = 20,
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


def diff_workspace_themes(
    previous: list[Theme],
    updated: list[Theme],
) -> list[dict[str, Any]]:
    """
    Compare the previous and updated workspace by stable IDs.

    The returned changes are used only to narrate what Jordan changed.
    They are not exposed to the frontend.
    """
    changes: list[dict[str, Any]] = []

    old_themes = {theme.id: theme for theme in previous}
    new_themes = {theme.id: theme for theme in updated}

    old_detail_locations: dict[str, tuple[str, str, ThemeDetail]] = {}
    new_detail_locations: dict[str, tuple[str, str, ThemeDetail]] = {}

    for theme in previous:
        for detail in theme.details:
            old_detail_locations[detail.id] = (
                theme.id,
                theme.label,
                detail,
            )

    for theme in updated:
        for detail in theme.details:
            new_detail_locations[detail.id] = (
                theme.id,
                theme.label,
                detail,
            )

    # Themes created, renamed, or removed.
    for theme_id, new_theme in new_themes.items():
        old_theme = old_themes.get(theme_id)

        if old_theme is None:
            changes.append({
                "type": "created_theme",
                "theme_id": theme_id,
                "theme_label": new_theme.label,
            })
            continue

        if old_theme.label != new_theme.label:
            changes.append({
                "type": "renamed_theme",
                "theme_id": theme_id,
                "old_label": old_theme.label,
                "theme_label": new_theme.label,
            })

        if old_theme.summary != new_theme.summary:
            changes.append({
                "type": "updated_summary",
                "theme_id": theme_id,
                "theme_label": new_theme.label,
            })

    for theme_id, old_theme in old_themes.items():
        if theme_id not in new_themes:
            changes.append({
                "type": "removed_theme",
                "theme_id": theme_id,
                "old_label": old_theme.label,
            })

    # Details added, moved, revised, or removed.
    for detail_id, (
        new_theme_id,
        new_theme_label,
        new_detail,
    ) in new_detail_locations.items():
        old_location = old_detail_locations.get(detail_id)

        if old_location is None:
            changes.append({
                "type": "added_detail",
                "detail_id": detail_id,
                "detail": new_detail.text,
                "theme_id": new_theme_id,
                "theme_label": new_theme_label,
            })
            continue

        old_theme_id, old_theme_label, old_detail = old_location

        if old_theme_id != new_theme_id:
            changes.append({
                "type": "moved_detail",
                "detail_id": detail_id,
                "detail": new_detail.text,
                "old_theme_label": old_theme_label,
                "theme_id": new_theme_id,
                "theme_label": new_theme_label,
            })

        if old_detail.text != new_detail.text:
            changes.append({
                "type": "updated_detail",
                "detail_id": detail_id,
                "old_detail": old_detail.text,
                "detail": new_detail.text,
                "theme_id": new_theme_id,
                "theme_label": new_theme_label,
            })

    for detail_id, (
        old_theme_id,
        old_theme_label,
        old_detail,
    ) in old_detail_locations.items():
        if detail_id not in new_detail_locations:
            changes.append({
                "type": "removed_detail",
                "detail_id": detail_id,
                "detail": old_detail.text,
                "theme_id": old_theme_id,
                "old_theme_label": old_theme_label,
            })

    if not changes:
        changes.append({"type": "unchanged"})

    return changes


def fallback_workspace_message(
    changes: list[dict[str, Any]],
) -> str:
    """Create a safe narration if the narration model call fails."""
    meaningful = [
        change
        for change in changes
        if change.get("type") not in {
            "unchanged",
            "updated_summary",
        }
    ]

    if not meaningful:
        return "I kept the board as it was because this information fits the ideas already there."

    first = meaningful[0]
    change_type = first.get("type")

    if change_type == "created_theme":
        return (
            f'I added a new idea called "{first["theme_label"]}" '
            "to capture what this information helps explain."
        )

    if change_type == "renamed_theme":
        return (
            f'I renamed "{first["old_label"]}" to '
            f'"{first["theme_label"]}" to better reflect the larger idea.'
        )

    if change_type == "added_detail":
        return (
            f'I added "{first["detail"]}" under '
            f'"{first["theme_label"]}" on the board.'
        )

    if change_type == "moved_detail":
        return (
            f'I moved "{first["detail"]}" into '
            f'"{first["theme_label"]}" because it fits that idea better.'
        )

    if change_type == "updated_detail":
        return (
            f'I revised one note under "{first["theme_label"]}" '
            "to make the idea clearer."
        )

    return "I reorganized the board so the new information fits the clearest larger idea."


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
                    "message": "1 or 2 short sentences that explain what Alex is about to investigate and why that information will help answer the user's question."
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

    def build_evidence_source_cards(
        sources: list[dict],
        evidence_snippets: list[dict],
    ) -> list[dict]:
        """
        Return only source cards that contain validated evidence.

        Each source card's `id` must correspond to the retrieved result index
        used by EvidenceSnippet.source_id.
        """

        evidence_by_source_id: dict[int, list[dict]] = {}

        for evidence in evidence_snippets:
            source_id = evidence.get("source_id")

            if not isinstance(source_id, int):
                continue

            evidence_by_source_id.setdefault(source_id, []).append(evidence)

        evidence_source_cards = []

        for source in sources:
            source_id = source.get("id")
            matching_evidence = evidence_by_source_id.get(source_id, [])

            # This retrieved source was not used as validated evidence.
            if not matching_evidence:
                continue

            source_copy = dict(source)
            source_copy["evidence_snippets"] = matching_evidence
            source_copy["evidence_snippet"] = matching_evidence[0]["snippet"]

            evidence_source_cards.append(source_copy)

        return evidence_source_cards

    def validate_evidence_snippets(
        snippets: list[EvidenceSnippet],
        results: list[Any],
    ) -> list[dict]:
        validated = []

        for item in snippets:
            if item.source_id < 0 or item.source_id >= len(results):
                continue

            source_text = results[item.source_id]["text"]
            snippet = item.snippet.strip()

            if not snippet:
                continue

            if snippet not in source_text:
                print(
                    "*** DROPPING INVALID EVIDENCE SNIPPET:",
                    {
                        "source_id": item.source_id,
                        "snippet": snippet,
                    },
                )
                continue

            metadata = results[item.source_id]["meta"]

            validated.append({
                "source_id": item.source_id,
                "snippet": snippet,
                "relevance": item.relevance,
                "source": metadata.get("source", ""),
                "title": metadata.get(
                    "title",
                    metadata.get("file", ""),
                ),
                "url": metadata.get("url", ""),
            })

        return validated

    async def prepare_alex_search(
        *,
        information_need: str,
        history: list[AdaptiveChatTurn],
    ) -> dict:
        preprocess = await preprocess_question(
            information_need,
            history,
        )

        print("*** ADAPTIVE SEARCH QUERY:", preprocess.search_query)

        results = rag.retrieve(
            preprocess.search_query,
            k=8,
        )

        return {
            "search_query": preprocess.search_query,
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
            You are Alex, a clinical trials educator and information-foraging assistant.

            Use conversation history only to understand what the user means.
            Use ONLY the provided retrieved context as factual evidence.

            Answer the user's current question in plain, conversational language.
            When the concern behind the question is clear, respond to that concern
            while staying within the provided facts.

            If there is no single general answer or the context cannot answer the
            question exactly:
            - Do not stop at saying the sources lack the answer.
            - Briefly explain why the answer varies or remains unknown.
            - Give any useful general information supported by the context.
            - Name 2 to 3 things that would need to be checked for a specific trial,
            but only when those things are supported by the retrieved context.

            Do not ask for personal health information.
            Do not recommend a trial, judge eligibility, choose a treatment,
            or give medical advice.

            Choose exactly one answer_scope:

            - general_answer:
            The retrieved context directly answers the user's general question.

            - varies_by_trial:
            The general idea is known, but important details depend on the
            specific clinical trial.

            - personalized_decision:
            The user asks what is best, safest, appropriate, or recommended
            for them personally.

            - insufficient_context:
            The retrieved context provides almost no useful information
            for answering the question.

            ANSWER:
            - Report what the retrieved evidence says rather than giving a final explanation from your own knowledge.
            - Ground every statement in the extracted evidence.
            - Briefly explain why the retrieved information is relevant to the user's question.
            - Do not infer, integrate, or interpret beyond what the evidence supports.
            - Use simple words and short sentences.
            - Explain medical terms in plain language.
            - Be friendly, direct, and reassuring.
            - Write one conversational paragraph under 90 words.
            - Do not use headings, lists, citations, source names, or line breaks.
            - Write phases as Phase 1, Phase 2, Phase 3, and Phase 4.
            - Do not claim more than the extracted evidence supports.
            - Do not speak as Jordan.
            - Do not organize the user's overall understanding.
            - Do not suggest an unrelated new direction.

            EVIDENCE SNIPPETS:
            - Return 1 to 3 evidence snippets.
            - Include only evidence that directly supports claims made in the answer.
            - Each snippet must be copied verbatim from the CONTENT of one retrieved source.
            - Do not paraphrase, combine, clean up, or complete a snippet.
            - Preserve the source's original wording.
            - Select the shortest passage that still makes sense on its own.
            - Each snippet must include the matching source ID.
            - Never assign a snippet to a source unless that exact text appears in
            that source's CONTENT.
            - Do not use text from SOURCE, TITLE, TYPE, FILE, or URL as a snippet.
            - Do not return duplicate or substantially overlapping snippets.
            - If no retrieved passage supports a useful answer, return an empty list.

            For each evidence snippet:
            - source_id must be the exact retrieved source ID.
            - snippet must be the exact supporting passage.
            - relevance should briefly explain which answer claim the passage supports.

            TALKING POINTS:
            - Return at most 3.
            - Each should be 4 to 9 words.
            - Use plain language.
            - Keep them in the same order as the answer.
            - Do not include citations or source names.
            - Return an empty list only when no useful supported information can be given.

            CONFIDENCE:
            - Base confidence only on how directly the retrieved evidence answers
            the information need.

            Return:
            - answer
            - evidence_snippets
            - confidence
            - talking_points
            - answer_scope
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
                RETRIEVED CONTEXT:
                {context}

                ORIGINAL USER MESSAGE:
                {original_message}

                INFORMATION NEED:
                {information_need}

                SEARCH QUERY:
                {search_query}
            """,
        })

        response = await client_chat.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            response_format=rag_response_model,
        )

        parsed = response.choices[0].message.parsed

        if parsed is None:
            raise ValueError("Alex returned no parsed response.")

        evidence_snippets = validate_evidence_snippets(
            parsed.evidence_snippets or [],
            results,
        )

        retrieved_source_cards = build_resource_cards(results)

        evidence_source_cards = build_evidence_source_cards(
            sources=retrieved_source_cards,
            evidence_snippets=evidence_snippets,
        )

        return {
            "answer": clean_alex_answer(parsed.answer),
            "search_query": search_query,
            "sources": evidence_source_cards,
            "evidence_snippets": evidence_snippets,
            "confidence": parsed.confidence,
            "talking_points": parsed.talking_points or [],
            "answer_scope": parsed.answer_scope,
        }

    # ---------------------------------------------------------------
    # 4. JORDAN INTERPRETS THE NEW EVIDENCE
    # ---------------------------------------------------------------

    async def integrate_answer(
        *,
        route: str,
        original_message: str,
        alex_answer: str,
        history: list[AdaptiveChatTurn],
        themes: list[Theme],
        running_summary: str | None,
    ) -> JordanAfter:
        print("*IN JORDAN AFTER, ROUTE IS:", route)

        # -----------------------------------------------------------
        # A. ORGANIZE THE WORKSPACE
        # -----------------------------------------------------------
        workspace_messages = [
            {
                "role": "system",
                "content": """
                    You are Jordan, a warm guide helping someone make sense of
                    clinical trial information that Alex retrieves.

                    Update the shared workspace using:
                    - the conversation history
                    - Alex's newest answer
                    - the current themes

                    Return the COMPLETE updated themes collection.

                    THEMES
                    - Keep at most 3 themes.
                    - Prefer updating an existing theme.
                    - Create a new theme only for a genuinely different big idea.
                    - Preserve existing theme and detail IDs whenever possible.
                    - Rename, merge, or reorganize only when it makes the user's
                      mental model clearer.
                    - A theme label must state an insight or larger idea, not merely
                      name a subject.
                    - Someone should understand the main lesson by reading only the
                      label.
                    - Avoid topic labels such as "Trial Safety", "Costs",
                      "Eligibility", or "Randomization".
                    - Prefer labels such as "Multiple layers protect participants",
                      "Participation remains voluntary", or
                      "Fair comparisons require chance".

                    DETAILS
                    - Each detail is one memory cue under 8 words.
                    - Use note fragments, not sentences.
                    - Include only facts supported by Alex's answer.
                    - Do not combine multiple ideas in one detail.

                    RUNNING SUMMARY
                    - Keep it under 60 words.
                    - Reflect the user's current understanding.
                    - Revise it when new evidence changes earlier understanding.

                    Return only JSON matching the JordanWorkspaceUpdate schema.
                """,
            },
            {
                "role": "user",
                "content": f"""
                    ROUTE:
                    {route}

                    CONVERSATION:
                    {format_history(history)}

                    CURRENT USER MESSAGE:
                    {original_message}

                    ALEX'S ANSWER:
                    {alex_answer}

                    CURRENT THEMES:
                    {json.dumps(
                        [theme.model_dump() for theme in themes],
                        indent=2,
                    )}

                    CURRENT RUNNING SUMMARY:
                    {running_summary or "(none yet)"}
                """,
            },
        ]

        try:
            print("*** CALLING JORDAN WORKSPACE ORGANIZER")

            workspace_response = (
                await client_chat.beta.chat.completions.parse(
                    model=model_name,
                    messages=workspace_messages,
                    response_format=JordanWorkspaceUpdate,
                    temperature=0,
                )
            )

            workspace_update = (
                workspace_response.choices[0].message.parsed
            )

            if workspace_update is None:
                raise ValueError(
                    "Jordan returned no parsed workspace update."
                )

        except (ValidationError, ValueError) as error:
            print(
                "*** JORDAN WORKSPACE PARSE FAILED:",
                error,
            )
            print("*** RETRYING JORDAN WORKSPACE UPDATE")

            retry_messages = [
                *workspace_messages,
                {
                    "role": "user",
                    "content": """
                        Retry and follow the JordanWorkspaceUpdate schema exactly.

                        Return:
                        - themes
                        - running_summary

                        Every theme must have:
                        - id
                        - label
                        - summary
                        - details

                        Every detail must have:
                        - id
                        - text
                        - from_question
                        - from_answer

                        Preserve existing string IDs.
                        Never use numeric IDs.
                        Never use "summary" instead of detail "text".
                    """,
                },
            ]

            retry_response = (
                await client_chat.beta.chat.completions.parse(
                    model=model_name,
                    messages=retry_messages,
                    response_format=JordanWorkspaceUpdate,
                    temperature=0,
                )
            )

            workspace_update = (
                retry_response.choices[0].message.parsed
            )

            if workspace_update is None:
                raise ValueError(
                    "Jordan returned no workspace update after retry."
                )

        # -----------------------------------------------------------
        # B. COMPUTE EXACTLY WHAT CHANGED
        # -----------------------------------------------------------
        workspace_changes = diff_workspace_themes(
            previous=themes,
            updated=workspace_update.themes,
        )

        print(
            "*** JORDAN WORKSPACE CHANGES:",
            json.dumps(workspace_changes, indent=2),
        )

        # -----------------------------------------------------------
        # C. NARRATE THE ACTUAL BOARD UPDATE
        # -----------------------------------------------------------
        narration_messages = [
            {
                "role": "system",
                "content": """
                    You are Jordan.
                    You have just updated the shared workspace while the user watched.
                    Now briefly tell the user what you did and why.

                    Guidelines:
                    - Write 1–2 conversational sentences (under 30 words).
                    - Speak naturally, like you're thinking out loud while organizing notes.
                    - Focus on the biggest change you made to the board.
                    - Mention the theme naturally when helpful.
                    - Explain why you grouped, renamed, connected, or moved ideas that way.
                    - If nothing important changed, explain why you kept the existing organization.
                    - For hypothesis testing, mention how the new evidence affected the organization of the user's thinking.
                    - Do not repeat Alex's explanation.
                    - Do not introduce new facts.
                    - Avoid sounding like a changelog or report.
                    - Avoid phrases like "Added a new...", "Updated...", "Created...", or "Renamed...".

                    Return only JSON matching the JordanNarration schema.
                """,
            },
            {
                "role": "user",
                "content": f"""
                    ROUTE:
                    {route}

                    USER'S QUESTION:
                    {original_message}

                    ALEX'S ANSWER:
                    {alex_answer}

                    WORKSPACE CHANGES:
                    {json.dumps(workspace_changes, indent=2)}
                """,
            },
        ]

        try:
            print("*** CALLING JORDAN WORKSPACE NARRATOR")

            narration_response = (
                await client_chat.beta.chat.completions.parse(
                    model=model_name,
                    messages=narration_messages,
                    response_format=JordanNarration,
                    temperature=0,
                )
            )

            narration = narration_response.choices[0].message.parsed

            if narration is None or not narration.message.strip():
                raise ValueError(
                    "Jordan returned no workspace narration."
                )

            message = narration.message.strip()

        except Exception as error:
            # Narration failure should not discard the successfully
            # organized themes or break the frontend stream.
            print(
                "*** JORDAN NARRATION FAILED; USING FALLBACK:",
                repr(error),
            )
            message = fallback_workspace_message(
                workspace_changes
            )

        return JordanAfter(
            message=message,
            themes=workspace_update.themes,
            running_summary=workspace_update.running_summary,
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
            route=route,
            original_message=request.message,
            alex_answer=alex_result["answer"],
            history=request.history,
            themes=request.themes,
            running_summary=request.running_summary,
        )

        return AdaptiveChatResponse(
            route=route,
            jordan_before=jordan_before,
            information_need=information_need,
            search_query=alex_result["search_query"],
            alex_answer=alex_result["answer"],
            jordan_after=jordan_result.message,
            themes=jordan_result.themes,
            running_summary=jordan_result.running_summary,
            sources=alex_result["sources"],
            evidence_snippets=alex_result["evidence_snippets"],
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
                    evidence_snippets=alex_result["evidence_snippets"],
                    talking_points=alex_result["talking_points"],
                    confidence=alex_result["confidence"],
                    answer_scope=alex_result["answer_scope"],
                )

                jordan_result = await integrate_answer(
                    route=route,
                    original_message=request.message,
                    alex_answer=alex_result["answer"],
                    history=request.history,
                    themes=request.themes,
                    running_summary=request.running_summary,
                )

                if jordan_result:
                    yield stream_part(
                        "jordan_after",
                        route=route,
                        message=jordan_result.message,
                        themes=[
                            theme.model_dump()
                            for theme in jordan_result.themes
                        ],
                        running_summary=jordan_result.running_summary,
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

    return router