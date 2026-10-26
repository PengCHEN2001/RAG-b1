"""Shared graph state and the structured plan produced for each turn.

These are type contracts, not runtime validators. The plan node validates
model output before publishing it to the graph state.

Retriever dataclasses are reused directly. The graph's checkpoint serializer
registers those types and restores their tuple fields on reads.
"""

from typing import Literal, TypedDict

from langgraph.graph import MessagesState

from retrieval.rulebook_retriever import RetrievedRulebookContext, RulebookQuerySet
from sql_retriever.result import RetrievalResult


Route = Literal["rulebook", "sql", "both", "direct", "clarify", "unsupported"]
EvidenceStatus = Literal[
    "ok", "empty", "partial", "needs_clarification", "unsupported", "error", "skipped",
]
Outcome = Literal[
    "completed", 
    "partial", 
    "needs_clarification", 
    "unsupported", 
    "error", 
    "cancelled",
]


class RoutePlan(TypedDict):
    """One turn's source selection and source-specific queries.

    Unused queries are None. depends_on_sql applies to the both route: when
    true, the rulebook query is provisional until SQL evidence is available.
    message holds the user-facing explanation for clarify / unsupported.
    """

    route: Route
    reason: str
    sql_question: str | None
    rulebook_query_set: RulebookQuerySet | None
    depends_on_sql: bool
    message: str


class ContextValue(TypedDict):
    """A user-supplied or verified value and the turn it came from."""

    value: str
    source_run_id: str


class PendingClarification(TypedDict):
    """The request and exact entity candidates awaiting a user response."""

    question: str
    message: str
    candidates: list[dict[str, str]]
    source_run_id: str


class Citation(TypedDict):
    """An answer citation pointing to evidence in this turn's results."""

    source: Literal["rulebook", "sql"]
    source_id: str  # Rulebook chunk ID or SQL query ID.


class AgentState(MessagesState, total=False):
    """Conversation memory plus the working data for the current turn.

    messages is inherited with LangGraph's add_messages reducer and stores
    committed user / assistant turns. Other fields use replacement updates;
    absent fields have not been initialized, and None clears a working value.

    thread_id belongs to graph config. Clients, retrievers and event callbacks
    belong to runtime context, not this checkpointed state.
    """

    # Conversation memory, retained between turns.
    conversation_context: dict[str, ContextValue]
    pending_clarification: PendingClarification | None

    # Current turn; prepare_turn will reset these fields for every new input.
    run_id: str
    turn_index: int
    user_input: str
    standalone_question: str | None
    route_plan: RoutePlan | None
    rulebook_result: RetrievedRulebookContext | None
    sql_result: RetrievalResult | None
    evidence_status: EvidenceStatus | None
    retrieval_errors: dict[str, str]
    sql_context_truncated: bool
    answer_context_truncated: dict[str, bool]
    context_updates: dict[str, ContextValue]
    answer: str | None
    citations: dict[str, Citation]
    outcome: Outcome | None
