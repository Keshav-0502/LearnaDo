"""
Build and compile the LearnaDo LangGraph.

Graph topology:
  START → classify_intent → (router)
    ├── greeting / off_topic  → chat_node → END
    ├── learning_request      → extract_lesson_info → generate_outline
    │       → (outline ok?) → review_outline → (approved?)
    │           ├── yes → create_mission → (single/dual?)
    │           │     ├── single → deliver_lesson (immediate) ──────┐
    │           │     └── dual   → END (goal-setter done)           │
    │           └── no  → cancelled_node → END                      │
    └── command / lesson_answer → lesson_handler → (has mission?)   │
            ├── yes → deliver_lesson ←──────────────────────────────┘
            └── no  → END                    │
                                              ↓
                              wait_for_response (interrupt)
                                              ↓
                              classify_lesson_response → (type?)
                                ├── question / more_detail → tutor_respond
                                │                              → wait_for_response (loop)
                                ├── simplify_request → simplify_node
                                │                        → wait_for_response (loop)
                                └── lesson_answer → evaluate_response → (confused?)
                                      ├── yes → simplify_node → wait_for_response
                                      └── no  → advance_lesson → (more lessons?)
                                                  ├── yes → deliver_lesson (loop)
                                                  └── no  → mission_complete → END
"""

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.state import LearnaDoState


# ── Conditional edge functions ─────────────────────────────────────────────────


def _route_after_classify(state: LearnaDoState) -> str:
    intent = state.get("intent", "off_topic")
    if intent in ("greeting", "off_topic"):
        return "chat_node"
    if intent == "learning_request":
        return "extract_lesson_info"
    return "lesson_handler"


def _route_after_outline(state: LearnaDoState) -> str:
    if not state.get("outline"):
        return END
    return "review_outline"


def _route_after_review(state: LearnaDoState) -> str:
    if state.get("outline_approved"):
        return "create_mission"
    return "cancelled_node"


def _route_after_create_mission(state: LearnaDoState) -> str:
    flow = state.get("flow")
    if flow == "single":
        return "deliver_lesson"
    return END


def _route_after_wait_for_start(state: LearnaDoState) -> str:
    if state.get("flow") == "cancelled":
        return "cancelled_node"
    return "deliver_lesson"


def _route_after_lesson_handler(state: LearnaDoState) -> str:
    if state.get("mission_id"):
        return "deliver_lesson"
    return END


def _route_after_deliver(state: LearnaDoState) -> str:
    if state.get("lesson_id"):
        return "wait_for_response"
    return "mission_complete"


def _route_after_lesson_classify(state: LearnaDoState) -> str:
    rtype = state.get("lesson_response_type", "lesson_answer")
    if rtype in ("question", "more_detail"):
        return "tutor_respond"
    if rtype == "simplify_request":
        return "simplify_node"
    if rtype == "skip":
        return "advance_lesson"
    return "evaluate_response"


def _route_after_evaluate(state: LearnaDoState) -> str:
    if state.get("should_simplify"):
        return "simplify_node"
    return "advance_lesson"


def _route_after_advance(state: LearnaDoState) -> str:
    if state.get("has_next_lesson"):
        return "deliver_lesson"
    return "mission_complete"


# ── Graph construction ─────────────────────────────────────────────────────────


def build_graph(checkpointer: BaseCheckpointSaver | None = None):
    builder = StateGraph(LearnaDoState)

    builder.add_node("classify_intent", nodes.classify_intent)
    builder.add_node("chat_node", nodes.chat_node)
    builder.add_node("extract_lesson_info", nodes.extract_lesson_info)
    builder.add_node("generate_outline", nodes.generate_outline)
    builder.add_node("review_outline", nodes.review_outline)
    builder.add_node("cancelled_node", nodes.cancelled_node)
    builder.add_node("create_mission", nodes.create_mission)
    builder.add_node("wait_for_start", nodes.wait_for_start)
    builder.add_node("lesson_handler", nodes.lesson_handler)
    builder.add_node("deliver_lesson", nodes.deliver_lesson)
    builder.add_node("wait_for_response", nodes.wait_for_response)
    builder.add_node("classify_lesson_response", nodes.classify_lesson_response)
    builder.add_node("tutor_respond", nodes.tutor_respond)
    builder.add_node("evaluate_response", nodes.evaluate_response)
    builder.add_node("simplify_node", nodes.simplify_node)
    builder.add_node("advance_lesson", nodes.advance_lesson)
    builder.add_node("mission_complete", nodes.mission_complete)

    # ── Edges ──

    builder.add_edge(START, "classify_intent")

    builder.add_conditional_edges("classify_intent", _route_after_classify)
    builder.add_edge("chat_node", END)

    builder.add_edge("extract_lesson_info", "generate_outline")
    builder.add_conditional_edges("generate_outline", _route_after_outline)

    builder.add_conditional_edges("review_outline", _route_after_review)
    builder.add_edge("cancelled_node", END)

    builder.add_conditional_edges("create_mission", _route_after_create_mission)
    builder.add_conditional_edges("wait_for_start", _route_after_wait_for_start)

    builder.add_conditional_edges("lesson_handler", _route_after_lesson_handler)

    builder.add_conditional_edges("deliver_lesson", _route_after_deliver)
    builder.add_edge("wait_for_response", "classify_lesson_response")
    builder.add_conditional_edges("classify_lesson_response", _route_after_lesson_classify)
    builder.add_edge("tutor_respond", "wait_for_response")

    builder.add_conditional_edges("evaluate_response", _route_after_evaluate)
    builder.add_edge("simplify_node", "wait_for_response")

    builder.add_conditional_edges("advance_lesson", _route_after_advance)
    builder.add_edge("mission_complete", END)

    return builder.compile(checkpointer=checkpointer)
