"""
LangGraph state schema for LearnaDo.

Extends MessagesState so messages accumulate automatically via the
add_messages reducer — giving multi-turn memory from the checkpointer.
"""

from langgraph.graph import MessagesState


class LearnaDoState(MessagesState):
    phone_number: str

    # Set by classify_intent
    intent: str
    sentiment: str

    # Set by extract_lesson_info
    topic: str
    learner_phone: str

    # Set by generate_outline
    outline: list[dict]

    # Set by review_outline
    outline_approved: bool

    # Set by create_mission / lesson_handler
    mission_id: str
    flow: str  # "single" or "dual"

    # Set by deliver_lesson
    lesson_id: str
    lesson_content: str

    # Set by wait_for_response
    user_response: str

    # Set by evaluate_response
    confusion_score: float
    attempts: int
    should_simplify: bool

    # Set by advance_lesson
    has_next_lesson: bool
