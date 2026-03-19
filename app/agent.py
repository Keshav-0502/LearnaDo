import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from tavily import TavilyClient

load_dotenv()


def _get_gemini_api_key() -> str:
    """Prefer app config so webhook can use GEMINI_API_KEY or GOOGLE_API_KEY."""
    try:
        from app.config import settings
        key = settings.gemini_api_key or settings.google_api_key
        if key:
            return key
    except Exception:
        pass
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set")
    return key






class AgentState(TypedDict):
    user_question: str
    outline: list[str] | None
    user_approved: bool | None  # NEW: Whether user approved the outline
    retrieved_facts: dict[str, list[dict]] | None
    current_lesson_index: int | None  # NEW: Track which lesson to generate
    synthesized_lessons: list[dict[str, str]] | None
    current_lesson: dict[str, str] | None  # NEW: The lesson being delivered
    error: str | None


# class ModuleDesign(TypedDict):
#     title: str | None
#     content: str | None





def outline_generator(state: AgentState):
    """
    Generate a logical, hierarchical lesson plan (outline) with 4-6 key topics
    based on the user's learning goal.
    """
    user_question = state["user_question"]

    prompt = f"""You are an expert curriculum designer. A user wants to learn about: {user_question}

Generate a logical, hierarchical lesson plan (outline) with 4-6 key topics that will help them understand this subject comprehensively.

IMPORTANT: Respond ONLY with a valid JSON list of strings, nothing else. No markdown, no explanations, just the JSON array.

Example format: ["Introduction to Topic", "Core Concepts", "Practical Applications", "Advanced Techniques"]

Generate the outline now:"""

    try:
        tool_llm = get_tool_llm()
        response = tool_llm.invoke(prompt)

        # Extract JSON from response
        raw = response.content
        content = (raw if isinstance(raw, str) else str(raw)).strip()

        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        # Parse JSON
        outline = json.loads(content)

        if not isinstance(outline, list) or len(outline) < 1:
            return {"error": "Failed to generate a valid outline"}

        print(f"\n📋 Generated Outline ({len(outline)} topics):")
        for i, topic in enumerate(outline, 1):
            print(f"  {i}. {topic}")

        return {"outline": outline}

    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse outline JSON: {str(e)}"}
    except Exception as e:
        return {"error": f"Error generating outline: {str(e)}"}


def source_harvester(state: AgentState):
    """
    Search for verifiable facts for each topic in the outline using parallel Tavily searches.
    """
    outline = state.get("outline")

    if not outline:
        return {"error": "No outline available for harvesting sources"}

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    tavily_client = TavilyClient(tavily_api_key)

    all_facts = {}
    log_path = "websearch_logs.json"

    def search_topic(topic: str) -> tuple[str, dict]:
        """Search a single topic and return (topic, results)."""
        try:
            response = tavily_client.search(
                query=topic,
                search_depth="advanced",
                max_results=5
            )

            # Log the search
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "query": topic,
                "response": response
            }

            # Thread-safe logging
            if not os.path.exists(log_path):
                with open(log_path, "w") as f:
                    json.dump([log_entry], f, indent=4)
            else:
                with open(log_path, "r+") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = []
                    data.append(log_entry)
                    f.seek(0)
                    f.truncate()
                    json.dump(data, f, indent=4)

            return (topic, response)

        except Exception as e:
            print(f"  ⚠️  Error searching '{topic}': {str(e)}")
            return (topic, {"error": str(e)})

    print(f"\n🔍 Harvesting sources for {len(outline)} topics (parallel search)...")

    # Perform parallel searches using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(len(outline), 5)) as executor:
        future_to_topic = {executor.submit(search_topic, topic): topic for topic in outline}

        for i, future in enumerate(as_completed(future_to_topic), 1):
            topic, results = future.result()
            all_facts[topic] = results
            print(f"  ✓ [{i}/{len(outline)}] {topic}")

    return {"retrieved_facts": all_facts}


# Lazy initialization of LLMs
_tool_llm = None
_main_llm = None

def get_tool_llm():
    """Get or initialize the tool LLM instance."""
    global _tool_llm
    if _tool_llm is None:
        api_key = _get_gemini_api_key()
        # 2026 recommendation: Gemini 2.5 Flash for low-latency/high-volume tasks.
        _tool_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    return _tool_llm


def get_main_llm():
    """Get or initialize the main LLM instance."""
    global _main_llm
    if _main_llm is None:
        api_key = _get_gemini_api_key()
        _main_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    return _main_llm


def generate_outline_from_topic(topic: str) -> list[dict]:
    """
    Callable by the webhook bridge. Returns outline as list of dicts with title/description.
    """
    initial_state: AgentState = {
        "user_question": topic,
        "outline": None,
        "user_approved": None,
        "retrieved_facts": None,
        "current_lesson_index": None,
        "synthesized_lessons": None,
        "current_lesson": None,
        "error": None,
    }
    try:
        result = outline_generator(initial_state)
        if result.get("error"):
            raise RuntimeError(result["error"])
        outline_strings = result.get("outline") or []
        if outline_strings:
            return [{"title": t, "description": ""} for t in outline_strings]
    except Exception:
        # Fallback outline if Gemini is rate-limited / misconfigured.
        pass

    base = topic.strip() or "the topic"
    fallback = [
        f"Introduction to {base}",
        f"Key concepts in {base}",
        f"Common mistakes & pitfalls in {base}",
        f"Best practices for {base}",
        f"Real-world examples of {base}",
    ]
    return [{"title": t, "description": ""} for t in fallback]



def lesson_synthesizer(state: AgentState):
    """
    Generate ONE lesson at a time based on current_lesson_index.
    """
    outline = state.get("outline")
    retrieved_facts = state.get("retrieved_facts")
    current_index: int = state.get("current_lesson_index") or 0
    synthesized_lessons: list[dict[str, str]] = state.get("synthesized_lessons") or []

    if not outline or not retrieved_facts:
        return {"error": "Missing outline or retrieved facts for synthesis"}

    # Check if we've generated all lessons
    if current_index >= len(outline):
        return {"current_lesson": None}  # No more lessons

    # Get the current topic
    current_topic = outline[current_index]
    topic_facts = retrieved_facts.get(current_topic, {})

    # Extract source URLs and scores from Tavily results
    sources = []
    if isinstance(topic_facts, dict) and "results" in topic_facts:
        for result in topic_facts["results"]:
            if "url" in result:
                sources.append({
                    "title": result.get("title", "Unknown"),
                    "url": result["url"],
                    "score": result.get("score", 0.0)  # Relevance score (0.0-1.0)
                })

    prompt = f"""You are a helpful teacher. Your task is to write ONE engaging micro-learning lesson.

Topic: {current_topic}

Research material for this topic:
{json.dumps(topic_facts, indent=2)}

Your task:
1. Write 2-3 paragraphs explaining this topic clearly
2. Use ONLY the provided research material
3. Break down difficult technical terms
4. Make it digestible in 2-3 minutes
5. Include inline citations (e.g., "According to [Source Name]..." or "As noted by [Source]...")
6. Keep the tone engaging and educational

Return ONLY a JSON object in this EXACT format (no markdown, just the JSON):
{{
  "title": "Lesson title based on the topic",
  "content": "Lesson content with inline citations and clear explanations..."
}}

Generate the lesson now:"""

    try:
        tool_llm = get_tool_llm()  # Use Tool LLM for better quality
        response = tool_llm.invoke(prompt)

        # Extract JSON from response
        raw = response.content
        content = (raw if isinstance(raw, str) else str(raw)).strip()

        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        # Parse JSON
        lesson = json.loads(content)

        if not isinstance(lesson, dict) or "title" not in lesson or "content" not in lesson:
            return {"error": "Failed to generate valid lesson"}

        # Add sources to the lesson
        lesson["sources"] = sources
        lesson["topic"] = current_topic

        # Update synthesized_lessons list
        synthesized_lessons.append(lesson)

        return {
            "current_lesson": lesson,
            "synthesized_lessons": synthesized_lessons,
            "current_lesson_index": current_index + 1
        }

    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse lesson JSON: {str(e)}"}
    except Exception as e:
        return {"error": f"Error synthesizing lesson: {str(e)}"}


#================================================================================================================
# LEARNADO GRAPH: Interactive Outline-Search-Synthesize Pipeline
#================================================================================================================

def should_proceed_to_search(state: AgentState) -> str:
    """Conditional edge: Check if user approved the outline."""
    if state.get("user_approved"):
        return "source_harvester"
    return END  # User rejected, end the flow

def should_continue_lessons(state: AgentState) -> str:
    """Conditional edge: Check if there are more lessons to generate."""
    outline: list[str] = state.get("outline") or []
    current_index: int = state.get("current_lesson_index") or 0

    if current_index < len(outline):
        return "lesson_synthesizer"
    return END  # All lessons generated


builder = StateGraph(AgentState)

# Add nodes
builder.add_node("outline_generator", outline_generator)
builder.add_node("source_harvester", source_harvester)
builder.add_node("lesson_synthesizer", lesson_synthesizer)

# Define flow
builder.add_edge(START, "outline_generator")

# After outline, check if user approved (this will be handled externally)
# For now, we'll create a simpler flow for external control
builder.add_edge("outline_generator", END)  # Return to user for approval
# Note: The graph will be re-invoked after user approval

# Compile the graph
learnado_graph = builder.compile()


# Separate graph for the search + lesson generation phase
builder_phase2 = StateGraph(AgentState)
builder_phase2.add_node("source_harvester", source_harvester)
builder_phase2.add_node("lesson_synthesizer", lesson_synthesizer)

builder_phase2.add_edge(START, "source_harvester")
builder_phase2.add_edge("source_harvester", "lesson_synthesizer")
builder_phase2.add_edge("lesson_synthesizer", END)

search_and_lesson_graph = builder_phase2.compile()


#==========================================================================================


def run_cli():
    """Run the LearnaDo agent in simple CLI mode."""
    print("=" * 80)
    print("🎓 LearnaDo - Automatic Micro-Course Generator")
    print("=" * 80)

    user_question = input("\n💭 What would you like to learn about?\n> ")

    if not user_question.strip():
        print("❌ Please provide a valid learning topic.")
        return

    print("\n🚀 Generating your personalized micro-course...\n")

    # Run the pipeline
    final_state = learnado_graph.invoke({
        "user_question": user_question,
        "outline": None,
        "user_approved": None,
        "retrieved_facts": None,
        "current_lesson_index": None,
        "synthesized_lessons": None,
        "current_lesson": None,
        "error": None,
    })

    # Check for errors
    if final_state.get("error"):
        print(f"\n❌ Error: {final_state['error']}")
        return

    # Display the synthesized lessons
    synthesized_lessons = final_state.get("synthesized_lessons", [])

    if synthesized_lessons:
        print("\n" + "=" * 80)
        print("📚 YOUR MICRO-COURSE")
        print("=" * 80)

        for i, lesson in enumerate(synthesized_lessons, 1):
            print(f"\n{'─' * 80}")
            print(f"📖 Lesson {i}: {lesson.get('title', 'Untitled')}")
            print(f"{'─' * 80}")
            print(lesson.get('content', ''))

        print("\n" + "=" * 80)
        print(f"✅ Course complete! ({len(synthesized_lessons)} lessons)")
        print("=" * 80)
    else:
        print("\n⚠️  No lessons were generated.")


def synthesize_single_lesson(topic: str, lesson_title: str, description: str) -> str:
    """
    Callable by agent_bridge. Runs Tavily search + Gemini synthesis
    for a single lesson and returns the lesson content as a plain string.
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY") or ""
    try:
        from app.config import settings as _s
        tavily_api_key = _s.tavily_api_key or tavily_api_key
    except Exception:
        pass

    query = f"{topic} {lesson_title}"
    topic_facts: dict = {}
    try:
        client = TavilyClient(tavily_api_key)
        topic_facts = client.search(query=query, search_depth="advanced", max_results=5)
    except Exception as e:
        topic_facts = {"error": str(e)}

    prompt = f"""You are a helpful teacher writing a short WhatsApp-friendly micro-lesson.

Topic: {topic}
Lesson: {lesson_title}
{f"Context: {description}" if description else ""}

Research material:
{json.dumps(topic_facts, indent=2)}

Instructions:
1. Write 2-3 short paragraphs explaining this lesson clearly
2. Use plain, simple language — no jargon
3. Include one real-life example
4. End with ONE short question to check understanding (e.g. "Quick check: ...")
5. Keep total length under 800 characters (WhatsApp-friendly)

Return ONLY the lesson text — no JSON, no markdown headers, no extra formatting."""

    try:
        llm = get_tool_llm()
        response = llm.invoke(prompt)
        raw = response.content
        return (raw if isinstance(raw, str) else str(raw)).strip()
    except Exception as e:
        return (
            f"*{lesson_title}*\n\n"
            f"This lesson covers {lesson_title} as part of {topic}.\n\n"
            f"_(Full content unavailable right now: {e})_\n\n"
            "Quick check: What do you already know about this topic?"
        )


if __name__ == "__main__":
    # Run the CLI if this file is executed directly
    run_cli()
