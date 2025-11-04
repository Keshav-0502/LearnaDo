#!/usr/bin/env python3
"""
CLI interface to run the LearnaDo agent.
Provides an interactive command-line interface for step-by-step micro-course generation.
"""

import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from app.agent import learnado_graph, search_and_lesson_graph

console = Console()


def print_welcome():
    """Print welcome message."""
    welcome_text = """
# 🎓 Welcome to LearnaDo!

**Interactive Micro-Course Generator**

Tell me what you want to learn, and I'll:
1. 📋 Create a structured learning outline for your approval
2. 🔍 Research each topic with verified sources (after you approve)
3. ✍️  Deliver lessons one at a time with citations

Type 'exit' or 'quit' to end the session.
    """
    console.print(Panel(Markdown(welcome_text), border_style="bold green"))


def display_outline(outline):
    """Display the generated outline in a table."""
    table = Table(title="📋 Proposed Learning Outline", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=6)
    table.add_column("Topic", style="cyan")

    for i, topic in enumerate(outline, 1):
        table.add_row(str(i), topic)

    console.print("\n")
    console.print(table)
    console.print()


def display_lesson(lesson, lesson_num, total_lessons):
    """Display a single lesson with sources."""
    title = lesson.get('title', 'Untitled Lesson')
    content = lesson.get('content', 'No content available')
    sources = lesson.get('sources', [])

    # Display lesson content
    lesson_header = f"📖 Lesson {lesson_num}/{total_lessons}: {title}"
    console.print("\n")
    console.print(Panel(
        Markdown(content),
        title=lesson_header,
        border_style="cyan",
        padding=(1, 2)
    ))

    # Display sources with relevance scores (only show sources >= 0.7)
    if sources:
        # Filter sources by threshold
        filtered_sources = [s for s in sources if s.get('score', 0.0) >= 0.7]

        if filtered_sources:
            console.print("\n[bold]📚 Sources:[/bold]")
            for i, source in enumerate(filtered_sources, 1):
                score = source.get('score', 0.0)
                console.print(f"  {i}. [{source['title']}]({source['url']}) {score:.2f}")
                console.print(f"     [dim]{source['url']}[/dim]")
        else:
            # No sources met the 0.7 threshold
            console.print("\n[dim]📚 Sources: None met the 0.7 relevance threshold[/dim]")
    console.print()


def main():
    """Run the interactive agent CLI."""
    print_welcome()

    try:
        while True:
            # ========== PHASE 1: GET USER QUESTION ==========
            try:
                user_input = console.input("\n[bold green]💭 What would you like to learn about?[/bold green]\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n\n[yellow]Session ended. Goodbye! 👋[/yellow]")
                break

            if user_input.lower() in ["exit", "quit", "bye"]:
                console.print("\n[yellow]Thank you for using LearnaDo! Goodbye! 👋[/yellow]")
                break

            if not user_input:
                continue

            # ========== CHECK FOR META/IDENTITY QUESTIONS ==========
            # Quick rule-based check first: if query contains "about [topic]", it's learning
            import re

            # Check if query contains "about [something]" or "of [something]" (excluding "yourself", "you")
            has_topic_phrase = False
            if re.search(r'\babout\s+(?!yourself|you\b)\w+', user_input.lower()):
                has_topic_phrase = True

            if has_topic_phrase:
                # Query mentions "about [topic]" - it's learning, skip LLM classification
                is_identity = False
            else:
                # Use Flash LLM to intelligently detect if this is about the agent itself
                from app.agent import get_main_llm

                classifier_prompt = f"""You are an expert query classifier for an educational agent named LearnaDo.

Analyze the user's query and determine if it's asking about YOU (LearnaDo, the assistant/agent/system itself) or if it's a genuine request to LEARN about a topic.

User query: "{user_input}"

Classification rules:
1. IDENTITY queries ask about:
   - Who/what you are (identity, nature, purpose)
   - Your capabilities, features, or limitations without mentioning a specific topic
   - How you work or what you can do
   - Instructions to introduce yourself
   - Meta questions about the system itself

2. LEARNING queries ask to:
   - Learn, understand, or explain a topic
   - Get information about a subject (even if it's "AI" or "chatbots")
   - Questions about YOUR knowledge OF A SPECIFIC TOPIC (e.g., "what do you know about X" = asking to learn about X)
   - Create educational content
   - Any knowledge-seeking question about a domain or subject

CRITICAL RULE: If the query mentions "about [TOPIC]" or "of [TOPIC]", it's asking to LEARN about that topic, NOT asking about your capabilities.

Examples:
IDENTITY queries:
- "who are you?" -> IDENTITY
- "what can you do?" -> IDENTITY (no topic specified)
- "tell me about yourself" -> IDENTITY
- "introduce yourself" -> IDENTITY
- "what are your features?" -> IDENTITY
- "how do you work?" -> IDENTITY

LEARNING queries:
- "what is machine learning?" -> LEARNING
- "teach me about AI assistants" -> LEARNING
- "how do chatbots work?" -> LEARNING
- "explain quantum computing" -> LEARNING
- "what do I need to know about Python?" -> LEARNING
- "I want to learn about RAG" -> LEARNING
- "tell me about Graph-based RAG" -> LEARNING
- "what do you know about graph RAG?" -> LEARNING (asking to learn about graph RAG)
- "what do know graph RAG" -> LEARNING (grammatically broken but mentions topic "graph RAG")

Edge cases:
- "what do you know about X" = LEARNING (asking to learn about topic X)
- "what do you know" (no topic) = IDENTITY (asking about your knowledge base)
- "what should I learn" -> LEARNING (asking for guidance)
- Empty or single-word queries -> LEARNING (default to teaching)

Respond with ONLY one word: IDENTITY or LEARNING"""

                try:
                    classifier_llm = get_main_llm()
                    classification = classifier_llm.invoke(classifier_prompt).content.strip().upper()

                    # More robust classification check
                    is_identity = (
                        "IDENTITY" in classification or
                        classification == "IDENTITY" or
                        classification.startswith("IDENTITY")
                    )
                except Exception as e:
                    # If classification fails, proceed with normal flow (assume LEARNING)
                    console.print(f"[dim]Classification check skipped: {e}[/dim]")
                    is_identity = False

            # Handle identity queries
            if is_identity:
                console.print("\n[bold cyan]🎓 About LearnaDo:[/bold cyan]")
                console.print(Panel(
                    Markdown("""
I'm **LearnaDo**, an interactive micro-course generator designed to help you learn any topic effectively!

**What I do:**
- 📋 Create structured learning outlines tailored to your topic
- 🔍 Research each topic using verified web sources
- ✍️  Generate comprehensive lessons with citations
- 📚 Deliver content one lesson at a time for better retention

**How it works:**
1. You tell me what you want to learn
2. I propose a learning outline for your approval
3. After you approve, I research all topics in parallel
4. I deliver lessons one by one, with sources cited

Just ask me what you'd like to learn about, and I'll create a personalized course for you!
                    """),
                    title="🎓 LearnaDo",
                    border_style="cyan"
                ))
                continue

            # ========== PHASE 2: GENERATE OUTLINE ==========
            console.print(f"\n[bold cyan]🎯 Analyzing your topic and creating a learning plan...[/bold cyan]\n")

            try:
                # Step 1: Generate outline
                state = learnado_graph.invoke({"user_question": user_input})

                if state.get("error"):
                    console.print(f"\n[bold red]❌ Error:[/bold red] {state['error']}")
                    continue

                outline = state.get("outline")
                if not outline:
                    console.print("\n[yellow]⚠️  Failed to generate outline. Please try again.[/yellow]")
                    continue

                # Display outline
                display_outline(outline)

                # ========== PHASE 3: GET USER APPROVAL ==========
                approval = console.input("[bold yellow]Do you want to proceed with this outline? (yes/no):[/bold yellow] ").strip().lower()

                if approval not in ["yes", "y", "sure", "ok", "okay", "yeah", "yep"]:
                    console.print("\n[yellow]📝 Outline rejected. Let's try again![/yellow]")
                    continue

                # ========== PHASE 4: RESEARCH TOPICS (ONE TIME) ==========
                console.print(f"\n[bold cyan]🔍 Great! Researching all topics...[/bold cyan]\n")

                # Initialize state for phase 2
                state["current_lesson_index"] = 0
                state["synthesized_lessons"] = []

                # Run ONLY the search phase (not lesson generation yet)
                from app.agent import source_harvester
                search_result = source_harvester(state)

                if search_result.get("error"):
                    console.print(f"\n[bold red]❌ Error during research:[/bold red] {search_result['error']}")
                    continue

                # Update state with search results
                state.update(search_result)

                # ========== PHASE 5: DELIVER LESSONS ONE BY ONE ==========
                console.print(f"\n[bold cyan]✍️  Generating lesson 1...[/bold cyan]")

                # Generate first lesson
                from app.agent import lesson_synthesizer
                lesson_result = lesson_synthesizer(state)

                if lesson_result.get("error"):
                    console.print(f"\n[bold red]❌ Error:[/bold red] {lesson_result['error']}")
                    continue

                state.update(lesson_result)
                current_lesson = state.get("current_lesson")

                if not current_lesson:
                    console.print("\n[yellow]⚠️  No lesson was generated. Please try again.[/yellow]")
                    continue

                # Display the first lesson
                display_lesson(current_lesson, 1, len(outline))

                # Continue generating and displaying remaining lessons
                for i in range(2, len(outline) + 1):
                    # Ask if user wants the next lesson
                    next_input = console.input(f"[bold green]Ready for lesson {i}? (yes/no/exit):[/bold green] ").strip().lower()

                    if next_input in ["exit", "quit", "stop"]:
                        console.print("\n[yellow]📚 Course paused. You can start a new topic anytime![/yellow]")
                        break

                    if next_input not in ["yes", "y", "sure", "ok", "okay", "yeah", "yep", ""]:
                        console.print("\n[yellow]⏸️  Pausing here. Type a new topic to start fresh.[/yellow]")
                        break

                    console.print(f"\n[bold cyan]✍️  Generating lesson {i}...[/bold cyan]")

                    # Generate next lesson (using cached search results)
                    lesson_result = lesson_synthesizer(state)

                    if lesson_result.get("error"):
                        console.print(f"\n[bold red]❌ Error:[/bold red] {lesson_result['error']}")
                        break

                    state.update(lesson_result)
                    current_lesson = state.get("current_lesson")

                    if current_lesson:
                        display_lesson(current_lesson, i, len(outline))
                    else:
                        break

                # Course complete
                total_generated = len(state.get("synthesized_lessons", []))
                console.print(f"\n[bold green]🎉 Course section complete! You've learned {total_generated} topic(s).[/bold green]")
                console.print("[dim]Start a new topic or type 'exit' to quit.[/dim]\n")

            except Exception as e:
                console.print(f"\n[bold red]Error during generation:[/bold red] {str(e)}")
                console.print("[yellow]Please check your API keys in .env file[/yellow]")

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Session interrupted. Goodbye! 👋[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Fatal error:[/bold red] {str(e)}")
        console.print("\n[yellow]Please check your API keys in .env file[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    main()
