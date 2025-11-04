# LearnaDo CLI Usage Guide

## Running the Agent

You have multiple ways to run the LearnaDo learning agent:

### Method 1: Using the run_agent.py script (Recommended)
```bash
uv run python run_agent.py
```

This provides a beautiful, Rich-formatted CLI interface with:
- ✨ Colored output and panels
- 📚 Markdown rendering for learning modules
- 🔄 Loading indicators
- 💬 Interactive chat experience

### Method 2: Using the installed command
```bash
uv run learnado
```

This runs the same script as Method 1.

### Method 3: Direct execution
```bash
uv run python -m app.agent
```

This runs the original, simpler CLI interface.

## How It Works

The agent automatically generates a complete micro-course through a three-phase pipeline:
1. **📋 Outline Generation**: Creates a structured 4-6 topic learning plan
2. **🔍 Source Harvesting**: Performs parallel web searches for each topic
3. **✍️ Lesson Synthesis**: Combines research into coherent, cited learning modules

No permission needed - just ask what you want to learn!

## Example Session

```
🎓 Welcome to LearnaDo!

Automatic Micro-Course Generator

Simply tell me what you want to learn, and I'll automatically:
1. 📋 Create a structured learning outline
2. 🔍 Research each topic with verified sources
3. ✍️ Synthesize a complete micro-course

💭 What would you like to learn about?
> quantum computing

🚀 Generating your personalized micro-course...

📋 Generated Outline (5 topics):
  1. Introduction to Quantum Computing
  2. Quantum Bits and Superposition
  3. Quantum Entanglement
  4. Quantum Algorithms
  5. Real-World Applications

🔍 Harvesting sources for 5 topics (parallel search)...
  ✓ [1/5] Introduction to Quantum Computing
  ✓ [2/5] Quantum Bits and Superposition
  ...

✍️ Synthesizing lessons...
  ✓ Successfully synthesized 5 lessons

================================================================================
                    📚 YOUR PERSONALIZED MICRO-COURSE
================================================================================

┌─ 📖 Lesson 1: Introduction to Quantum Computing ──────────────────────┐
│ Quantum computing represents a revolutionary approach to computation...│
│ [Full lesson content with citations]                                   │
└────────────────────────────────────────────────────────────────────────┘

[... additional lessons ...]

================================================================================
                        ✅ Course complete! (5 lessons)
================================================================================
```

## Commands

- Type your question or topic to learn about
- Type `exit`, `quit`, or `bye` to end the session
- Press `Ctrl+C` to interrupt at any time

## Features

- 📋 **Intelligent Outlining**: Automatically creates structured learning plans (4-6 topics)
- 🔍 **Parallel Web Research**: Concurrent Tavily searches for faster information gathering
- 🤖 **Dual LLM Architecture**:
  - Gemini 2.5 Pro for curriculum design and synthesis
  - Gemini 2.5 Flash for quick lesson generation
- 📝 **Automatic Course Generation**: Complete micro-courses without manual intervention
- 📚 **Cited Content**: All lessons reference source material
- 🎨 **Beautiful Output**: Rich-formatted panels and markdown rendering

## Configuration

Make sure your `.env` file has the required API keys:
```env
GOOGLE_API_KEY=your-google-api-key
TAVILY_API_KEY=your-tavily-api-key
```

## Troubleshooting

**Error: "GOOGLE_API_KEY not found"**
- Ensure your `.env` file exists in the project root
- Check that `GOOGLE_API_KEY` is set correctly

**Error: SSL certificate verification failed**
- This may happen with Whisper model downloads
- The agent CLI doesn't use Whisper, so this shouldn't affect you
- If it persists, check your Python SSL certificates

**Slow responses**
- The agent performs web searches for each query
- First response includes model initialization time
- Subsequent responses should be faster
