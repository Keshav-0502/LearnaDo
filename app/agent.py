from tavily import TavilyClient
import json
import os
from datetime import datetime
from typing import Annotated, Any, TypedDict
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode






class AgentState(TypedDict):
    # messages: Annotated[list, add_messages]
    messages: list[dict[str,str]]
    user_question: str | None
    websearch_results: dict[str, Any] | None
    final_answer: str | None
    modules: str | None
    last_topic: str | None 


# class ModuleDesign(TypedDict):
#     title: str | None
#     content: str | None





def websearch(state: AgentState):
    """
    This websearch tool searches the internet for refining responses
    to the user query with accurate and up-to-date information.
    """

    tavily_client = TavilyClient("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")

    user_query = state.get("user_question","")

    response = tavily_client.search(
        query=user_query,
        search_depth="advanced",
        max_results=5
    )

    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": user_query,
        "response": response
    }

    log_path = "websearch_logs.json"

    
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            json.dump([log_entry], f, indent=4)
    else:
        with open(log_path, "r+") as f:
            data = json.load(f)
            data.append(log_entry)
            f.seek(0)
            json.dump(data, f, indent=4)

    return {"websearch_results": response}




tool_llm = ChatGoogleGenerativeAI(model = "gemini-2.5-pro", api_key="AIzaSyCDQjzDQFszCg75IvHpgu3xgbsjqXOHVRw")
main_llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash", api_key="AIzaSyCDQjzDQFszCg75IvHpgu3xgbsjqXOHVRw")



def module_llm(state: AgentState):

    """This tool is to be used for designing modules for the user with the help of an llm. Only to be called when the user agrees to module designing"""



    prompt = f"""You are a helpful teacher who creates modules for people to learn all kinds of topics in simple and understandable terms.
    
    You key responsibilities are-
    
    - Always ask user if they want a learning module collection or no
    - Make very small modules suitable for micro-learning and serve them to the user one by one
    - If no modules are requested, greet user and END.
    - Always use the structured websearch results to create modules and cite any and all information that you use in making the modules.
    - Breakdown difficult technical terminologies to make them easier to digest.
    - The modules will be served on whatsapp text, so manage content of each module accordingly so that its not overwhelming to read from a text in one go.


    The module structure must look like this: 

    {{
    "title": "...",
    "content": "..."
    }}

    

    Topic: {state.get("last_topic", state.get("user_question", ""))}
    Websearch results: {state.get("websearch_results")}
    
    """

    # tool_llm_with_structure = tool_llm.with_structured_output(ModuleDesign)

    response = tool_llm.invoke(prompt)

    module_response = response.content

    return {"modules": module_response, "messages": [{"role": "module_llm", "content": module_response}]}


tools = [module_llm]

def assistant(state: AgentState):
    """Main assistant that decides when to call the module LLM."""

    assistant_prompt = f"""
    You are LearnaDo, a learning companion and explainer bot.

    The user may ask a question or request a topic explanation.
    - If the user asks a question, explain it clearly using websearch results.
    - After answering, politely ask if they’d like a structured learning module collection about that topic.
    - If the user responds in any positive way (e.g. “yes”, “sure”, “okay”, “go ahead”, “make it”, “yup”, etc.), call the tool `module_llm` with:
        {{
          "topic": "<the relevant topic>",
          "user_request": "<user's last message>"
        }}
    - If the user declines, just continue the chat normally.
    - Do NOT ask again unless user re-requests modules.
    - Keep answers concise (3-4 lines max).

    The user's query: {state["user_question"]}
    The websearch results: {state["websearch_results"]}
    """
    

    llm_with_tools = main_llm.bind_tools(tools)
    assistant_response = llm_with_tools.invoke(assistant_prompt)

    # result = {
    #     "final_answer": assistant_response.content,
    #     "messages": [{"role": "assistant", "content": assistant_response.content}],
    # }

    topic_guess = state.get("user_question")
    if state.get("websearch_results"):
        topic_guess = state["websearch_results"].get("query", state.get("user_question"))


    result = {
        "final_answer": assistant_response.content,
        "messages": [{"role": "assistant", "content": assistant_response.content}],
        "last_topic": topic_guess,  # store last discussed topic
    }


    if getattr(assistant_response, "tool_calls", None):
        result["tool_calls"] = assistant_response.tool_calls
        print("\nTool trigger detected:", assistant_response.tool_calls) 

    return result





#================================================================================================================



builder = StateGraph(AgentState)

builder.add_node("websearch", websearch)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))


builder.add_edge(START, "websearch")
builder.add_edge("websearch", "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

react_graph = builder.compile()


#==========================================================================================



initial_state = {
    "user_question": "Hi",
    "messages": []
}

main_state = initial_state

print("Running the agent...")


while True:

    # for step in react_graph.stream(initial_state):
    #     for node_name, _ in step.items():
    #         print(f"\nNode executed: {node_name}")
    #         # print("Output:", output)










    # final_state = react_graph.invoke(initial_state)

    # if "final_answer" in main_state:
    #     print("\nLearnaDo: ", main_state["final_answer"][0]["text"])


    if "final_answer" in main_state:
        fa = main_state["final_answer"]
        if isinstance(fa, list) and len(fa) > 0 and isinstance(fa[0], dict):
            print("\nLearnaDo:", fa[0].get("text", ""))
        else:
            print("\nLearnaDo:", fa)






    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    main_state = react_graph.invoke({
        "user_question": user_input,
        "messages": main_state.get("messages", []) + [{"role": "user", "content": user_input}]
    })

    # print("\nFinal Answer:")
    # print(final_state["final_answer"][-1]["text"])