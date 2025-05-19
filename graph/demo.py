from typing import Annotated

from typing_extensions import TypedDict
import os
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
from IPython.display import Image, display
from langchain_community.tools import DuckDuckGoSearchRun

from graph.utils import DuckDuckGoSearchTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.graph import StateGraph, START, END
duck_tool = DuckDuckGoSearchTool(max_results=6)


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

llm = init_chat_model("google_genai:gemini-2.0-flash")

tools = [duck_tool]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools=[duck_tool])
graph_builder.add_node("tools", tool_node)

# def chatbot(state: State):
#     # 演示：如果用户消息以 "search:" 开头，则调用duckduckgo工具
#     last_msg = state["messages"][-1].content
#     if last_msg.startswith("search:"):
#         query = last_msg[len("search:") :].strip()
#         search_result = duck_tool(query)
#         return {"messages": [f"搜索结果: {search_result}"]}
#     # 否则走原有llm
#     return {"messages": [llm.invoke(state["messages"])]}


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()











def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()



graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
try:
    img_data = graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(img_data)
    display(Image(img_data))
except Exception:
    # This requires some extra dependencies and is optional
    pass





def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except Exception as e:
        # fallback if input() is not available
        print(f"发生异常: {e}")  # 输出异常日志
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
