from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END, MessagesState

load_dotenv(".env")

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

if api_key is None:
    print("Please set the OPENAI_API_KEY environment variable.")

else:
    print(api_key)
    print(base_url)


from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """获取特定位置的天气"""
    weather_data = {"Beijing": "晴天", "上海": "多云", "广州": "大雨"}
    return weather_data.get(city, "未知城市")


from langgraph.prebuilt import ToolNode

tool_node = ToolNode([get_weather])
model = ChatOpenAI(
    model="gpt-3.5-turbo", api_key=api_key, base_url=base_url
).bind_tools([get_weather])

def call_llm(state: MessagesState):
    """
    Calls a language model with the last message from the message graph state.

    Args:
        state (MessageGraph): The current message graph state containing message history

    Returns:
        dict: A dictionary containing the model's response message
    """
    messages = state["messages"]
    response = model.invoke(messages[-1].content)

    if response.tool_calls:
        tool_result = tool_node.invoke({"messages": [response]})
        tool_message = tool_result["messages"][-1].content
        response.content += f"\n工具调用 {tool_message}"
    return {"messages": [response]}


workflow = StateGraph(MessagesState)

workflow.add_node("call_llm", call_llm)

workflow.add_edge(START, "call_llm")
workflow.add_edge("call_llm", END)

app = workflow.compile()

# # 单一消息
# input_message = {
#     "messages":[("human","北京的天气怎么样")]
# }

# for chunk in app.stream(input_message,stream_mode="values"):
#     chunk["messages"][-1].pretty_print()


# 持续消息
def interact_with_agent():
    while True:
        user_input = input("Human: ")
        if user_input.lower() in ["exit", "quit"]:
            print("对话结束")
            break

        input_message = {"messages": [("human", user_input)]}

        for chunk in app.stream(input_message, stream_mode="values"):
            chunk["messages"][-1].pretty_print()


# interact_with_agent()

# ToolNode
@tool
def get_user_profile(user_id: str) -> str:
    """按用户id获取用户信息"""
    user_data = {
        "101":  {"name": "张三","age": 18, "gender": "男","location": "北京"},
        "102": {"name": "李四","age": 20, "gender": "女","location": "上海"},
        "103": {"name": "王五","age": 22, "gender": "男","location": "广州"}
        
    }
    return user_data.get(user_id, "未知用户")

tools = [get_user_profile]

tool_node = ToolNode(tools)


from langchain_core.messages import AIMessage

#AI代理工具
message_with_tool_call = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "get_user_profile",
            "args": {"user_id": "101"},
            "id":"tool_call_id",
            "type":"tool_call"
        }
    ],
)
#设置stategraph
state = {
    "messages": [message_with_tool_call]
}
#使用toolnode调用
result = tool_node.invoke(state)
print(result)
