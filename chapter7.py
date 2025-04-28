import asyncio
from chapter6 import (
    model,
    MessagesState,
    StateGraph,
    ToolNode,
    START,
    END,
    tools_condition,
)


def multiply(x: int, y: int) -> int:
    """
    Multiply two integers.

    Args:
        x (int): First integer number
        y (int): Second integer number

    Returns:
        int: Product of x and y
    """
    "两个数相乘"
    return x * y


llm_with_tools = model.bind_tools([multiply])


def tool_calling_llm(state: MessagesState):

    messages = state["messages"]
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


builder = StateGraph(MessagesState)

builder.add_node("tool_calling_llm", tool_calling_llm)

builder.add_node("tools", ToolNode([multiply]))


builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", END)


graph = builder.compile()


def simulate():

    user_input = {"messages": [("human", "你能把3 乘以 5 吗")]}

    result = graph.invoke(user_input)
    return result["messages"][-1].pretty_print()


# print(simulate())

# 简单流


def weather_node(state: MessagesState):

    return {"messages": ["天气晴朗，气温为25c"]}


def calculator_node(state: MessagesState):
    user_query = state["messages"][-1].content.lower()

    if "add" in user_query:
        numbers = [int(s) for s in user_query.split() if s.isdigit()]
        result = sum(numbers)
        return {"messages": [f"结果是{result}"]}

    return {"messages": ["我只会做加法"]}


def default_node(state: MessagesState):
    return {"messages": ["我听不懂你说什么"]}


def routing_function(state: MessagesState):
    last_message = state["messages"][-1].content.lower()
    if "天气" in last_message:
        return "weather_node"
    elif "加" in last_message:
        return "calculator_node"
    else:
        return "default_node"


builder = StateGraph(MessagesState)
builder.add_node("routing_function", routing_function)
builder.add_node("weather_node", weather_node)
builder.add_node("calculator_node", calculator_node)
builder.add_node("default_node", default_node)

builder.add_conditional_edges(START, routing_function)
# builder.add_conditional_edges("routing_function", tools_condition)
builder.add_edge("weather_node", END)
builder.add_edge("calculator_node", END)
builder.add_edge("default_node", END)

app = builder.compile()


def simulate_interactive():
    while True:
        user_input = input("Human: ")
        if user_input.lower() in ["exit", "quit"]:
            print("对话结束")
            break
        input_message = {"messages": [("human", user_input)]}
        for chunk in app.stream(input_message, stream_mode="values"):
            chunk["messages"][-1].pretty_print()


# simulate_interactive()

# 完整状态流式处理
import operator

from typing import Annotated, TypedDict


class State(TypedDict):
    messages: Annotated[list, operator.add]


def weather_node(state: State):
    return {"messages": ["天气晴朗，气温为25c"]}


def calculator_node(state: State):
    return {"messages": ["我只会做加法"]}


builder = StateGraph(State)

builder.add_node("weather_node", weather_node)
builder.add_node("calculator_node", calculator_node)
builder.add_edge(START, "weather_node")
builder.add_edge("weather_node", "calculator_node")
builder.add_edge("calculator_node", END)

app = builder.compile()


def simulate_interactive():

    user_input = input("Human: ")

    input_message = {"messages": [("human", user_input)]}
    ##values 累加
    for chunk in app.stream(input_message, stream_mode="updates"):
        print(chunk)


# simulate_interactive()

# LLM令牌流式处理
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

from langchain_core.messages import HumanMessage, AIMessageChunk

load_dotenv(".env")

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

if api_key is None:
    print("Please set the OPENAI_API_KEY environment variable.")

else:
    print(api_key)
    print(base_url)


class State(TypedDict):
    messages: Annotated[list, add_messages]


# 短期记忆
model = ChatOpenAI(
    model="gpt-3.5-turbo", api_key=api_key, base_url=base_url, streaming=False
)


async def call_llm(state: State):
    """
    Calls a language model with the last message from the message graph state.

    Args:
        state (MessageGraph): The current message graph state containing message history

    Returns:
        dict: A dictionary containing the model's response message
    """
    messages = state["messages"]
    response = await model.ainvoke(messages)

    return {"messages": [response]}


workflow = StateGraph(State)
workflow.add_node("call_llm", call_llm)
workflow.add_edge(START, "call_llm")
workflow.add_edge("call_llm", END)
app = workflow.compile()


async def simulate_interactive():

    input_message = {"messages": [HumanMessage(content="给我讲一个很长的故事")]}
    first = True
    ##values 累加
    async for msg, metadata in app.astream(
        input_message, stream_mode=["messages", "updates"]
    ):
        # if msg.content and not isinstance(msg, HumanMessage):
        #     print(msg.content, end="|", flush=True)
        # if isinstance(msg, AIMessageChunk):
        #     if first:
        #         gathered = msg
        #         first = False
        #     else:
        #         gathered = gathered + msg

        #     if msg.tool_call_chunks:
        #         print(gathered.tool_calls)
        if isinstance(metadata, dict) and "call_llm" in metadata:
            ai_message = metadata["call_llm"]["messages"][0]
            if ai_message.content:
                print(ai_message.content, end="|", flush=True)


# asyncio.run(simulate_interactive())


# 流式处理自定义数据
from time import sleep
from langgraph.types import StreamWriter


def long_running_node(state: MessagesState, writer: StreamWriter):
    for i in range(1, 6):
        sleep(1)
        writer({"progress": [f"正在进行第{i}步"]})

    return {"messages": ["任务完成"]}


workflow = StateGraph(MessagesState)
workflow.add_node("long_running_node", long_running_node)
workflow.add_edge(START, "long_running_node")
workflow.add_edge("long_running_node", END)
app = workflow.compile()


def simulate_interactive():
    input_message = {"messages": [("human", "执行长时间运行的任务")]}
    for chunk in app.stream(input_message, stream_mode=["custom", "updates"]):
        if "progress" in chunk[-1]:
            print(chunk[-1]["progress"][-1])
        else:
            print(chunk[-1])


# simulate_interactive()


# 为不支持的模型禁用流

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm_streaming = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=api_key,
    base_url=base_url,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

llm_no_streaming = ChatOpenAI(
    model="gpt-3.5-turbo", api_key=api_key, base_url=base_url, streaming=False
)


def create_graph(llm):
    workflow = StateGraph(MessagesState)

    def chat_box(state: MessagesState):
        messages = state["messages"]
        if not isinstance(messages, list):
            messages = [messages]

        return {"messages": llm.invoke(messages)}

    workflow.add_node("chat_box", chat_box)
    workflow.add_node("call_llm", call_llm)
    workflow.add_edge(START, "chat_box")

    workflow.add_edge("chat_box", END)
    app = workflow.compile()
    return app


input = {"messages": [{"role": "user", "content": "你能给我讲个笑话吗？"}]}

# graph_streaming = create_graph(llm_streaming)
# for output in graph_streaming.stream(input):
#     if isinstance(output, dict) and "chat_box" in output:
#       pass

# print("禁用流式处理")
# graph_no_streaming = create_graph(llm_no_streaming)
# for output in graph_no_streaming.stream(input):
#     if isinstance(output, dict) and "chat_box" in output:
#        message = output["chat_box"]["messages"]
#        print(message.content,end="|",flush=True)


import requests


def live_weather_node(state):
    city = "上海"
    url = "http://t.weather.itboy.net/api/weather/city/101010100"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather = data["data"]["forecast"][0]["type"]
        temperature = data["data"]["wendu"]
        return {"messages": [f"{city}的天气是{weather}，气温是{temperature}°C"]}

    else:
        return {"messages": ["无法获取天气数据"]}


builder = StateGraph(MessagesState)
builder.add_node("live_weather_node", live_weather_node)
builder.add_edge(START, "live_weather_node")
builder.add_edge("live_weather_node", END)
app = builder.compile()


def simulate_interactive():
    input_message = {"messages": [("human", "上海的天气怎么样？")]}
    for chunk in app.stream(input_message, stream_mode="updates"):
        if "messages" in chunk["live_weather_node"]:
            print(
                chunk["live_weather_node"]["messages"][-1], end="|", flush=True
            )
        else:
            print(chunk[-1])


simulate_interactive()
