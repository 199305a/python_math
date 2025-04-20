# 在AI代理中引入内存  短期记忆


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


# 短期记忆
model = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, base_url=base_url)


# def call_llm(state: MessagesState):
#     """
#     Calls a language model with the last message from the message graph state.

#     Args:
#         state (MessageGraph): The current message graph state containing message history

#     Returns:
#         dict: A dictionary containing the model's response message
#     """
#     messages = state["messages"]
#     response = model.invoke(messages[-1].content)

#     return {"messages": [response]}


# app = workflow.compile()


# # 持续消息
# def interact_with_agent():
#     while True:
#         user_input = input("Human: ")
#         if user_input.lower() in ["exit", "quit"]:
#             print("对话结束")
#             break

#         input_message = {"messages": [("human", user_input)]}

#         for chunk in app.stream(input_message, stream_mode="values"):
#             chunk["messages"][-1].pretty_print()


# interact_with_agent()


# 使用短期记忆增强代理
from langgraph.checkpoint.memory import MemorySaver


def call_llm(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)

    return {"messages": [response]}


workflow = StateGraph(MessagesState)

workflow.add_node("call_llm", call_llm)

workflow.add_edge(START, "call_llm")
workflow.add_edge("call_llm", END)

checkpointer = MemorySaver()


app_with_memory = workflow.compile(checkpointer=checkpointer)


# 持续消息
def interact_with_agent_with_memory():
    # 必须用数字
    thread_id = 49196
    while True:
        user_input = input("Human: ")
        if user_input.lower() in ["exit", "quit"]:
            print("对话结束")
            break

        input_message = {"messages": [("human", user_input)]}
        config = {"configurable": {"thread_id": thread_id}}

        for chunk in app_with_memory.stream(
            input_message, config=config, stream_mode="values"
        ):
            chunk["messages"][-1].pretty_print()


# interact_with_agent_with_memory()


# 跨会话内存持久化


def interact_with_agent_across_sessions():
    while True:
        thread_id = input("请输入线程ID：")
        if thread_id.lower() in ["exit", "quit"]:
            print("对话结束")
            break
        if thread_id.lower() == "new":
            import os

            thread_id = os.urandom(4).hex()
        while True:
            user_input = input("Human: ")
            if user_input.lower() in ["exit", "quit"]:
                print("对话结束")
                break
            input_message = {"messages": [("human", user_input)]}
            config = {"configurable": {"thread_id": thread_id}}
            for chunk in app_with_memory.stream(
                input_message, config=config, stream_mode="values"
            ):
                # 这里假设chunk["messages"][-1] 有 pretty_print 方法
                chunk["messages"][-1].pretty_print()


# interact_with_agent_across_sessions()

# 持久化的内存存储

from langgraph.store.memory import InMemoryStore


import uuid

in_memory_store = InMemoryStore()


def store_user_info(state: MessagesState, config, *, store=in_memory_store):
    user_id = config["configurable"]["user_id"]

    namespace = (user_id, "memories")

    memory_id = str(uuid.uuid4())
    memory = {"user_name": state["user_name"]}
    store.put(namespace, memory_id, memory)
    return {"messages": ["用户信息已保存"]}


def retrieve_user_info(state: MessagesState, config, *, store=in_memory_store):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")

    memorys = store.search(namespace)
    if memorys:
        info = f"Hello {memorys[-1].value['user_name']} 欢迎回来"
    else:
        info = "我还没有关于你的任何信息"
    return {"messages": [info]}


def call_model(state: MessagesState, config):
    last_message = state["messages"][-1].content.lower()
    if "记住我的名字" in last_message:
        user_name = last_message.split("记住我的名字是")[-1].strip()
        state["user_name"] = user_name
        return store_user_info(state, config)
    if "我的名字叫" in last_message:
        return retrieve_user_info(state, config)

    return {"messages": ["我不知道你在说什么"]}


workflow = StateGraph(MessagesState)


workflow.add_node("call_model", call_model)

workflow.add_edge(START, "call_model")
workflow.add_edge("call_model", END)


app_with_memory = workflow.compile(checkpointer=MemorySaver(), store=in_memory_store)


def simulate_sessions():
    config = {"configurable": {"user_id": "123456789", "thread_id": "111"}}
    input_message = {"messages": [{"type": "user", "content": "记住我的名字是张三"}]}
    for chunk in app_with_memory.stream(
        input_message, config=config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    config = {"configurable": {"user_id": "123456789", "thread_id": "222"}}
    input_message = {"messages": [{"type": "user", "content": "我的名字叫什么"}]}
    for chunk in app_with_memory.stream(
        input_message, config=config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()


# simulate_sessions()


# 捕获检查点

from typing import TypedDict


class State(TypedDict):
    foo: str
    bar: list[str]


def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}


def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}


workflow = StateGraph(State)
workflow.add_node("node_a", node_a)
workflow.add_node("node_b", node_b)

workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

# checkpointer = MemorySaver()

# graph = workflow.compile(checkpointer=checkpointer)

# config = {"configurable": {"thread_id": "123456789"}}

# graph.invoke({"foo": "", "bar": []}, config=config)


# # 获取最新状态
# latest_state = graph.get_state(config=config)


# print("State Snapshot:", latest_state.values)

# 获取历史状态

# state_history = graph.get_state_history(config=config)

# for state in state_history:
#     print(state.values)


from langgraph.store.memory import InMemoryStore
in_memory_store = InMemoryStore()

user_id = "123456789"
namespace_for_memory = (user_id, "memories")
memory_id = str(uuid.uuid4())
memory = {"food_perference":"我爱吃牛肉面"}
in_memory_store.put(namespace_for_memory, memory_id, memory)

memories = in_memory_store.search(namespace_for_memory)

print(memories[-1].dict)

#一起使用

from langchain.schema.runnable.config import RunnableConfig
from langgraph.store.base import BaseStore
def update_memory(state:MessagesState,config:RunnableConfig,*,store:BaseStore):
     user_id = config["configurable"]["user_id"]
     namespace_for_memory = (user_id, "memories")
     memory_id = str(uuid.uuid4())
     store.put(namespace_for_memory, memory_id, {"food_perference":"我爱吃披萨"})
     memories = store.search(namespace_for_memory)
     
     return {"messages":[f"我记得你喜欢吃{memories[-1].value['food_perference']}"]}
 

