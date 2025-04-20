from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI


from langgraph.graph import MessagesState, Graph, START, END, StateGraph

from langgraph.checkpoint.memory import MemorySaver

load_dotenv(".env")

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")


if api_key is None:
    print("Please set the OPENAI_API_KEY environment variable.")
else:
    print(api_key)
    print(base_url)


model = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, base_url=base_url)


def call_llm(state: MessagesState):
    """
    Calls a language model with the last message from the message graph state."""

    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


builder = StateGraph(input=MessagesState, output=MessagesState).add_node(
    "agent", call_llm
)

builder.add_edge(START, "agent")
builder.add_edge("agent", END)

# checkpointer = MemorySaver()
# app = builder.compile(checkpointer=checkpointer)


def interact_with_agent():
    thread_id =  1111
    while True:
        user_input = input("Human: ")
        if user_input.lower() in ["exit", "quit"]:
            print("对话结束")
            break

        input_message = {"messages": [("human", user_input)]}
        config = {"configurable": {"thread_id": thread_id}}
        for chunk in app.stream(input_message, config=config, stream_mode="values"):
            print(chunk["messages"][-1].content)


# interact_with_agent()


from langgraph.store.memory import InMemoryStore
import uuid
in_memory_store = InMemoryStore()

def store_user_info(state: MessagesState, config, *, store=in_memory_store):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id,"memories")
    memory_id = str(uuid.uuid4())
    memory  = {"user_name":state["user_name"]}
    store.put(namespace, memory_id, memory)
    return {"messages": ["用户信息已保存"]}


def retrieve_user_info(state: MessagesState, config, *, store=in_memory_store):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id,"memories")
   
    memories  = store.search(namespace)
    if memories:
        info = f"Hello {memories[-1].value['user_name']} back"
    else:
        info = "没有关于他的消息"    
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

builder.add_node("call_model", call_model)
builder.add_edge(START, "call_model")
builder.add_edge("call_model", END)

app_with_memory = builder.compile(checkpointer=MemorySaver(), store=in_memory_store)

def simulate_sessions():

    config = {"configurable": {"user_id": "123456789","thread_id": "111"}}

    input_message = {"messages": [("human", "记住我的名字是小崔")]}
    for chunk in app_with_memory.stream(
        input_message, config=config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    config = {"configurable": {"user_id": "123456789","thread_id": "222"}}

    input_message = {"messages": [("human", "我的名字叫什么")]}
    for chunk in app_with_memory.stream(
        input_message, config=config, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

simulate_sessions()        
