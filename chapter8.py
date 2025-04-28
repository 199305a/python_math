import asyncio
import json
from chapter6 import (
    model,
    MessagesState,
    StateGraph,
    ToolNode,
    START,
    END,
    tools_condition,
    TypedDict,
)

# ReAct
from langgraph.prebuilt import create_react_agent


def add(x: int, y: int) -> int:
    """
    Add two integers.
    """

    return x + y


def multiply(x: int, y: int) -> int:
    """
    Multiply two integers.
    """

    return x * y


tools = [add, multiply]

llm = model.bind_tools(tools)

graph = create_react_agent(llm, tools)
input_message = {"messages": [("human", "你能把3  加 2 乘以 5 吗")]}

# message = graph.invoke(input_message)
# for chunk in message["messages"]:
#     print(chunk.content)

# 使用单线程内存进行产品查询
from langgraph.checkpoint.memory import MemorySaver


def product_info(product_name: str) -> str:
    """
    Get product information from a product name.
    """
    product_catalog = {
        "iphone": "苹果手机有芯片",
        "ipad": "苹果平板",
        "macbook": "苹果笔记本",
        "airpods": "苹果耳机",
    }
    return product_catalog.get(product_name, "没有找到该产品")


# checkpointer = MemorySaver()


# graph = create_react_agent(model=model, tools=[product_info], checkpointer=checkpointer)
# config = {"configurable": {"thread_id": 1234}}


# input_message = {"messages": [("human", "你能告诉我 iphone 手机的信息")]}

# message = graph.invoke(input_message, config=config)
# for chunk in message["messages"]:
#     chunk.pretty_print()

# input_message2 = {"messages": [("human", "告诉我更多关于 iphone 的信息吗")]}

# message2 = graph.invoke(input_message2, config=config)

# for chunk2 in message2["messages"]:
#     chunk2.pretty_print()


def check_stock(product_name: str) -> str:
    """
    Check stock of a product.
    """
    stock_catalog = {
        "iphone": "有货",
        "ipad": "有货",
        "macbook": "缺货",
        "airpods": "有货",
    }

    return stock_catalog.get(product_name, "没有找到该产品")


# checkpointer = MemorySaver()

# tools = [product_info, check_stock]
# graph = create_react_agent(model=model, tools=tools, checkpointer=checkpointer)

# config = {"configurable": {"thread_id": 1234}}
# input_message = {"messages": [("human", "你能告诉我 iphone 手机的信息")]}
# message = graph.invoke(input_message, config=config)
# for chunk in message["messages"]:
#     chunk.pretty_print()

# input_message2 = {"messages": [("human", "告诉我更多关于 iphone  库存")]}
# message2 = graph.invoke(input_message2, config=config)
# for chunk2 in message2["messages"]:
#     chunk2.pretty_print()


class ReActAgentState(TypedDict):
    message: str
    action: str
    sub_action: str
    memory: str


def reasoning_node(state: ReActAgentState):
    query = state["message"]
    past_input = state["memory"]
    if "天气" in query:
        return {"action": "fetch_weather"}

    elif "新闻" in query:
        return {"action": "fetch_news"}

    elif "推荐" in query:
        if past_input.get("爱好") == "电影":
            return {"action": "recommendation", "sub_action": "movie"}
        else:
            return {"action": "recommendation", "sub_action": "book"}

    else:
        return {"action": "unknown"}


def weather_subgraph_node(state: ReActAgentState):
    return {"message": "天气晴朗，气温为25c"}


def new_subgraph_node(state: ReActAgentState):
    """
    Creates a new subgraph node with news content.
    
    Args:
        state (ReActAgentState): The current state of the ReAct agent.
    
    Returns:
        dict: A dictionary containing a message with news content.
    """
    return {"message": "今天的新闻是：..."}


def generate_recommendation_node(state: ReActAgentState):
    return {"message": "推荐的书籍是：一般书籍"}
def movie_recommendation_node(state: ReActAgentState):
    return {"message": "推荐的书籍是：电影"}

def update_memory_node(state: ReActAgentState):

    if "推荐" in state:
        state["memory"] = {"爱好": "电影"}
    return state


def recommendation_subgraph_node(state: ReActAgentState):
    if state.get("sub_action") == "book":
        return {"message": "推荐的书籍是：..."}

    else:
        return {"message": "我目前只推荐书籍"}


weather_subgraph_builder = StateGraph(ReActAgentState)
weather_subgraph_builder.add_node("weather_action", weather_subgraph_node)
weather_subgraph_builder.set_entry_point("weather_action")
weather_subgraph = weather_subgraph_builder.compile()


new_subgraph_builder = StateGraph(ReActAgentState)
new_subgraph_builder.add_node("new_action", new_subgraph_node)
new_subgraph_builder.set_entry_point("new_action")

new_subgraph = new_subgraph_builder.compile()

generate_subgraph_builder = StateGraph(ReActAgentState)
generate_subgraph_builder.add_node(
    "generate_recommendation_node", generate_recommendation_node
)
generate_subgraph_builder.set_entry_point("generate_recommendation_node")
generate_subgraph = generate_subgraph_builder.compile()
movie_subgraph_builder = StateGraph(ReActAgentState)
movie_subgraph_builder.add_node("movie_recommendation_node", movie_recommendation_node)
movie_subgraph_builder.set_entry_point("movie_recommendation_node")
movie_subgraph = movie_subgraph_builder.compile()

memory_update_builder = StateGraph(ReActAgentState)
memory_update_builder.add_node(
    "update_memory_node", update_memory_node
)
memory_update_builder.set_entry_point("update_memory_node")
memory_update_subgraph = memory_update_builder.compile()


def reasoning_state_manager(state: ReActAgentState):
    if state["action"] == "fetch_weather":
        return weather_subgraph
    elif state["action"] == "fetch_news":
        return new_subgraph
    elif state["action"] == "recommendation":
        if state["sub_action"] == "movie":
            return movie_subgraph
        return generate_subgraph
    else:
        return None


parent_builder = StateGraph(ReActAgentState)
parent_builder.add_node("reasoning", reasoning_node)
parent_builder.add_node("action_dispatch", reasoning_state_manager)
parent_builder.add_node("memory_update", memory_update_subgraph)
parent_builder.add_edge(START, "reasoning")
parent_builder.add_edge("reasoning", "action_dispatch")
parent_builder.add_edge("action_dispatch", "memory_update")

react_agent_graph = parent_builder.compile()

checkpointer = MemorySaver()


inputs_weather = {"message": "今天天气怎么样","memory": {}}
result_weather = react_agent_graph.invoke(inputs_weather)
print(result_weather["message"])
# inputs_news = {"message": "今天有什么新闻"}
# result_news = react_agent_graph.invoke(inputs_news)
# print(result_news["message"])
inputs_recommendation = {"message": "推荐一本书", "memory": {}}


result_recommendation = react_agent_graph.invoke(inputs_recommendation)
print(result_recommendation["message"])

inputs_recommendation2 = {"message": "推荐一本书", "memory": {"爱好": "电影"}}


result_recommendation2 = react_agent_graph.invoke(inputs_recommendation2)
print(result_recommendation2["message"])

# 定义price agent
def get_demand_data(product_id:str)->dict:
    """
    Get demand data for a product.
    """
    return {"product_id": product_id, "demand_level": "high"}


def get_competitor_price(product_id:str)->dict:
    """
    Get competitor price for a product.
    """
    return {"product_id": product_id, "competitor_price": 100}


tools = [get_demand_data, get_competitor_price]

graph = create_react_agent(model=model, tools=tools)

initial_messages = [("system","你是一个AI代理，可以根据市场需求和竞争对手的价格动态调整产品价格"),("user","请告诉我产品 12345 的价格")]

inputs = {"messages": initial_messages}

# for state in graph.stream(inputs, stream_mode="values"):
#     messages =  state["messages"][-1]
#     if isinstance(messages, tuple):
#         print(messages)
#     else:
#         messages.pretty_print()
from typing import Annotated, Sequence, TypedDict    
from langchain_core.messages import BaseMessage     
from langgraph.graph.message import add_messages  
class AgentState(TypedDict):
    message: Annotated[Sequence[BaseMessage],add_messages]

from langchain_core.tools import tool
from textblob import TextBlob

@tool
def analyze_sentiment(feeback: str) -> str:
    """
    Analyze the sentiment of the given text.
    """
    analysis = TextBlob(feeback) 
    if analysis.sentiment.polarity > 0.5:
        return "正面情感"
    elif analysis.sentiment.polarity == 0.5:
        return "中性情感"
    else:
        return "负面情感"


@tool
def respond_based_on_sentiment(sentiment: str) -> str:
    """
    Respond based on the sentiment of the feedback.
    """    
    if sentiment == "正面情感":
        return "谢谢你的积极反馈！"
    elif sentiment == "中性情感":
        return "谢谢你的反馈！"
    else:
        return "我们会努力改进！"

tools = [analyze_sentiment, respond_based_on_sentiment]
llm = model.bind_tools(tools)

tools_by_name = {tool.name: tool for tool in tools}

from langchain_core.messages import ToolMessage,SystemMessage
def tool_node(state: AgentState):
    """
    Tool node that selects the appropriate tool based on the state.
    """
    outputs = []
    for tool_call in state["message"][-1].tool_calls:
        tool_name = tool_call["name"]
        tool = tools_by_name[tool_name]
        output = tool.invoke(tool_call["args"])
        outputs.append(ToolMessage(content=json.dumps(output),tool_call_id=tool_call["id"],name=tool_name))
         
        return {"message": outputs}


from langchain_core.runnables import RunnableConfig

def call_model(state: AgentState,config: RunnableConfig):
    """
    Call the model with the current state.
    """
    system_prompt = SystemMessage(
        content="你是一个AI助手，负责处理用户的反馈信息。"
        
    )
    response = llm.invoke([system_prompt]+state["message"],config)
    return {"message": [response]} 

def should_continue(state: AgentState):
    """
    Check if the conversation should continue.
    """
    last_message = state["message"][-1]
    if  not last_message.tool_calls:
        return "end"
    else:
        return "continue"


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tool_node", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent", should_continue, {"continue": "tool_node", "end": END}
)

workflow.add_edge("tool_node", "agent")

graph = workflow.compile()

# import os

# from display_graph import display_graph

initial_state = {"message": [("user", "产品很棒，我不喜欢")]}

def print_stream(stream):
    for state in stream:
        messages  = state["message"][-1]
        if isinstance(messages, tuple):
            print(messages)
        else:
            messages.pretty_print()

# print_stream(graph.stream(initial_state, stream_mode="values"))

class RecommendationState(TypedDict):
    user_id:str
    preference:str
    reasoning:str
    recommendation:str
    memory: str


@tool
def recommendation_product(user_id: str, preference: str) -> str:
    """
    Recommend a product based on user ID and preference.
    """
    # Simulate a recommendation process
    return f"推荐的产品是：{preference}，用户ID：{user_id}"    


from openai import OpenAI

# 初始化客户端，指向 Ollama 的本地服务
client = OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama API 地址
    api_key="ollama",  # Ollama 默认无需真实 API Key，填任意值即可
)

# 发送请求
response = client.chat.completions.create(
    model="deepseek-r1",  # 指定模型
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "你好，什么是大模型？"},
    ],
    temperature=0.7,  # 控制生成多样性
    max_tokens=512,  # 最大生成 token 数
)

# 打印结果
print(response.choices[0].message.content)
