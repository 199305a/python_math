# 定义state结构体
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from langchain_core.runnables.graph import MermaidDrawMethod
import random
import os
import sys

import subprocess


class HelloWorldState(TypedDict):
    greeting: str


# 定义node函数
def hello_world_note(state: HelloWorldState) -> HelloWorldState:
    state["greeting"] = "Hello, world!" + state["greeting"]
    return state


# 初始化图并添加节点 使用Langgraph
# 定义一个新的node函数


def exclamation_note(state: HelloWorldState) -> HelloWorldState:
    state["greeting"] = state["greeting"] + "!"
    return state


builder = StateGraph(HelloWorldState)
builder.add_node("greet", hello_world_note)
builder.add_node("exclaim", exclamation_note)


# 使用edges定义执行流程
builder.add_edge(START, "greet")
builder.add_edge("greet", "exclaim")
builder.add_edge("exclaim", END)

# 编译并运行图形
graph = builder.compile()
result = graph.invoke({"greeting": "来自 LangGraph 的问候!"})
print(result)
# {"greeting": "Hello, world!来自 LangGraph 的问候!"}

# 可视化你的图表
mermaid_png = graph.get_graph(xray=1).draw_mermaid_png(
    draw_method=MermaidDrawMethod.API
)

output_folder = "."

os.makedirs(output_folder, exist_ok=True)
filename = os.path.join(output_folder, f"graph_{random.randint(0,1000000)}.png")
with open(filename, "wb") as f:
    f.write(mermaid_png)
print(f"图已保存到 {filename}")

if sys.platform.startswith("drawin"):
    subprocess.call(["open", filename])
elif sys.platform.startswith("linux"):
    subprocess.call(["xdg-open", filename])
elif sys.platform.startswith("win"):
    os.startfile(filename)


# 核心元素

# state
from typing_extensions import TypedDict


class HelloWorldState(TypedDict):
    message: str


state = {"message": "Hello, world!"}


def hello_world_node(state: HelloWorldState) -> HelloWorldState:
    state["message"] = "Hello, world!" + state["message"]
    return state


graph_builder = StateGraph(HelloWorldState)

# 开始节点
graph_builder.add_edge(START,"greet_user")

graph_builder.set_entry_point("greet_user")

# 结束节点
graph_builder.add_edge("say_goodbye", END)


graph_builder.set_finish_point("say_goodbye")

# 决策点
def check_subscription(state: HelloWorldState):
    if state["message"] == "Hello, world!" and state["subscription"]:
        return "greet_user"
    else:
        return "say_goodbye"


graph_builder.add_conditional_edges("greet_user", check_subscription)

# 边缘原则 顺序流 条件流 完成控制 错误处理
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class UserState(TypedDict):
    is_premium:bool
    message:str


def greet_user(state:UserState):
     state["message"]= "Welcome!"
     return state   

def premium_greeting(state:UserState):
    state["message"]= "感谢你成为高级用户！"
    return state

def  regular_greeting(state:UserState):
    state["message"]= "祝你在这里度过愉快的时光"
    return state

# 定义决策树
def check_subscription(state:UserState):
    if state["is_premium"]:
        return "premium_greeting"
    else:
        return "regular_greeting"


# 构建图
graph_builder = StateGraph(UserState)
graph_builder.add_node("greet_user", greet_user)
graph_builder.add_node("check_subscription", check_subscription)
graph_builder.add_node("premium_greeting", premium_greeting)
graph_builder.add_node("regular_greeting", regular_greeting)    

# 添加边来控制流
graph_builder.add_edge(START,"greet_user")
graph_builder.add_conditional_edges("greet_user",check_subscription)

graph_builder.add_edge("premium_greeting",END)
graph_builder.add_edge("regular_greeting",END)
result = graph_builder.compile().invoke({"is_premium":True,"message":"Hello, is_premium!"})

print(result)
# {'is_premium': True, 'message': '感谢你成为高级用户！'}

# 普通用户
result = graph_builder.compile().invoke({"is_premium":False,"message":"Hello, is_premium!"})


class OverallState(TypedDict):
    partial_message:str
    user_input:str
    message_output:str


# 多个状态架构的图形
class InputState(TypedDict):
    user_input:str
class OutputState(TypedDict):
    message_output:str

class PrivateState(TypedDict):
    private_message:str

def add_world(state:InputState) -> OverallState:
    partial_message  = state["user_input"] + " world!"
    print("节点1添加了add world")
    return {"partial_message":partial_message,"user_input":state["user_input"],"message_output":""}

def add_exclamation(state:OverallState) -> PrivateState:
    private_message = state["partial_message"] + "!"
    print("节点二转换了 partial_message")
    return {"private_message":private_message}

def finalize_message(state:PrivateState) -> OutputState:
    message_output = state["private_message"]
    print("节点三完成了")
    return {"message_output":message_output}

# 构建图
graph_builder = StateGraph(OverallState,input=InputState,output=OutputState)
graph_builder.add_node("add_world", add_world)
graph_builder.add_node("add_exclamation", add_exclamation)
graph_builder.add_node("finalize_message", finalize_message)

# 定义边
graph_builder.add_edge(START,"add_world")
graph_builder.add_edge("add_world","add_exclamation")
graph_builder.add_edge("add_exclamation","finalize_message")
graph_builder.add_edge("finalize_message",END)

# 编译并运行图
graph = graph_builder.compile()

result= graph.invoke({"user_input":"Hello"})
print(result)


