# from typing_extensions import TypedDict

# age = 25

# price = 19.99


# name = "John"

# is_active = True


# unknown = None

# my_tuple = (1, "hello", 3.14)


# print(type(age))


# numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# first, *middle, last = numbers
# print(first)
# print(middle)


# squared = list(map(lambda x: x**2, numbers))

# print(squared)


# # 装饰器
# def log_decorator(func):
#     def wrapper(*args, **kwargs):
#         print(f"Executing {func.__name__} with {args}, {kwargs}")
#         return func(*args, **kwargs)

#     return wrapper


# @log_decorator
# def greet(name):
#     print(f"Hello {name}")


# greet("John")


# # 递归函数
# def factorial(n):
#     if n == 1:
#         return 1
#     else:
#         return n * factorial(n - 1)


# def greet(name):
#     """问候人通过名字"""
#     print(f"Hello {name}")


# help(greet)


# user_info = {"name": "John", "age": 25, "is_active": True}

# user_info.pop("age")

# print(user_info)

# del user_info["name"]

# print(user_info)

# user_info.clear()
# user_info.setdefault("name", "Tom")

# squares = {x: x**2 for x in range(10)}

# print(squares)

# Person = TypedDict("Person", {"name": str, "age": int})


# class ServerConfig(TypedDict):

#     host: str
#     port: int
#     use_ssl: bool


# def create_default_server_config() -> ServerConfig:
#     return {"host": "localhost", "port": 8080, "use_ssl": False}


# config = create_default_server_config()

# class Dog:
#     #类属性
#     species = "Canis familiaris"
#     def __init__(self, name, age,type):
#         self.name = name
#         self.age = age
#         self._type  = type # 私有属性
#     #实例方法
#     def bark(self):
#         return f"{self.name} says woof!"

# dog = Dog("Fido", 3,"")
# print(dog.bark())


# class HaSHiQi(Dog):
#     def speak(self):
#         return "哈士奇说话了"

#     @classmethod
#     def set_species(cls,new_species):
#          cls.species = new_species

#     @staticmethod
#     def is_haSHiQi():
#         return True

#     def __str__(self):
#         return f"{self.name} is a {self.species}"
#     #魔法方法
#     def __repr__(self):
#         """
#         Returns a string representation of the object in the format '{name} is a {species}'.

#         Returns:
#             str: A formatted string containing the name and species of the object.
#         """
#         return f"{self.name} is a {self.species}"


# hs = HaSHiQi("哈士奇", 3,"")
# print(hs.speak())
# HaSHiQi.set_species("哈士奇")
# print(HaSHiQi.species)
# print(HaSHiQi.is_haSHiQi())

# print(repr(hs))
# #m魔法方法重载内置运算符

# class Vector2D:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

#     def __add__(self, other):
#         return Vector2D(self.x + other.x, self.y + other.y)

#     @property  # 属性装饰器
#     def magnitude(self):
#         return (self.x ** 2 + self.y ** 2) ** 0.5


#     def width(self):
#         return self.x
#     @width.setter
#     def width(self, value):
#         self.x = value


# v1 = Vector2D(5, 7)

# v2 = Vector2D(3, 9)

# result = v1 + v2

# print(result.x, result.y)
# print(result.magnitude)


# #常见类型
# from typing  import List,Dict,Optional

# def process_data(data:Dict[str,Optional[List[int]]]) -> None:
#     print(f"Data: {data}")

# #异步
# import   asyncio
# async def fetch_data():
#       await asyncio.sleep(1)
#       return "data"

# async def main():
#       result = await asyncio.gather(fetch_data(),fetch_data())
#       print(result)

# asyncio.run(main())


# #装饰器
import requests

try:
    response = requests.get("https://api.apihubs.cn/holiday/get")
    if response.status_code == 200:
        data = response.json()
        print(data)
except requests.exceptions.HTTPError as errh:
    print("Http Error:", errh)
except requests.exceptions.ConnectionError as errc:
    print("Error Connecting:", errc)


# 验证数据
from pydantic import BaseModel, Field, ValidationError


class User(BaseModel):
    name: str
    age: int = Field(..., gt=0)


try:
    user = User(name="John", age=-1)

except ValidationError as e:
    print(e)


# 日志记录
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("这是一个详细的消息")


# 子进程
import sys
import subprocess

# 用于执行shell命令
result = subprocess.run(
    ["echo", "Hello World"], capture_output=True, text=True, executable=sys.executable
)

print(result.stdout)

# JSON
import json


with open("data.json", "r") as f:
    loaded_data = json.load(f)
print(loaded_data)


# yaml
import yaml

config = {"name": "Alice", "roles": ["admin", "user"]}

try:
    with open("config.yaml", "w") as f:
        yaml.dump(config, f)

    with open("config.yaml", "r") as f:
        loaded_config = yaml.safe_load(f)
        print(loaded_config)
except Exception as e:
    print(e)

# 处理和装换大数据
import pandas as pd

df = pd.read_csv("data.csv")
print(df.head())
# 可视化数据  seaborn建立在matplotlib上的高级绘画
import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4], [10, 20, 25, 30])
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# 可视化与数据库交互
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

try:

    engine = create_engine("sqlite:///data.db")
    Session = sessionmaker(bind=engine)
    session = Session()
except Exception as e:
    print(e)

# 使用OS
import os

try:
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    api_key = os.environ.get("API_KEY", "default_value")

except Exception as e:
    print(e)

# sys 用来处理命令行及控制解释器 输入输出流
import sys

if len(sys.argv) < 2:
    print("Usage: python script.py <name>")
    sys.exit("请输入一个有效的文件地址")
name = sys.argv[1]
print(f"Hello {name}!")


def process_data(data):
    if not data:
        print("No data received!")
        sys.exit(1)
    print(f"Data received: {data}")
    sys.exit(0)


data = None
process_data(data)


# 随机数
import random

random_number = random.randint(1, 100)
print(random_number)

options = ["apple", "banana", "orange"]
fruit = random.choice(options)

print(fruit)
# 随机数种子

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

random.shuffle(numbers)
print(numbers)

# 随机浮点数
random_float = random.uniform(0, 1)
print(random_float)

# 时间模块
from datetime import datetime, timedelta

now = datetime.now()

future_date = now + timedelta(days=1)

print(future_date.strftime("%Y-%m-%d"))

# collections 专用数据结构
from collections import Counter, defaultdict

words = ["apple", "banana", "apple", "orange", "banana", "apple"]

word_count = Counter(words)

fruits = defaultdict(int)

fruits["apple"] += 1
print(fruits)

# itertools 迭代生成器
from itertools import cycle

colors = cycle(["red", "green", "blue"])

for _ in range(10):
    print(next(colors))

# MAP
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

squared = list(map(lambda x: x**2, numbers))

print(squared)

evens = list(filter(lambda x: x % 2 == 0, numbers))

print(evens)

from functools import reduce

sum = reduce(lambda x, y: x + y, numbers)
print(sum)

# 异步编程
import asyncio


async def fetch_data():
    await asyncio.sleep(1)
    return "data"


result = asyncio.run(fetch_data())


# 异步
async def task1():
    await asyncio.sleep(1)
    print("Task 1 is done")


async def task2():
    await asyncio.sleep(2)
    print("Task 2 is done")


async def main():
    await task1()
    await task2()


asyncio.run(main())


# 并发     gather
async def main():
    await asyncio.gather(task1(), task2())


# 异步IO操作
async def fetch_data():
    print("Fetching data")

    await asyncio.sleep(1)
    print("Data fetched")


async def main():
    await asyncio.gather(*(fetch_data() for _ in range(10)))


asyncio.run(main())


# 异步代码错误处理
async def faulty_task():
    try:
        raise ValueError("Something went wrong")
    except ValueError as e:
        print(f"Caught an exception: {e}")


asyncio.run(faulty_task())


# 使用gather处理错误
async def task1():
    raise ValueError("Error in Task1")


async def task2():
    await asyncio.sleep(1)
    return "任务2已完成"


async def main():
    results = await asyncio.gather(task1(), task2(), return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            print(f"Caught an exception: {result}")

        else:
            print(f"Results: {results}")


asyncio.run(main())

# 获取多个节点的数据
import asyncio


async def fetch_data_for_node(node_id, api_url):
    print(f"Fetching data for node {node_id}")
    await asyncio.sleep(1)
    return f"Data for node {node_id}"


# 字符串操作 +  * 重复
greeting = "Hello" + " " + "World"

echo = greeting * 5

# find split  join upper

# 异常  raise
# from langgraph.graph import Graph
# from langchain_core.runnables.graph import Node


# class APIRequestNode(Node):
#     def run(self):
#         try:
#             data = self.make_request()
#             self.send_output(data)
#         except TimeoutError as e:
#             self.send_output("请求超时")

#         except ValueError as e:
#             self.send_output("无效的请求")
#         finally:
#             self.log("请求完成")


# graph = Graph()
# node = APIRequestNode()
# graph.add_node(node)


