from fastapi import FastAPI

# 创建FastAPI应用实例
app = FastAPI()
app.docs_url = "/docs"  # 设置Swagger UI的文档路径


@app.get("/")
def read_root():
    return {"message": "Hello World"}


@app.post("/goodbye")
def goodbye(data: dict):
    name = data.get("name", "World")
    return f"Goodbye {name}"

if __name__ == "__main__":
    import uvicorn

    # 运行FastAPI应用
    uvicorn.run(app, host="127.0.0.1", port=8001)
