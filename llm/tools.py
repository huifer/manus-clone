from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi


# 初始化LLM，使用Gemini-2.0-flash
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )
llm = ChatTongyi(
        model="qwen-turbo",
        dashscope_api_key=os.getenv("DASH_SCOPE_API_KEY"),
        top_p=0.95,
        temperature=0.7,
    )
import requests
from typing import Any, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate

from langchain.tools import BaseTool
from math import pi
from typing import Union

from langchain.chains.conversation.memory import ConversationBufferWindowMemory


# 圆周长计算工具
class CircumferenceTool(BaseTool):
    name: str = "圆周长计算器"
    description: str = "当你需要根据圆的半径计算圆周长时，请使用此工具"

    def _run(self, radius: Union[int, float]):
        # 计算圆周长
        return float(radius) * 2.0 * pi

    def _arun(self, radius: int):
        # 不支持异步
        raise NotImplementedError("该工具不支持异步")


# 假设的 Swagger 接口描述
swagger_apis = [
    {
        "name": "userinfo",
        "description": "根据用户ID查询用户基本信息",
        "method": "GET",
        "url": "http://your-api/userinfo",
        "params": ["user_id"],
    },
    {
        "name": "userList",
        "description": "获取所有用户列表",
        "method": "GET",
        "url": "http://your-api/userList",
        "params": [],
    },
    {
        "name": "UserPage",
        "description": "分页获取用户信息",
        "method": "GET",
        "url": "http://your-api/UserPage",
        "params": ["page", "size"],
    },
]


# 动态创建 HTTP 请求工具
class DynamicHTTPTool(BaseTool):
    name: str
    description: str
    api_info: dict

    def _run(self, param: Optional[Union[int, str]] = None,
):
        # MOCK: 根据接口名称返回模拟数据
        if self.api_info["name"] == "userinfo":
            # 模拟返回用户基本信息
            return {
                "user_id": param,
                "name": "张三",
                "age": 28,
                "email": "zhangsan@example.com",
            }
        elif self.api_info["name"] == "userList":
            # 模拟返回用户列表
            return [{"user_id": 1, "name": "张三"}, {"user_id": 2, "name": "李四"}]
        elif self.api_info["name"] == "UserPage":
            # 模拟返回分页用户信息
            page = param
            size =param
            return {
                "page": page,
                "size": size,
                "total": 2,
                "users": [
                    {"user_id": 1, "name": "张三"},
                    {"user_id": 2, "name": "李四"},
                ],
            }
        # 其他接口返回空
        return {}

    def _arun(self, param: str):
        raise NotImplementedError("该工具不支持异步")


# 根据 swagger_apis 动态生成工具列表
http_tools = [
    DynamicHTTPTool(name=api["name"], description=api["description"], api_info=api)
    for api in swagger_apis
]


# 工具选择工具：让 LLM 选择合适的接口
class SwaggerSelectorTool(BaseTool):
    name: str = "Swagger接口选择器" 
    description: str = "根据用户意图选择最合适的Swagger接口" 

    def _run(self, query: str):
        # 简单实现：根据描述关键字匹配
        for api in swagger_apis:
            if "用户ID" in query or "基本信息" in query:
                if api["name"] == "userinfo":
                    return api["name"]
            if "列表" in query:
                if api["name"] == "userList":
                    return api["name"]
            if "分页" in query:
                if api["name"] == "UserPage":
                    return api["name"]
        return "未找到合适的接口"

    def _arun(self, query: str):
        raise NotImplementedError("该工具不支持异步")


# 初始化对话记忆，最多保留5轮对话
conversational_memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=5, return_messages=True
)

from langchain.agents import initialize_agent

# 工具列表：包含接口选择器和所有HTTP工具
tools = [SwaggerSelectorTool()] + http_tools

# 初始化智能体
agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    memory=conversational_memory,
)

# 示例：用户输入
v = agent("我要查询用户ID是1的基本信息")
print(v)
