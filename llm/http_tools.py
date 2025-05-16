from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import FakeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.llms import Ollama

from llm.llm_factory import get_llm



llm = get_llm()
import requests
from typing import Any, Dict, Optional
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

    def _run(
        self,
        param: Optional[Union[int, Dict]] = None,
    ):
        print(param)
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
            size = param
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
        # 直接在swagger_apis中搜索最匹配的接口，返回接口名和参数列表
        # 简单关键字匹配（可根据实际需求优化匹配逻辑）
        best_api = None
        for api in swagger_apis:
            if api["name"].lower() in query.lower() or api["description"][:6] in query:
                best_api = api
                break
        if not best_api:
            # 没有匹配到，返回空
            return '{"name": "", "params": {}}'
        # 构造参数字典，参数值为空字符串
        params_dict = {param: "" for param in best_api["params"]}
        import json

        return json.dumps(
            {"name": best_api["name"], "params": params_dict}, ensure_ascii=False
        )

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
v = agent("我要看第3页的用户数据,每页10条")
print(v)
