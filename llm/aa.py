from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.llms.tongyi import Tongyi
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
import sys
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_community.llms import Tongyi
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field  
from langchain.output_parsers import PydanticOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# 初始化 ConversationBufferMemory
memory = ConversationBufferMemory(
  memory_key="history",
  return_messages=True,
  chat_memory=ChatMessageHistory()
)

class SubTask(BaseModel):
    task_id: str = Field(description="Unique identifier for the sub-task, e.g., 't1', 't2'.")
    description: str = Field(description="Detailed description of what the sub-task entails.")
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of task_ids this sub-task depends on. Empty if no dependencies."
    )

class TaskList(BaseModel):
    """A list of sub-tasks to accomplish a larger goal.""" 
    tasks: List[SubTask] = Field(description="A list of sub-tasks.")

# 定义会话历史获取函数
def get_session_history(session_id: str):
  return memory.chat_memory


# 初始化 ChatOpenAI
llm = Tongyi()
# parser = PydanticOutputParser(pydantic_object=TaskList)
parser = StrOutputParser()

# 定义提示模板
prompt = ChatPromptTemplate.from_messages([
  ("system", """你是一名任务分解专家。首先要理解用户输入的任务意图，思考需要用到哪些领域知识或技能，然后再将任务拆解为一系列可执行的子任务，输出严格符合下方 schema 的 JSON 对象。
每个子任务需包含唯一的 task_id、详细的 description，以及依赖的 dependencies（如无依赖则为 []）。
task_id 必须唯一（如 t1, t2, t3...），dependencies 仅引用已存在的 task_id。
请确保结构清晰、步骤合理、依赖关系准确。

"""  ),
  MessagesPlaceholder(variable_name="history"),
  ("human", "{input}")
])




# 创建对话链
chain = prompt | llm | parser

# 包装链以支持消息历史
chain_with_history = RunnableWithMessageHistory(
  runnable=chain,
  get_session_history=get_session_history,
  input_messages_key="input",
  history_messages_key="history"
)

# 测试 ConversationBufferMemory 和对话链
print("测试 ConversationBufferMemory 和对话链：")
try:
  session_id = "ai_chat_001"

  # 第一轮对话
  question1 = "如何泡一杯好喝的拿铁咖啡？"
  result1 = chain_with_history.invoke(
    {"input": question1,},
    config={"configurable": {"session_id": session_id}}
  )
  print(f"问题 1: {question1}")
  print(f"回答 1: {result1}")

  # 第二轮对话（依赖上下文）
  question2 = "如果没有咖啡机怎么办？"
  result2 = chain_with_history.invoke(
    {"input": question2,},
    config={"configurable": {"session_id": session_id}}
  )
  print(f"\n问题 2: {question2}")
  print(f"回答 2: {result2}")

  # 显示消息历史
  print("\n消息历史：")
  history = memory.load_memory_variables({})["history"]
  for i, msg in enumerate(history):
    print(f"消息 {i + 1} ({msg.__class__.__name__}): {msg.content}")
except Exception as e:
  print(f"错误: {e}")
1