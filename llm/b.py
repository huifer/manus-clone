from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.llms.tongyi import Tongyi
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

# 初始化 ConversationBufferMemory
memory = ConversationBufferMemory(
  memory_key="history",
  return_messages=True,
  chat_memory=ChatMessageHistory()
)


# 定义会话历史获取函数
def get_session_history(session_id: str):
  return memory.chat_memory


# 初始化 ChatOpenAI
llm = Tongyi()

# 定义提示模板
prompt = ChatPromptTemplate.from_messages([
  ("system", "你是一个人工智能专家，回答用户问题。"),
  MessagesPlaceholder(variable_name="history"),
  ("human", "{input}")
])

# 定义输出解析器
parser = StrOutputParser()

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
  question1 = "什么是人工智能？"
  result1 = chain_with_history.invoke(
    {"input": question1},
    config={"configurable": {"session_id": session_id}}
  )
  print(f"问题 1: {question1}")
  print(f"回答 1: {result1}")

  # 第二轮对话（依赖上下文）
  question2 = "它有哪些应用？"
  result2 = chain_with_history.invoke(
    {"input": question2},
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
