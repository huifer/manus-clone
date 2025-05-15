import os
import sys
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_community.llms import Tongyi
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# LangChain 组件
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field  
from langchain.output_parsers import PydanticOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory


from llm.llm_generatory import get_llm_model

load_dotenv()


class SubTask(BaseModel):
    task_id: str = Field(description="Unique identifier for the sub-task, e.g., 't1', 't2'.")
    description: str = Field(description="Detailed description of what the sub-task entails.")
    dependencies: List[str] = Field(
        default_factory=list, # 默认为空列表
        description="List of task_ids this sub-task depends on. Empty if no dependencies."
    )

class TaskList(BaseModel):
    """A list of sub-tasks to accomplish a larger goal.""" 
    tasks: List[SubTask] = Field(description="A list of sub-tasks.")

class PlannerLLM:
    def __init__(self):
        self.parser = PydanticOutputParser(pydantic_object=TaskList)
        self.llm = Tongyi(
           
        )
        self.system_prompt_template_str = """
你是一名任务分解专家。首先要理解用户输入的任务意图，思考需要用到哪些领域知识或技能，然后再将任务拆解为一系列可执行的子任务，输出严格符合下方 schema 的 JSON 对象。
每个子任务需包含唯一的 task_id、详细的 description，以及依赖的 dependencies（如无依赖则为 []）。
task_id 必须唯一（如 t1, t2, t3...），dependencies 仅引用已存在的 task_id。
请确保结构清晰、步骤合理、依赖关系准确。

{format_instructions}
"""
        self.human_prompt_template_str = "User Request: {input}"
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt_template_str),
            HumanMessagePromptTemplate.from_template(self.human_prompt_template_str)
        ])
        # 初始化对话历史内存
        self.memory = ChatMessageHistory()
        # 使用 RunnableWithMessageHistory 包装 chain
        self.chain = RunnableWithMessageHistory(
            self.prompt | self.llm | self.parser,
            lambda session_id: self.memory,
            input_messages_key="input",
        )

    def generate_subtasks(self, user_query: str) -> List[Dict[str, Any]]:
        """
        Generates a list of sub-tasks based on user query, supporting multi-turn conversation.

        Args:
            user_query: The user's main task or question.
            history: Optional. List of previous messages (dicts) for multi-turn context.

        Returns:
            A list of dictionaries, where each dictionary represents a sub-task
            conforming to the specified JSON structure.
            Returns an empty list on failure.
        """
        try:
            response_pydantic_object = self.chain.invoke(
                {
                    "input": user_query,
                    "format_instructions": self.parser.get_format_instructions()
                },
                config={"configurable": {"session_id": "default"}}
            )
            # 返回当前历史消息（可用于后续多轮）
            self.latest_history = messages_to_dict(self.memory.chat_memory.messages)
            return [task.dict() for task in response_pydantic_object.tasks]
        except Exception as e:
            print(f"Error generating subtasks: {e}")
            return []

# 7. 示例用法
if __name__ == "__main__":
    planner = PlannerLLM()
    # 示例1: 制作咖啡
    user_input_1 = "如何泡一杯好喝的拿铁咖啡？"

    print(f"--- Decomposing task: {user_input_1} ---")
    subtasks_1 = planner.generate_subtasks(user_input_1)

    if subtasks_1:
        import json
        print("Generated Subtasks (JSON):")
        print(json.dumps(subtasks_1, indent=4, ensure_ascii=False))
        # 多轮对话示例
        user_input_2 = "如果我没有咖啡机怎么办？"
        print(f"\n--- Follow-up: {user_input_2} ---")
        subtasks_2 = planner.generate_subtasks(user_input_2)
        print("Generated Subtasks (JSON):")
        print(json.dumps(subtasks_2, indent=4, ensure_ascii=False))
    else:
        print("Failed to generate subtasks for example 1.")