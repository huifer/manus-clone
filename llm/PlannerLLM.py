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

from llm.llm_generatory import get_llm_model

load_dotenv()

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

class PlannerLLM:
    def __init__(self):
        self.parser = PydanticOutputParser(pydantic_object=TaskList)
        self.llm = Tongyi()
        self.system_prompt_template_str = """
你是一名任务分解专家。首先要理解用户输入的任务意图，思考需要用到哪些领域知识或技能，然后再将任务拆解为一系列可执行的子任务，输出严格符合下方 schema 的 JSON 对象。
每个子任务需包含唯一的 task_id、详细的 description，以及依赖的 dependencies（如无依赖则为 []）。
task_id 必须唯一（如 t1, t2, t3...），dependencies 仅引用已存在的 task_id。
请确保结构清晰、步骤合理、依赖关系准确。

{format_instructions}
"""
        self.human_prompt_template_str = "User Request: {input}"

        # 优化后的内存和历史
        self.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True,
            chat_memory=ChatMessageHistory()
        )

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_prompt_template_str),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template(self.human_prompt_template_str)
        ])

        # 会话历史获取函数
        def get_session_history(session_id: str):
            return self.memory.chat_memory
        self.get_session_history = get_session_history

        self.chain = RunnableWithMessageHistory(
            runnable=self.prompt | self.llm | self.parser,
            get_session_history=self.get_session_history,
            input_messages_key="input",
            history_messages_key="history"
        )

    def generate_subtasks(self, user_query: str) -> List[Dict[str, Any]]:
        """
        Generates a list of sub-tasks based on user query, supporting multi-turn conversation.
        """
        try:
            response_pydantic_object = self.chain.invoke(
                {
                    "input": user_query,
                    "format_instructions": self.parser.get_format_instructions(),
                },
                config={"configurable": {"session_id": "default"}}
            )
            return [task.model_dump() for task in response_pydantic_object.tasks]
        except Exception as e:
            print(f"Error generating subtasks: {e}")
            return []

if __name__ == "__main__":
    planner = PlannerLLM()
    user_input_1 = "如何泡一杯好喝的拿铁咖啡？"
    print(f"--- Decomposing task: {user_input_1} ---")
    subtasks_1 = planner.generate_subtasks(user_input_1)

    if subtasks_1:
        import json
        print("Generated Subtasks (JSON):")
        print(json.dumps(subtasks_1, indent=4, ensure_ascii=False))
        user_input_2 = "如果我没有咖啡机怎么办？"
        print(f"\n--- Follow-up: {user_input_2} ---")
        subtasks_2 = planner.generate_subtasks(user_input_2)
        print("Generated Subtasks (JSON):")
        print(json.dumps(subtasks_2, indent=4, ensure_ascii=False))

        history = planner.memory.load_memory_variables({})["history"]
        for i, msg in enumerate(history):
            print(f"消息 {i + 1} ({msg.__class__.__name__}): {msg.content}")
    else:
        print("Failed to generate subtasks for example 1.")