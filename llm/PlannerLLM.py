import os
from dotenv import load_dotenv
from typing import List, Dict, Any

# LangChain 组件
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field # 使用 pydantic_v1 兼容 LangChain
from langchain.output_parsers import PydanticOutputParser

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

parser = PydanticOutputParser(pydantic_object=TaskList)

llm = get_llm_model()

system_prompt_template_str = """
You are an expert task decomposition assistant.
Your goal is to break down a user's request into a sequence of actionable sub-tasks.
The output MUST be a JSON object strictly conforming to the provided schema.
Each sub-task must have a unique 'task_id', a 'description', and a list of 'dependencies'.
'dependencies' should be a list of 'task_id's that must be completed before this task can start.
If a task has no dependencies, its 'dependencies' list should be empty ([]).
Ensure task_ids are unique (e.g., t1, t2, t3...) and dependencies correctly reference existing task_ids.

{system_prompt_addon}

{format_instructions}
"""

human_prompt_template_str = "User Request: {input}"

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt_template_str),
    HumanMessagePromptTemplate.from_template(human_prompt_template_str)
])

# 5. 构建 LangChain Chain (LCEL - LangChain Expression Language)
# 链式调用：Prompt -> LLM -> Parser
chain = prompt | llm | parser

# 6. 定义主函数来调用 Chain
def generate_subtasks(user_query: str, system_prompt_addon: str) -> List[Dict[str, Any]]:
    """
    Generates a list of sub-tasks based on user query and system prompt.

    Args:
        user_query: The user's main task or question.
        system_prompt_addon: Additional instructions for the system prompt.

    Returns:
        A list of dictionaries, where each dictionary represents a sub-task
        conforming to the specified JSON structure.
        Returns an empty list on failure.
    """
    try:
        # 调用 chain 并传入所有需要的参数
        # format_instructions 来自 parser
        response_pydantic_object = chain.invoke({
            "input": user_query,
            "system_prompt_addon": system_prompt_addon,
            "format_instructions": parser.get_format_instructions()
        })

        # response_pydantic_object 将会是 TaskList 类型的一个实例
        # 我们需要将其转换为题目要求的 List[Dict]
        # Pydantic 模型的 .dict() 方法可以将其转换为字典
        return [task.dict() for task in response_pydantic_object.tasks]

    except Exception as e:
        print(f"Error generating subtasks: {e}")
        # 可以根据需要添加更复杂的错误处理或重试逻辑
        # 例如，如果 LLM 输出不是有效的 JSON，PydanticOutputParser 会抛出 OutputParserException
        # 可以捕获这个异常并尝试修复或提示用户
        return []

# 7. 示例用法
if __name__ == "__main__":
    # 示例1: 制作咖啡
    user_input_1 = "如何泡一杯好喝的拿铁咖啡？"
    custom_system_prompt_1 = "请确保步骤清晰，并包含准备工具和材料的步骤。任务ID请使用 step1, step2... 的格式。"

    print(f"--- Decomposing task: {user_input_1} ---")
    subtasks_1 = generate_subtasks(user_input_1, custom_system_prompt_1)

    if subtasks_1:
        import json
        print("Generated Subtasks (JSON):")
        print(json.dumps(subtasks_1, indent=4, ensure_ascii=False))
    else:
        print("Failed to generate subtasks for example 1.")

    print("\n" + "="*50 + "\n")

    # 示例2: 开发 Web 应用
    user_input_2 = "我要写一个Python FastAPI的Web应用，包含用户认证和基本的CRUD操作。"
    custom_system_prompt_2 = "重点考虑API端点的设计和数据库模型的初步规划。任务ID请使用 tsk_auth_01, tsk_db_01, tsk_crud_01... 的格式。"

    print(f"--- Decomposing task: {user_input_2} ---")
    subtasks_2 = generate_subtasks(user_input_2, custom_system_prompt_2)

    if subtasks_2:
        import json
        print("Generated Subtasks (JSON):")
        print(json.dumps(subtasks_2, indent=4, ensure_ascii=False))
    else:
        print("Failed to generate subtasks for example 2.")

    print("\n" + "="*50 + "\n")

    # 示例3: 简单任务，无特定ID格式要求
    user_input_3 = "计划一次周末东京两日游"
    custom_system_prompt_3 = "主要景点包括浅草寺、东京塔和新宿御苑。考虑交通和餐饮。"
    print(f"--- Decomposing task: {user_input_3} ---")
    subtasks_3 = generate_subtasks(user_input_3, custom_system_prompt_3)

    if subtasks_3:
        import json
        print("Generated Subtasks (JSON):")
        print(json.dumps(subtasks_3, indent=4, ensure_ascii=False))
    else:
        print("Failed to generate subtasks for example 3.")