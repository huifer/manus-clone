from typing import Any, Dict, List, Optional


# 工具库示例
def web_search(query: str, num_results: int = 3) -> List[str]:
    # ...实际实现应调用搜索API...
    return [f"搜索结果 {i+1} for '{query}'" for i in range(num_results)]


def browse_webpage(url: str, selectors: Optional[List[str]] = None) -> str:
    return f"页面内容: {url} (selectors={selectors})"


def code_interpreter(code: str, language: str = "python") -> str:
    return f"执行{language}代码: {code}"


def file_read(path: str) -> str:
    return f"读取文件: {path}"


def file_write(path: str, content: str) -> bool:
    return True


def summarize_text(text: str, max_length: Optional[int] = None) -> str:
    return f"摘要: {text[:max_length] if max_length else text[:50]}..."


def extract_entities(text: str, entity_types: List[str]) -> Dict[str, List[str]]:
    return {etype: [f"{etype}_1", f"{etype}_2"] for etype in entity_types}


TOOL_LIBRARY = {
    "web_search": web_search,
    "browse_webpage": browse_webpage,
    "code_interpreter": code_interpreter,
    "file_read": file_read,
    "file_write": file_write,
    "summarize_text": summarize_text,
    "extract_entities": extract_entities,
}


# 状态与历史记录器
class StateHistoryLogger:
    def __init__(self):
        self.history = []

    def log(self, entry: Dict[str, Any]):
        self.history.append(entry)

    def get_history(self):
        return self.history


# 中央信息池通信（伪实现）
class CentralInfoPool:
    def __init__(self):
        self.data = {}

    def get_context(self, task_id: str) -> Dict[str, Any]:
        return self.data.get(task_id, {})

    def update(self, task_id: str, info: Dict[str, Any]):
        self.data[task_id] = info


# 执行用LLM（伪实现）
def executor_llm(prompt: str) -> Dict[str, Any]:
    # 实际应调用LLM，这里返回固定结构
    return {
        "thought": "分析当前情况，决定调用web_search工具。",
        "action": {
            "tool": "web_search",
            "params": {"query": "AI Agent", "num_results": 2},
        },
    }


class ExecutorAgent:
    def __init__(self, cip: CentralInfoPool):
        self.cip = cip
        self.logger = StateHistoryLogger()

    def receive_subtask(self, task_id: str, subtask: Dict[str, Any]):
        context = self.cip.get_context(task_id)
        self.logger.log(
            {"event": "receive_subtask", "subtask": subtask, "context": context}
        )
        return context

    def build_prompt(
        self,
        subtask: Dict[str, Any],
        history: List[Dict[str, Any]],
        tools: Dict[str, Any],
        context: Dict[str, Any],
    ) -> str:
        return f"目标: {subtask['goal']}\n历史: {history}\n工具: {list(tools.keys())}\n上下文: {context}"

    def parse_action(self, action: Dict[str, Any]):
        tool = action.get("tool")
        params = action.get("params", {})
        return tool, params

    def execute(self, task_id: str, subtask: Dict[str, Any]):
        context = self.receive_subtask(task_id, subtask)
        finished = False
        while not finished:
            prompt = self.build_prompt(
                subtask, self.logger.get_history(), TOOL_LIBRARY, context
            )
            llm_output = executor_llm(prompt)
            self.logger.log({"event": "llm_output", "output": llm_output})

            thought = llm_output.get("thought")
            action = llm_output.get("action")
            self.logger.log({"event": "thought", "thought": thought})

            tool, params = self.parse_action(action)
            if tool in TOOL_LIBRARY:
                try:
                    result = TOOL_LIBRARY[tool](**params)
                    self.logger.log(
                        {
                            "event": "action",
                            "tool": tool,
                            "params": params,
                            "result": result,
                        }
                    )
                    observation = {"result": result}
                except Exception as e:
                    observation = {"error": str(e)}
                    self.logger.log({"event": "error", "error": str(e)})
            else:
                observation = {"error": f"未知工具: {tool}"}
                self.logger.log({"event": "error", "error": f"未知工具: {tool}"})

            self.logger.log({"event": "observation", "observation": observation})
            # 简化：只执行一轮
            finished = True
            self.cip.update(
                task_id,
                {"final_result": observation, "history": self.logger.get_history()},
            )


# 用法示例
if __name__ == "__main__":
    cip = CentralInfoPool()
    agent = ExecutorAgent(cip)
    subtask = {"goal": "查找AI Agent相关信息"}
    agent.execute("task_001", subtask)
    print("历史记录：")
    for entry in agent.logger.get_history():
        print(entry)
