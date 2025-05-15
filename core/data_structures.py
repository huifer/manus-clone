# 核心数据结构定义

from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


# 任务状态枚举类，表示任务的不同状态
class TaskStatus(Enum):
    PENDING = "pending"  # 待处理
    ASSIGNED = "assigned"  # 已分配
    IN_PROGRESS = "in_progress"  # 执行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败

    def to_json(self):
        return self.value

    @staticmethod
    def from_json(value):
        return TaskStatus(value)


# 子任务类，表示一个具体的子任务及其属性
@dataclass
class SubTask:
    task_id: str  # 子任务的唯一标识符
    description: str  # 子任务的描述
    dependencies: List[str]  # 子任务的依赖列表
    assigned_agent: Optional[str] = None  # 分配的执行Agent
    status: TaskStatus = TaskStatus.PENDING  # 子任务的当前状态
    input_params: Optional[Dict[str, Any]] = None  # 子任务的输入参数
    output: Optional[Any] = None  # 子任务的输出结果

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "description": self.description,
            "dependencies": self.dependencies,
            "assigned_agent": self.assigned_agent,
            "status": self.status.to_json(),
            "input_params": self.input_params,
            "output": self.output,
        }

    @staticmethod
    def from_dict(data):
        return SubTask(
            task_id=data["task_id"],
            description=data["description"],
            dependencies=data["dependencies"],
            assigned_agent=data.get("assigned_agent"),
            status=TaskStatus.from_json(data["status"]),
            input_params=data.get("input_params"),
            output=data.get("output"),
        )


# 任务计划类，管理所有子任务及其状态
@dataclass
class TaskPlan:
    tasks: Dict[str, SubTask] = field(default_factory=dict)  # 存储所有子任务的字典

    def add_task(self, task: SubTask):
        self.tasks[task.task_id] = task  # 添加一个子任务到任务计划中

    def get_ready_tasks(self) -> List[SubTask]:
        """返回所有已准备好执行的子任务（其依赖已满足）。"""
        ready_tasks = []
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING and all(
                self.tasks[dep].status == TaskStatus.COMPLETED
                for dep in task.dependencies
            ):
                ready_tasks.append(task)
        return ready_tasks

    def update_task_status(self, task_id: str, status: TaskStatus):
        if task_id in self.tasks:
            self.tasks[task_id].status = status  # 更新子任务的状态

    def to_dict(self):
        return {
            "tasks": {tid: task.to_dict() for tid, task in self.tasks.items()}
        }

    @staticmethod
    def from_dict(data):
        tasks = {tid: SubTask.from_dict(task_data) for tid, task_data in data["tasks"].items()}
        return TaskPlan(tasks=tasks)


# 中央信息池类，管理任务计划和执行结果
class CentralInfoPool:
    def __init__(self):
        self.task_plan = TaskPlan()  # 全局任务计划
        self.execution_results: Dict[str, Any] = {}  # 存储子任务的执行结果

    def store_result(self, task_id: str, result: Any):
        self.execution_results[task_id] = result  # 存储子任务的执行结果

    def get_result(self, task_id: str) -> Optional[Any]:
        return self.execution_results.get(task_id)  # 获取子任务的执行结果

    def to_dict(self):
        return {
            "task_plan": self.task_plan.to_dict(),
            "execution_results": self.execution_results,
        }

    @staticmethod
    def from_dict(data):
        cip = CentralInfoPool()
        cip.task_plan = TaskPlan.from_dict(data["task_plan"])
        cip.execution_results = data.get("execution_results", {})
        return cip
