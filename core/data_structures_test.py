from core.data_structures import SubTask, TaskStatus, TaskPlan, CentralInfoPool


def main():
    # 创建子任务
    subtask1 = SubTask(task_id="t1", description="子任务1", dependencies=[])
    subtask2 = SubTask(task_id="t2", description="子任务2", dependencies=["t1"])
    plan = TaskPlan()
    plan.add_task(subtask1)
    plan.add_task(subtask2)

    print("任务列表：", list(plan.tasks.keys()))
    print("t1描述：", plan.tasks["t1"].description)

    # 获取可执行任务
    ready = plan.get_ready_tasks()
    print("可执行任务：", [t.task_id for t in ready])

    # 完成t1后，t2才可执行
    plan.update_task_status("t1", TaskStatus.COMPLETED)
    ready = plan.get_ready_tasks()
    print("t1完成后可执行任务：", [t.task_id for t in ready])

    # 更新任务状态
    plan.update_task_status("t1", TaskStatus.IN_PROGRESS)
    print("t1状态：", plan.tasks["t1"].status)

    # CentralInfoPool演示
    pool = CentralInfoPool()
    pool.task_plan.add_task(subtask1)
    pool.store_result("t1", {"result": 123})
    print("t1结果：", pool.get_result("t1"))
    print("不存在任务结果：", pool.get_result("not_exist"))


if __name__ == "__main__":
    main()
