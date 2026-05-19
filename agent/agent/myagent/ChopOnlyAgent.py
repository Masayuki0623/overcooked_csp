from .TaskAgent import TaskAgent


class ChopOnlyAgent:
    def __init__(self, speed=2.5, replay=None):
        self.task_agent = TaskAgent(speed=speed, replay=replay, task_name=None)

    def __getattr__(self, name):
        return getattr(self.task_agent, name)

    def _get_available_chop_tasks(self, env):
        tasks = []
        for order_tuple in getattr(env.order, 'current_orders', []):
            goal_obj = order_tuple[0]
            name = getattr(goal_obj, 'full_name', '').lower()
            for ingredient in ('lettuce', 'onion', 'tomato'):
                task_name = f"chop_{ingredient}"
                if ingredient in name and task_name not in tasks:
                    tasks.append(task_name)
        return tasks

    def __call__(self, env, dynamic_obstacles=None):
        task_name = self.task_agent.task_name
        available_tasks = self._get_available_chop_tasks(env)
        if task_name is not None:
            if task_name not in available_tasks:
                self.task_agent.task_name = None
                task_name = None

        selected_new_task = False
        if task_name is None:
            task_name = self.task_agent.choose_random_chop_task_name(env)
            selected_new_task = task_name is not None

        if task_name is None:
            self.task_agent.task_name = None
            return (0, 0), "chop候補なし"

        self.task_agent.task_name = task_name
        if selected_new_task:
            print(f"[ChopOnlyAgent] TaskAgentをそのまま再利用: {self.task_agent.task_name}")

        action, reason = self.task_agent(env, dynamic_obstacles=dynamic_obstacles)
        if "完了" in reason or "Done" in reason or "done" in reason:
            self.task_agent.task_name = None
        return action, reason