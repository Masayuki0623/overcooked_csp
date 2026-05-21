from copy import copy, deepcopy

from agent.executor.low import bfs_reachable, bfs_search_all, GridSquare


class DualAgentController:
    def __init__(self, agent0, agent1):
        self.agent0 = agent0
        self.agent1 = agent1

    def _clone_env_for_agent(self, env, agent_idx):
        cloned_env = copy(env)
        cloned_env.agent_idx = agent_idx

        cloned_env.all_obj_h = cloned_env.all_obj + ([cloned_env.hold] if cloned_env.hold is not None else [])
        cloned_env.all_obj_a = cloned_env.all_obj + [agent.holding for agent in cloned_env.agents if agent.holding is not None]

        cloned_env.to_grid = [[1 for _ in range(cloned_env.world_height)] for _ in range(cloned_env.world_width)]
        for obj in cloned_env.world_all:
            if obj.collidable:
                cloned_env.to_grid[obj.location[0]][obj.location[1]] = 0

        cloned_env.to_grid_a = deepcopy(cloned_env.to_grid)
        for other_agent in cloned_env.agents[:cloned_env.agent_idx] + cloned_env.agents[cloned_env.agent_idx + 1:]:
            cloned_env.to_grid_a[other_agent.location[0]][other_agent.location[1]] = 0

        cloned_env.rch_map = bfs_reachable(cloned_env.to_grid, cloned_env.self_pos)
        cloned_env.rch_obj = [obj for obj in cloned_env.all_obj if cloned_env.rch_map[obj.location[0]][obj.location[1]]]
        cloned_env.rch_obj_h = cloned_env.rch_obj + ([cloned_env.hold] if cloned_env.hold is not None else [])
        cloned_env.rch_grid = [
            grid_obj for grid_obj in cloned_env.world_all
            if isinstance(grid_obj, GridSquare) and cloned_env.rch_map[grid_obj.location[0]][grid_obj.location[1]]
        ]

        cloned_env.bfs_search_a = bfs_search_all(cloned_env.to_grid_a, cloned_env.self_pos)
        cloned_env.bfs_search = bfs_search_all(cloned_env.to_grid, cloned_env.self_pos)

        return cloned_env

    def _normalize_action(self, result, preferred_key=None):
        if isinstance(result, tuple):
            return result
        if isinstance(result, dict):
            if preferred_key is not None and preferred_key in result:
                return result[preferred_key]
            if "ai" in result:
                return result["ai"]
            if result:
                return next(iter(result.values()))
        return (0, 0)

    def __call__(self, env):
        env0 = self._clone_env_for_agent(env, 0)
        env1 = self._clone_env_for_agent(env, 1)

        move0, reason0 = self.agent0(env0)
        move1, reason1 = self.agent1(env1)

        action0 = self._normalize_action(move0)
        action1 = self._normalize_action(move1, preferred_key="ai_1")

        return {"ai_0": action0, "ai_1": action1}, f"{reason0} | {reason1}"

    def high_level_infer(self, env, chat):
        if hasattr(self.agent1, "high_level_infer"):
            return self.agent1.high_level_infer(env, chat)
        return None

    def get_assigned_counters(self):
        if hasattr(self.agent1, "get_assigned_counters"):
            return self.agent1.get_assigned_counters()
        return {}

    @property
    def skill_estimator(self):
        return getattr(self.agent1, "skill_estimator", None)

    @property
    def skill_estimation_log(self):
        return getattr(self.agent1, "skill_estimation_log", None)

    @property
    def skill_estimation_alpha(self):
        return getattr(self.agent1, "skill_estimation_alpha", 0.3)

    def calculate_skill_estimation_from_log(self, skill_estimation_log=None, emit_logs=False):
        if hasattr(self.agent1, "calculate_skill_estimation_from_log"):
            return self.agent1.calculate_skill_estimation_from_log(
                skill_estimation_log=skill_estimation_log,
                emit_logs=emit_logs,
            )
        raise AttributeError("agent1 does not support deferred skill estimation calculation")
