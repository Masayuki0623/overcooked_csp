import unittest
from types import SimpleNamespace

from ortools.sat.python import cp_model

from agent.agent.myagent.CSPAgent import CSPAgent


class InstructionInterruptTests(unittest.TestCase):
    def test_deadline_zero_instruction_preempts_current_task(self):
        agent = CSPAgent(deadline_seconds=0)
        agent.schedule = [
            {"id": ("chop", "tomato", 0), "verb": "chop", "obj": "tomato", "order": 0, "res": ("cutboard", None), "end": 10},
            {"id": ("chop", "onion", 0), "verb": "chop", "obj": "onion", "order": 0, "fixed_task_id": ("task", "chop", "onion", 0), "res": ("cutboard", None), "end": 20},
        ]
        agent.current_task_idx = 0

        class DummyTaskAgent:
            def __init__(self):
                self.task_name = ""
                self.assigned_counter = None
                self.assigned_cutboard = None
                self.assigned_pot = None
                self.assigned_plate = None
                self.assigned_serve_loc = None

            def __call__(self, env):
                return (0, 0), "Done"

        agent.task_agent = DummyTaskAgent()

        env = SimpleNamespace(
            agents=[SimpleNamespace(location=(0, 0), holding=None)],
            time=0.0,
            order=SimpleNamespace(current_orders=[]),
        )

        pending = {
            "id": 1,
            "task": {"fixed_task_id": ("task", "chop", "onion", 0)},
            "accepted_env_time": 0.0,
            "status": "pending",
            "execution_logged": False,
            "deadline_constraint_applied": False,
        }
        agent._pending_instructions = [pending]
        env._pending_instructions = [pending]

        agent._build_order_tasks = lambda env: []
        agent._stabilize_task_ids_for_held_progress = lambda env, current_task_ids: current_task_ids
        agent._should_defer_holding_reschedule = lambda env, added, removed: False
        agent._get_active_task_ids = lambda: {("chop", "tomato", 0)}
        agent._apply_instruction_deadline_constraints = lambda *args, **kwargs: None
        agent.solve_csp_scheduling = lambda env, orders: agent.schedule

        action, reason = agent(env)

        self.assertEqual(action, (0, 0))
        self.assertEqual(reason, "Done")
        self.assertEqual(agent.current_task_idx, 1)

    def test_execution_logged_does_not_cancel_pending_instruction(self):
        agent = CSPAgent(deadline_seconds=0)
        model = cp_model.CpModel()
        start_var = model.NewIntVar(0, 1000, "start")
        pending = {
            "id": 2,
            "task": {"fixed_task_id": ("task", "chop", "onion", 1)},
            "accepted_env_time": 0.0,
            "status": "pending",
            "execution_logged": True,
            "deadline_constraint_applied": False,
        }
        agent._pending_instructions = [pending]
        env = SimpleNamespace(time=0.0, _pending_instructions=[pending])
        tasks = [{
            "id": ("chop", "onion", 1),
            "verb": "chop",
            "obj": "onion",
            "order": 1,
            "fixed_task_id": ("task", "chop", "onion", 1),
        }]
        agent._apply_instruction_deadline_constraints(model, tasks, {0: start_var}, env)

        self.assertIn(pending["status"], {"pending", "started"})
        self.assertTrue(pending["deadline_constraint_applied"])


if __name__ == "__main__":
    unittest.main()
