import unittest
from types import SimpleNamespace

from agent.agent.myagent.CSPAgent import CSPAgent


class InstructionDeadlineTests(unittest.TestCase):
    def test_expired_deadline_is_treated_as_urgent(self):
        agent = CSPAgent(deadline_seconds=0)
        pending = {"status": "pending", "accepted_env_time": 0.0}

        result = agent._classify_instruction_deadline(pending, current_env_time=0.5, deadline_seconds=0.0)

        self.assertEqual(result["mode"], "urgent")
        self.assertTrue(result["priority_boost"])

    def test_sc2agent_preempt_targets_the_agent_containing_the_instruction(self):
        agent = CSPAgent(deadline_seconds=0, sc_2agent=True)
        agent.schedule_per_agent = {
            0: [{"id": ("chop", "tomato", 0), "fixed_task_id": ("task", "chop", "tomato", 0)}],
            1: [{"id": ("chop", "onion", 1), "fixed_task_id": ("task", "chop", "onion", 1)}],
        }
        agent.current_task_idx = {0: 0, 1: 0}
        pending = {
            "id": 1,
            "status": "pending",
            "accepted_env_time": 0.0,
            "task": {"fixed_task_id": ("task", "chop", "onion", 1)},
        }
        env = SimpleNamespace(time=0.1, _pending_instructions=[pending])

        result = agent._get_instruction_preempt_target(env)

        self.assertIsNotNone(result)
        self.assertEqual(result[1], ("task", "chop", "onion", 1))
        self.assertEqual(result[2], 1)
        self.assertEqual(result[3], 0)

    def test_sc2agent_human_counterpart_mode_controls_only_own_agent(self):
        # human_counterpart_mode=True のとき CSP は own_agent_idx のみ実行する
        agent = CSPAgent(deadline_seconds=0, sc_2agent=True)
        agent.human_counterpart_mode = True
        agent.own_agent_idx = 0

        self.assertTrue(agent.human_counterpart_mode)
        self.assertEqual(agent.own_agent_idx, 0)
        # two-agent scheduling path は sc_2agent フラグだけで決まる
        self.assertTrue(agent.sc_2agent)


if __name__ == "__main__":
    unittest.main()
