import unittest
from types import SimpleNamespace

from gym_cooking.utils.core import Onion, Pot, Tomato

from agent.agent.myagent.CSPAgent import CSPAgent


class DependencyWorldStateTests(unittest.TestCase):
    def test_dependency_checks_traverse_agent_and_grid_holding(self):
        agent = CSPAgent()

        fresh_tomato = Tomato()
        fresh_onion = Onion()

        pot = Pot((2, 2))
        pot.holding = fresh_onion

        env = SimpleNamespace(
            pos_obj={},
            hold=None,
            agents=[SimpleNamespace(holding=fresh_tomato), SimpleNamespace(holding=None)],
            world=SimpleNamespace(objects={"Pot": [pot]}),
        )

        self.assertTrue(agent._owns_world_ingredient(env, "tomato"))
        self.assertTrue(agent._owns_world_ingredient(env, "onion"))
        self.assertTrue(agent._cook_dependency_ready_from_world(env, "tomato-onion soup"))


if __name__ == "__main__":
    unittest.main()
