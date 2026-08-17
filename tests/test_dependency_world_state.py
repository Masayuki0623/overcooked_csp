import unittest
from types import SimpleNamespace

from gym_cooking.utils.core import FoodState, Onion, Pot, Tomato

from agent.agent.myagent.CSPAgent import CSPAgent


def build_env(tomato, onion):
    """agent.holding と gridsquare.holding の両方に食材を置いた env スタブ。"""
    pot = Pot((2, 2))
    pot.holding = onion
    return SimpleNamespace(
        pos_obj={},
        hold=None,
        agents=[SimpleNamespace(holding=tomato), SimpleNamespace(holding=None)],
        world=SimpleNamespace(objects={"Pot": [pot]}),
    )


class DependencyWorldStateTests(unittest.TestCase):
    def test_dependency_checks_traverse_agent_and_grid_holding(self):
        """在庫判定が agent.holding と gridsquare.holding までたどれること。"""
        agent = CSPAgent()
        env = build_env(Tomato(), Onion())

        self.assertTrue(agent._owns_world_ingredient(env, "tomato"))
        self.assertTrue(agent._owns_world_ingredient(env, "onion"))

    def test_cook_dependency_traverses_holdings_when_chopped(self):
        """刻んだ食材なら、持ち物や鍋の中にあっても cook の前提を満たす。"""
        agent = CSPAgent()
        chopped_tomato = Tomato()
        chopped_tomato.set_state(FoodState.CHOPPED)
        chopped_onion = Onion()
        chopped_onion.set_state(FoodState.CHOPPED)

        env = build_env(chopped_tomato, chopped_onion)

        self.assertTrue(agent._cook_dependency_ready_from_world(env, "tomato-onion soup"))

    def test_cook_dependency_rejects_unchopped_holdings(self):
        """未カットのままでは cook の前提を満たさない。

        process_cook_task は Chopped* しか消費できないため、ここで True を返すと
        AI は cook タスクに入ったまま何も出来ず永久に待機してしまう。
        """
        agent = CSPAgent()
        env = build_env(Tomato(), Onion())

        self.assertFalse(agent._cook_dependency_ready_from_world(env, "tomato-onion soup"))


if __name__ == "__main__":
    unittest.main()
