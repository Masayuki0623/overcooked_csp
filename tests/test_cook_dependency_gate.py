"""cook タスクの前提判定が、TaskAgent が実際に消費できる状態だけを数えることの検証。

CSPAgent._cook_dependency_ready_from_world が Fresh(未カット)を「材料あり」と
数えてしまうと、人間が未カット食材をカウンターに置いただけで cook タスクが
開始可能と判定される。しかし TaskAgent.process_cook_task は Chopped* しか
消費できないため、AI はその場から動かず「必要な食材 (Chopped) を待機中」を
延々と返し続けて停止する。この不整合が起きないことを確認する。
"""
import unittest
from types import SimpleNamespace

from gym_cooking.utils.core import Onion, Tomato, Object, FoodState

from agent.agent.myagent.CSPAgent import CSPAgent


def make_env(objects):
    """pos_obj だけを持つ最小の env スタブ。"""
    return SimpleNamespace(
        pos_obj={obj.location: obj for obj in objects},
        pos_gs={},
        world=None,
        agents=[],
    )


def food(cls, state=None):
    item = cls()
    if state is not None:
        item.set_state(state)
    return item


class CookDependencyGateTests(unittest.TestCase):
    def setUp(self):
        self.agent = CSPAgent(sc_2agent=True)
        self.agent.human_counterpart_mode = True

    def test_fresh_ingredient_does_not_satisfy_cook_dependency(self):
        """未カット食材だけがある状態で cook を開始可能と判定してはいけない。"""
        env = make_env([
            Object(location=(1, 0), contents=[food(Onion)]),
            Object(location=(2, 0), contents=[food(Tomato)]),
        ])
        self.assertFalse(
            self.agent._cook_dependency_ready_from_world(env, 'onion-tomato soup')
        )

    def test_chopped_ingredients_satisfy_cook_dependency(self):
        """刻んだ食材が揃っていれば cook を開始できる。"""
        env = make_env([
            Object(location=(1, 0), contents=[food(Onion, FoodState.CHOPPED)]),
            Object(location=(2, 0), contents=[food(Tomato, FoodState.CHOPPED)]),
        ])
        self.assertTrue(
            self.agent._cook_dependency_ready_from_world(env, 'onion-tomato soup')
        )

    def test_partially_fresh_does_not_satisfy_cook_dependency(self):
        """片方だけ刻まれている場合も cook は開始できない。"""
        env = make_env([
            Object(location=(1, 0), contents=[food(Onion, FoodState.CHOPPED)]),
            Object(location=(2, 0), contents=[food(Tomato)]),
        ])
        self.assertFalse(
            self.agent._cook_dependency_ready_from_world(env, 'onion-tomato soup')
        )

    def test_fresh_still_counts_for_general_inventory_query(self):
        """cook 前提以外の在庫問い合わせでは、従来どおり Fresh も数える。"""
        env = make_env([Object(location=(1, 0), contents=[food(Onion)])])
        self.assertTrue(self.agent._owns_world_ingredient(env, 'onion'))
        self.assertFalse(
            self.agent._owns_world_ingredient(env, 'onion', require_ready_to_cook=True)
        )


if __name__ == '__main__':
    unittest.main()
