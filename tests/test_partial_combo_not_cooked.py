"""作りかけの組み合わせを、そのまま鍋に入れてしまわないことの検証。

3種スープ(レタス・玉ねぎ・トマト)を作る途中、AIは「レタス+玉ねぎ」を持って
3つ目のトマトを取りに行く。このとき持ち物から cook タスクを合成する経路が
「レタス+玉ねぎスープ」という2種のレシピを作ってしまうと、そのまま鍋に入れて
調理が確定し、注文が永久に完成しなくなる。

作りかけの場合は、その注文の完成レシピを目標にすることを確認する。
"""
import unittest
from types import SimpleNamespace

from agent.agent.myagent.CSPAgent import CSPAgent


def make_env(recipe_names, holding_name):
    """current_orders と agents だけを持つ最小の env スタブ。"""
    orders = [(SimpleNamespace(full_name=name),) for name in recipe_names]
    holding = SimpleNamespace(full_name=holding_name) if holding_name else None
    return SimpleNamespace(
        order=SimpleNamespace(current_orders=orders),
        agents=[SimpleNamespace(holding=holding, location=(1, 1))],
    )


class PartialComboTests(unittest.TestCase):
    def setUp(self):
        self.agent = CSPAgent(sc_2agent=True)
        self.agent.sc_2agent = False
        self.agent.carry_task_by_agent = None
        self.agent.active_order_entries = []

    def _override(self, env, scheduled_task=None):
        return self.agent._get_carry_override_task(env, 0, scheduled_task)

    def test_partial_combo_targets_full_recipe(self):
        """レタス+玉ねぎを持っているとき、3種スープを目標にする。"""
        env = make_env(['CookedLettuce-CookedOnion-CookedTomato-Plate'],
                       'ChoppedLettuce-ChoppedOnion')
        task = self._override(env)
        self.assertEqual(task['id'][0], 'cook')
        self.assertEqual(task['id'][1], 'lettuce-onion-tomato soup')

    def test_complete_combo_is_cooked_as_is(self):
        """3種そろっていれば、そのまま調理してよい。"""
        env = make_env(['CookedLettuce-CookedOnion-CookedTomato-Plate'],
                       'ChoppedLettuce-ChoppedOnion-ChoppedTomato')
        task = self._override(env)
        self.assertEqual(task['id'][1], 'lettuce-onion-tomato soup')

    def test_exact_match_preferred_over_superset(self):
        """2種の注文が実在するなら、その2種スープとして扱う。"""
        env = make_env(['CookedLettuce-CookedOnion-Plate',
                        'CookedLettuce-CookedOnion-CookedTomato-Plate'],
                       'ChoppedLettuce-ChoppedOnion')
        task = self._override(env)
        self.assertEqual(task['id'][1], 'lettuce-onion soup')

    def test_no_matching_order_falls_back_to_held_combo(self):
        """どの注文にも該当しない組み合わせは従来どおり持ち物のまま扱う。"""
        env = make_env(['CookedTomato-Plate'], 'ChoppedLettuce-ChoppedOnion')
        task = self._override(env)
        self.assertEqual(task['id'][1], 'lettuce-onion soup')

    def test_helper_finds_smallest_superset(self):
        env = make_env(['CookedLettuce-CookedOnion-CookedTomato-Plate',
                        'CookedLettuce-CookedOnion-Plate'], None)
        parts, _counter = self.agent._find_order_recipe_for_partial(env, ['lettuce'])
        self.assertEqual(parts, ['lettuce', 'onion'])


if __name__ == '__main__':
    unittest.main()
