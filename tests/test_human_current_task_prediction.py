"""「人間がいま手をつけているタスク」の推測と、それを人間の最初のタスクへ
強制割り当てすることの検証。

推測の手がかりは2段階:
  1. 持ち物(未カット/刻んだ食材を持っていれば、その食材の作業中とみなす)
  2. 分からなければ、位置から一番早く終わるタスク(既存の貪欲予測を1件だけ再利用)
"""
import unittest
from types import SimpleNamespace

from agent.agent.myagent.CSPAgent import CSPAgent


def make_tasks():
    return [
        {'id': ('chop', 'onion', 0), 'verb': 'chop', 'obj': 'onion', 'order': 0},
        {'id': ('chop', 'tomato', 0), 'verb': 'chop', 'obj': 'tomato', 'order': 0},
        {'id': ('cook', 'onion-tomato soup', 0), 'verb': 'cook',
         'obj': 'onion-tomato soup', 'order': 0},
    ]


def make_env(human_holding):
    holding = SimpleNamespace(full_name=human_holding) if human_holding else None
    return SimpleNamespace(agents=[
        SimpleNamespace(holding=None, location=(1, 1)),          # AI (own_agent_idx=0)
        SimpleNamespace(holding=holding, location=(5, 5)),       # 人間
    ])


class HumanCurrentTaskPredictionTests(unittest.TestCase):
    def setUp(self):
        self.agent = CSPAgent(sc_2agent=True)
        self.agent.human_counterpart_mode = True
        self.agent.own_agent_idx = 0

    def test_fresh_ingredient_identifies_chop_task(self):
        """未カットの玉ねぎを持っていれば『玉ねぎを切る』作業中と推測する。"""
        task = self.agent._predict_human_current_task(
            make_env('FreshOnion'), make_tasks(), (5, 5))
        self.assertIsNotNone(task)
        self.assertEqual(task['id'], ('chop', 'onion', 0))

    def test_chopped_ingredient_identifies_same_ingredient(self):
        """刻んだトマトを運んでいる途中も、その食材の作業とみなす。"""
        task = self.agent._predict_human_current_task(
            make_env('ChoppedTomato'), make_tasks(), (5, 5))
        self.assertEqual(task['id'], ('chop', 'tomato', 0))

    def test_falls_back_to_greedy_when_empty_handed(self):
        """手ぶらなら、既存の貪欲予測へフォールバックする。"""
        calls = []

        def fake_greedy(env, tasks, human_start_pos, limit=None):
            calls.append((human_start_pos, limit))
            return [{'id': tasks[1]['id'], 'start': 0, 'end': 5, 'task': tasks[1]}]

        self.agent._predict_human_greedy_tasks = fake_greedy
        task = self.agent._predict_human_current_task(
            make_env(None), make_tasks(), (5, 5))
        self.assertEqual(task['id'], ('chop', 'tomato', 0))
        # 人間の実座標が使われていること(AI の座標ではない)
        self.assertEqual(calls, [((5, 5), 1)])

    def test_prediction_is_sticky_while_task_remains(self):
        """一度推測したタスクは、残タスクにある限り毎回変えない。

        位置ベースの推測は人間が歩くたびに結果が変わる。毎回変えると
        人間スロットに固定するタスクが入れ替わり、CSPの解ごとAIの計画が
        組み替わって、まな板に置いた直後に別タスクへ飛ばされる等の
        不安定な動きを招く。
        """
        tasks = make_tasks()
        calls = []

        def fake_greedy(env, tasks_, human_start_pos, limit=None):
            calls.append(human_start_pos)
            return [{'id': tasks_[0]['id'], 'start': 0, 'end': 5, 'task': tasks_[0]}]

        self.agent._predict_human_greedy_tasks = fake_greedy
        first = self.agent._predict_human_current_task(make_env(None), tasks, (5, 5))
        # 人間が移動しても推測は変わらない
        second = self.agent._predict_human_current_task(make_env(None), tasks, (1, 1))
        self.assertEqual(first['id'], second['id'])
        self.assertEqual(len(calls), 1, "2回目は貪欲予測を呼び直さない")

    def test_prediction_updates_when_task_disappears(self):
        """推測したタスクが残タスクから消えたら、改めて推測し直す。"""
        tasks = make_tasks()
        self.agent._predict_human_greedy_tasks = (
            lambda env, t, human_start_pos, limit=None:
            [{'id': t[0]['id'], 'start': 0, 'end': 5, 'task': t[0]}])
        first = self.agent._predict_human_current_task(make_env(None), tasks, (5, 5))
        remaining = [t for t in tasks if t['id'] != first['id']]
        again = self.agent._predict_human_current_task(make_env(None), remaining, (5, 5))
        self.assertNotEqual(again['id'], first['id'])

    def test_no_tasks_returns_none(self):
        self.assertIsNone(self.agent._predict_human_current_task(make_env('FreshOnion'), [], (5, 5)))

    def test_holding_unrelated_item_falls_back(self):
        """レシピに無い物(皿など)を持っていても落ちずに貪欲予測へ回る。"""
        self.agent._predict_human_greedy_tasks = lambda *a, **k: []
        self.assertIsNone(
            self.agent._predict_human_current_task(make_env('Plate'), make_tasks(), (5, 5)))


if __name__ == '__main__':
    unittest.main()
