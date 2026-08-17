"""指示が「注文番号なしの行動単位」でまとめられることの検証。

同じ (動詞, 対象) のタスクが複数の注文にまたがっていても、指示として選べるのは
1つだけにする。CSP 側はそのうち「どれか1つ」が d 以内に実行されればよい、
という制約として扱う。
"""
import unittest

from agent.agent.myagent.CSPAgent import CSPAgent


class InstructionGroupingTests(unittest.TestCase):
    def setUp(self):
        self.agent = CSPAgent(sc_2agent=True)

    def _candidates_from(self, tids):
        """get_instruction_candidates の集約部分だけを、既知の tids で再現する。"""
        self.agent.get_remaining_tids = lambda env, orders: set(tids)
        self.agent._build_order_tasks = lambda env: []
        return self.agent.get_instruction_candidates(env=None)

    def test_same_action_across_orders_is_one_candidate(self):
        """cut onion が2つの注文にあっても候補は1つ。"""
        cands = self._candidates_from([
            ('chop', 'onion', 0),
            ('chop', 'onion', 1),
            ('chop', 'tomato', 1),
        ])
        displays = sorted(d for d, _ in cands)
        self.assertEqual(displays, ['chop_onion', 'chop_tomato'])

    def test_display_has_no_order_number(self):
        cands = self._candidates_from([('chop', 'onion', 2), ('cook', 'onion-tomato soup', 2)])
        for display, _payload in cands:
            self.assertNotIn('order', display)
            self.assertNotIn('(', display)

    def test_payload_keeps_every_matching_order(self):
        """まとめた候補は、対象となる全注文を保持する(制約でOR条件にするため)。"""
        cands = self._candidates_from([('chop', 'onion', 0), ('chop', 'onion', 1)])
        self.assertEqual(len(cands), 1)
        _display, payload = cands[0]
        self.assertEqual(payload['verb'], 'chop')
        self.assertEqual(payload['obj'], 'onion')
        self.assertEqual(sorted(payload['order_uids']), [0, 1])
        self.assertEqual(len(payload['fixed_task_ids']), 2)

    def test_extract_action_ignores_order_number(self):
        pending = {'task': ('chop_onion', {'verb': 'chop', 'obj': 'onion', 'order_uids': [0, 1]})}
        self.assertEqual(self.agent._extract_instruction_action(pending), ('chop', 'onion'))

    def test_extract_action_falls_back_to_legacy_payload(self):
        """旧形式(単一 fixed_task_id)の指示からも行動を復元できる。"""
        pending = {'task': ('chop_onion', {'fixed_task_id': ('task', 'chop', 'onion', 3)})}
        self.assertEqual(self.agent._extract_instruction_action(pending), ('chop', 'onion'))

    def test_group_indices_match_every_order(self):
        tasks = [
            {'verb': 'chop', 'obj': 'onion', 'order': 0},
            {'verb': 'chop', 'obj': 'tomato', 'order': 0},
            {'verb': 'chop', 'obj': 'onion', 'order': 1},
        ]
        self.assertEqual(
            self.agent._find_group_task_indices(tasks, ('chop', 'onion')), [0, 2]
        )
        self.assertEqual(
            self.agent._find_group_task_indices(tasks, ('chop', 'lettuce')), []
        )


if __name__ == '__main__':
    unittest.main()
