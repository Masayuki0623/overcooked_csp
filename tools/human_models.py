"""シミュレーション用の「人間役」の行動モデル。

CSPAgent は human_counterpart_mode=True のとき自分の担当ぶんしか動かさない
(exec_indices = [own_agent_idx])。空いた人間側のキャラクターを、ここで定義した
方策で動かす。AI 側のコードには一切触れないので、「AIは人間が最適に動くと
予測して計画する / 実際の人間はそう動かない」というズレをそのまま再現できる。

方策は3つ。
  follow_plan : AIが人間スロットに置いた計画どおりに動く。ズレが起きない上限条件
  greedy      : いま着手できるタスクのうち、自分から一番近いものを選ぶ
  random      : いま着手できるタスクからランダムに1つ選ぶ

greedy / random は「愚かだが仕事は回る」水準に留めてある。具体的には
  * 前提(依存タスク)が揃っているタスクだけを選ぶ
  * 一度選んだら、そのタスクが消えるか手詰まりになるまで続ける
だけで、「AIの計画を先読みして待つ」「全体のバランスを見る」といった賢さは
持たせない。賢くすると optimal に近づいてしまい、見たいズレが消える。
"""
import random

from agent.myagent.CSPAgent import dish_ingredients
from agent.myagent.TaskAgent import TaskAgent

MODELS = ('follow_plan', 'greedy', 'random')

# 同じタスクを続けても状況が変わらないまま何フレーム我慢するか。
# これを超えたら「手詰まり」とみなして選び直す。短すぎると迷い続け、
# 長すぎると詰まったまま固まる。
STUCK_LIMIT = 30


def task_name_of(task):
    """CSP のタスクを TaskAgent が解釈できる名前にする(CSPAgent と同じ規則)。"""
    verb, obj, _uid = task['id']
    if verb == 'chop':
        return f'chop_{obj}'
    return f"{verb}_{'_'.join(dish_ingredients(obj))}"


class HumanModel:
    """人間役のキャラクターを動かす。"""

    def __init__(self, model, ai, human_idx, replay, seed=0):
        assert model in MODELS, model
        self.model = model
        self.ai = ai
        self.human_idx = human_idx
        self.ta = TaskAgent(10, replay)
        self.rng = random.Random(seed)
        self.current_id = None
        self.stuck = 0
        self.last_signature = None
        # 予測と実際のズレを測るための記録
        self.pred_hits = 0
        self.pred_total = 0

    # ------------------------------------------------------------------
    def available_tasks(self, env, tasks):
        """いま自分が着手できるタスク。

        前提が揃っていないもの(まだ材料が刻まれていない cook など)を選ぶと、
        延々と待つだけで仕事が回らない。CSPAgent が仮想状態で使っている
        判定をそのまま借りて、着手可能なものだけに絞る。
        """
        remaining = {t['id'] for t in tasks}
        out = []
        for t in tasks:
            if self.human_idx not in self.ai._task_allowed_agents(env, t):
                continue
            if not self.ai._task_is_available_in_virtual_state(t, remaining):
                continue
            out.append(t)
        return out

    def pick(self, env, tasks):
        """方策に従ってタスクを1つ選ぶ。"""
        if self.model == 'follow_plan':
            plan = (self.ai.schedule_per_agent or {}).get(self.human_idx, [])
            for entry in plan:
                match = next((t for t in tasks if t['id'] == entry['id']), None)
                if match is not None:
                    return match
            return tasks[0] if tasks else None

        if self.model == 'random':
            return self.rng.choice(tasks) if tasks else None

        # greedy: 自分の現在地から一番近いものを選ぶだけ。先読みはしない。
        def distance(t):
            d = self.ai.astar_distance(env, env.self_pos,
                                       t.get('start_pos') or env.self_pos)
            return d if d is not None else 10 ** 6

        return min(tasks, key=distance) if tasks else None

    def _signature(self, env):
        """状況が動いたかを見るための指紋。手持ちと位置だけで十分。"""
        holding = getattr(env.hold, 'full_name', None)
        return (tuple(env.self_pos), holding)

    # ------------------------------------------------------------------
    def act(self, env, other_pos):
        """1フレーム分の行動を返す。"""
        tasks = [t for o in self.ai._build_order_tasks(env) for t in o['tasks']]
        if not tasks:
            self.current_id = None
            return (0, 0), None

        self.ai._annotate_task_geometry(env, tasks, env.self_pos)
        options = self.available_tasks(env, tasks) or tasks

        # 予測が当たったかを、選び直す前の時点で記録する。
        predicted = {p['id'] for p in (self.ai.predicted_human_tasks or [])}

        # 進んでいないなら手詰まりとみなす。
        signature = self._signature(env)
        if signature == self.last_signature:
            self.stuck += 1
        else:
            self.stuck = 0
        self.last_signature = signature

        keep = next((t for t in options if t['id'] == self.current_id), None)
        if keep is None or self.stuck >= STUCK_LIMIT:
            if self.stuck >= STUCK_LIMIT:
                # 同じタスクで詰まったら、それ以外から選び直す。
                options = [t for t in options if t['id'] != self.current_id] or options
                self.stuck = 0
            keep = self.pick(env, options)
        if keep is None:
            self.current_id = None
            return (0, 0), None

        self.current_id = keep['id']
        if predicted:
            self.pred_total += 1
            if keep['id'] in predicted:
                self.pred_hits += 1

        self.ta.task_name = task_name_of(keep)
        self.ta.assigned_counter = keep.get('assigned_counter')
        action, _reason = self.ta(env, dynamic_obstacles={tuple(other_pos)})
        return action, keep['id']

    @property
    def prediction_match_rate(self):
        """AIの予測と、実際に人間役が取り掛かったタスクの一致率。"""
        if not self.pred_total:
            return None
        return self.pred_hits / self.pred_total
