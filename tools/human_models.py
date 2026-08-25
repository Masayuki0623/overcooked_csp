"""シミュレーション用の「人間役」の行動モデル。

AI(CSPAgent)は human_counterpart_mode=False で動かす。つまり「相手も CSP と
して全体最適に動く」と仮定して2体分の計画を立てる。ただし実行に使うのは
AI 自身の分だけで、人間側のキャラクターはここで定義した方策で動かす。

こうすると「AIは相手が賢いと信じて計画を立てるが、実際の相手はそう動かない」
という、現実の人間-AI協調に近い状況になる。AI 側のコードには一切触れない。

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

# タスク一覧を作り直す間隔(フレーム)。毎フレームだと重すぎる。
TASK_REFRESH_EVERY = 10


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
        # タスク一覧の組み立ては重い(1回 10〜30ms)。毎フレームやると
        # エピソード1本が数倍の時間になるので、状況が動いたときと
        # 一定間隔でだけ作り直す。人間役の判断が数フレーム古い一覧に
        # 基づくことになるが、「愚かな人間」の表現としてはむしろ自然。
        self._tasks_cache = None
        self._cache_age = 0
        # 予測と実際のズレを測るための記録
        self.pred_hits = 0
        self.pred_total = 0
        # 時間の使われ方。「判断が愚かで遅い」のか「動作に無駄があって遅い」のかを
        # 切り分けるために、1フレームずつ何をしていたかを数える。
        self.frames = 0
        self.frames_idle = 0      # 行動を返せなかった(0,0)
        self.frames_moved = 0     # 実際に位置が変わった
        self.frames_blocked = 0   # 行動は出したのに位置が変わらなかった
        self.task_switches = 0
        self._prev_pos = None
        self._prev_task = None

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

    def planned_for_me(self):
        """AI が「相手(=自分)はこれをやる」と計画している、いまのタスク。"""
        plan = (self.ai.schedule_per_agent or {}).get(self.human_idx, [])
        idx = self.ai.current_task_idx
        idx = idx.get(self.human_idx, 0) if isinstance(idx, dict) else 0
        if idx < len(plan):
            return plan[idx]['id']
        return plan[0]['id'] if plan else None

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

    def record(self, env, action):
        """1フレームぶんの結果を数える。env.step の前に呼ぶ。"""
        self.frames += 1
        pos = tuple(env.self_pos)
        if action == (0, 0):
            self.frames_idle += 1
        elif self._prev_pos is not None and pos != self._prev_pos:
            self.frames_moved += 1
        elif self._prev_pos is not None:
            # 行動は出したが位置が変わらない。壁に向かってインタラクトして
            # いる(=仕事をしている)か、相手にふさがれて進めていないか。
            self.frames_blocked += 1
        self._prev_pos = pos
        if self.current_id != self._prev_task:
            if self._prev_task is not None:
                self.task_switches += 1
            self._prev_task = self.current_id

    def time_breakdown(self):
        n = max(1, self.frames)
        return {
            'human_frames': self.frames,
            'human_idle_pct': round(100 * self.frames_idle / n, 1),
            'human_move_pct': round(100 * self.frames_moved / n, 1),
            'human_interact_pct': round(100 * self.frames_blocked / n, 1),
            'human_task_switches': self.task_switches,
        }

    def _signature(self, env):
        """状況が動いたかを見るための指紋。手持ちと位置だけで十分。"""
        holding = getattr(env.hold, 'full_name', None)
        return (tuple(env.self_pos), holding)

    # ------------------------------------------------------------------
    def _tasks(self, env, refresh):
        if refresh or self._tasks_cache is None or self._cache_age >= TASK_REFRESH_EVERY:
            tasks = [t for o in self.ai._build_order_tasks(env) for t in o['tasks']]
            if tasks:
                self.ai._annotate_task_geometry(env, tasks, env.self_pos)
            self._tasks_cache = tasks
            self._cache_age = 0
        else:
            self._cache_age += 1
        return self._tasks_cache

    def act(self, env, other_pos):
        """1フレーム分の行動を返す。"""
        # 進んでいないなら手詰まりとみなす。
        previous = self.last_signature
        signature = self._signature(env)
        if signature != previous:
            self.stuck = 0
        else:
            self.stuck += 1
        self.last_signature = signature

        # 手持ちが変わったときは世界が動いているので、一覧を作り直す。
        holding_changed = (previous or (None, None))[1] != signature[1]
        tasks = self._tasks(env, refresh=holding_changed)
        if not tasks:
            self.current_id = None
            return (0, 0), None

        options = self.available_tasks(env, tasks) or tasks

        # 何かを手に持っているなら、まずそれを片付ける。持ち物と無関係な
        # タスクを選ぶと、何をしようにも手が塞がっていて動けない。
        # 「持っているものを置きに行く/使い切る」は先読みでも全体最適でもなく、
        # 人間なら誰でもする最低限の行動なので、既存の判定をそのまま借りる。
        if env.hold is not None:
            carry = self.ai._get_carry_override_task(env, self.human_idx, None)
            if carry is not None:
                self.current_id = carry['id']
                self.ta.task_name = task_name_of(carry)
                self.ta.assigned_counter = carry.get('assigned_counter')
                action, _reason = self.ta(env, dynamic_obstacles={tuple(other_pos)})
                planned = self.planned_for_me()
                if planned is not None:
                    self.pred_total += 1
                    if carry['id'] == planned:
                        self.pred_hits += 1
                return action, carry['id']

        # AI が「相手はこれをやる」と計画しているタスク。
        # human_counterpart_mode=False では2体分の計画を立てるので、
        # 人間側スロットの現在タスクがそのまま AI の見込みになる。
        planned = self.planned_for_me()

        keep = next((t for t in options if t['id'] == self.current_id), None)
        if keep is None or self.stuck >= STUCK_LIMIT:
            # 選び直す前に、一覧が古いかもしれないので作り直す。
            tasks = self._tasks(env, refresh=True)
            options = self.available_tasks(env, tasks) or tasks
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
        if planned is not None:
            self.pred_total += 1
            if keep['id'] == planned:
                self.pred_hits += 1

        self.ta.task_name = task_name_of(keep)
        self.ta.assigned_counter = keep.get('assigned_counter')
        action, _reason = self.ta(env, dynamic_obstacles={tuple(other_pos)})
        return action, keep['id']

    def observe_plan_match(self, task_id):
        """follow_plan 用。定義上つねに計画どおりなので一致として数える。"""
        if task_id is not None:
            self.pred_total += 1
            self.pred_hits += 1

    @property
    def prediction_match_rate(self):
        """AIの予測と、実際に人間役が取り掛かったタスクの一致率。"""
        if not self.pred_total:
            return None
        return self.pred_hits / self.pred_total
