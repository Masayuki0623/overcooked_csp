"""シミュレーション用の「人間役」の行動モデル。

AI(CSPAgent)は human_counterpart_mode=False で動かす。つまり「相手も CSP と
して全体最適に動く」と仮定して2体分の計画を立てる。ただし実行に使うのは
AI 自身の分だけで、人間側のキャラクターはここで定義した方策で動かす。

こうすると「AIは相手が賢いと信じて計画を立てるが、実際の相手はそう動かない」
という、現実の人間-AI協調に近い状況になる。AI 側のコードには一切触れない。

方策は3つ。いずれも「選択できるタスクの中から1つ選び、TaskAgent に実行
させる」形で、選び方だけが違う。
  follow_plan : AIが人間スロットに置いた計画どおりに選ぶ。ズレが起きない上限条件
  greedy      : 完了までが最短のタスクを選ぶ(移動 + 作業時間)
  random      : 選択できるタスクから一様にランダムで選ぶ

選び直すのは次の2つの場合だけ。
  * 実行中のタスクが終わったとき
  * 他のエージェントがやってしまい、もうやらなくてよくなったとき
どちらも「そのタスクが残タスク一覧から消える」という同じ形で判定できる。

greedy / random は「愚かだが仕事は回る」水準に留めてある。前提が揃った
タスクだけを選び、手が塞がっていれば片付ける、それだけ。「AIの計画を
先読みして待つ」「全体のバランスを見る」といった賢さは持たせない。
賢くすると optimal に近づいてしまい、見たいズレが消える。
"""
import random

from agent.myagent.CSPAgent import dish_ingredients
from agent.myagent.TaskAgent import TaskAgent

MODELS = ('follow_plan', 'greedy', 'random')

# 選び直すのは「タスクが終わった/不要になった」ときだけ。ただし何らかの
# 理由で永久に進めなくなると計測が潰れるので、安全網として、まったく状況が
# 動かないまま これだけのフレームが過ぎたら選び直す。実際に発動した回数は
# stuck_switches として記録し、頻発するなら実装側の問題として扱う。
STUCK_LIMIT = 100

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
        self.stuck_switches = 0
        # 進まないタスクを一定時間だけ避けるための記録。
        # 人は「これは今できない」と分かれば別の仕事に移る。
        self._blocked_since = {}
        self._cooldown = {}
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

        # greedy: 完了までが最短のタスクを選ぶ(そこまでの移動 + 作業時間)。
        # 先読みはせず、いまの自分の位置だけから見た短さで決める。
        def finish_cost(t):
            _approach, total = self.ai._estimate_virtual_task_finish(
                env, t, env.self_pos)
            return total if total is not None else 10 ** 6

        return min(tasks, key=finish_cost) if tasks else None

    # 鍋の調理は 15 秒(150フレーム)かかり、その間じっと待つのは正常な動作。
    # 誤って中断しないよう、それより長く取る。
    BLOCKED_LIMIT = 200     # これだけ動けなければ、そのタスクは今できないとみなす
    COOLDOWN_FRAMES = 300   # 避けておく長さ

    def note_progress(self, task_id, action):
        """同じタスクで動けない状態が続いていないかを見る。"""
        if task_id is None:
            return
        if action and tuple(action) != (0, 0):
            self._blocked_since.pop(task_id, None)
            return
        n = self._blocked_since.get(task_id, 0) + 1
        self._blocked_since[task_id] = n
        if n >= self.BLOCKED_LIMIT:
            self._blocked_since.pop(task_id, None)
            self._cooldown[task_id] = self.COOLDOWN_FRAMES

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
            'human_stuck_switches': self.stuck_switches,
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

        # 着手できるものが1つも無いことがある。この地図では果物が相手側に
        # しかないので、相手が刻み終わるまで人間側は手が出せない。そこで
        # 前提を無視して選ぶと、器具の前で永久に立ち尽くすことになる。
        # 人なら「まだできないから待つ」ので、そのまま待たせる。
        options = self.available_tasks(env, tasks)
        if not options and env.hold is None:
            self.current_id = None
            return (0, 0), None

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
                # 注文外の食材を合流させないための材料一覧。渡し忘れると
                # 判定が素通りし、別注文の山に混ぜてしまう。
                self.ta.order_ingredients = self.ai._order_ingredients_of(carry)
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

        # 実行中のタスクがまだ残っていれば続ける。終わった/不要になった
        # ときだけ選び直す(どちらも一覧から消えるので同じ判定でよい)。
        # しばらく進まなかったタスクは、いったん避けて別の仕事に移る。
        for tid in list(self._cooldown):
            self._cooldown[tid] -= 1
            if self._cooldown[tid] <= 0:
                del self._cooldown[tid]
        if self._cooldown:
            # 進まないと分かったタスクは、本当に候補から外す。ここで
            # 「他に無いなら戻す」ことにすると、同じ場所で止まり続ける。
            options = [t for t in options if t['id'] not in self._cooldown]

        # 実行中のタスクは、終わるか着手できなくなるまで続ける。
        keep = next((t for t in options if t['id'] == self.current_id), None)
        if keep is None:
            # 一覧が古いかもしれないので作り直してから確認する。
            tasks = self._tasks(env, refresh=True)
            options = self.available_tasks(env, tasks)
            keep = next((t for t in options if t['id'] == self.current_id), None)

        if keep is not None and self.stuck >= STUCK_LIMIT:
            # 安全網。まったく進まないまま長く経ったので別のものにする。
            self.stuck_switches += 1
            self.stuck = 0
            options = [t for t in options if t['id'] != self.current_id] or options
            keep = None

        if keep is None:
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
        self.ta.order_ingredients = self.ai._order_ingredients_of(keep)
        action, _reason = self.ta(env, dynamic_obstacles={tuple(other_pos)})
        self.note_progress(keep['id'], action)
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
