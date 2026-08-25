import random
import re
import time
from dataclasses import dataclass
from copy import deepcopy
from ortools.sat.python import cp_model
from gym_cooking.utils.config import BLENDING_NUM_STEPS
from .csp.model import CSPModel
from .csp.solver import solve as solve_csp
from .TaskAgent import TaskAgent
from gym_cooking.utils.config import COOKING_TIME_SECONDS


@dataclass
class VirtualHumanState:
    current_time: int
    current_pos: tuple[int, int]
    remaining_task_ids: set

# 人間の予測タスクを乗り換える判定のしきい値。
# 「別タスクの方がこのフレーム数以上早く終わる」かつ「その状態が連続でこの回数続いた」
# ときだけ予測ミスとみなす。人間はうろうろするので、すぐ乗り換えるとそのつど
# CSP の再計算が走って解が変わってしまう。
HUMAN_PREDICTION_COST_MARGIN = 15   # フレーム (10fps で 1.5 秒相当)
HUMAN_PREDICTION_CONFIRM_FRAMES = 3

# 料理名(タスクの obj)の接尾辞。
# スープは「刻む → 鍋で調理 → 皿に移して提供」だが、サラダは鍋を使わず
# 「刻む → 皿に乗せて提供」で完成する。両者を同じ obj 文字列で扱うと
# サラダにも cook タスクが生成されてしまう(XYSalad が XYSoup として
# 提供される不具合の原因)ため、料理の種類を接尾辞で区別する。
SOUP_SUFFIX = ' soup'
SALAD_SUFFIX = ' salad'
JUICE_SUFFIX = ' juice'
DISH_SUFFIXES = (SOUP_SUFFIX, SALAD_SUFFIX, JUICE_SUFFIX)

# 料理の系統。工程が違うので、どの系統かで作るタスクが変わる。
#   salad: 刻む -> 皿に盛って提供
#   soup : 刻む -> 鍋で煮る -> 皿に移して提供
#   juice: 刻む -> ミキサーで混ぜる -> コップに注いで提供
KIND_SALAD, KIND_SOUP, KIND_JUICE = 'salad', 'soup', 'juice'

VEGETABLES = ['lettuce', 'onion', 'tomato']
FRUITS = ['apple', 'orange', 'banana']
ALL_INGREDIENTS = VEGETABLES + FRUITS

# 材料名 -> 供給台の名前。材料を増やすときはここだけ足せばよい。
INGREDIENT_TILE = {ing: f'Fresh{ing.capitalize()}Tile' for ing in ALL_INGREDIENTS}


def strip_dish_suffix(name):
    """'onion-tomato soup' / 'onion-tomato salad' → 'onion-tomato'"""
    text = str(name)
    for suffix in DISH_SUFFIXES:
        if text.endswith(suffix):
            return text[:-len(suffix)]
    return text


def is_salad_dish(name):
    return str(name).endswith(SALAD_SUFFIX)


def is_juice_dish(name):
    return str(name).endswith(JUICE_SUFFIX)


def dish_kind_of(name):
    """内部の料理名('apple-orange juice' 等)から系統を判定する。

    goal_dish_kind は注文ゴールの名前(材料の状態が入ったもの)用。
    内部の料理名は接尾辞で系統が決まるので、そちらを見る。
    """
    text = str(name)
    if text.endswith(JUICE_SUFFIX):
        return KIND_JUICE
    if text.endswith(SOUP_SUFFIX):
        return KIND_SOUP
    if text.endswith(SALAD_SUFFIX):
        return KIND_SALAD
    return goal_dish_kind(text)


def goal_dish_kind(goal_full_name):
    """注文ゴールの full_name から料理の系統を判定する。

    材料の状態が Chopped / Cooked / Mixed のどれで登録されるかで決まる。
    'mixed' を先に見ること(ジュースは 'cooked' を含まないため、先に
    'cooked' で分けるとサラダに紛れる)。
    """
    name = str(goal_full_name).lower()
    if 'mixed' in name:
        return KIND_JUICE
    if 'cooked' in name:
        return KIND_SOUP
    return KIND_SALAD


def dish_ingredients(name):
    """料理名から材料名のリストを取り出す。"""
    return [part.strip() for part in strip_dish_suffix(name).split('-') if part.strip()]


class CSPAgent:
    """
    CSP(制約充足問題)ベースのエージェント
    """

    # 「提供」系の動詞。
    #   serve       : 鍋の調理済み料理を皿に移して提供する(スープ)
    #   serve_salad : 刻んだ食材をそのまま皿に乗せて提供する(サラダ、鍋を使わない)
    #   serve_juice : ミキサーで混ぜた中身をコップに注いで提供する(ジュース)
    SERVE_VERBS = ('serve', 'serve_salad', 'serve_juice', 'serve_from_counter')
    # 同一注文内の実行順序。提供系はどれも「最後の工程」。
    VERB_PRIORITY = {'chop': 0, 'cook': 1, 'mix': 1, 'serve': 2, 'serve_salad': 2,
                     'serve_juice': 2, 'handover': 2, 'serve_from_counter': 3}

    def __init__(self, speed=2.5, replay=None, no_reschedule=False, sc_2agent=False, deadline_seconds: float | None = None, skip_budget: int | None = None):
        self.speed = speed
        self.replay = replay
        self.no_reschedule = no_reschedule
        self.sc_2agent = sc_2agent
        self.initialized = False
        
        # CSP関連の変数
        self.variables = []  
        self.domains = {}
        self.constraints = [] 
        # 入力フレーム間隔
        self.frames_per_action = 1 

        # FPS
        self.fps = 10
        # 期限 (frames)
        self.deadline_seconds = deadline_seconds
        self.deadline_frames = int(75 * self.fps) if deadline_seconds is None else int(deadline_seconds * self.fps)
        # skip_budget: 指示タスク前に同エージェントが実行してよい他タスクの上限個数 (None=使用しない)
        # 秒数ベースの deadline_seconds / deadline_frames は当面未使用だが削除しない
        self.skip_budget = skip_budget
        # 30秒の選択予算
        self.budget_frames = 30 * self.fps
        # 実行状態管理
        self.current_task_idx = {0: 0, 1: 0} if self.sc_2agent else 0
        self.holding_state = None 
        # 交代制のためのターン管理
        self.turn = 0
        self.completed_task_ids = set() # 追加：完了したタスクのID集合（同期用）
        self.pending_reschedule_reason = "initial"
        self._last_event_history_len = 0
        # Pickup/Put/Chop などのイベントが短時間に連発したとき、毎回スケジュール全体を
        # 再計算すると、その連続イベントの間の一時的で曖昧な世界状態(食材がどこにも
        # 見えない/複数の注文から同時に見える瞬間)を拾ってしまい、同じ食材を複数の
        # 注文が奪い合うような揺れを引き起こす。直近の再計算からこの秒数(シミュレーション
        # 内時間)経っていなければ、今回は再計算を見送り理由だけ保持する。
        self._min_reschedule_interval_seconds = 0.3
        self._last_reschedule_time = None
        self.stall_threshold = 8
        self.stall_counts = {0: 0, 1: 0} if self.sc_2agent else 0
        self.active_order_entries = []
        self.next_order_uid = 0
        self.counter_policy_by_order = {}
        self.counter_invalid_since_by_order = {}
        self.order_display_labels = []
        self.carry_task_by_agent = {0: None, 1: None} if self.sc_2agent else None
        # 詳細トレースの既定は OFF。
        # これらは1回の判断ごとに数十行を出力するため、常時ONにすると実コンソールへの
        # 書き込みだけで判断1回が数百msかかり(実測: 8.5ms -> 約350ms)、AI が
        # 毎フレーム動けなくなる。play_main.py が --debug のときだけ True にする。
        self.debug_counter_trace = False
        
        self.task_agent = TaskAgent()
        self.task_agent.strict_counter_management = True
        if self.sc_2agent:
            self.task_agents = {0: TaskAgent(), 1: TaskAgent()}
            for task_agent in self.task_agents.values():
                task_agent.strict_counter_management = True
        
        # 優先度重み（GUI等で設定）
        self.priority_weights = {}
        # 制約指示テキスト（GUI等で設定）
        self.gui_constraint_input = ""
        # 適用する動的制約リスト (JSON format)
        self.active_constraints = []
        # 人間が「いま手をつけているタスク」を推測し、それだけを人間スロットの
        # 最初のタスクとして強制割り当てするか。
        self.use_predicted_human_model = True
        self.predicted_human_tasks = []
        # 直近に推測した人間のタスク(毎フレーム変わらないよう保持する)
        self._predicted_human_task_id = None
        self._human_prediction_doubt = 0
        self.human_counterpart_mode = False
        # 相手が CSP ではなく外部(人間・別方策)に動かされているか。
        # 計画は2体分を立てたままにしたいが、相手が計画どおり動く保証がない
        # ときは「相手の担当タスクを待ち続ける」わけにいかない。この旗が
        # 立っていると、手待ちになったときに相手の担当も引き受ける。
        self.partner_is_external = False
        # 「いま即座に着手できる cook タスク」を (動詞, 対象) で保持する。
        # __call__ ごとに更新し、GamePlay の指示タイミング監視(enable_cook)が読む。
        self.ready_cook_actions = set()
        # CSP が実際に操作するプレイヤー番号 (0 or 1)。
        # sc_2agent=True かつ human_counterpart_mode=True のとき有効。
        # play_main.py が ai_idx を設定する。
        self.own_agent_idx = 0

        # print("[CSPAgent] 初期化完了")

    def _mark_reschedule_needed(self, reason):
        if not self.no_reschedule:
            self.pending_reschedule_reason = reason

    def _emit_counter_debug(self, message):
        if getattr(self, 'debug_counter_trace', False):
            print(message)

    def _log_reschedule_event(self, reason, env, added=None, removed=None):
        current_time = getattr(env, 'time', None)
        if current_time is None:
            current_time = getattr(env, 'current_time', None)
        parts = [f"[CSPAgent][RESCHEDULE] time={current_time}", f"reason={reason}"]
        if added:
            parts.append(f"added={sorted(added)}")
        if removed:
            parts.append(f"removed={sorted(removed)}")
        self._emit_counter_debug(" ".join(parts))

    def _get_active_task_ids(self):
        if self.sc_2agent:
            active = set()
            if hasattr(self, 'schedule_per_agent') and isinstance(self.current_task_idx, dict):
                for agent_idx in [0, 1]:
                    schedule = self.schedule_per_agent.get(agent_idx, [])
                    task_idx = self.current_task_idx.get(agent_idx, 0)
                    if task_idx < len(schedule):
                        active.add(schedule[task_idx]['id'])
            return active

        if hasattr(self, 'schedule') and self.schedule and isinstance(self.current_task_idx, int) and self.current_task_idx < len(self.schedule):
            return {self.schedule[self.current_task_idx]['id']}
        return set()

    def _extract_instruction_fixed_task_id(self, pending):
        task_payload = pending.get('task')
        if isinstance(task_payload, (list, tuple)) and len(task_payload) >= 2:
            payload = task_payload[1]
        elif isinstance(task_payload, dict):
            payload = task_payload
        else:
            payload = task_payload

        if isinstance(payload, dict):
            return payload.get('fixed_task_id')
        if isinstance(payload, (list, tuple)) and len(payload) >= 1:
            return payload[0]
        return payload

    def _find_order_recipe_for_partial(self, env, held_parts):
        """持っている組み合わせが、どの注文の作りかけかを探す。

        戻り値は (完成レシピの材料リスト, その注文の置き場, 系統) 。
        系統は 'salad' / 'soup' / 'juice' のいずれか(該当なしは None)。
        完全一致する注文があればそれを優先し、無ければ held_parts を
        真に含む注文(まだ材料が足りない作りかけ)を返す。
        どれにも当てはまらなければ (None, None, False)。
        """
        held = set(held_parts)
        if not held:
            return None, None, None

        current_orders = []
        if hasattr(env, 'order') and hasattr(env.order, 'current_orders'):
            current_orders = env.order.current_orders or []

        uid_by_idx = {}
        for entry in getattr(self, 'active_order_entries', []) or []:
            if entry.get('order_idx') is not None:
                uid_by_idx[entry['order_idx']] = entry.get('uid')

        supersets = []
        for order_idx, order_tuple in enumerate(current_orders):
            goal = order_tuple[0] if order_tuple else None
            name = str(getattr(goal, 'full_name', '')).lower()
            ings = [ing for ing in ALL_INGREDIENTS if ing in name]
            if not ings:
                continue
            kind = goal_dish_kind(name)
            ing_set = set(ings)
            counter = None
            uid = uid_by_idx.get(order_idx)
            if uid is not None:
                counter = self._get_assigned_counter(uid)
            if ing_set == held:
                return sorted(ings), counter, kind
            if held < ing_set:
                supersets.append((len(ing_set), sorted(ings), counter, kind))

        if supersets:
            # 一番少ない材料で済む注文(=完成が近い)を選ぶ
            supersets.sort(key=lambda e: e[0])
            return supersets[0][1], supersets[0][2], supersets[0][3]
        return None, None, None

    @staticmethod
    def _goal_name_is_salad(goal_full_name):
        """注文ゴールの full_name からサラダかスープかを判定する。

        レシピ定義上、サラダは材料が Chopped(state_index=2)、スープは
        Cooked(state_index=4) で登録される。したがってゴール名は
        'ChoppedOnion-ChoppedTomato-Plate' か 'CookedTomato-Plate' となり、
        'cooked' を含むかどうかで確実に区別できる。
        """
        return 'cooked' not in str(goal_full_name).lower()

    def _extract_instruction_action(self, pending):
        """指示から (動詞, 対象) を取り出す。注文番号は持たない。"""
        task_payload = pending.get('task')
        if isinstance(task_payload, (list, tuple)) and len(task_payload) >= 2:
            payload = task_payload[1]
        else:
            payload = task_payload
        if isinstance(payload, dict):
            verb, obj = payload.get('verb'), payload.get('obj')
            if verb is not None and obj is not None:
                return str(verb), str(obj)
        # 旧形式(単一の fixed_task_id = ("task", verb, obj, order_uid))からの復元
        fixed_task_id = self._extract_instruction_fixed_task_id(pending)
        if isinstance(fixed_task_id, (list, tuple)) and len(fixed_task_id) >= 3:
            return str(fixed_task_id[1]), str(fixed_task_id[2])
        return None

    def _find_group_task_indices(self, tasks, action):
        """指示された行動 (動詞, 対象) に一致するタスクを全部探す。

        「cut onion が2つあるなら、そのどちらか1つを d 以内に」を実現するための
        グループ。注文番号は見ない。
        """
        if not action:
            return []
        verb, obj = action
        indices = []
        for idx, t in enumerate(tasks):
            if str(t.get('verb', '')) == verb and str(t.get('obj', '')) == obj:
                indices.append(idx)
        return indices

    def _derive_fixed_task_id(self, task):
        if task is None:
            return None

        fixed_task_id = task.get('fixed_task_id')
        if fixed_task_id is not None:
            return fixed_task_id

        task_id = task.get('id')
        if isinstance(task_id, (list, tuple)) and len(task_id) >= 3:
            return self._make_fixed_task_id(task_id[0], task_id[1], task_id[2])

        verb = task.get('verb')
        obj = task.get('obj')
        order = task.get('order')
        if order is None:
            order = task.get('order_uid')
        if verb is None or obj is None or order is None:
            return None
        return self._make_fixed_task_id(verb, obj, order)

    def _find_schedule_index_by_fixed_id(self, schedule, fixed_task_id, action=None):
        """指示に対応するタスクをスケジュールから探す。

        action=(動詞, 対象) が与えられた場合は注文番号を問わずそれに一致する
        最初のタスクを返す(指示は行動単位で、どの注文のものでもよいため)。
        """
        if not schedule:
            return None
        if action is not None:
            verb, obj = action
            for idx, task in enumerate(schedule):
                task_id = task.get('id')
                if isinstance(task_id, tuple) and len(task_id) >= 2:
                    if str(task_id[0]) == verb and str(task_id[1]) == obj:
                        task['fixed_task_id'] = self._derive_fixed_task_id(task)
                        return idx
        for idx, task in enumerate(schedule):
            task_fixed_id = self._derive_fixed_task_id(task)
            if task_fixed_id is None:
                continue
            if task_fixed_id == fixed_task_id:
                task['fixed_task_id'] = task_fixed_id
                return idx
        return None

    def _classify_instruction_deadline(self, pending, current_env_time, deadline_seconds):
        if deadline_seconds is None:
            return {'mode': 'unspecified', 'priority_boost': False}

        accepted_env_time = pending.get('accepted_env_time', None)
        if accepted_env_time is None:
            return {'mode': 'unspecified', 'priority_boost': False}

        elapsed_seconds = max(0.0, float(current_env_time) - float(accepted_env_time))
        remaining_deadline_seconds = max(0.0, float(deadline_seconds) - elapsed_seconds)
        if float(deadline_seconds) <= 0.0:
            return {'mode': 'urgent', 'priority_boost': True}
        if remaining_deadline_seconds <= 0.0:
            return {'mode': 'urgent', 'priority_boost': True}
        return {'mode': 'deadline', 'priority_boost': False, 'remaining_seconds': remaining_deadline_seconds}

    # ──────────────────────────────────────────────────────────
    # skip_budget (タスク数ベース) 関連メソッド
    # ──────────────────────────────────────────────────────────

    def _is_urgent_by_skip_budget(self, pending):
        """remaining_skip_budget が 0 以下のとき urgent とみなす。"""
        if self.skip_budget is None:
            return False
        remaining = pending.get('remaining_skip_budget', self.skip_budget)
        return remaining is not None and remaining <= 0

    def _get_dep_indices_for_target(self, tasks, matched_idx):
        """対象タスクの依存タスク(前提)のインデックス集合を返す。
        同じ order_uid 内で、動詞優先度(chop<cook<serve)が低いタスクを依存とみなす。"""
        dep_indices = set()
        target = tasks[matched_idx]
        target_order = target.get('order')
        verb_priority = self.VERB_PRIORITY
        target_prio = verb_priority.get(target.get('verb', ''), 9)
        for j, t in enumerate(tasks):
            if j == matched_idx:
                continue
            if t.get('order') != target_order:
                continue
            if verb_priority.get(t.get('verb', ''), 9) < target_prio:
                dep_indices.add(j)
        return dep_indices

    def _apply_instruction_skip_budget_constraints(self, model, tasks, starts_by_idx, env, is_a1=None):
        """skip_budget(タスク数)ベースの順序制約を CP-SAT モデルへ反映する。
        is_a1: 2エージェントモードの割り当て変数リスト(BoolVar) or None(1エージェント)。"""
        try:
            pending_instr = list(getattr(env, '_pending_instructions', []))
            agent_pending = getattr(self, '_pending_instructions', [])
            if agent_pending:
                for p in agent_pending:
                    if not any(e.get('id') == p.get('id') for e in pending_instr):
                        pending_instr.append(p)
            if not pending_instr:
                return
            # print(f"[SkipBudget] 保留指示数: {len(pending_instr)}")

            task_index_by_fixed_id = {}
            for idx, t in enumerate(tasks):
                fid = t.get('fixed_task_id')
                if fid is None:
                    fid = self._make_fixed_task_id(t.get('verb', ''), t.get('obj', ''), t.get('order', 0))
                    t['fixed_task_id'] = fid
                task_index_by_fixed_id[fid] = idx

            for pending in list(pending_instr):
                try:
                    if pending.get('status') in {'done', 'canceled'}:
                        continue
                    init_budget = pending.get('skip_budget')
                    if init_budget is None:
                        continue
                    remaining = pending.get('remaining_skip_budget', init_budget)
                    if remaining is None:
                        remaining = init_budget

                    # 指示は「(動詞, 対象)」単位。同じ行動のタスクが複数の注文に
                    # またがっている場合(例: cut onion が2つ)は、そのうち
                    # 「どれか1つ」が d 以内に実行されればよい。
                    action = self._extract_instruction_action(pending)
                    group_indices = self._find_group_task_indices(tasks, action)
                    if not group_indices:
                        fixed_task_id = self._extract_instruction_fixed_task_id(pending)
                        matched_idx = task_index_by_fixed_id.get(fixed_task_id)
                        if matched_idx is None:
                            pending['status'] = 'done'
                            continue
                        group_indices = [matched_idx]

                    group_indices = [
                        idx for idx in group_indices
                        if tasks[idx].get('id') not in self.completed_task_ids
                        and starts_by_idx.get(idx) is not None
                    ]
                    if not group_indices:
                        pending['status'] = 'done'
                        continue

                    budget_bound = max(0, remaining)
                    overage = max(0, -remaining)

                    # メンバー k について「k より前に走る非依存タスクが budget 以下」を
                    # ok_k として表し、最後に少なくとも1つの ok_k が真であることを課す。
                    ok_vars = []
                    for matched_idx in group_indices:
                        dep_indices = self._get_dep_indices_for_target(tasks, matched_idx)
                        target_start = starts_by_idx[matched_idx]
                        counts_vars = []

                        for j in range(len(tasks)):
                            if j == matched_idx or j in dep_indices:
                                continue
                            if starts_by_idx.get(j) is None:
                                continue

                            prec_j = model.NewBoolVar(f'prec_sb_{matched_idx}_{j}')
                            model.Add(starts_by_idx[j] <= target_start - 1).OnlyEnforceIf(prec_j)
                            model.Add(starts_by_idx[j] >= target_start).OnlyEnforceIf(prec_j.Not())

                            if is_a1 is None:
                                # 1エージェント: すべて同エージェント
                                counts_vars.append(prec_j)
                            else:
                                # 2エージェント: 同エージェント割り当てのみカウント
                                a1_t = is_a1[matched_idx]
                                a1_j = is_a1[j]
                                if a1_t is None or a1_j is None:
                                    continue
                                same_j = model.NewBoolVar(f'same_ag_{matched_idx}_{j}')
                                # same_j=1 iff is_a1[target]==is_a1[j] (両方0か両方1)
                                model.Add(a1_t - a1_j == 0).OnlyEnforceIf(same_j)
                                model.Add(a1_t + a1_j == 1).OnlyEnforceIf(same_j.Not())
                                count_j = model.NewBoolVar(f'count_sb_{matched_idx}_{j}')
                                model.AddBoolAnd([prec_j, same_j]).OnlyEnforceIf(count_j)
                                model.AddBoolOr([prec_j.Not(), same_j.Not()]).OnlyEnforceIf(count_j.Not())
                                counts_vars.append(count_j)

                        if not counts_vars:
                            # 前に置けるタスクが無い = 無条件に満たされる
                            ok_vars = None
                            break
                        ok_k = model.NewBoolVar(f'sb_ok_{matched_idx}')
                        model.Add(sum(counts_vars) <= budget_bound).OnlyEnforceIf(ok_k)
                        ok_vars.append(ok_k)

                    if ok_vars:
                        # どれか1つが d 以内であればよい
                        model.AddBoolOr(ok_vars)
                    if overage > 0:
                        # print(f"[SkipBudget] 警告: fixed_id={fixed_task_id} 超過中(超過量={overage}), 最善優先化")
                        pass
                    else:
                        # print(f"[SkipBudget] 制約: fixed_id={fixed_task_id} 残りbudget={remaining}, 対象前タスク≤{budget_bound} (非依存{len(counts_vars)}個)")
                        pass
                    pending['skip_budget_constraint_applied'] = True
                except Exception:
                    pass
        except Exception:
            pass

    def _update_skip_budget_on_completion(self, completed_tid, completed_agent_idx, completed_dur_frames):
        """タスク完了時に pending_instructions の remaining_skip_budget を更新しログする。"""
        pending_instr = list(getattr(self, '_pending_instructions', []))
        for pending in pending_instr:
            if pending.get('status') in {'done', 'canceled'}:
                continue
            if pending.get('skip_budget') is None:
                continue
            fixed_task_id = self._extract_instruction_fixed_task_id(pending)
            if fixed_task_id is None:
                continue
            # fixed_task_id = ("task", verb, obj, order_uid) → target_tid = (verb, obj, order_uid)
            if not (isinstance(fixed_task_id, (list, tuple)) and len(fixed_task_id) >= 4):
                continue
            target_tid = (str(fixed_task_id[1]), str(fixed_task_id[2]), int(fixed_task_id[3]))

            # 対象タスク自体が完了した場合
            if completed_tid == target_tid:
                pending['status'] = 'done'
                continue

            # エージェントが異なる場合はカウントしない
            target_agent_idx = None
            if self.sc_2agent and hasattr(self, 'schedule_per_agent'):
                for aidx in [0, 1]:
                    for t in self.schedule_per_agent.get(aidx, []):
                        if t.get('id') == target_tid:
                            target_agent_idx = aidx
                            break
                    if target_agent_idx is not None:
                        break
            else:
                target_agent_idx = 0
            if target_agent_idx != completed_agent_idx:
                continue

            # 依存タスク(同じ注文で動詞優先度が低い)はカウントしない
            verb_prio = self.VERB_PRIORITY
            if (len(completed_tid) >= 3 and len(target_tid) >= 3 and
                    completed_tid[2] == target_tid[2] and
                    verb_prio.get(completed_tid[0], 9) < verb_prio.get(target_tid[0], 9)):
                continue

            # remaining_skip_budget を減らす
            old_remaining = pending.get('remaining_skip_budget', pending.get('skip_budget', 0))
            if old_remaining is None:
                old_remaining = 0
            pending['remaining_skip_budget'] = old_remaining - 1
            new_remaining = pending['remaining_skip_budget']
            completed_secs = completed_dur_frames / float(self.fps) if self.fps > 0 else 0.0
            log_list = pending.setdefault('tasks_before_target_log', [])
            log_list.append({'task_id': completed_tid, 'duration_seconds': completed_secs})
            initial_budget = pending.get('skip_budget', 0)
            overage = max(0, -new_remaining)
            # print(
            #     f"[SkipBudget] 対象前完了: {completed_tid} ({completed_secs:.1f}s) "
            #     f"初期d={initial_budget} 残り={new_remaining} 超過={overage} 累計={len(log_list)}件"
            # )
            if overage > 0:
                # print("[SkipBudget] 警告: skip_budget超過 → 次回再計画で即時優先化")
                self._mark_reschedule_needed('skip_budget_exceeded')

    def _get_immediate_instruction_preempt(self, env):
        pending_instr = list(getattr(env, '_pending_instructions', []))
        agent_pending = getattr(self, '_pending_instructions', [])
        if agent_pending:
            for pending in agent_pending:
                if not any(existing.get('id') == pending.get('id') for existing in pending_instr):
                    pending_instr.append(pending)

        if not pending_instr:
            return None

        deadline_seconds = self.deadline_seconds
        if deadline_seconds is None:
            deadline_seconds = self.deadline_frames / float(self.fps)

        current_env_time = getattr(env, 'time', None)
        if current_env_time is None:
            current_env_time = getattr(env, 'current_time', None)
        if current_env_time is None:
            return None

        for pending in pending_instr:
            if pending.get('status') != 'pending':
                continue

            fixed_task_id = self._extract_instruction_fixed_task_id(pending)
            action = self._extract_instruction_action(pending)
            if fixed_task_id is None and action is None:
                continue

            deadline_info = self._classify_instruction_deadline(pending, current_env_time, deadline_seconds)
            if self.skip_budget is not None:
                is_urgent = self._is_urgent_by_skip_budget(pending)
            else:
                is_urgent = (deadline_info.get('mode') == 'urgent')
            if not is_urgent:
                continue

            target_idx = self._find_schedule_index_by_fixed_id(getattr(self, 'schedule', []), fixed_task_id)
            if target_idx is None:
                continue

            if isinstance(self.current_task_idx, int) and self.current_task_idx == target_idx:
                continue

            return pending, fixed_task_id, target_idx

        return None

    def _get_instruction_preempt_target(self, env):
        pending_instr = list(getattr(env, '_pending_instructions', []))
        agent_pending = getattr(self, '_pending_instructions', [])
        if agent_pending:
            for pending in agent_pending:
                if not any(existing.get('id') == pending.get('id') for existing in pending_instr):
                    pending_instr.append(pending)

        if not pending_instr:
            return None

        deadline_seconds = self.deadline_seconds
        if deadline_seconds is None:
            deadline_seconds = self.deadline_frames / float(self.fps)

        current_env_time = getattr(env, 'time', None)
        if current_env_time is None:
            current_env_time = getattr(env, 'current_time', None)
        if current_env_time is None:
            return None

        for pending in pending_instr:
            if pending.get('status') != 'pending':
                continue

            fixed_task_id = self._extract_instruction_fixed_task_id(pending)
            action = self._extract_instruction_action(pending)
            if fixed_task_id is None and action is None:
                continue

            deadline_info = self._classify_instruction_deadline(pending, current_env_time, deadline_seconds)
            if self.skip_budget is not None:
                is_urgent = self._is_urgent_by_skip_budget(pending)
            else:
                is_urgent = (deadline_info.get('mode') == 'urgent')
            if not is_urgent:
                continue

            if not self.sc_2agent:
                target_idx = self._find_schedule_index_by_fixed_id(getattr(self, 'schedule', []), fixed_task_id, action=action)
                if target_idx is None:
                    continue
                if isinstance(self.current_task_idx, int) and self.current_task_idx == target_idx:
                    continue
                return pending, fixed_task_id, None, target_idx

            schedule_per_agent = getattr(self, 'schedule_per_agent', None)
            if not schedule_per_agent:
                continue

            remaining = self._remaining_ids_in_schedule(schedule_per_agent, self.current_task_idx)
            for agent_idx in [0, 1]:
                schedule = schedule_per_agent.get(agent_idx, [])
                if not schedule:
                    continue
                target_idx = self._find_schedule_index_by_fixed_id(schedule, fixed_task_id, action=action)
                if target_idx is None:
                    continue
                # 前提の工程が済んでいないタスクに飛ばしてはいけない。混ぜる前に
                # 「注ぐ」へ飛ぶと、ミキサーの前で永久に立ち尽くすことになる。
                # 指示は「次にやること」を変えるものであって、工程の順序を
                # 壊してよいという意味ではない。
                if not self._task_is_available_in_virtual_state(schedule[target_idx], remaining):
                    continue
                return pending, fixed_task_id, agent_idx, target_idx

        return None

    @staticmethod
    def _remaining_ids_in_schedule(schedule_per_agent, current_task_idx=None):
        """まだ終わっていないタスクの id 集合。

        各エージェントの現在位置より前は済んでいるので数えない。
        """
        remaining = set()
        for agent_idx, tasks in (schedule_per_agent or {}).items():
            start = 0
            if isinstance(current_task_idx, dict):
                start = current_task_idx.get(agent_idx, 0)
            elif isinstance(current_task_idx, int):
                start = current_task_idx
            for t in tasks[start:]:
                tid = t.get('id')
                if tid is not None:
                    remaining.add(tid)
        return remaining

    def _get_reschedule_reason(self, current_task_ids, added, removed):
        if not self.initialized:
            return "initial"

        if self.pending_reschedule_reason:
            return self.pending_reschedule_reason

        if added:
            return f"task_added:{sorted(added)}"

        if removed:
            return f"task_removed:{sorted(removed)}"

        active_task_ids = self._get_active_task_ids()
        missing_active = sorted(tid for tid in active_task_ids if tid not in current_task_ids)
        if missing_active:
            return f"active_task_missing:{missing_active}"

    def _collect_event_replan_reason(self, env):
        """直近のイベント履歴から、再スケジュールを要する人間/世界状態変更だけを拾う。"""
        event_history = list(getattr(env, 'event_history', []) or [])
        last_len = getattr(self, '_last_event_history_len', 0)
        if last_len > len(event_history):
            last_len = 0

        new_events = event_history[last_len:]
        self._last_event_history_len = len(event_history)

        if not new_events:
            return None

        relevant_prefixes = (
            'Pickup_',
            'Put_',
            'Drop_',
            'Chop_',
            'Cook_',
            'Assemble_',
            'Deliver_',
            'Putout_Fire',
        )

        for event in new_events:
            event_name = getattr(event, 'event', None)
            player_name = getattr(event, 'playerA', None)
            if not event_name or not player_name:
                continue
            if event_name == 'No-op':
                continue
            if event_name.startswith(relevant_prefixes):
                return f"event:{player_name}:{event_name}"

        return None

    def _should_defer_holding_reschedule(self, env, added, removed):
        if not self.initialized or self.pending_reschedule_reason:
            return False
        if not added and not removed:
            return False

        def iter_active_tasks():
            if self.sc_2agent:
                for agent_idx in [0, 1]:
                    sc = getattr(self, 'schedule_per_agent', {}).get(agent_idx, [])
                    task_idx_val = getattr(self, 'current_task_idx', {})
                    task_idx = task_idx_val.get(agent_idx, 0) if isinstance(task_idx_val, dict) else 0
                    if task_idx < len(sc):
                        yield agent_idx, sc[task_idx]
            else:
                schedule = getattr(self, 'schedule', [])
                task_idx = getattr(self, 'current_task_idx', 0)
                if not isinstance(task_idx, int):
                    task_idx = 0
                if task_idx < len(schedule):
                    yield 0, schedule[task_idx]

        for agent_idx, task in iter_active_tasks():
            agents = getattr(env, 'agents', [])
            if agent_idx >= len(agents):
                continue
            holding = getattr(agents[agent_idx], 'holding', None)
            holding_name = getattr(holding, 'full_name', None) if holding is not None else None
            if not holding_name or 'Chopped' not in holding_name:
                continue

            verb, obj, order_uid = task['id']
            if verb == 'chop':
                chopped_name = f"Chopped{obj.capitalize()}"
                if chopped_name in holding_name and task['id'] in removed:
                    return True

            if verb in ('cook', 'serve_salad'):
                recipe_ings = dish_ingredients(obj)
                if any(f"Chopped{ing.capitalize()}" in holding_name for ing in recipe_ings):
                    if any(tid[0] == 'chop' and tid[2] == order_uid for tid in added):
                        return True

        return False

        if current_task_ids and not self._get_active_task_ids():
            return "no_active_task_with_remaining_work"

        return None

    def _resolve_mutual_block(self, env, actions):
        """相手に道をふさがれて進めない状態を解く。

        2つのパターンがある。
          (1) 2体が同じマスへ同時に入ろうとして、環境の衝突判定で両方の移動が
              取り消され続ける
          (2) 片方が通りたいマスに、もう片方が用事もなく立ち止まっている
        既存の停止検知は「行動が (0,0) のとき」しか働かないため、行動は出して
        いるのに進めないこれらの状態は素通りしてしまう。数フレーム続いたら
        片方を強制的にどかして解く。
        """
        agents = getattr(env, 'agents', None)
        if not agents or len(agents) < 2:
            return actions

        positions = [tuple(a.location) for a in agents[:2]]
        moved = positions != getattr(self, '_last_agent_positions', None)
        self._last_agent_positions = list(positions)
        if moved:
            self._block_counts = {0: 0, 1: 0}
            return actions

        counts = getattr(self, '_block_counts', None) or {0: 0, 1: 0}
        for i in (0, 1):
            if actions.get(f'ai_{i}', (0, 0)) != (0, 0):
                counts[i] = counts.get(i, 0) + 1
            else:
                counts[i] = 0
        self._block_counts = counts

        for i in (0, 1):
            if counts.get(i, 0) < 3:
                continue
            other = 1 - i
            act = actions.get(f'ai_{i}', (0, 0))
            target = (positions[i][0] + act[0], positions[i][1] + act[1])

            if target == positions[other]:
                # 相手が目的のマスに居座っている -> 相手をどかす
                escape = self._find_escape_step(env, positions[other], positions[i])
                if escape is not None:
                    actions[f'ai_{other}'] = escape
                    self._block_counts = {0: 0, 1: 0}
                    self._emit_counter_debug(
                        f"[CSPAgent] AI{i} の進路をふさぐ AI{other} を {escape} へどかす")
                    return actions

            if actions.get(f'ai_{other}', (0, 0)) == act or target == (
                    positions[other][0] + actions.get(f'ai_{other}', (0, 0))[0],
                    positions[other][1] + actions.get(f'ai_{other}', (0, 0))[1]):
                # 同じマスを取り合っている -> 片方(常に AI1)を1フレーム譲らせる
                actions['ai_1'] = (0, 0)
                self._block_counts = {0: 0, 1: 0}
                self._emit_counter_debug("[CSPAgent] 同じマスの取り合いを検知: AI1 を譲らせる")
                return actions

        return actions

    def _find_escape_step(self, env, pos, avoid):
        """pos にいるエージェントが avoid 以外の隣接マスへ1歩どく方向を返す。"""
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = pos[0] + dx, pos[1] + dy
            if not (0 <= nx < env.world_width and 0 <= ny < env.world_height):
                continue
            if env.to_grid[nx][ny] != 1 or (nx, ny) == avoid:
                continue
            return (dx, dy)
        return None

    def _update_stall_state(self, agent_idx, action, reason):
        if self.sc_2agent:
            if action == (0, 0) and reason not in ("待機(相手のターン)", "AI0:Idle", "AI1:Idle"):
                self.stall_counts[agent_idx] += 1
                if self.stall_counts[agent_idx] >= self.stall_threshold:
                    self._mark_reschedule_needed(f"stall_agent_{agent_idx}")
            else:
                self.stall_counts[agent_idx] = 0
            return

        if action == (0, 0) and "タスクなし" not in reason and "アイドル" not in reason:
            self.stall_counts += 1
            if self.stall_counts >= self.stall_threshold:
                self._mark_reschedule_needed("stall_single_agent")
        else:
            self.stall_counts = 0

    def _refresh_active_order_uids(self, current_orders):
        previous_entries = list(getattr(self, 'active_order_entries', []))
        current_uids = [None] * len(current_orders)
        next_entries = []

        for order_idx, order_tuple in enumerate(current_orders):
            goal = order_tuple[0]
            name = getattr(goal, 'full_name', '').lower()
            if not any(ing in name for ing in ALL_INGREDIENTS):
                continue

            rest_time = order_tuple[1] if len(order_tuple) > 1 else None
            time_limit = order_tuple[2] if len(order_tuple) > 2 else None

            order_uid = None
            matched = None

            # 1. 同じ index の注文を優先して再利用する。
            #    これは同一レシピが複数あるときに、名前だけで別注文へ紐づけてしまうのを防ぐ。
            for prev_idx, prev in enumerate(previous_entries):
                if prev.get('order_idx') == order_idx:
                    matched = previous_entries.pop(prev_idx)
                    order_uid = matched['uid']
                    break

            if order_uid is None:
                best_prev_idx = None
                best_score = None
                for prev_idx, prev in enumerate(previous_entries):
                    if prev['name'] != name:
                        continue

                    prev_rest_time = prev.get('rest_time')
                    if prev_rest_time is not None and rest_time is not None and rest_time > prev_rest_time + 1e-6:
                        continue

                    delta = 0.0 if prev_rest_time is None or rest_time is None else prev_rest_time - rest_time
                    score = (abs(delta), abs(prev.get('order_idx', order_idx) - order_idx))
                    if best_score is None or score < best_score:
                        best_score = score
                        best_prev_idx = prev_idx

                if best_prev_idx is not None:
                    matched = previous_entries.pop(best_prev_idx)
                    order_uid = matched['uid']

            if order_uid is None:
                order_uid = self.next_order_uid
                self.next_order_uid += 1

            current_uids[order_idx] = order_uid
            next_entries.append({
                'uid': order_uid,
                'name': name,
                'rest_time': rest_time,
                'time_limit': time_limit,
                'order_idx': order_idx,
            })

        active_uids = {entry['uid'] for entry in next_entries}
        stale_uids = [uid for uid in list(self.counter_policy_by_order) if uid not in active_uids]
        for uid in stale_uids:
            self.counter_policy_by_order.pop(uid, None)

        self.active_order_entries = next_entries
        return current_uids

    def get_remaining_tids(self, env, current_orders):
        """現在の環境から残存タスクのID集合(tid)を抽出する（インベントリ照合）"""
        inv_chopped = []
        inv_pots_ings = []
        inv_plates_ings = []
        cutboard_locs = set(env.get_pos_by_obj_gs(gs="Cutboard"))
        pot_locs = set(env.get_pos_by_obj_gs(gs="Pot"))

        def extract_food_names(obj):
            names = []
            for content in getattr(obj, 'contents', []):
                name = getattr(content, 'name', None)
                if name in ('Lettuce', 'Onion', 'Tomato'):
                    names.append(name.lower())
            return names

        def has_plate(obj):
            return any(getattr(content, 'name', None) == 'Plate' for content in getattr(obj, 'contents', []))
        
        inv_blender_ings = []
        blender_locs = env.get_pos_by_obj_gs(gs="Blender")

        items = []
        for pos, obj in env.pos_obj.items():
            if obj is not None:
                items.append(obj)
                
        for obj in items:
            if getattr(obj, 'location', None) in cutboard_locs:
                continue

            if hasattr(obj, 'is_chopped') and obj.is_chopped():
                for ing in extract_food_names(obj):
                    inv_chopped.append(ing)

            obj_foods = set(extract_food_names(obj))
            if not obj_foods:
                continue

            obj_loc = getattr(obj, 'location', None)
            if obj_loc in pot_locs:
                inv_pots_ings.append(obj_foods)
            elif obj_loc in blender_locs:
                # 入れただけ(Mixing)と混ぜ終わり(Mixed)を区別する。
                # 入れた時点で完了扱いにすると、まだ回している最中に
                # 提供タスクへ移ってしまい、誰も回さなくなる。
                inv_blender_ings.append(
                    (obj_foods, bool(getattr(obj, 'is_mixed', lambda: False)())))
            elif has_plate(obj):
                inv_plates_ings.append(obj_foods)
                    
        remaining_tids = set()
        
        for order in current_orders:
            order_uid = order['order']
            order_name = order['name']
            is_juice = is_juice_dish(order_name)
            is_salad = is_salad_dish(order_name)
            if not (is_salad or is_juice) and SOUP_SUFFIX not in order_name:
                continue

            raw_parts = dish_ingredients(order_name)
            req_set = set(raw_parts)

            needs_chop = list(raw_parts)
            needs_cook = not (is_salad or is_juice)
            # ジュースはミキサーで混ぜる工程。鍋(cook)と同じ位置づけ。
            needs_mix = is_juice
            needs_serve = True

            if is_juice:
                for entry in list(inv_blender_ings):
                    b, mixed = entry
                    if b == req_set or (b.issubset(req_set) and b):
                        inv_blender_ings.remove(entry)
                        # 混ぜ終わって初めて mix タスクは不要になる。
                        if b == req_set and mixed:
                            needs_mix = False
                        for ing in b:
                            if ing in needs_chop:
                                needs_chop.remove(ing)
                        break

            # 1. 皿にある完成品
            plate_match = None
            for p in inv_plates_ings:
                if p == req_set:
                    plate_match = p
                    break
            if plate_match:
                inv_plates_ings.remove(plate_match)
                needs_chop = []
                needs_cook = False
            elif not is_salad:
                # 2. 鍋にあるか？(サラダは鍋を使わないので見ない。
                #    人間が誤って鍋へ入れた分を「調理済み」と数えると、
                #    サラダの chop タスクが消えて完成できなくなる)
                pot_match = None
                for pot_s in inv_pots_ings:
                    if pot_s == req_set:
                        pot_match = pot_s
                        needs_cook = False
                        break
                    if pot_s.issubset(req_set) and len(pot_s) > 0:
                        pot_match = pot_s
                        break
                if pot_match:
                    inv_pots_ings.remove(pot_match)
                    # 鍋にあるものはchop完了済み
                    for ing in pot_match:
                        if ing in needs_chop:
                            needs_chop.remove(ing)

            # 3. chop済み確認
            final_needs_chop = []
            for ing in needs_chop:
                if ing in inv_chopped:
                    inv_chopped.remove(ing)
                else:
                    final_needs_chop.append(ing)

            # tids構築
            for ing in final_needs_chop:
                raw_ing = ing.replace('Chopped', '').lower()
                remaining_tids.add(('chop', raw_ing, order_uid))

            base_name = "-".join([
                i.replace('Chopped', '').replace('Mixed', '').replace('Cooked', '').lower()
                for i in raw_parts])
            suffix = (JUICE_SUFFIX if is_juice
                      else SALAD_SUFFIX if is_salad else SOUP_SUFFIX)
            dish_name = base_name + suffix
            if needs_cook:
                remaining_tids.add(('cook', dish_name, order_uid))
            if needs_mix:
                remaining_tids.add(('mix', dish_name, order_uid))
            if needs_serve:
                serve_verb = ('serve_juice' if is_juice
                              else 'serve_salad' if is_salad else 'serve')
                # 仕切りのあるマップでは、鍋の側から提供口へ行けない。
                # 実際に作られるタスクは serve ではなく handover +
                # serve_from_counter なので、候補としてもそちらを出す。
                # (ここで serve を出すと、実在しないタスクが指示の候補に並び、
                #  選ばれても制約が効かない)
                if serve_verb == 'serve' and self._map_is_partitioned(env):
                    pot_side = set()
                    for pot in env.get_pos_by_obj_gs(gs='Pot'):
                        pot_side |= self._components_touching(env, pot)
                    reach = any(
                        pot_side & self._components_touching(env, d)
                        for d in env.get_pos_by_obj_gs(gs='Delivery'))
                    if not reach:
                        remaining_tids.add(('handover', dish_name, order_uid))
                        remaining_tids.add(('serve_from_counter', dish_name, order_uid))
                        continue
                remaining_tids.add((serve_verb, dish_name, order_uid))

        return remaining_tids

    def _make_fixed_task_id(self, verb, obj, order_uid):
        return ("task", str(verb), str(obj), int(order_uid))

    def get_instruction_candidates(self, env):
        """現在の環境で未実行のタスク候補を返す。

        指示は「注文いくつ目の玉ねぎを切る」ではなく「玉ねぎを切る」という行動単位で
        選ばせる。そのため同じ (動詞, 対象) のタスクが複数の注文にまたがっていても
        候補は1つにまとめる。CSP 側は「そのうちのどれか1つ」を d 以内に実行すれば
        よいという制約として扱う(_apply_instruction_skip_budget_constraints)。
        """
        current_orders = self._build_order_tasks(env)
        remaining_tids = self.get_remaining_tids(env, current_orders)

        verb_priority = self.VERB_PRIORITY
        sorted_tids = sorted(
            remaining_tids,
            key=lambda tid: (verb_priority.get(tid[0], 9), tid[1], tid[2])
        )

        grouped = {}
        for verb, obj, order_uid in sorted_tids:
            grouped.setdefault((verb, obj), []).append(order_uid)

        candidates = []
        for (verb, obj), order_uids in grouped.items():
            display = f"{verb}_{obj.replace(' ', '').replace('-', '_')}"
            payload = {
                # 後方互換のため代表IDも持たせる(グループ先頭)
                'fixed_task_id': self._make_fixed_task_id(verb, obj, order_uids[0]),
                'fixed_task_ids': [self._make_fixed_task_id(verb, obj, uid) for uid in order_uids],
                'verb': verb,
                'obj': obj,
                'order_uids': order_uids,
            }
            candidates.append((display, payload))

        return candidates

    def high_level_infer(self, env, chat: str):
        """Handle a high-level chat/instruction input from gameplay.

        Minimal implementation to avoid AttributeError when GUI/Gameplay
        calls this on all agents. Records the instruction in `_int_hist`.
        """
        try:
            if not hasattr(self, '_int_hist'):
                self._int_hist = []
            self._int_hist.append({'time': time.time(), 'chat': chat})
        except Exception:
            pass

    def _stabilize_task_ids_for_held_progress(self, env, current_task_ids):
        def get_holding_name(agent_idx):
            agents = getattr(env, 'agents', [])
            if agent_idx >= len(agents):
                return None
            holding = getattr(agents[agent_idx], 'holding', None)
            return getattr(holding, 'full_name', None) if holding is not None else None

        def apply_for_task(agent_idx, task):
            if not task:
                return

            holding_name = get_holding_name(agent_idx)
            if not holding_name:
                return

            task_id = None
            if isinstance(task, dict):
                task_id = task.get('id')
            if not isinstance(task_id, tuple):
                return

            verb, obj, order_uid = task_id
            if verb == 'chop':
                chopped_name = f"Chopped{obj.capitalize()}"
                if chopped_name in holding_name:
                    current_task_ids.add(task_id)
            elif verb in ('cook', 'serve_salad'):
                for ing in dish_ingredients(obj):
                    chopped_name = f"Chopped{ing.capitalize()}"
                    if chopped_name in holding_name:
                        current_task_ids.discard(('chop', ing, order_uid))

        if self.sc_2agent:
            for agent_idx in [0, 1]:
                sc = getattr(self, 'schedule_per_agent', {}).get(agent_idx, [])
                task_idx_val = getattr(self, 'current_task_idx', {})
                task_idx = task_idx_val.get(agent_idx, 0) if isinstance(task_idx_val, dict) else 0
                task = sc[task_idx] if task_idx < len(sc) else None
                apply_for_task(agent_idx, task)
                carry_task = getattr(self, 'carry_task_by_agent', {}).get(agent_idx) if isinstance(getattr(self, 'carry_task_by_agent', None), dict) else None
                apply_for_task(agent_idx, carry_task)
        else:
            schedule = getattr(self, 'schedule', [])
            task_idx = getattr(self, 'current_task_idx', 0)
            if not isinstance(task_idx, int):
                task_idx = 0
            task = schedule[task_idx] if task_idx < len(schedule) else None
            apply_for_task(0, task)
            apply_for_task(0, getattr(self, 'carry_task_by_agent', None))

        return current_task_ids

    def _hold_before_for_log(self, env_like):
        return getattr(env_like, 'hold', None)

    def _hold_hint_for_log(self, hold_before, reason):
        if hold_before is None:
            if reason == '皿の取得':
                return 'after_interact_may_be_plate'
            return 'no_hold_before'

        hold_str = str(hold_before)
        if hold_str == 'p':
            if reason in ('調理済み料理の取得', '調理完了待ち'):
                return 'holding_plate'
            return 'plate_like'

        return 'carrying_item'

    def _get_carry_override_task(self, env, agent_idx, scheduled_task):
        agents = getattr(env, 'agents', [])
        if agent_idx >= len(agents):
            return scheduled_task

        holding = getattr(agents[agent_idx], 'holding', None)
        holding_name = getattr(holding, 'full_name', None) if holding is not None else None
        if getattr(self, 'debug_counter_trace', False) and holding_name:
            sched_id = scheduled_task.get('id') if scheduled_task else None
            carry_now = self.carry_task_by_agent.get(agent_idx) if self.sc_2agent and isinstance(self.carry_task_by_agent, dict) else getattr(self, 'carry_task_by_agent', None)
            self._emit_counter_debug(f"[CarryOverride] agent={agent_idx} holding_name={holding_name!r} scheduled_id={sched_id} carry_task_id={carry_now.get('id') if carry_now else None}")
        if not holding_name:
            if self.sc_2agent:
                self.carry_task_by_agent[agent_idx] = None
            else:
                self.carry_task_by_agent = None
            return scheduled_task

        def finish_or_handover(verb, dish, kind):
            """完成品を持っているときの行き先。

            提供口が仕切りの向こうにあると自分では配膳できない。その場合は
            受け渡し台に置いて相手に渡す。サラダ・スープ・ジュースのどれでも
            事情は同じなので、ここに集約する。
            """
            if not self._can_reach_delivery(env, agent_idx):
                counter = self._find_shared_counter(env, env.self_pos)
                if counter is not None:
                    return {
                        'id': ('handover', dish, -1),
                        'res': ('delivery', None),
                        'assigned_counter': counter,
                        'dish_kind': kind,
                    }
            return {'id': (verb, dish, -1), 'res': ('delivery', None)}

        if 'Plate' in holding_name and 'Cooked' in holding_name:
            cooked_parts = []
            for part in holding_name.split('-'):
                if part.startswith('Cooked'):
                    cooked_parts.append(part.replace('Cooked', '').lower())
            if cooked_parts:
                cooked_parts.sort()
                return finish_or_handover(
                    'serve', f"{'-'.join(cooked_parts)}{SOUP_SUFFIX}", KIND_SOUP)

        # コップに混ぜたものが入っている = ジュースの完成品。提供口へ運ぶだけ。
        if 'Cup' in holding_name and 'Mixed' in holding_name:
            mixed_parts = sorted(
                part.replace('Mixed', '').lower()
                for part in holding_name.split('-') if part.startswith('Mixed'))
            if mixed_parts:
                return finish_or_handover(
                    'serve_juice', f"{'-'.join(mixed_parts)}{JUICE_SUFFIX}", KIND_JUICE)

        # 皿の上に刻んだ食材が乗っている = サラダの完成品。あとは提供口へ運ぶだけ。
        if 'Plate' in holding_name and 'Chopped' in holding_name:
            chopped_parts = []
            for part in holding_name.split('-'):
                if part.startswith('Chopped'):
                    chopped_parts.append(part.replace('Chopped', '').lower())
            # フルーツは皿ではなくコップに入れるので、サラダ扱いにしない。
            if any(part in FRUITS for part in chopped_parts):
                chopped_parts = []
            if chopped_parts:
                chopped_parts.sort()
                return finish_or_handover(
                    'serve_salad', f"{'-'.join(chopped_parts)}{SALAD_SUFFIX}", KIND_SALAD)

        chopped_combo_parts = []
        if 'Plate' not in holding_name and '-' in holding_name:
            parts = holding_name.split('-')
            if parts and all(part.startswith('Chopped') for part in parts):
                chopped_combo_parts = sorted(part.replace('Chopped', '').lower() for part in parts)

        carried_ing = None
        carried_ing_is_fresh = False
        if '-' not in holding_name:
            if holding_name.startswith('Fresh'):
                carried_ing = holding_name.replace('Fresh', '').lower()
                carried_ing_is_fresh = True
            elif holding_name.startswith('Chopped'):
                carried_ing = holding_name.replace('Chopped', '').lower()

        carry_task = self.carry_task_by_agent[agent_idx] if self.sc_2agent else self.carry_task_by_agent

        def matches_single_chopped(task):
            if carried_ing is None or not holding_name.startswith('Chopped') or task is None:
                return False
            verb, obj, _ = task['id']
            if verb not in ('cook', 'serve_salad'):
                return False
            needed_parts = dish_ingredients(obj)
            return carried_ing in needed_parts

        if scheduled_task:
            verb, obj, _ = scheduled_task['id']
            if verb in ('cook', 'serve_salad') and chopped_combo_parts:
                scheduled_parts = sorted(dish_ingredients(obj))
                if scheduled_parts == chopped_combo_parts:
                    if self.sc_2agent:
                        self.carry_task_by_agent[agent_idx] = deepcopy(scheduled_task)
                    else:
                        self.carry_task_by_agent = deepcopy(scheduled_task)
                    return scheduled_task
            if matches_single_chopped(scheduled_task):
                if self.sc_2agent:
                    self.carry_task_by_agent[agent_idx] = deepcopy(scheduled_task)
                else:
                    self.carry_task_by_agent = deepcopy(scheduled_task)
                return scheduled_task
            if verb == 'chop':
                food_names = (f"Fresh{obj.capitalize()}", f"Chopped{obj.capitalize()}")
                if any(food_name in holding_name for food_name in food_names):
                    if self.sc_2agent:
                        self.carry_task_by_agent[agent_idx] = deepcopy(scheduled_task)
                    else:
                        self.carry_task_by_agent = deepcopy(scheduled_task)
                    return scheduled_task

        if carry_task:
            verb, obj, _ = carry_task['id']
            if verb in ('cook', 'serve_salad') and chopped_combo_parts:
                carry_parts = sorted(dish_ingredients(obj))
                if carry_parts == chopped_combo_parts:
                    return deepcopy(carry_task)
            if matches_single_chopped(carry_task):
                return deepcopy(carry_task)
            food_names = (f"Fresh{obj.capitalize()}", f"Chopped{obj.capitalize()}")
            # 複数の食材がマージ済みのものを持っているときは、単品の chop タスクでは
            # 扱えない(process_chop_task は「1食材を切って置く」しかできず、
            # 置き場が決まっていないと永久に待機する)。この場合は下の
            # chopped_combo_parts の処理に任せて、注文に対応した cook / serve_salad
            # タスクへ読み替える。
            if verb == 'chop' and not chopped_combo_parts and any(food_name in holding_name for food_name in food_names):
                return deepcopy(carry_task)

        if chopped_combo_parts:
            assigned_counter = None
            if carry_task:
                assigned_counter = carry_task.get('assigned_counter')
            if assigned_counter is None and scheduled_task:
                assigned_counter = scheduled_task.get('assigned_counter')

            # 持っている組み合わせをそのままレシピとして鍋に入れてはいけない。
            # 3種スープの途中(レタス+玉ねぎを持って3つ目を取りに行く状態)で
            # ここに来ることがあり、そのまま入れると2種のまま調理が確定して
            # 注文が永久に完成しなくなる。
            # まだ材料が足りない注文の一部なら、その注文の完成レシピを目標にする。
            # process_cook_task 側は「一部しか持っていない」と判断して、
            # 残りが集まる置き場へマージしに行く。
            recipe_parts, recipe_counter, recipe_kind = self._find_order_recipe_for_partial(
                env, chopped_combo_parts
            )
            if recipe_parts is None:
                recipe_parts = chopped_combo_parts
            elif recipe_counter is not None:
                assigned_counter = recipe_counter

            # サラダ注文の作りかけなら、鍋ではなく皿へ運ぶタスクとして扱う。
            # ここで cook を返すと、刻んだ食材がそのまま鍋に入れられて
            # サラダがスープとして提供されてしまう。
            if recipe_kind == KIND_JUICE:
                # ジュースの作りかけは鍋でも皿でもなく、ミキサーへ運ぶ。
                return {
                    'id': ('mix', f"{'-'.join(recipe_parts)}{JUICE_SUFFIX}", -1),
                    'res': ('blender', None),
                    'assigned_counter': assigned_counter,
                }

            if recipe_kind == KIND_SALAD:
                return {
                    'id': ('serve_salad', f"{'-'.join(recipe_parts)}{SALAD_SUFFIX}", -1),
                    'res': ('delivery', None),
                    'assigned_counter': assigned_counter,
                }

            return {
                'id': ('cook', f"{'-'.join(recipe_parts)}{SOUP_SUFFIX}", -1),
                'res': ('pot', None),
                'assigned_counter': assigned_counter,
            }

        # 「刻む前(Fresh)」の食材は、どの注文にもまだ紐づいていなくても
        # 「とりあえず切ってしまう」のが常に安全(切った後の置き場は次サイクルで
        # 改めて決まる)。しかし「刻んだ後(Chopped)」の食材でここまで来たもの
        # (carry_task にも scheduled_task にも一致しなかったもの)は、既にどこかの
        # 注文の要求が別経路で満たされたことによる余剰品である可能性が高い。
        # ここで chop タスクとして再解釈して assigned_counter(他タスクから借用した、
        # 無関係な場所)へ運ぼうとすると、既に埋まっている置き場へ届けようとして
        # マージできず詰まる。scheduled_task をそのまま返し、通常の「不要品」経路
        # (現在のタスクに合わない持ち物は空きカウンターへ自動的に置く)に任せる。
        if carried_ing and carried_ing_is_fresh:
            assigned_counter = None
            if carry_task:
                assigned_counter = carry_task.get('assigned_counter')
            if assigned_counter is None and scheduled_task:
                assigned_counter = scheduled_task.get('assigned_counter')
            return {
                'id': ('chop', carried_ing, -1),
                'res': ('cutboard', None),
                'assigned_counter': assigned_counter,
            }

        if self.sc_2agent:
            self.carry_task_by_agent[agent_idx] = None
        else:
            self.carry_task_by_agent = None

        return scheduled_task

    def _recipe_ingredient_set(self, dish_name):
        return {
            self._normalize_ingredient_name(part)
            for part in dish_ingredients(dish_name)
            if self._normalize_ingredient_name(part)
        }

    def _has_usable_pot_for_cook(self, env, soup_name):
        """この cook タスクが実際に使える鍋があるか。

        エンジンの interact() は「空の鍋」にしか食材を投入できないため、
        使えるのは (a) 空の鍋 か (b) 既にこのレシピが入っている鍋だけ。
        """
        expected = self._recipe_ingredient_set(soup_name)
        for pot in self._get_resources(env).get('pots', []):
            if env.pos_obj.get(pot) is None:
                return True
            if expected and self._get_counter_food_names(env, pot) == expected:
                return True
        return False

    def _collect_ready_cook_actions(self, env, current_orders):
        """「いま即座に着手できる cook タスク」を (動詞, 対象) の集合で返す。

        指示タイミングを cook に固定する実験モード(--instruction_request_timing
        enable_cook)のための監視用。GamePlay 側から毎フレーム世界を評価し直すと
        重いうえに _build_order_tasks の副作用(置き場の割り当て)まで走ってしまうため、
        AI が通常の判断サイクルで作った current_orders を使ってここで求めておき、
        GamePlay は結果を読むだけにする。

        「即座に着手できる」= 材料が刻み終わって世界に存在し、かつ実際に
        投入できる鍋がある状態。どちらか欠けていると、選ばせても AI はその場で
        待つことしかできない。

        注文番号(order_uid)を含めないのは、注文が1件配達されて補充されるたびに
        番号が振り直され、中身が同じタスクなのに別物として再発火してしまうため。
        指示パネルの候補も (動詞, 対象) 単位でまとめているので、粒度も揃う。
        """
        ready = set()
        for order in current_orders:
            for task in order.get('tasks', []):
                task_id = task.get('id')
                if not (isinstance(task_id, tuple) and len(task_id) >= 3):
                    continue
                verb, obj, _order_uid = task_id
                if verb != 'cook' or task_id in self.completed_task_ids:
                    continue
                if not self._cook_dependency_ready_from_world(env, obj):
                    continue
                if not self._has_usable_pot_for_cook(env, obj):
                    continue
                ready.add((verb, obj))
        return ready

    def _find_ready_serve_task(self, env, agent_idx):
        """鍋が塞がって cook が進めないとき、鍋を空けられる serve タスクを探す。

        スケジュールは先頭から順に実行するため、cook が鍋待ちで止まると、
        その鍋を空けるはずの serve が後ろにあって永久に実行されない。
        結果として「鍋が空くまで待機中」のまま両方が固まる典型的な
        デッドロックになる。取り出せる状態の鍋があるなら serve を先に行う。
        """
        pots = self._get_resources(env).get('pots', [])
        for candidate in self.schedule_per_agent.get(agent_idx, []):
            task_id = candidate.get('id')
            if not (isinstance(task_id, tuple) and len(task_id) >= 3):
                continue
            verb, obj, _order_uid = task_id
            if verb != 'serve':
                continue
            expected = self._recipe_ingredient_set(obj)
            if not expected:
                continue
            for pot in pots:
                pot_obj = env.pos_obj.get(pot)
                if pot_obj is None:
                    continue
                is_cooked = getattr(pot_obj, 'is_cooked', None)
                if not callable(is_cooked) or not is_cooked():
                    continue
                if self._get_counter_food_names(env, pot) == expected:
                    return candidate
        return None

    def _find_takeover_task_for_deps(self, missing_deps, agent_idx, env=None):
        """待ちの原因になっている前提タスクを、自分で引き受けるために探す。

        CSP は常に2エージェント分のスケジュールを作るが、human_counterpart_mode
        では「もう一方」は CSP が指示できない人間であり、人間スロットに割り当てられた
        タスクは誰も実行しない。そのため自分のタスクの前提が人間スロットにあると、
        AI はその場で永久に待ち続けて停止してしまう(実プレイで後半に固まる原因)。

        足りない前提タスク(missing_deps)に一致するタスクをスケジュール全体から探す。
        自分のスケジュールを優先し、無ければ人間スロットのものを引き受ける。
        見つからなければ None。
        """
        if not missing_deps:
            return None
        missing_set = set(missing_deps)
        partitioned = env is not None and self._map_is_partitioned(env)
        if partitioned:
            # 位置が無いままだと「誰でもできる」と判定されてしまう。
            for tasks in self.schedule_per_agent.values():
                self._annotate_task_geometry(env, tasks, env.self_pos)
        for search_idx in (agent_idx, 1 - agent_idx):
            for candidate in self.schedule_per_agent.get(search_idx, []):
                if candidate['id'] not in missing_set:
                    continue
                # 仕切りの向こうのタスクを引き受けても、資材に手が届かない。
                # 引き受けた側がそこで止まるだけなので、できるものだけにする。
                if partitioned and agent_idx not in self._task_allowed_agents(env, candidate):
                    continue
                return candidate
        return None

    def __call__(self, env):
        """
        環境から呼ばれるメイン関数
        """
        # 常にタスクリストを構築して変化をチェック
        current_orders = self._build_order_tasks(env)
        current_task_ids = set()
        for o in current_orders:
            for t in o['tasks']:
                current_task_ids.add(t['id'])
        current_task_ids = self._stabilize_task_ids_for_held_progress(env, current_task_ids)
        event_reason = self._collect_event_replan_reason(env)

        # 指示タイミング固定モード(enable_cook)の監視用。GamePlay が読む。
        self.ready_cook_actions = self._collect_ready_cook_actions(env, current_orders)

        # 人間が推測と違うことをしていたら、予測ミスとして再スケジューリングする
        if self.sc_2agent and self.human_counterpart_mode and self.use_predicted_human_model:
            self._check_human_prediction(env)
            self._check_human_prediction_by_cost(
                env, [t for o in current_orders for t in o['tasks']])

        if not hasattr(self, 'prev_task_ids'):
            self.prev_task_ids = set()

        conflicts = self._detect_counter_conflicts(env, current_orders)
        if conflicts:
            self._emit_counter_debug(f"  [カウンター衝突検知] {conflicts}")
            for conflict in conflicts:
                self._set_assigned_counter(conflict['order_uid'], None)
                self._log_counter_policy(conflict['order_uid'], "release", conflict['counter'], "reason=counter_conflict_detected")
            self._mark_reschedule_needed("counter_conflict_detected")

        added = current_task_ids - self.prev_task_ids
        removed = self.prev_task_ids - current_task_ids

        if self._should_defer_holding_reschedule(env, added, removed):
            added = set()
            removed = set()

        reschedule_reason = None if self.no_reschedule and self.initialized else event_reason or self._get_reschedule_reason(current_task_ids, added, removed)

        # Pickup/Put/Chop などのイベントが短時間に連発すると、その連続イベントの
        # 合間の一時的で曖昧な世界状態(食材がどこにも見えない/複数の注文から
        # 同時に見える瞬間)を拾って毎回スケジュール全体を再計算してしまい、同じ
        # 食材を複数の注文が奪い合うような揺れを引き起こす。直近の実際の再計算
        # からこの秒数(シミュレーション内時間)経っていなければ、今回は再計算を
        # 見送り、理由だけ pending として保持して後でまとめて再計算する。
        if reschedule_reason is not None and self.initialized:
            current_env_time = getattr(env, 'time', None)
            last_resched = self._last_reschedule_time
            if (
                current_env_time is not None
                and last_resched is not None
                and (current_env_time - last_resched) < self._min_reschedule_interval_seconds
            ):
                self._mark_reschedule_needed(reschedule_reason)
                reschedule_reason = None

        # 必要なタイミングだけリスケジュールする
        if reschedule_reason is not None:
            self._log_reschedule_event(reschedule_reason, env, added=added, removed=removed)
            if removed and self.initialized:
                physically_done = {t for t in removed if t[0] == 'cook'}
                if physically_done:
                    self._emit_counter_debug(f"  [完了検知] 物理完了タスク: {physically_done}")
                    self.completed_task_ids |= physically_done
                    self._mark_reschedule_needed("physical_completion")

            if self.initialized and self.debug_counter_trace:
                self._emit_counter_debug(f"\n[タスク更新] 時間: {env.time}")
                self._emit_counter_debug(f"  [再計算理由] {reschedule_reason}")
                if added:
                    self._emit_counter_debug(f"  (+) 追加: {added}")
                if removed:
                    self._emit_counter_debug(f"  (-) 削除: {removed}")
                self._emit_counter_debug("  -> スケジュール再計算中...")

            in_progress_tasks = {}
            if self.sc_2agent and hasattr(self, 'schedule_per_agent'):
                for aidx in [0, 1]:
                    sc = self.schedule_per_agent.get(aidx, [])
                    t_idx = self.current_task_idx.get(aidx, 0) if isinstance(self.current_task_idx, dict) else 0
                    if t_idx < len(sc):
                        in_progress_tasks[aidx] = deepcopy(sc[t_idx])
                        self._emit_counter_debug(f"  [継続確認] AI{aidx} 実行中タスク: {sc[t_idx]['id']}")

            try:
                start_time = time.time()
                self.schedule = self.solve_csp_scheduling(env, orders=current_orders)
                self._last_reschedule_time = getattr(env, 'time', self._last_reschedule_time)
                elapsed_time = time.time() - start_time
                self._emit_counter_debug(f"[CSPAgent] スケジューリング時間: {elapsed_time:.4f} 秒")

                self._print_schedule(self.schedule)

                if self.sc_2agent:
                    new_idx = {0: 0, 1: 0}
                    for aidx in [0, 1]:
                        in_prog_task = in_progress_tasks.get(aidx)
                        if in_prog_task:
                            in_prog_tid = in_prog_task['id']
                            new_sc = self.schedule_per_agent.get(aidx, [])
                            found = False
                            for i, t in enumerate(new_sc):
                                if t['id'] == in_prog_tid:
                                    new_idx[aidx] = i
                                    self._emit_counter_debug(f"  [継続] AI{aidx}: {in_prog_tid} → 新スケジュール idx={i} から再開")
                                    found = True
                                    break
                            if not found:
                                holding = getattr(env.agents[aidx], 'holding', None)
                                holding_name = getattr(holding, 'full_name', None) if holding is not None else None
                                if in_prog_tid[0] == 'chop' and holding_name and f"{in_prog_tid[1].capitalize()}" in holding_name:
                                    # スケジュールから消えた理由は2通りある:
                                    # (1) 自分が持っている分でこの注文が満たされた → 継続して届けるべき
                                    # (2) 既に別の経路(他の食材の置き場)でこの注文が満たされていた
                                    #     → 自分が持っているのは余剰品で、届け先(counter)は既に埋まっている
                                    # (2) なのに carry task として再アタッチすると、満杯のcounterへ
                                    # 届けようとしてマージできず永久に詰まる。届け先に既に同じ食材が
                                    # あるかを見て、あれば余剰品とみなし carry task にしない
                                    # (通常の「不要品」経路でどこか空きcounterへ自動的に置かれる)。
                                    order_uid_for_carry = in_prog_tid[2]
                                    assigned_counter_for_order = self._get_assigned_counter(order_uid_for_carry)
                                    already_satisfied = False
                                    if assigned_counter_for_order is not None:
                                        counter_foods = self._get_counter_food_names(env, assigned_counter_for_order)
                                        if self._normalize_ingredient_name(in_prog_tid[1]) in counter_foods:
                                            already_satisfied = True
                                    if not already_satisfied:
                                        self.carry_task_by_agent[aidx] = deepcopy(in_prog_task)
                                        self._emit_counter_debug(f"  [保持継続] AI{aidx}: {in_prog_tid} を carry task として継続")
                                    else:
                                        self._emit_counter_debug(f"  [余剰品] AI{aidx}: {in_prog_tid} の届け先は既に満たされているため carry task にしない")
                                self._emit_counter_debug(f"  [スキップ] AI{aidx}: {in_prog_tid} が新スケジュールに存在しない → idx=0から開始")
                                new_idx[aidx] = 0
                    self.current_task_idx = new_idx
                else:
                    self.current_task_idx = 0

                if self.sc_2agent:
                    preempt_target = self._get_instruction_preempt_target(env)
                    if preempt_target is not None:
                        pending, fixed_task_id, agent_idx, target_idx = preempt_target
                        if isinstance(self.current_task_idx, dict):
                            if self.current_task_idx.get(agent_idx, 0) != target_idx:
                                self.current_task_idx[agent_idx] = target_idx
                                self.carry_task_by_agent[agent_idx] = None
                                task_agent = self.task_agents[agent_idx]
                                task_agent.assigned_counter = None
                                task_agent.assigned_cutboard = None
                                task_agent.assigned_pot = None
                                task_agent.assigned_plate = None
                                task_agent.assigned_serve_loc = None
                                pending['status'] = 'started'
                                pending['execution_logged'] = True
                                pending['deadline_constraint_applied'] = True
                                self._mark_reschedule_needed('instruction_preempt')
                                self._emit_counter_debug(f"[CSPAgent] 2エージェント即時切替: fixed_id={fixed_task_id} agent={agent_idx} idx={target_idx}")
                else:
                    preempt_target = self._get_immediate_instruction_preempt(env)
                    if preempt_target is not None:
                        pending, fixed_task_id, target_idx = preempt_target
                        if self.current_task_idx != target_idx:
                            self._just_preempted = True
                            self.current_task_idx = target_idx
                            self.carry_task_by_agent = None
                            self.task_agent.assigned_counter = None
                            self.task_agent.assigned_cutboard = None
                            self.task_agent.assigned_pot = None
                            self.task_agent.assigned_plate = None
                            self.task_agent.assigned_serve_loc = None
                            pending['status'] = 'started'
                            pending['execution_logged'] = True
                            pending['deadline_constraint_applied'] = True
                            self._mark_reschedule_needed('instruction_preempt')
                            self._emit_counter_debug(f"[CSPAgent] 期限0指示により即時切替: fixed_id={fixed_task_id} idx={target_idx}")

                self.pending_reschedule_reason = None
                if self.sc_2agent:
                    self.stall_counts = {0: 0, 1: 0}
                else:
                    self.stall_counts = 0

            except Exception as e:
                self._emit_counter_debug(f"[CSPAgent] CSPスケジュール中に例外: {e}")
                import traceback
                traceback.print_exc()

            self.initialized = True

        self.prev_task_ids = current_task_ids

        # ====== スケジュール実行 ======
        if not self.sc_2agent:
            # 単一エージェントモード
            schedule = getattr(self, 'schedule', None) or []
            task_idx = self.current_task_idx
            if task_idx >= len(schedule):
                # 担当が尽きていても、持ち物から継続すべき作業が決まることがある
                # (2エージェントモード側の同じ分岐のコメントを参照)。
                # self.current_task_idx は進めない(進めると次フレームで
                # 本来のスケジュールを先頭からやり直してしまう)。
                fallback_task = self._get_carry_override_task(env, 0, None)
                if fallback_task is None:
                    return (0, 0), "タスクなし"
                schedule = [fallback_task]
                task_idx = 0

            preempted_this_frame = bool(getattr(self, '_just_preempted', False))
            scheduled_task = schedule[task_idx]
            task = self._get_carry_override_task(env, 0, scheduled_task)
            scheduled_tid = scheduled_task['id']
            tid = task['id']
            verb, obj, order_uid = tid
            res = task['res'] 

            task_name = None
            if verb == 'chop':
                task_name = f"chop_{obj}"
                if getattr(self.task_agent, 'assigned_task_id', None) != tid or getattr(self.task_agent, 'assigned_counter', None) is None:
                    self.task_agent.assigned_counter = task.get('assigned_counter')
            elif verb == 'cook':
                parts = dish_ingredients(obj)
                task_name = f"cook_{'_'.join(parts)}"
                if getattr(self.task_agent, 'assigned_task_id', None) != tid or getattr(self.task_agent, 'assigned_counter', None) is None:
                    self.task_agent.assigned_counter = task.get('assigned_counter')
            elif verb == 'serve_salad':
                # サラダは置き場に集めた刻んだ食材を取りに行くので、
                # serve と違って assigned_counter を保持する。
                parts = dish_ingredients(obj)
                task_name = f"serve_salad_{'_'.join(parts)}"
                if getattr(self.task_agent, 'assigned_task_id', None) != tid or getattr(self.task_agent, 'assigned_counter', None) is None:
                    self.task_agent.assigned_counter = task.get('assigned_counter')
            elif verb == 'mix':
                # ミキサーへ入れる材料は置き場に集めてあるので、chop/cook と
                # 同じく assigned_counter を保持する。
                parts = dish_ingredients(obj)
                task_name = f"mix_{'_'.join(parts)}"
                if getattr(self.task_agent, 'assigned_task_id', None) != tid or getattr(self.task_agent, 'assigned_counter', None) is None:
                    self.task_agent.assigned_counter = task.get('assigned_counter')
            elif verb == 'serve_juice':
                parts = dish_ingredients(obj)
                task_name = f"serve_juice_{'_'.join(parts)}"
                self.task_agent.assigned_counter = None
            elif verb == 'handover':
                parts = dish_ingredients(obj)
                task_name = f"handover_{'_'.join(parts)}"
                self.task_agent.assigned_counter = task.get('assigned_counter')
                self.task_agent.dish_kind = task.get('dish_kind') or dish_kind_of(obj)
            elif verb == 'serve_from_counter':
                parts = dish_ingredients(obj)
                task_name = f"serve_from_counter_{'_'.join(parts)}"
                self.task_agent.assigned_counter = task.get('assigned_counter')
                self.task_agent.dish_kind = task.get('dish_kind') or dish_kind_of(obj)
            elif verb == 'serve':
                parts = dish_ingredients(obj)
                task_name = f"serve_{'_'.join(parts)}"
                self.task_agent.assigned_counter = None

            if task_name:
                self.task_agent.task_name = task_name
                self.task_agent.assigned_task_id = tid
                # 「切らずに運ぶだけ」の指定(既に切られた物が別テーブルにある場合)
                self.task_agent.carry_from = task.get('carry_from') if verb == 'chop' else None
                action, reason = self.task_agent(env)
                hold_before = self._hold_before_for_log(env)
                hold_hint = self._hold_hint_for_log(hold_before, reason)
                self._emit_counter_debug(
                    f"[ACTION] AI task={task_name} tid={tid} action={action} reason='{reason}' "
                    f"hold_before={hold_before} hold_hint={hold_hint} counter={self.task_agent.assigned_counter}"
                )
                
                # chop/serve の "(完了)" は「置く/配膳すると決定した瞬間」に返される信号であり、
                # 実際に環境へ反映された(=本当に手放した)ことの確認ではない。これを即座に
                # 完了扱いにすると、非同期反映が追いつく前に次のタスクへ進んでしまい、
                # 持っている食材が別の置き場へ付け替えられて「置いては別の場所へ運び直す」
                # 挙動を引き起こす。cook の "(Done)" は鍋の中身という実観測に基づくため、
                # 即座に完了扱いにして問題ない。chop/serve は実際に手放したことを確認できる
                # イベント(event_history の Put_/Deliver_)に基づく既存の再スケジュール検知に任せ、
                # ここでは先走って completed_task_ids やタスク進行を進めない。
                reason_is_done = "Done" in reason or "done" in reason or "完了" in reason
                if reason_is_done and verb not in ('chop',) + self.SERVE_VERBS:
                    self._emit_counter_debug(f"[CSPAgent] タスク {task_name} 完了。次へ移動。")
                    self.completed_task_ids.add(tid)
                    self._update_skip_budget_on_completion(tid, 0, scheduled_task.get('dur', 0))
                    if tid == scheduled_tid and not preempted_this_frame:
                        self.current_task_idx += 1
                    self._mark_reschedule_needed("task_completed_single")
                    self.carry_task_by_agent = None
                    self.task_agent.assigned_cutboard = None
                    self.task_agent.assigned_pot = None
                    self.task_agent.assigned_plate = None
                    self.task_agent.assigned_serve_loc = None
                    self.task_agent.assigned_counter = None
                    self.task_agent.assigned_task_id = None
                    self._just_preempted = False
                else:
                    self._just_preempted = False
                
                return action, reason

            return (0, 0), "アイドル"

        else:
            # ====== 2エージェントモード ======
            # human_counterpart_mode=True のときは CSP は own_agent_idx の行動だけ返す。
            # human_counterpart_mode=False のときは両方の行動を返す。
            if not hasattr(self, 'schedule_per_agent') or not self.schedule_per_agent:
                if self.human_counterpart_mode:
                    return {f"ai_{self.own_agent_idx}": (0, 0)}, "タスクなし"
                return {"ai_0": (0, 0), "ai_1": (0, 0)}, "タスクなし"

            exec_indices = [self.own_agent_idx] if self.human_counterpart_mode else [0, 1]

            actions = {}
            reasons = []
            for agent_idx in exec_indices:

                sc = self.schedule_per_agent.get(agent_idx, [])
                t_idx = self.current_task_idx[agent_idx]
                if t_idx >= len(sc):
                    # 自分の担当が尽きたら、人間スロットに置いたタスクを引き受ける。
                    # 人間がいま手をつけていると推測して人間スロットへ回したタスクは、
                    # 推測が外れると誰も実行しない。AI が手待ちのまま止まるより、
                    # 自分でやってしまう方が常に良い(人間が先にやれば、その結果が
                    # 世界に現れて次の再スケジューリングでタスクから消える)。
                    # 担当が尽きていても、手に持っているものから「やるべきこと」が
                    # 決まる場合がある(例: 材料を全部集め終えて手に持っているが、
                    # その注文の提供タスクは相手側に割り当てられている)。
                    # そのまま Idle にすると、持ち物を抱えたまま止まり、相手も
                    # その食材を見つけられずに双方が固まる。
                    takeover = self._get_carry_override_task(env, agent_idx, None)
                    if takeover is not None:
                        self._emit_counter_debug(
                            f"[DEBUG] AI{agent_idx} 担当が尽きたが持ち物から継続: {takeover['id']}")
                    if takeover is None and (self.human_counterpart_mode
                                             or self.partner_is_external):
                        # 人間がいま手をつけていると推測して人間スロットへ回したタスクは、
                        # 推測が外れると誰も実行しない。AI が手待ちのまま止まるより、
                        # 自分でやってしまう方が常に良い(人間が先にやれば、その結果が
                        # 世界に現れて次の再スケジューリングでタスクから消える)。
                        other_sc = self.schedule_per_agent.get(1 - agent_idx, [])
                        # 仕切りのあるマップでは、相手側にしかできないタスクがある。
                        # それを引き受けると資材に到達できず永久に止まるので、
                        # 自分で実行できるものだけを候補にする。
                        # 位置が入っていないタスクは「誰でもできる」と判定されて
                        # しまうため、判定の前に必ず位置を入れ直す。
                        self._annotate_task_geometry(env, other_sc, env.self_pos)
                        takeover = next(
                            (t for t in other_sc
                             if agent_idx in self._task_allowed_agents(env, t)),
                            None)
                        if takeover is not None:
                            self._emit_counter_debug(
                                f"[DEBUG] AI{agent_idx} 手待ちのため人間スロットのタスクを引き受け: {takeover['id']}")
                    if takeover is None:
                        actions[f"ai_{agent_idx}"] = (0, 0)
                        reasons.append(f"AI{agent_idx}:Idle")
                        continue
                    sc = [takeover]
                    t_idx = 0

                scheduled_task = sc[t_idx]
                task = self._get_carry_override_task(env, agent_idx, scheduled_task)
                scheduled_tid = scheduled_task['id']
                tid = task['id']
                verb, obj, order_uid = tid
                if getattr(self, 'debug_counter_trace', False) and tid != scheduled_tid:
                    self._emit_counter_debug(f"[CarryOverride] agent={agent_idx} scheduled={scheduled_tid} overridden_to={tid}")

                import copy
                e_agent = copy.copy(env)
                e_agent.agent_idx = agent_idx
                # EnvState は生成時に「そのエージェントから到達できるマス」を
                # 計算して持っている。浅いコピーで agent_idx だけ差し替えると、
                # 中身は元のエージェント(0番)のものが残る。仕切りのあるマップでは
                # 「相手側の資材が自分から使える」と誤認して動けなくなるので、
                # 視点を変えたら作り直す。
                if agent_idx != getattr(env, 'agent_idx', 0):
                    from agent.executor.low import bfs_reachable
                    e_agent.rch_map = bfs_reachable(e_agent.to_grid, e_agent.self_pos)
                # 相手の現在位置を dynamic_obstacles として先に定義（can_start前に必要）
                other_pos = env.agents[1 - agent_idx].location
                dynamic_obstacles = {other_pos}

                # --- 先行する依存タスクが終わっているか（フライング実行エラーの防止） ---
                # スケジュール上に存在しないchopタスクは「すでに食材がある」ため完了済みとみなす
                all_scheduled_ids = {t['id'] for agent_sc in self.schedule_per_agent.values() for t in agent_sc}

                # 鍋が全部埋まっていて cook が進めないなら、鍋を空けられる serve を先に行う。
                # そうしないと「cook は鍋待ち → その鍋を空ける serve は cook の後ろで
                # 永久に実行されない」というデッドロックになる。
                if verb == 'cook' and not self._has_usable_pot_for_cook(env, obj):
                    ready_serve = self._find_ready_serve_task(env, agent_idx)
                    if ready_serve is not None:
                        self._emit_counter_debug(
                            f"[DEBUG] AI{agent_idx} 鍋が塞がっているため serve を先行実行: "
                            f"{ready_serve['id']} (元のタスク {tid} は保留)"
                        )
                        task = ready_serve
                        tid = task['id']
                        verb, obj, order_uid = tid

                can_start = True
                if verb == 'cook':
                    # 依存判定は「古い task id」ではなく、「実際の食材の存在」を正とする。
                    # これにより、古い完了履歴によって cook が止まる問題を防ぐ。
                    ingredient_ready = self._cook_dependency_ready_from_world(env, obj)
                    if not ingredient_ready:
                        can_start = False
                elif verb == 'serve_salad':
                    # サラダは鍋を経由せず刻んだ食材をそのまま皿に乗せるので、
                    # 前提は cook と同じ「刻んだ食材が世界にあるか」。
                    if not self._salad_dependency_ready_from_world(env, obj):
                        can_start = False
                elif verb == 'serve':
                    # serve は皿の先取りができるので、cook 完了前でも TaskAgent に進める。
                    # 鍋前待機や実際の取得タイミングは process_serve_task 側で判定する。
                    can_start = True

                if not can_start:
                    missing_deps = []
                    if verb in ('cook', 'serve_salad'):
                        parts = dish_ingredients(obj)
                        missing_deps = [('chop', p, order_uid) for p in parts if ('chop', p, order_uid) not in self.completed_task_ids]
                    elif verb == 'serve':
                        if ('cook', obj, order_uid) not in self.completed_task_ids:
                            missing_deps = [('cook', obj, order_uid)]
                    self._emit_counter_debug(f"[DEBUG] AI{agent_idx} 待機中: {verb} '{obj}' の前提タスク未完了 -> {missing_deps}")
                    self._emit_counter_debug(f"[DEBUG]   完了済み: {self.completed_task_ids}")

                    # human_counterpart_mode では人間スロットのタスクは誰も実行しない。
                    # 前提タスクがそこに割り当たっていると永久に待ち続けて停止するため、
                    # 待つ前に自分で引き受けられないか探す。
                    takeover_task = None
                    if self.human_counterpart_mode or self.partner_is_external:
                        takeover_task = self._find_takeover_task_for_deps(
                            missing_deps, agent_idx, env)

                    if takeover_task is not None:
                        self._emit_counter_debug(
                            f"[DEBUG] AI{agent_idx} 前提タスクを自分で引き受け: {takeover_task['id']} "
                            f"(元のタスク {tid} は前提待ちのため保留)"
                        )
                        task = takeover_task
                        tid = task['id']
                        verb, obj, order_uid = tid
                    else:
                        # じゃまにならない場所に移動する
                        # 他エージェントが現在実行中のタスクを取得（避けるべきリソースを特定）
                        other_idx = 1 - agent_idx
                        other_sc = self.schedule_per_agent.get(other_idx, [])
                        other_t_idx = self.current_task_idx.get(other_idx, 0)
                        other_current_task = other_sc[other_t_idx] if other_t_idx < len(other_sc) else None

                        ta = self.task_agents[agent_idx]
                        action = ta.move_to_safe_position(
                            env,
                            blocking_task=other_current_task,  # 他エージェントが使っているリソース（避ける対象）
                            own_next_task=task,                # 自分が次に実行するタスク（近くで待機する基準）
                            dynamic_obstacles=dynamic_obstacles
                        )
                        actions[f"ai_{agent_idx}"] = action
                        reasons.append(f"AI{agent_idx}:MovingToSafePosition")
                        continue

                # -------------------------------------------------------------
                
                ta = self.task_agents[agent_idx]
                task_name = None
                ta.protected_counters = {
                    entry['counter']
                    for entry in self.counter_policy_by_order.values()
                    if entry.get('counter') is not None
                }
                
                # どのタスクでもassigned_counterを引き継ぐ（CSPでスケジュールされたカウンターがあるならそれを使う）
                if getattr(ta, 'assigned_task_id', None) != tid or getattr(ta, 'assigned_counter', None) is None:
                    ta.assigned_counter = task.get('assigned_counter')
                ta.assigned_task_id = tid
                # 「切らずに運ぶだけ」の指定(既に切られた物が別テーブルにある場合)
                ta.carry_from = task.get('carry_from') if verb == 'chop' else None

                # 置き場(assigned_counter)は「刻んだ材料を1か所に集める」ための
                # 指定で、chop/cook/serve_salad/mix はこれを見て動く。単体エージェント
                # 経路では渡していたが、2エージェント経路では渡していなかったため、
                # 指定なしのフォールバックに落ちて材料を持ったまま止まることがあった。
                if verb == 'chop':
                    task_name = f"chop_{obj}"
                    ta.assigned_counter = task.get('assigned_counter')
                elif verb == 'cook':
                    parts = dish_ingredients(obj)
                    task_name = f"cook_{'_'.join(parts)}"
                    ta.assigned_counter = task.get('assigned_counter')
                elif verb == 'serve_salad':
                    parts = dish_ingredients(obj)
                    task_name = f"serve_salad_{'_'.join(parts)}"
                    ta.assigned_counter = task.get('assigned_counter')
                elif verb == 'mix':
                    parts = dish_ingredients(obj)
                    task_name = f"mix_{'_'.join(parts)}"
                    ta.assigned_counter = task.get('assigned_counter')
                elif verb == 'serve_juice':
                    parts = dish_ingredients(obj)
                    task_name = f"serve_juice_{'_'.join(parts)}"
                elif verb == 'handover':
                    parts = dish_ingredients(obj)
                    task_name = f"handover_{'_'.join(parts)}"
                    ta.assigned_counter = task.get('assigned_counter')
                    ta.dish_kind = task.get('dish_kind') or dish_kind_of(obj)
                elif verb == 'serve_from_counter':
                    parts = dish_ingredients(obj)
                    task_name = f"serve_from_counter_{'_'.join(parts)}"
                    ta.assigned_counter = task.get('assigned_counter')
                    ta.dish_kind = task.get('dish_kind') or dish_kind_of(obj)
                elif verb == 'serve':
                    parts = dish_ingredients(obj)
                    task_name = f"serve_{'_'.join(parts)}"
                


                if task_name:
                    ta.task_name = task_name
                    if verb == 'cook':
                        self._emit_counter_debug(f"[DEBUG] AI{agent_idx} cook_task_start tid={tid} assigned_counter={ta.assigned_counter} completed={sorted(self.completed_task_ids)} hold={getattr(e_agent, 'hold', None)}")
                    
                    # 交互ターン待機は使わず、毎フレーム実行する。
                    action, reason = ta(e_agent, dynamic_obstacles=dynamic_obstacles)

                    hold_before = self._hold_before_for_log(e_agent)
                    hold_hint = self._hold_hint_for_log(hold_before, reason)

                    self._emit_counter_debug(
                        f"[ACTION] AI{agent_idx} task={task_name} tid={tid} action={action} reason='{reason}' "
                        f"hold_before={hold_before} hold_hint={hold_hint} counter={ta.assigned_counter} turn={self.turn}"
                    )
                    
                    if action == (0, 0) and reason not in ("待機(相手のターン)",):
                        self._emit_counter_debug(f"[DEBUG] AI{agent_idx} 停止: task={task_name} reason='{reason}' counter={ta.assigned_counter} hold_before={hold_before} hold_hint={hold_hint}")

                    self._update_stall_state(agent_idx, action, reason)

                    
                    # chop/serve の "(完了)" は「置く/配膳すると決定した瞬間」の信号であり、
                    # 実際に環境へ反映された確認ではない(cook の "(Done)" は鍋の中身という
                    # 実観測に基づくため即時確定してよい)。詳細は単一エージェント側の
                    # 同じ分岐のコメントを参照。ここで先走って完了扱いにしないことで、
                    # 手放す前に次のタスクへ進み食材の置き場が付け替わってしまう不具合を防ぐ。
                    reason_is_done = reason.endswith("(Done)") or reason.endswith("(完了)")
                    if reason_is_done and verb not in ('chop',) + self.SERVE_VERBS:
                        self._emit_counter_debug(f"[CSPAgent] AI{agent_idx} タスク {task_name} 完了。")
                        self.completed_task_ids.add(tid)
                        self._update_skip_budget_on_completion(tid, agent_idx, scheduled_task.get('dur', 0))
                        if tid == scheduled_tid:
                            self.current_task_idx[agent_idx] += 1
                        self._mark_reschedule_needed(f"task_completed_agent_{agent_idx}")
                        self.carry_task_by_agent[agent_idx] = None
                        ta.assigned_cutboard = None
                        ta.assigned_pot = None
                        ta.assigned_plate = None
                        ta.assigned_serve_loc = None
                        ta.assigned_counter = None
                        ta.assigned_task_id = None

                    actions[f"ai_{agent_idx}"] = action
                    reasons.append(reason)
                else:
                    actions[f"ai_{agent_idx}"] = (0, 0)
                    reasons.append("アイドル")

            actions = self._resolve_mutual_block(env, actions)
            return actions, " | ".join(reasons)

    # ============ OR-Tools: 0-1選択問題（予算内で重み最大化） ============ 
    def solve_csp_knapsack_with_ortools(self, env):
        orders = self._build_order_tasks(env)
        tasks = []
        for o in orders:
            tasks.extend(o['tasks'])

        model = CSPModel()
        var_names = []
        durations = {}
        benefits = {}
        for idx, t in enumerate(tasks):
            name = f"x_{t['verb']}_{t['obj']}_{t['order']}_{idx}"
            model.add_bool_var(name)
            var_names.append(name)
            durations[name] = int(t['dur'])
            benefits[name] = 100 if t['verb'] in self.SERVE_VERBS else 0

        model.add_linear_le(durations, self.budget_frames)

        name_by_task_id = {}
        for name, t in zip(var_names, tasks):
            name_by_task_id[id(t)] = name
        for o in orders:
            chops = [t for t in o['tasks'] if t['verb'] == 'chop']
            cooks = [t for t in o['tasks'] if t['verb'] == 'cook']
            serves = [t for t in o['tasks'] if t['verb'] in self.SERVE_VERBS]

            if cooks:
                c_name = name_by_task_id[id(cooks[0])]
                for ch in chops:
                    ch_name = name_by_task_id[id(ch)]
                    model.model.Add(model.vars[c_name] <= model.vars[ch_name])
            if serves:
                s_name = name_by_task_id[id(serves[0])]
                if cooks:
                    c_name = name_by_task_id[id(cooks[0])]
                    model.model.Add(model.vars[s_name] <= model.vars[c_name])
                else:
                    # サラダは cook が無いので、chop が全部選ばれていることを条件にする
                    for ch in chops:
                        ch_name = name_by_task_id[id(ch)]
                        model.model.Add(model.vars[s_name] <= model.vars[ch_name])

        model.maximize_linear(benefits)

        result = solve_csp(model, time_limit=5.0)

        selected = []
        if result.status_name in ("OPTIMAL", "FEASIBLE"):
            for name, t in zip(var_names, tasks):
                if result.solution.get(name, 0) == 1:
                    selected.append(t)
        return selected

    def _print_selection(self, selected_tasks):
        # print("\n=== OR-Tools 選択結果（予算内最大化） ===")
        total = 0
        # for t in selected_tasks:
        #     verb = t['verb']; obj = t['obj']; order = t.get('display_order', t.get('slot_idx', t['order']))
        #     dur = t['dur']
        #     total += dur
        #     print(f"選択: {verb} {obj} (注文{order+1}) 所要={dur}")
        # print(f"合計投入フレーム(選択分): {total}")
        # print("===================================\n")

    # ============ CSP（選択問題 A解釈） ============ 
    def _get_resources(self, env):
        cutboards = env.get_pos_by_obj_gs(gs="Cutboard")
        pots = env.get_pos_by_obj_gs(gs="Pot")
        deliveries = env.get_pos_by_obj_gs(gs="Delivery")
        plates = env.get_pos_by_obj_gs(gs="Plate") 
        if not plates:
            plates = env.get_pos_by_obj_gs(gs="PlateTile") 
        counters = env.get_pos_by_obj_gs(gs="Counter")
        blenders = env.get_pos_by_obj_gs(gs="Blender")
        cups = env.get_pos_by_obj_gs(gs="Cup")
        if not cups:
            cups = env.get_pos_by_obj_gs(gs="CupTile")

        return {
            'blenders': blenders,
            'cups': cups,
            'cup': cups[0] if cups else (0, 0),
            'cutboards': cutboards,
            'pots': pots,
            'delivery': deliveries[0] if deliveries else (0,0),
            'plate': plates[0] if plates else (0,0),
            # 仕切りのあるマップでは皿置き場が側ごとに要るため、一覧も持たせる。
            # 'plate' は後方互換のための代表1枚。
            'plates': plates,
            'counters': counters,
        }

    def _adjacent_walkable_positions(self, env, pos_list):
        width = env.world_width
        height = env.world_height
        grid = env.to_grid
        out = []
        for x, y in pos_list:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height and grid[nx][ny] == 1:
                    out.append((nx, ny))
        return list(set(out))

    def _nearest_by_path(self, env, start_pos, candidates):
        if not candidates:
            return None

        best = None
        best_dist = None
        for candidate in candidates:
            dist = self.astar_distance(env, start_pos, candidate)
            if dist is None:
                continue
            if best is None or dist < best_dist:
                best = candidate
                best_dist = dist
        return best

    def _annotate_task_geometry(self, env, tasks, default_start_pos):
        resources = self._get_resources(env)

        def raw_base_name(item):
            if item is None:
                return None
            if getattr(item, 'is_held', False):
                return None
            name = getattr(item, 'name', '')
            if name.startswith('Fresh'):
                return name.replace('Fresh', '')
            full_name = getattr(item, 'full_name', '')
            if full_name.startswith('Fresh'):
                return full_name.replace('Fresh', '')
            return None

        def chopped_base_name(item):
            if item is None:
                return None
            if getattr(item, 'is_held', False):
                return None
            if hasattr(item, 'is_chopped') and item.is_chopped():
                contents = getattr(item, 'contents', [])
                if contents:
                    return getattr(contents[0], 'name', None)
            name = getattr(item, 'name', '')
            if name.startswith('Chopped'):
                return name.replace('Chopped', '')
            return None

        def get_nearest(start_pos, candidates):
            if not candidates:
                return start_pos
            return min(candidates, key=lambda p: abs(p[0] - start_pos[0]) + abs(p[1] - start_pos[1]))

        for t in tasks:
            # 持ち物から作った臨時のタスクには verb/obj/order が無い。
            # id から取れる分だけ取り、取れないものは位置を入れずに飛ばす。
            tid = t.get('id')
            verb = t.get('verb') or (tid[0] if tid else None)
            obj = t.get('obj') if t.get('obj') is not None else (tid[1] if tid else None)
            if verb is None or obj is None:
                continue
            order_idx = t.get('slot_idx')
            if order_idx is None:
                order_idx = t.get('order')
            if order_idx is None:
                order_idx = (tid[2] if tid and isinstance(tid[2], int) and tid[2] >= 0 else 0)

            if verb == 'chop':
                tile_map = INGREDIENT_TILE
                ing_pos_list = env.get_pos_by_obj_gs(gs=tile_map.get(obj, ""))
                raw_candidates = []
                for pos, world_obj in env.pos_obj.items():
                    if world_obj is None:
                        continue
                    base_name = raw_base_name(world_obj)
                    if base_name is not None and base_name.lower() == obj:
                        raw_candidates.append(pos)

                if raw_candidates:
                    ing_pos = self._nearest_by_path(env, default_start_pos, raw_candidates) or get_nearest(default_start_pos, raw_candidates)
                else:
                    ing_pos = ing_pos_list[0] if ing_pos_list else default_start_pos

                cutboards = resources['cutboards']
                best_cb = get_nearest(ing_pos, cutboards)

                if t.get('assigned_counter'):
                    target = t['assigned_counter']
                else:
                    counters = env.get_pos_by_obj_gs(gs="Counter")
                    target = get_nearest(best_cb, counters) if counters else best_cb

                t['start_pos'] = ing_pos
                t['end_pos'] = target
                t['fixed_res'] = ('cutboard', best_cb)

            elif verb == 'cook':
                pots = resources['pots']
                pot = pots[order_idx % len(pots)] if pots else default_start_pos
                needed_ings = dish_ingredients(obj)
                start_candidates = []

                for pos, world_obj in env.pos_obj.items():
                    if world_obj is None:
                        continue
                    base_name = chopped_base_name(world_obj)
                    if base_name is not None and base_name.lower() in needed_ings:
                        start_candidates.append(pos)

                if start_candidates:
                    start_pos = self._nearest_by_path(env, default_start_pos, start_candidates) or get_nearest(default_start_pos, start_candidates)
                else:
                    counters = env.get_pos_by_obj_gs(gs="Counter")
                    start_pos = get_nearest(pot, counters) if counters else pot

                t['start_pos'] = start_pos
                t['end_pos'] = pot
                t['fixed_res'] = ('pot', pot)

            elif verb == 'serve_salad':
                # サラダ: 刻んだ食材のある置き場 → 皿タイル → 提供口。鍋は使わない。
                delivery = resources['delivery']
                plate = self._pick_plate(env, resources, delivery)
                needed_ings = dish_ingredients(obj)

                start_pos = t.get('assigned_counter')
                if start_pos is None:
                    start_candidates = []
                    for pos, world_obj in env.pos_obj.items():
                        if world_obj is None:
                            continue
                        base_name = chopped_base_name(world_obj)
                        if base_name is not None and base_name.lower() in needed_ings:
                            start_candidates.append(pos)
                    if start_candidates:
                        start_pos = self._nearest_by_path(env, default_start_pos, start_candidates) or get_nearest(default_start_pos, start_candidates)
                    else:
                        start_pos = plate

                t['start_pos'] = start_pos
                t['end_pos'] = delivery
                t['fixed_res'] = ('delivery', delivery)

            elif verb == 'serve':
                pots = resources['pots']
                pot = pots[order_idx % len(pots)] if pots else default_start_pos
                delivery = resources['delivery']
                plate = self._pick_plate(env, resources, delivery)

                t['start_pos'] = plate
                t['end_pos'] = delivery
                t['fixed_res'] = ('pot', pot)

            elif verb == 'mix':
                blenders = resources['blenders']
                blender = blenders[order_idx % len(blenders)] if blenders else default_start_pos
                t['start_pos'] = (t.get('assigned_counter')
                                  or self._find_shared_counter(env, blender) or blender)
                t['end_pos'] = blender
                t['fixed_res'] = ('blender', blender)

            elif verb == 'serve_juice':
                blenders = resources['blenders']
                blender = blenders[order_idx % len(blenders)] if blenders else default_start_pos
                t['start_pos'] = self._pick_cup(env, resources, blender)
                t['end_pos'] = resources['delivery']
                t['fixed_res'] = ('blender', blender)

            elif verb == 'handover':
                pots = resources['pots']
                pot = pots[order_idx % len(pots)] if pots else default_start_pos
                counter = t.get('assigned_counter') or self._find_shared_counter(env, pot)
                t['start_pos'] = self._pick_plate(env, resources, pot)
                t['end_pos'] = counter
                t['fixed_res'] = ('pot', pot)

            elif verb == 'serve_from_counter':
                delivery = resources['delivery']
                counter = t.get('assigned_counter') or self._find_shared_counter(env, delivery)
                t['start_pos'] = counter
                t['end_pos'] = delivery
                t['fixed_res'] = ('delivery', delivery)

    def _task_is_available_in_virtual_state(self, task, remaining_task_ids):
        """前提となる工程が済んでいて、いま着手できるタスクか。

        知らない動詞に False を返すと「永久に着手できない」扱いになるので、
        工程を増やしたらここにも足すこと。
        """
        verb, obj, order_uid = task['id']
        if verb == 'chop':
            return True
        if verb in ('cook', 'mix', 'serve_salad'):
            # 材料を全部刻み終えていること。mix(ミキサー)は cook と同じ位置づけ。
            needed_ings = dish_ingredients(obj)
            return all(('chop', ing, order_uid) not in remaining_task_ids for ing in needed_ings)
        if verb == 'serve':
            return ('cook', obj, order_uid) not in remaining_task_ids
        if verb == 'serve_juice':
            # 混ぜ終わってからでないとコップに注げない。
            return ('mix', obj, order_uid) not in remaining_task_ids
        if verb == 'handover':
            # 鍋を使う料理なら煮上がってから、使わないなら刻み終わってから渡せる。
            if ('cook', obj, order_uid) in remaining_task_ids:
                return False
            needed_ings = dish_ingredients(obj)
            return all(('chop', ing, order_uid) not in remaining_task_ids for ing in needed_ings)
        if verb == 'serve_from_counter':
            # 受け渡し台に置かれてからでないと取れない。
            return ('handover', obj, order_uid) not in remaining_task_ids
        return False

    def _estimate_virtual_task_finish(self, env, task, from_pos):
        start_pos = task.get('start_pos')
        if start_pos is None:
            return None, None

        approach = self.astar_distance(env, from_pos, start_pos)
        if approach is None:
            return None, None

        total_cost = int(approach + task['dur'])
        return int(approach), total_cost

    def _ingredients_of_task_id(self, task_id):
        if not (isinstance(task_id, tuple) and len(task_id) >= 2):
            return set()
        return {
            self._normalize_ingredient_name(part)
            for part in dish_ingredients(task_id[1])
            if self._normalize_ingredient_name(part)
        }

    def _check_human_prediction(self, env):
        """人間が推測と違うことに手を付けていたら、予測が外れたとみなす。

        予測が外れたまま人間スロットにタスクを固定し続けると、そのタスクは
        誰も実行しないまま計画に居座る。持ち物という明確な証拠が出た時点で
        推測を捨てて、次のパスで推測し直す(=再スケジューリングする)。

        逆に「人間が少し歩いた」程度では捨てない。予測自体は軽い(実測0.007ms)が、
        予測が変わるたびに CSP の再計算(実測23.6ms)が走り、しかも解が変わって
        AI の計画まで組み替わってしまうため。
        """
        predicted = getattr(self, '_predicted_human_task_id', None)
        if predicted is None:
            return

        human_idx = 1 - self.own_agent_idx
        agents = getattr(env, 'agents', None) or getattr(env, 'sim_agents', []) or []
        if human_idx >= len(agents):
            return
        holding = getattr(agents[human_idx], 'holding', None)
        holding_name = getattr(holding, 'full_name', None) if holding is not None else None
        if not holding_name:
            # 手ぶらは「別のことをしている」証拠にはならない
            return

        held = {
            self._normalize_ingredient_name(part)
            for part in re.split(r'[-_/]+', str(holding_name))
        }
        held.discard('')
        if not held:
            return

        if not (held & self._ingredients_of_task_id(predicted)):
            self._emit_counter_debug(
                f"[HumanModel] 予測が外れた: 予測={predicted} だが人間は {holding_name} を持っている")
            self._predicted_human_task_id = None
            self._human_prediction_doubt = 0
            self._mark_reschedule_needed('human_prediction_missed')

    def _check_human_prediction_by_cost(self, env, tasks):
        """所要時間の見積もりからも、予測が外れていないかを見る。

        持ち物は明確な証拠だが、手ぶらで別の場所へ向かっている人間は捉えられない。
        そこで人間の現在地から各タスクの所要時間を見積もり(既存の
        _estimate_virtual_task_finish)、予測タスクより明らかに早く終わるタスクが
        現れたら予測ミスとみなす。

        ただし人間はうろうろするので、1フレームでも早くなったら乗り換える、とすると
        そのつど CSP の再計算(実測23.6ms)が走り解が変わってしまう。
        「はっきり差がある(MARGIN)」かつ「続けて何度もそうだった(CONFIRM)」ときだけ
        予測ミスと判定する。
        """
        predicted = getattr(self, '_predicted_human_task_id', None)
        if predicted is None or not tasks:
            return

        predicted_task = next((t for t in tasks if t['id'] == predicted), None)
        if predicted_task is None:
            self._predicted_human_task_id = None
            self._human_prediction_doubt = 0
            return

        human_idx = 1 - self.own_agent_idx
        agents = getattr(env, 'agents', None) or getattr(env, 'sim_agents', []) or []
        if human_idx >= len(agents):
            return
        human_pos = getattr(agents[human_idx], 'location', None)
        if human_pos is None:
            return

        _, predicted_cost = self._estimate_virtual_task_finish(env, predicted_task, human_pos)

        best_id, best_cost = None, None
        for task in tasks:
            if task['id'] == predicted:
                continue
            _, cost = self._estimate_virtual_task_finish(env, task, human_pos)
            if cost is None:
                continue
            if best_cost is None or cost < best_cost:
                best_id, best_cost = task['id'], cost

        clearly_better = (
            best_cost is not None
            and (predicted_cost is None or best_cost + HUMAN_PREDICTION_COST_MARGIN < predicted_cost)
        )
        if not clearly_better:
            self._human_prediction_doubt = 0
            return

        self._human_prediction_doubt = getattr(self, '_human_prediction_doubt', 0) + 1
        if self._human_prediction_doubt >= HUMAN_PREDICTION_CONFIRM_FRAMES:
            self._emit_counter_debug(
                f"[HumanModel] 予測が外れた(所要時間): 予測={predicted}({predicted_cost}) "
                f"より {best_id}({best_cost}) の方が早い")
            self._predicted_human_task_id = None
            self._human_prediction_doubt = 0
            self._mark_reschedule_needed('human_prediction_missed_by_cost')

    def _predict_human_current_task(self, env, tasks, human_pos):
        """残りタスクの中から「人間がいま手をつけているタスク」を推測する。

        持ち物は行動を最もよく表す手がかりなので先に見る。
          * 未カットの食材を持っている -> その食材を切る作業の途中
          * 刻んだ食材を持っている     -> その食材を置き場へ運んでいる途中
        持ち物から分からなければ、既存の貪欲予測(人間は自分の位置から
        一番早く終わるタスクを取る)を1件だけ使う。
        """
        if not tasks:
            return None

        human_idx = 1 - self.own_agent_idx
        agents = getattr(env, 'agents', None) or getattr(env, 'sim_agents', []) or []
        holding_name = None
        if human_idx < len(agents):
            holding = getattr(agents[human_idx], 'holding', None)
            holding_name = getattr(holding, 'full_name', None) if holding is not None else None

        if holding_name:
            held = set()
            for part in re.split(r'[-_/]+', str(holding_name)):
                normalized = self._normalize_ingredient_name(part)
                if normalized:
                    held.add(normalized)
            # 持っている食材に対応する chop タスクがあれば、それに手をつけているとみなす
            for task in tasks:
                if task.get('verb') != 'chop':
                    continue
                if self._normalize_ingredient_name(str(task.get('obj', ''))) in held:
                    self._predicted_human_task_id = task['id']
                    self._emit_counter_debug(
                        f"[HumanModel] 持ち物から推測: {task['id']} (holding={holding_name})")
                    return task

        # 位置ベースの推測は人間が歩き回るたびに結果が変わる。毎回変えると、
        # そのつど「人間スロットに固定するタスク」が入れ替わって CSP の解が変わり、
        # AI の計画まで組み替わってしまう(まな板に食材を置いた直後に別タスクへ
        # 飛ばされ、置き場で置く/拾うを繰り返す等)。
        # 一度推測したタスクは、それが残タスクから消えるまで維持する。
        previous_id = getattr(self, '_predicted_human_task_id', None)
        if previous_id is not None:
            for task in tasks:
                if task['id'] == previous_id:
                    return task

        predicted = self._predict_human_greedy_tasks(
            env, tasks, human_start_pos=human_pos, limit=1
        )
        if predicted:
            task = predicted[0]['task']
            self._predicted_human_task_id = task['id']
            self._emit_counter_debug(
                f"[HumanModel] 位置から推測: {task['id']} (human_pos={human_pos})")
            return task
        self._predicted_human_task_id = None
        return None

    def _predict_human_greedy_tasks(self, env, tasks, human_start_pos, limit=None):
        tasks_by_id = {task['id']: task for task in tasks}
        state = VirtualHumanState(
            current_time=0,
            current_pos=human_start_pos,
            remaining_task_ids={task['id'] for task in tasks},
        )
        predicted = []
        verb_priority = self.VERB_PRIORITY

        while state.remaining_task_ids:
            candidates = []
            for tid in list(state.remaining_task_ids):
                task = tasks_by_id[tid]
                if not self._task_is_available_in_virtual_state(task, state.remaining_task_ids):
                    continue
                approach, total_cost = self._estimate_virtual_task_finish(env, task, state.current_pos)
                if total_cost is None:
                    continue
                candidates.append((
                    total_cost,
                    task.get('display_order', task.get('slot_idx', task['order'])),
                    verb_priority.get(task['verb'], 9),
                    task['obj'],
                    approach,
                    task,
                ))

            if not candidates:
                break

            candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            total_cost, _, _, _, approach, task = candidates[0]
            planned_start = state.current_time + approach
            planned_end = state.current_time + total_cost
            predicted.append({
                'id': task['id'],
                'start': planned_start,
                'end': planned_end,
                'task': task,
            })

            state.current_time = planned_end
            state.current_pos = task.get('end_pos', state.current_pos)
            state.remaining_task_ids.remove(task['id'])

            if limit is not None and len(predicted) >= limit:
                break

        return predicted

    def _task_duration_frames(self, env, verb, obj, order_idx, assigned_counter=None):
        resources = self._get_resources(env)

        def raw_base_name(item):
            if item is None:
                return None
            if getattr(item, 'is_held', False):
                return None
            name = getattr(item, 'name', '')
            if name.startswith('Fresh'):
                return name.replace('Fresh', '')
            full_name = getattr(item, 'full_name', '')
            if full_name.startswith('Fresh'):
                return full_name.replace('Fresh', '')
            return None

        def chopped_base_name(item):
            if item is None:
                return None
            if getattr(item, 'is_held', False):
                return None
            if hasattr(item, 'is_chopped') and item.is_chopped():
                contents = getattr(item, 'contents', [])
                if contents:
                    return getattr(contents[0], 'name', None)
            name = getattr(item, 'name', '')
            if name.startswith('Chopped'):
                return name.replace('Chopped', '')
            return None
        
        def get_nearest(start_pos, candidates):
            if not candidates: return None
            if not start_pos: return candidates[0]
            return min(candidates, key=lambda p: abs(p[0]-start_pos[0]) + abs(p[1]-start_pos[1]))

        if verb == 'chop':
            tile_map = INGREDIENT_TILE
            ing_pos_list = env.get_pos_by_obj_gs(gs=tile_map.get(obj, ""))
            raw_candidates = []
            for pos, world_obj in env.pos_obj.items():
                if world_obj is None:
                    continue
                base_name = raw_base_name(world_obj)
                if base_name is not None and base_name.lower() == obj:
                    raw_candidates.append(pos)

            if raw_candidates:
                ing_pos = get_nearest(resources['cutboards'][0] if resources['cutboards'] else None, raw_candidates)
                pickup_cost = 1
            else:
                if not ing_pos_list:
                    return None
                ing_pos = ing_pos_list[0]
                pickup_cost = 1

            cutboard_pos_list = resources['cutboards']
            if not cutboard_pos_list: return None
            
            cutboard_pos = get_nearest(ing_pos, cutboard_pos_list)
            
            if assigned_counter:
                target = assigned_counter
            else:
                counters = env.get_pos_by_obj_gs(gs="Counter")
                if not counters: return None
                target = get_nearest(cutboard_pos, counters)
#
            def adj(pos_list):
                width = env.world_width; height = env.world_height; grid = env.to_grid
                out=[]
                for x,y in pos_list:
                    for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nx,ny=x+dx,y+dy
                        if 0<=nx<width and 0<=ny<height and grid[nx][ny]==1:
                            out.append((nx,ny))
                return list(set(out))
            
            ing_adj=adj([ing_pos]); cut_adj=adj([cutboard_pos]); tgt_adj=adj([target])
            
            min_total=None
            for s in ing_adj:
                for m in cut_adj:
                    d1 = self.astar_distance(env, s, m)
                    if d1 is None: continue
                    
                    for e in tgt_adj:
                        d2 = self.astar_distance(env, m, e)
                        if d2 is None: continue
                        
                        tot = d1 + d2
                        if min_total is None or tot < min_total:
                            min_total = tot
            
            if min_total is None:
                return None
            return int(min_total + 8 + pickup_cost + 1 + 1)

        elif verb == 'cook':
            pot_pos_list = resources['pots']
            if not pot_pos_list: return None
            
            pot_pos = pot_pos_list[order_idx % len(pot_pos_list)]

            needed_ings = dish_ingredients(obj)
            # 材料は指定テーブルに集めるので、そこが実際の出発点。指定が無い
            # ときだけ、置かれている材料や適当なカウンターから見積もる。
            start_pos = assigned_counter
            if start_pos is None:
                start_candidates = []

                for pos, world_obj in env.pos_obj.items():
                    if world_obj is None:
                        continue
                    base_name = chopped_base_name(world_obj)
                    if base_name is not None and base_name.lower() in needed_ings:
                        start_candidates.append(pos)

                if start_candidates:
                    start_pos = get_nearest(pot_pos, start_candidates)
                else:
                    counters = env.get_pos_by_obj_gs(gs="Counter")
                    if not counters: return None
                    start_pos = get_nearest(pot_pos, counters)

            d = self.astar_distance(env, start_pos, pot_pos)
            if d is None: return None
            return int(d + 2)

        elif verb == 'serve_salad':
            # サラダ: 置き場の刻んだ食材を取る → 皿タイルへ寄って皿に乗せる → 提供口。
            delivery_pos = resources['delivery']
            plate_pos = self._pick_plate(env, resources, delivery_pos)

            needed_ings = dish_ingredients(obj)
            start_pos = assigned_counter
            if start_pos is None:
                start_candidates = []
                for pos, world_obj in env.pos_obj.items():
                    if world_obj is None:
                        continue
                    base_name = chopped_base_name(world_obj)
                    if base_name is not None and base_name.lower() in needed_ings:
                        start_candidates.append(pos)
                if start_candidates:
                    start_pos = get_nearest(plate_pos, start_candidates)
                else:
                    counters = env.get_pos_by_obj_gs(gs="Counter")
                    if not counters: return None
                    start_pos = get_nearest(plate_pos, counters)

            d1 = self.astar_distance(env, start_pos, plate_pos)
            d2 = self.astar_distance(env, plate_pos, delivery_pos)

            if d1 is None or d2 is None: return None
            # 食材の取得 + 皿に乗せる + 提供 の3インタラクト分
            return int(d1 + d2 + 3)

        elif verb == 'serve':
            pot_pos_list = resources['pots']
            delivery_pos = resources['delivery']
            plate_pos = self._pick_plate(env, resources, delivery_pos)

            if not pot_pos_list: return None
            pot_pos = pot_pos_list[order_idx % len(pot_pos_list)]

            d1 = self.astar_distance(env, plate_pos, pot_pos)
            d2 = self.astar_distance(env, pot_pos, delivery_pos)

            if d1 is None or d2 is None: return None
            return int(d1 + d2 + 3)

        elif verb == 'mix':
            # 置き場の刻んだフルーツを取ってミキサーへ入れ、規定回数まわす。
            # 鍋(cook)と同じ位置づけだが、待ち時間ではなくインタラクト回数で進む。
            blenders = resources['blenders']
            if not blenders: return None
            blender = blenders[order_idx % len(blenders)]
            start_pos = assigned_counter or self._find_shared_counter(env, blender) or blender
            d = self.astar_distance(env, start_pos, blender)
            if d is None: return None
            # 材料を取る + 入れる + 混ぜる回数
            return int(d + 2 + BLENDING_NUM_STEPS)

        elif verb == 'serve_juice':
            # ミキサーの中身をコップに注いで提供口へ。serve(鍋->皿)と同じ形。
            blenders = resources['blenders']
            delivery_pos = resources['delivery']
            if not blenders: return None
            blender = blenders[order_idx % len(blenders)]
            cup_pos = self._pick_cup(env, resources, blender)
            d1 = self.astar_distance(env, cup_pos, blender)
            d2 = self.astar_distance(env, blender, delivery_pos)
            if d1 is None or d2 is None: return None
            # コップを取る + 注ぐ + 提供する の3インタラクト
            return int(d1 + d2 + 3)

        elif verb == 'handover':
            # 仕切りの向こうへ渡すための工程。皿を取り、鍋から盛り、受け渡し台に置く。
            pot_pos_list = resources['pots']
            if not pot_pos_list: return None
            pot_pos = pot_pos_list[order_idx % len(pot_pos_list)]
            plate_pos = self._pick_plate(env, resources, pot_pos)
            counter = assigned_counter or self._find_shared_counter(env, pot_pos)
            if counter is None: return None

            d1 = self.astar_distance(env, plate_pos, pot_pos)
            d2 = self.astar_distance(env, pot_pos, counter)
            if d1 is None or d2 is None: return None
            # 皿を取る + 鍋から盛る + 台に置く の3インタラクト
            return int(d1 + d2 + 3)

        elif verb == 'serve_from_counter':
            # 受け渡し台に置かれた完成品を取って提供口へ運ぶだけ。
            delivery_pos = resources['delivery']
            counter = assigned_counter or self._find_shared_counter(env, delivery_pos)
            if counter is None: return None
            d = self.astar_distance(env, counter, delivery_pos)
            if d is None: return None
            # 台から取る + 提供する の2インタラクト
            return int(d + 2)
        else:
            return None
    def get_assigned_counters(self):
        return getattr(self, 'assigned_counters_display_map', {})

    def get_order_display_labels(self):
        return getattr(self, 'order_display_labels', [])

    def _get_counter_policy_entry(self, order_uid):
        return self.counter_policy_by_order.setdefault(order_uid, {
            'counter': None,
            'armed': False,
            'last_state': None,
        })

    def _get_assigned_counter(self, order_uid):
        return self._get_counter_policy_entry(order_uid)['counter']

    def _set_assigned_counter(self, order_uid, counter_pos):
        entry = self._get_counter_policy_entry(order_uid)
        entry['counter'] = counter_pos
        if counter_pos is None:
            entry['invalid_since'] = None
            entry['last_invalid_reason'] = None

    def _should_release_invalid_counter(self, order_uid, assigned_counter, reason, env):
        """暫定的な不整合は即座に捨てず、数フレーム継続したときだけ解除する。"""
        if assigned_counter is None:
            return False

        entry = self._get_counter_policy_entry(order_uid)
        now = getattr(env, 'time', 0)
        invalid_since = entry.get('invalid_since')
        if invalid_since is None:
            entry['invalid_since'] = now
            entry['last_invalid_reason'] = reason
            return False

        if now - invalid_since < 2:
            return False

        entry['invalid_since'] = None
        entry['last_invalid_reason'] = None
        return True

    def _normalize_ingredient_name(self, ingredient_name):
        if ingredient_name is None:
            return ''

        text = str(ingredient_name).strip().lower()
        if not text:
            return ''

        text = text.replace('-', '').replace('_', '').replace(' ', '')
        for prefix in ('fresh', 'chopped', 'cooked', 'cooking', 'raw', 'cut'):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break

        return text

    def _get_counter_food_names(self, env, counter_pos):
        counter_obj = env.pos_obj.get(counter_pos)
        if counter_obj is None or not hasattr(counter_obj, 'contents'):
            return set()

        names = set()

        def add_name(raw_name):
            normalized = self._normalize_ingredient_name(raw_name)
            if normalized:
                names.add(normalized)

        def walk(value):
            if value is None:
                return
            if isinstance(value, str):
                for sub in re.split(r'[-_/]+', value):
                    add_name(sub)
                return
            full_name = getattr(value, 'full_name', getattr(value, 'name', ''))
            if full_name not in ('', 'Plate', 'plate'):
                for sub in re.split(r'[-_/]+', str(full_name)):
                    add_name(sub)
            for child in getattr(value, 'contents', []):
                walk(child)

        for food in counter_obj.contents:
            walk(food)

        return names

    def _get_counter_blocking_food_names(self, env, counter_pos):
        """マージ先として使えなくする食材(未カット等)を counter から拾う。

        ゲームエンジンの mergeable() は ChoppedX 同士でなければマージを許さない。
        一方 _get_counter_food_names は Fresh/Chopped の区別を落とす(どちらも
        'onion' になる)ため、人間が「切っていない食材」を置き場に置くと、CSP は
        「必要な食材が既に置いてある」と誤認したまま、AI はマージできない置く操作を
        永久に繰り返して停止してしまう。
        ここでは実際にマージを妨げる(= Chopped でない)食材名だけを返す。
        """
        counter_obj = env.pos_obj.get(counter_pos)
        if counter_obj is None or not hasattr(counter_obj, 'contents'):
            return set()

        blocking = set()

        def walk(value):
            if value is None:
                return
            full_name = str(getattr(value, 'full_name', getattr(value, 'name', '')) or '')
            for part in re.split(r'[-_/]+', full_name):
                if not part or part in ('Plate', 'FireExtinguisher'):
                    continue
                if part.startswith('Chopped'):
                    continue
                normalized = self._normalize_ingredient_name(part)
                if normalized:
                    blocking.add(normalized)
            for child in getattr(value, 'contents', []):
                walk(child)

        for food in counter_obj.contents:
            walk(food)

        return blocking

    def _evaluate_order_counter_state(self, env, order_uid, order_ingredient_names, assigned_counter):
        """counter の状態を設計上の意味で分類する。

        - valid: 期待材料が counter にあり、余計な材料が混ざっていない
        - incomplete: 期待材料が一部しかなく、未完了状態だが再割り当て対象ではない
        - unexpected: 予期しない材料が混ざっているので破棄対象
        - owner_conflict: 別注文と同じ counter を共有しているので破棄対象

        重要な設計原則:
        - missing は invalid ではない
        - 余計な材料や owner conflict だけが unsafe である
        - valid な割り当ては、別の counter に誤った材料があっても再割り当てしない
        """
        result = {
            'status': 'unassigned',
            'unexpected': set(),
            'missing': set(),
            'owner_conflict': False,
            'counter_food_names': set(),
        }

        if assigned_counter is None:
            return result
        if env.pos_obj.get(assigned_counter) is None:
            return result

        counter_food_names = self._get_counter_food_names(env, assigned_counter)
        if not counter_food_names:
            return result

        result['counter_food_names'] = counter_food_names
        expected = {
            self._normalize_ingredient_name(name)
            for name in order_ingredient_names
            if self._normalize_ingredient_name(name)
        }
        if not expected:
            return result

        result['unexpected'] = counter_food_names - expected
        result['missing'] = expected - counter_food_names

        for other_uid, entry in self.counter_policy_by_order.items():
            if other_uid == order_uid:
                continue
            if entry.get('counter') == assigned_counter:
                result['owner_conflict'] = True
                break

        has_required_material = bool(counter_food_names & expected)
        if result['owner_conflict']:
            result['status'] = 'owner_conflict'
        elif result['unexpected'] and not has_required_material:
            # 既に要求食材が counter 上に存在しているなら、異なる食材の余計な置き場は
            # 「再割り当て対象」ではなく「未完了の valid partial state」として保持する。
            result['status'] = 'unexpected'
        elif result['missing'] or has_required_material:
            result['status'] = 'incomplete'
        else:
            result['status'] = 'valid'

        return result

    def _classify_counter_conflict(self, env, order_ingredient_names, assigned_counter, current_orders=None, order_uid=None):
        """counter の再割り当てが必要かを判定する。未完了は保持し、unsafe だけ破棄する。"""
        if assigned_counter is None:
            return None
        if env.pos_obj.get(assigned_counter) is None:
            return None

        state = self._evaluate_order_counter_state(env, order_uid, order_ingredient_names, assigned_counter)
        if state['owner_conflict']:
            return 'owner_conflict'

        # 未カット食材が置き場に乗っていると、そこへは何もマージできない。
        # 期待食材と同じ種類に見えても使えないので、置き場ごと解除して
        # 別のカウンターへ retarget させる(人間が切らずに置いた場合の救済)。
        if self._get_counter_blocking_food_names(env, assigned_counter):
            return 'counter_blocked_by_unchopped_food'

        expected = {
            self._normalize_ingredient_name(name)
            for name in order_ingredient_names
            if self._normalize_ingredient_name(name)
        }
        has_required_material = bool(state['counter_food_names'] & expected)
        if state['unexpected'] and not has_required_material:
            return 'counter_occupied_by_other_order'
        return None

    def _is_counter_conflict_for_order(self, env, order_ingredient_names, assigned_counter, current_orders=None):
        return self._classify_counter_conflict(env, order_ingredient_names, assigned_counter, current_orders=current_orders) is not None

    def _detect_counter_conflicts(self, env, current_orders):
        """unsafe な counter 状態だけを再割り当て対象にする。未完了状態は維持する。"""
        conflicts = []
        for order in current_orders:
            order_uid = order.get('order')
            if order_uid is None:
                continue
            assigned_counter = self._get_assigned_counter(order_uid)
            if assigned_counter is None:
                continue
            if env.pos_obj.get(assigned_counter) is None:
                continue

            order_ingredients = order.get('ingredients', [])
            if not order_ingredients:
                continue

            state = self._evaluate_order_counter_state(env, order_uid, order_ingredients, assigned_counter)
            expected = {
                self._normalize_ingredient_name(ing)
                for ing in order_ingredients
                if self._normalize_ingredient_name(ing)
            }
            has_required_material = bool(state['counter_food_names'] & expected)
            if (state['owner_conflict'] or (state['unexpected'] and not has_required_material)):
                conflicts.append({
                    'order_uid': order_uid,
                    'counter': assigned_counter,
                    'actual': sorted(state['counter_food_names']),
                    'expected': sorted(expected),
                    'reason': 'owner_conflict' if state['owner_conflict'] else 'counter_occupied_by_other_order',
                })
        return conflicts

    def _iter_dependency_world_objects(self, env):
        """依存判定に使う実世界オブジェクトを、pos_obj/world.objects/agent.holding/grid-square.holding まで辿って列挙する。"""
        seen = set()

        def add_value(value):
            if value is None:
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    yield from add_value(item)
                return

            obj_id = id(value)
            if obj_id in seen:
                return
            seen.add(obj_id)
            yield value

            for child in getattr(value, 'contents', []):
                yield from add_value(child)

            holding = getattr(value, 'holding', None)
            if holding is not None:
                if isinstance(holding, (list, tuple, set)):
                    for item in holding:
                        yield from add_value(item)
                else:
                    yield from add_value(holding)

        for obj in list(getattr(env, 'pos_obj', {}).values()):
            yield from add_value(obj)

        for obj in list(getattr(env, 'pos_gs', {}).values()):
            yield from add_value(obj)

        world = getattr(env, 'world', None)
        if world is not None:
            for obj_list in getattr(world, 'objects', {}).values():
                for obj in obj_list:
                    yield from add_value(obj)

        for agent in getattr(env, 'agents', []):
            if agent is None:
                continue
            yield from add_value(getattr(agent, 'holding', None))

        env_hold = getattr(env, 'hold', None)
        if env_hold is not None:
            yield from add_value(env_hold)

    def _ingredient_is_already_available(self, env, ingredient_name):
        """要求食材が実際にワールド上にあるなら、完了タスク登録だけで待たせない。"""
        if ingredient_name is None:
            return False

        raw = str(ingredient_name).strip().lower()
        if not raw:
            return False

        normalized = self._normalize_ingredient_name(raw)
        if not normalized:
            return False

        targets = {f"Chopped{normalized.capitalize()}", f"Cooking{normalized.capitalize()}", f"Cooked{normalized.capitalize()}"}

        def matches_name(name):
            if not name:
                return False
            n = str(name).strip()
            return any(target in n for target in targets)

        for obj in self._iter_dependency_world_objects(env):
            if matches_name(getattr(obj, 'full_name', '')):
                return True
            for child in getattr(obj, 'contents', []):
                if matches_name(getattr(child, 'full_name', '')):
                    return True

        return False

    def _owns_world_ingredient(self, env, ingredient_name, require_ready_to_cook=False):
        """世界の中にその食材があるかを実体ベースで判定する。古いタスク ID には依存しない。

        require_ready_to_cook=True のときは Fresh(未カット)を数えない。
        cook タスクの前提判定に Fresh を含めてしまうと、人間が未カット食材を
        カウンターに置いただけで「材料は揃っている」と誤判定して cook を開始し、
        一方 TaskAgent の process_cook_task は Chopped* しか消費できないため、
        その場から動かず永久に待ち続けてしまう。
        """
        if not ingredient_name:
            return False
        name = str(ingredient_name).strip().lower()
        if not name:
            return False
        normalized = self._normalize_ingredient_name(name)
        if not normalized:
            return False

        candidate_tokens = {
            f"Chopped{normalized.capitalize()}",
            f"Cooking{normalized.capitalize()}",
            f"Cooked{normalized.capitalize()}",
        }
        if not require_ready_to_cook:
            candidate_tokens.add(f"Fresh{normalized.capitalize()}")

        def has_token(obj_or_name):
            if obj_or_name is None:
                return False
            name_text = str(obj_or_name)
            return any(token in name_text for token in candidate_tokens)

        for obj in self._iter_dependency_world_objects(env):
            if has_token(getattr(obj, 'full_name', '')):
                return True
            for child in getattr(obj, 'contents', []):
                if has_token(getattr(child, 'full_name', '')):
                    return True

        return False

    def _cook_dependency_ready_from_world(self, env, dish_name):
        """cook の前提となる具材が実世界に存在するなら true。任意の古い完了タスクより優先する。"""
        if not dish_name:
            return False
        parts = dish_ingredients(dish_name)
        if not parts:
            return True
        # cook が実際に消費できるのは Chopped 以降だけなので、Fresh は数えない。
        return all(
            self._owns_world_ingredient(env, p, require_ready_to_cook=True)
            for p in parts
        )

    def _salad_dependency_ready_from_world(self, env, dish_name):
        """serve_salad の前提となる「刻んだ食材」が実世界に存在するなら true。

        サラダは鍋を使わないだけで、必要な食材が Chopped 以降であることは
        cook と同じ。Fresh のまま置かれた食材で開始してしまうと、
        TaskAgent が消費できず永久に待つのも同様なので判定を共有する。
        """
        return self._cook_dependency_ready_from_world(env, dish_name)

    def _serve_dependency_ready_from_world(self, env, dish_name):
        """serve 直前の cook 依存は、実際の調理済み品があるかで判定する。"""
        if not dish_name:
            return False
        normalized = strip_dish_suffix(dish_name).strip()
        if not normalized:
            return False
        return self._owns_world_ingredient(env, normalized)

    def _log_counter_policy(self, order_uid, action, counter_pos, details=""):
        entry = self._get_counter_policy_entry(order_uid)
        state = (action, counter_pos, details)
        if entry['last_state'] == state:
            return
        entry['last_state'] = state

        suffix = f" {details}" if details else ""
        self._emit_counter_debug(f"[CounterPolicy] order_uid={order_uid} action={action} counter={counter_pos}{suffix}")

    def _debug_order_counter_state(self, env, current_orders, order_idx=None, order_uid=None, assigned_counter=None, reason=None):
        """order と counter の対応を追いやすくするためのデバッグ出力。"""
        if not hasattr(env, 'time'):
            return

        target_order_uid = order_uid
        if target_order_uid is None and current_orders and order_idx is not None and order_idx < len(current_orders):
            entry = current_orders[order_idx]
            if isinstance(entry, dict):
                target_order_uid = entry.get('order')
            elif isinstance(entry, (tuple, list)) and len(entry) >= 2:
                target_order_uid = entry[1] if isinstance(entry[1], (int, str)) else None

        if target_order_uid is None:
            return

        counter_state = {}
        for order in current_orders:
            if isinstance(order, dict):
                uid = order.get('order')
                if uid is None:
                    continue
                counter_state[uid] = self._get_assigned_counter(uid)
            elif isinstance(order, (tuple, list)) and len(order) >= 2:
                # raw order tuple は order_uid を別途保持しているので、この出力では安全にスキップする
                continue

        current_ingredients = None
        if order_idx is not None and order_idx < len(current_orders):
            entry = current_orders[order_idx]
            if isinstance(entry, dict):
                current_ingredients = entry.get('ingredients')
            elif isinstance(entry, (tuple, list)) and len(entry) >= 1:
                current_ingredients = None

        self._emit_counter_debug(f"[CounterDebug] time={env.time} order_uid={target_order_uid} assigned_counter={assigned_counter} reason={reason} "
              f"all_assignments={counter_state} current_order_ingredients={current_ingredients}")

    def _agent_holds_complete_set(self, env, ings_cap):
        """この注文の材料が「全部そろった状態」で誰かの手に持たれているか。

        置き場(マージ地点)は食材を1か所に集めるための作業場所なので、
        集め終わって手に取られた時点で用済みになる。
        """
        expected = {
            self._normalize_ingredient_name(ing)
            for ing in ings_cap
            if self._normalize_ingredient_name(ing)
        }
        if not expected:
            return False

        for agent in getattr(env, 'agents', []) or []:
            holding = getattr(agent, 'holding', None)
            if holding is None:
                continue
            held = set()
            usable = True
            for part in re.split(r'[-_/]+', str(getattr(holding, 'full_name', '') or '')):
                if not part or part in ('Plate', 'FireExtinguisher'):
                    continue
                if part.startswith('Fresh'):
                    # 未カットのものは鍋に入れられないので「集め終わった」とは言えない
                    usable = False
                    break
                normalized = self._normalize_ingredient_name(part)
                if normalized:
                    held.add(normalized)
            if usable and held == expected:
                return True
        return False

    def _resolve_assigned_counter(self, env, order_uid):
        assigned_counter = self._get_assigned_counter(order_uid)
        if assigned_counter is None:
            return None, False

        counter_food_names = self._get_counter_food_names(env, assigned_counter)
        details = f"foods={sorted(counter_food_names)}" if counter_food_names else "reason=fixed"
        self._log_counter_policy(order_uid, "fixed", assigned_counter, details)
        return assigned_counter, False

    def _find_unclaimed_matching_counter(self, env, order_uid, ings_lower, reserved_counters, used_counters, order_idx):
        """人間が(想定外の)別カウンターに置いてしまった、この注文向けの食材を探す。

        他注文の assigned_counter でも、今回のスケジューリングパスで既に
        他の注文に確保済みでもないカウンターのうち、この注文が求める食材だけが
        (部分一致でも)置かれている場所があれば、その位置を返す。
        見つからなければ None。呼び出し側はこれを新しい assigned_counter として
        採用することで、人間がどこに置いても次のパスで正しい置き場に追従できる。
        """
        expected = {
            self._normalize_ingredient_name(name)
            for name in ings_lower
            if self._normalize_ingredient_name(name)
        }
        if not expected:
            return None

        resources = self._get_resources(env)
        counters = resources.get('counters', [])

        candidates = []
        for pos in counters:
            if pos in reserved_counters or pos in used_counters:
                continue
            foods = self._get_counter_food_names(env, pos)
            if not foods:
                continue
            if not foods <= expected:
                # 余計な食材が混ざっている場所はこの注文向けとは断定できない
                continue
            if self._get_counter_blocking_food_names(env, pos):
                # 未カット食材が乗っている場所へは何もマージできない
                continue
            candidates.append(pos)

        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        # 複数候補があれば、鍋までの距離が最短のものを優先する
        pots = resources.get('pots', [])
        pot = pots[order_idx % len(pots)] if pots else None
        if pot is None:
            return candidates[0]

        best_pos, best_dist = None, float('inf')
        for pos in candidates:
            dist = self.astar_distance(env, pos, pot)
            if dist is not None and dist < best_dist:
                best_dist = dist
                best_pos = pos
        return best_pos if best_pos is not None else candidates[0]

    def _calculate_dynamic_merge_point(self, env, ings_lower, order_idx, pot_locs, used_counters, reserved_counters=None):
        reserved = set(reserved_counters or [])
        ing = ings_lower[0] if ings_lower else None
        tile_map = INGREDIENT_TILE
        ing_pos_list = env.get_pos_by_obj_gs(gs=tile_map.get(ing, ""))
        ing_pos = ing_pos_list[0] if ing_pos_list else (0, 0)

        resources = self._get_resources(env)
        cutboards = resources.get('cutboards', [])

        def get_nearest(start_pos, candidates):
            if not candidates: return start_pos
            return min(candidates, key=lambda p: abs(p[0]-start_pos[0]) + abs(p[1]-start_pos[1]))

        best_cb = get_nearest(ing_pos, cutboards)

        pots = resources.get('pots', [])
        pot = pots[order_idx % len(pots)] if pots else (0, 0)

        counters = resources.get('counters', [])

        # 置き場は2人で材料を持ち寄る場所なので、隣接して立てる床マスが1つしか
        # ないテーブルを選ぶと、片方が居座っている間もう片方が近づけず詰む。
        # 2マス以上から使えるテーブルを優先する(候補が無ければ従来どおり)。
        def access_count(loc):
            n = 0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = loc[0] + dx, loc[1] + dy
                if (0 <= nx < env.world_width and 0 <= ny < env.world_height
                        and env.to_grid[nx][ny] == 1):
                    n += 1
            return n

        shared_access = [c for c in counters if access_count(c) >= 2]
        if shared_access:
            counters = shared_access

        if self._map_is_partitioned(env):
            # 仕切りのあるマップでは、置き場は「両側から使えるカウンター」に
            # 限る。片側にしか届かない台を集合場所にすると、相手が材料を
            # 取りに来られず、その注文は永久に完成しない。
            # 壁に専用の受け渡し口を作るのではなく、既存の置き場割り当ての
            # 候補を絞るだけで受け渡しが成立する。
            shared = [c for c in counters if len(self._components_touching(env, c)) > 1]
            if shared:
                counters = shared

        empty_counters = []
        for c in counters:
            if c in reserved:
                continue
            if env.pos_obj.get(c) is None and c not in used_counters:
                empty_counters.append(c)

        if not empty_counters:
            # 完全に空いているカウンターがない場合は使用済みでも候補にする
            for c in counters:
                if c in reserved:
                    continue
                if env.pos_obj.get(c) is None:
                    empty_counters.append(c)
            if not empty_counters:
                empty_counters = [c for c in counters if c not in reserved]

        best_counter = None
        min_total_dist = float('inf')

        for c in empty_counters:
            d1 = self.astar_distance(env, best_cb, c)
            d2 = self.astar_distance(env, c, pot)
            if d1 is not None and d2 is not None:
                total = d1 + d2
                if total < min_total_dist:
                    min_total_dist = total
                    best_counter = c

        if best_counter is None:
            fallback = [c for c in counters if c not in reserved]
            best_counter = fallback[order_idx % len(fallback)] if fallback else None

        return best_counter

    def _build_order_tasks(self, env):
        available_chopped = {}
        available_chopped_by_pos = {}
        pot_states = []
        # ミキサーの中身(ジュースの作りかけ/完成品)。鍋と同じ扱い。
        blender_states = []
        # 皿の上に食材が乗っている状態(サラダの完成品/作りかけ)。
        # サラダは鍋を使わないので、pot_states の代わりにこれで進捗を見る。
        plate_states = []
        held_object_ids = {
            id(getattr(agent, 'holding', None))
            for agent in getattr(env, 'agents', [])
            if getattr(agent, 'holding', None) is not None
        }

        # エージェントが今まさに手に持っている「刻んだ食材」の在庫。
        # 世界(カウンター上など)に置かれた食材は available_chopped_by_pos で
        # 数えているが、held_object_ids はそこから保持中アイテムを除外しているため、
        # 「鍋へ運んでいる最中の刻んだ食材」がどの注文からも見えなくなり、
        # 既に確保済みなのに再度 chop タスクが追加されてしまう不具合があった。
        # 複数の注文が同じ食材を必要としている場合、どの注文向けかを取り違えると
        # 別の注文がその分を横取りしてしまい、結局 chop タスクが復活して同じ問題が
        # 別の食材で再発するため、「このエージェントは今どの注文向けに動いているか」
        # (carry_task、または現在のスケジュール済みタスクの order) に紐づけて管理する。
        # 紐づけ先が分からない場合のみ、注文を問わない共有プールにフォールバックする。
        held_chopped_by_order = {}
        held_chopped_unassigned = {}

        def _infer_agent_order(agent_idx, ing_names_lower):
            carry_task = None
            sched = []
            idx = 0
            if self.sc_2agent:
                carry_by_agent = getattr(self, 'carry_task_by_agent', None)
                if isinstance(carry_by_agent, dict):
                    carry_task = carry_by_agent.get(agent_idx)
                sched = getattr(self, 'schedule_per_agent', {}).get(agent_idx, [])
                idx_by_agent = getattr(self, 'current_task_idx', {})
                idx = idx_by_agent.get(agent_idx, 0) if isinstance(idx_by_agent, dict) else 0
            elif agent_idx == 0:
                carry_task = getattr(self, 'carry_task_by_agent', None)
                sched = getattr(self, 'schedule', [])
                idx = getattr(self, 'current_task_idx', 0)
                if not isinstance(idx, int):
                    idx = 0

            # carry_task(明示的に「この食材はこの注文向け」と紐付けたもの)は無条件に信用する。
            if carry_task:
                task_id = carry_task.get('id')
                if isinstance(task_id, tuple) and len(task_id) >= 3:
                    return task_id[2]

            # carry_task が無い場合のみ、現在のスケジュール済みタスクを参考にするが、
            # 頻繁な再スケジューリングでスケジュールは実際に持っている食材と無関係な
            # 注文を指していることがある(例: 今はchop onionが割り当てられているが
            # 実際に持っているのはtomato)。対象食材が一致する場合に限って採用し、
            # 一致しなければ「不明」として共有プールへフォールバックさせる。
            scheduled_task = sched[idx] if idx < len(sched) else None
            if scheduled_task:
                task_id = scheduled_task.get('id')
                if isinstance(task_id, tuple) and len(task_id) >= 3:
                    verb, obj, order = task_id
                    if verb == 'chop' and obj.lower() in ing_names_lower:
                        return order
            return None

        for agent_idx, agent in enumerate(getattr(env, 'agents', [])):
            holding = getattr(agent, 'holding', None)
            if holding is None:
                continue

            ing_names = []
            if hasattr(holding, 'is_chopped') and holding.is_chopped():
                for food in getattr(holding, 'contents', []):
                    food_name = getattr(food, 'name', None)
                    if food_name:
                        ing_names.append(food_name)
            else:
                holding_name = getattr(holding, 'name', '')
                if holding_name.startswith('Chopped'):
                    ing_names.append(holding_name.replace('Chopped', ''))

            if not ing_names:
                continue

            order_for_agent = _infer_agent_order(agent_idx, {n.lower() for n in ing_names})
            self._emit_counter_debug(
                f"[HeldInventory] agent={agent_idx} ing_names={ing_names} order_for_agent={order_for_agent} "
                f"carry_task_by_agent={self.carry_task_by_agent} current_task_idx={getattr(self, 'current_task_idx', None)}"
            )
            for ing_name in ing_names:
                if order_for_agent is not None:
                    bucket = held_chopped_by_order.setdefault(order_for_agent, {})
                else:
                    bucket = held_chopped_unassigned
                bucket[ing_name] = bucket.get(ing_name, 0) + 1

        def register_chopped_item(item, location=None):
            if item is None:
                return

            def add_chopped(base_name):
                if not base_name:
                    return
                available_chopped[base_name] = available_chopped.get(base_name, 0) + 1
                if location is not None:
                    pos_stock = available_chopped_by_pos.setdefault(location, {})
                    pos_stock[base_name] = pos_stock.get(base_name, 0) + 1

            if hasattr(item, 'is_chopped') and item.is_chopped():
                for food in getattr(item, 'contents', []):
                    if getattr(food, 'name', None):
                        add_chopped(food.name)
                return

            name = getattr(item, 'name', '')
            if name.startswith('Chopped'):
                base_name = name.replace('Chopped', '')
                add_chopped(base_name)

        def plated_food_names(item):
            """皿の上に乗っている食材名(sorted)。皿でなければ、または空皿なら None。"""
            contents = getattr(item, 'contents', [])
            if not any(getattr(c, 'name', '') == 'Plate' for c in contents):
                return None
            foods = sorted(
                c.name for c in contents
                if getattr(c, 'name', '') and c.name != 'Plate'
            )
            return foods or None

        # この注文で「切らずに運ぶだけ」にできる食材 -> 運び元カウンター
        carry_sources = {}

        def consume_chopped(ingredient_name, assigned_counter, reserved_counters, order_uid=None):
            preferred_positions = []
            if assigned_counter is not None:
                preferred_positions.append(assigned_counter)
            else:
                preferred_positions.extend(
                    pos for pos in available_chopped_by_pos.keys()
                    if pos not in reserved_counters
                )

            # 自分の置き場に無い場合、どの注文の置き場でもない「自由な」カウンターに
            # 既に切られた物があるなら、切り直さずそこから運べばよい。
            # 他注文の置き場(reserved_counters)からは絶対に取らない(取り合いと
            # テーブル間の移動合戦を防ぐため)。
            if assigned_counter is not None:
                for pos in available_chopped_by_pos.keys():
                    if pos == assigned_counter or pos in reserved_counters:
                        continue
                    if available_chopped_by_pos.get(pos, {}).get(ingredient_name, 0) > 0:
                        preferred_positions.append(pos)

            for pos in preferred_positions:
                pos_stock = available_chopped_by_pos.get(pos, {})
                if pos_stock.get(ingredient_name, 0) <= 0:
                    continue
                pos_stock[ingredient_name] -= 1
                if pos_stock[ingredient_name] <= 0:
                    del pos_stock[ingredient_name]
                if not pos_stock:
                    available_chopped_by_pos.pop(pos, None)
                available_chopped[ingredient_name] -= 1
                if available_chopped[ingredient_name] <= 0:
                    del available_chopped[ingredient_name]
                if pos != assigned_counter:
                    # 自分の置き場ではない = 運んで合流させる必要がある
                    self._emit_counter_debug(
                        f"[ConsumeChopped] order={order_uid} ing={ingredient_name} source=carry pos={pos}")
                    carry_sources[ingredient_name] = pos
                    return False
                self._emit_counter_debug(f"[ConsumeChopped] order={order_uid} ing={ingredient_name} source=world pos={pos}")
                return True

            # 世界(カウンター等)に見つからなくても、この注文向けに動いているエージェントが
            # 既に手に持っているなら chop 要求は満たされているとみなす。座標に紐づかないため
            # assigned_counter の一致は問わない。まずこの注文向けと分かっている分だけを見て、
            # 他の注文の分を横取りしないようにする。持ち主の注文が特定できなかった分だけ
            # 共有プールから消費する。
            own_bucket = held_chopped_by_order.get(order_uid, {}) if order_uid is not None else {}
            if own_bucket.get(ingredient_name, 0) > 0:
                own_bucket[ingredient_name] -= 1
                if own_bucket[ingredient_name] <= 0:
                    del own_bucket[ingredient_name]
                self._emit_counter_debug(f"[ConsumeChopped] order={order_uid} ing={ingredient_name} source=held_own")
                return True

            if held_chopped_unassigned.get(ingredient_name, 0) > 0:
                held_chopped_unassigned[ingredient_name] -= 1
                if held_chopped_unassigned[ingredient_name] <= 0:
                    del held_chopped_unassigned[ingredient_name]
                self._emit_counter_debug(f"[ConsumeChopped] order={order_uid} ing={ingredient_name} source=held_unassigned")
                return True

            self._emit_counter_debug(
                f"[ConsumeChopped] order={order_uid} ing={ingredient_name} source=NONE "
                f"held_by_order={held_chopped_by_order} held_unassigned={held_chopped_unassigned}"
            )
            return False

        def choose_counter_for_order(ingredient_names, current_counter=None, reserved_counters=None):
            reserved = set(reserved_counters or [])
            best_counter = None
            best_score = None

            for pos, pos_stock in available_chopped_by_pos.items():
                if pos in reserved:
                    continue
                valid_count = 0
                has_unwanted = False
                for stock_name, stock_count in pos_stock.items():
                    if stock_count <= 0:
                        continue
                    if stock_name in ingredient_names:
                        valid_count += stock_count
                    else:
                        has_unwanted = True

                if valid_count <= 0:
                    continue

                score = (
                    0 if has_unwanted else 1,
                    valid_count,
                    1 if pos == current_counter else 0,
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_counter = pos

            return best_counter

        if hasattr(env, 'world_all'):
            all_objects = env.world_all
        elif hasattr(env, 'world'):
            all_objects = env.world.get_object_list()
        else:
            all_objects = []

        pot_locs = []
        for o in all_objects:
            if getattr(o, 'name', '') == 'Pot':
                pot_locs.append(o.location)

        cutboard_locs = env.get_pos_by_obj_gs(gs="Cutboard")
        blender_locs = env.get_pos_by_obj_gs(gs="Blender")

        for obj in all_objects:
            if id(obj) in held_object_ids:
                continue
            if type(obj).__name__ == 'Object':
                if obj.location in pot_locs:
                    c_names = sorted([c.name for c in obj.contents])
                    pot_states.append({'names': c_names, 'obj': obj, 'used': False})
                elif obj.location in blender_locs:
                    # ミキサーに入っている時点で「材料は集め終わった」。
                    # ただし混ぜるのは待つだけでは進まないので、混ぜ終わって
                    # いるかどうかは別に持っておく。両方を一緒くたにすると、
                    # 入れただけの材料が「まだ刻まれていない」ことになり、
                    # 誰も回せないまま止まる。
                    c_names = sorted([c.name for c in obj.contents])
                    blender_states.append({
                        'names': c_names, 'obj': obj, 'used': False,
                        'mixed': bool(getattr(obj, 'is_mixed', lambda: False)()),
                    })
                else:
                    if obj.location not in cutboard_locs:
                        plated_names = plated_food_names(obj)
                        if plated_names is not None:
                            plate_states.append({'names': plated_names, 'obj': obj, 'used': False})
                        register_chopped_item(obj, obj.location)

        resources = self._get_resources(env)
        orders = []
        current_orders = env.order.current_orders if hasattr(env, 'order') and hasattr(env.order, 'current_orders') else []
        order_uids = self._refresh_active_order_uids(current_orders)
        self.order_display_labels = [
            (order_uid + 1) if order_uid is not None else (order_idx + 1)
            for order_idx, order_uid in enumerate(order_uids)
        ]

        raw_used_counters = [
            entry['counter']
            for entry in self.counter_policy_by_order.values()
            if entry.get('counter') is not None
        ]
        used_counters = list(dict.fromkeys(raw_used_counters))
        assigned_counters_display_map = {}

        for order_idx, order_tuple in enumerate(current_orders):
            goal = order_tuple[0]
            name = getattr(goal, 'full_name', '').lower()
            ings_lower = [ing for ing in ALL_INGREDIENTS if ing in name]
            if not ings_lower:
                continue

            # 料理の系統(サラダ/スープ/ジュース)。ここを見ずにゴール名の材料
            # だけを拾っていたため、サラダにも無条件で cook タスクが作られ、
            # スープとして提供されていた。
            dish_kind = goal_dish_kind(name)
            is_salad = (dish_kind == KIND_SALAD)
            is_juice = (dish_kind == KIND_JUICE)

            order_uid = order_uids[order_idx]
            self._emit_counter_debug(f"[StateCheck] order_slot={order_idx} order_uid={order_uid} recipe={name} is_salad={is_salad} ingredients={ings_lower}")
            display_order = order_uid
            ings_cap = [ing.capitalize() for ing in ings_lower]
            self._emit_counter_debug(f"[StateCheck] counter_snapshot order_uid={order_uid} assigned_counter={self._get_assigned_counter(order_uid)}")
            for pos, obj in sorted(getattr(env, 'pos_obj', {}).items(), key=lambda kv: kv[0]):
                if pos not in resources.get('counters', []):
                    continue
                foods = self._get_counter_food_names(env, pos)
                if foods:
                    self._emit_counter_debug(f"  [CounterState] counter={pos} foods={sorted(foods)}")
            reserved_counters = {
                other_counter
                for other_uid, entry in self.counter_policy_by_order.items()
                if other_uid != order_uid and (other_counter := entry.get('counter')) is not None
            }

            # --- (0) そもそもこの注文に置き場が要るかを先に決める ---
            # 置き場(マージ地点)は「刻んだ食材を1か所に集めて鍋へ運ぶ」ための
            # 一時的な作業場所であって、注文が永久に所有するものではない。
            #   * 材料が完全に揃って手に取られた  -> 集め終わったので用済み
            #   * 既に鍋に入っている              -> 集める段階が終わっている
            # このときは割り当てを解除し、他の注文がそのテーブルを使えるようにする。
            # その後また誰かがテーブルに置いた場合は、(3) の retarget が拾い直す。
            suffix = {KIND_SALAD: SALAD_SUFFIX, KIND_SOUP: SOUP_SUFFIX,
                      KIND_JUICE: JUICE_SUFFIX}[dish_kind]
            dish_name = '-'.join(ings_lower) + suffix
            sorted_ings = sorted(ings_cap)
            # assembly_needed: まだ「材料を集める」段階が残っているか。
            #   スープ: 鍋にこのレシピが入っていれば集め終わり
            #   サラダ: 皿にこのレシピが乗っていれば集め終わり
            assembly_needed = True
            # 混ぜ終わっているか。ミキサーに入っていない間は「まだ」なので True。
            mixing_unfinished = True
            #   ジュース: ミキサーにこのレシピが入っていれば集め終わり
            states = (blender_states if is_juice
                      else plate_states if is_salad else pot_states)
            for ps in states:
                if not ps['used'] and ps['names'] == sorted_ings:
                    ps['used'] = True
                    assembly_needed = False
                    mixing_unfinished = not ps.get('mixed', True)
                    break

            cook_needed = assembly_needed and dish_kind == KIND_SOUP
            # 集め終わっていても、混ぜ終わるまで mix タスクは残す。
            mix_needed = is_juice and (assembly_needed or mixing_unfinished)

            merge_point_needed = assembly_needed and not self._agent_holds_complete_set(env, ings_cap)

            assigned_counter, _ = self._resolve_assigned_counter(
                env,
                order_uid,
            )
            self._debug_order_counter_state(env, current_orders, order_idx=order_idx, order_uid=order_uid, assigned_counter=assigned_counter, reason='before_conflict_check')

            if not merge_point_needed and assigned_counter is not None:
                reason = "reason=already_assembled" if not assembly_needed else "reason=ingredients_collected"
                self._set_assigned_counter(order_uid, None)
                self._log_counter_policy(order_uid, "release", assigned_counter, reason)
                if assigned_counter in used_counters:
                    used_counters.remove(assigned_counter)
                assigned_counter = None

            # state は unexpected / missing / owner_conflict で分離する。
            # missing は不完全だが破棄対象ではない。unexpected または owner_conflict だけ解除する。
            state = self._evaluate_order_counter_state(env, order_uid, ings_lower, assigned_counter)
            conflict_reason = self._classify_counter_conflict(env, ings_lower, assigned_counter, current_orders=current_orders, order_uid=order_uid)
            if assigned_counter is not None and conflict_reason is not None:
                should_release = self._should_release_invalid_counter(order_uid, assigned_counter, conflict_reason, env)
                if should_release:
                    self._set_assigned_counter(order_uid, None)
                    self._log_counter_policy(order_uid, "release", assigned_counter, f"reason={conflict_reason} persisted")
                    assigned_counter = None
                else:
                    self._log_counter_policy(order_uid, "hold", assigned_counter, f"reason={conflict_reason} pending_release")
            elif assigned_counter is not None:
                if state['status'] == 'incomplete':
                    self._log_counter_policy(order_uid, "hold", assigned_counter, "reason=incomplete_but_valid_partial_state")
                else:
                    self._log_counter_policy(order_uid, "stable", assigned_counter, "reason=valid_assignment")

            if assigned_counter is not None and assigned_counter in reserved_counters:
                # 同一カウンターを複数注文が使っている場合だけ、該当注文だけを撤去する。
                # これは「別テーブルの誤配置」で正しい割当を巻き込むことを防ぐための例外条件。
                if conflict_reason is not None:
                    self._set_assigned_counter(order_uid, None)
                    self._log_counter_policy(order_uid, "release", assigned_counter, "reason=counter_reserved_by_other_order")
                    assigned_counter = None

            # 割当先がまだ無い、または割当先はあるがまだ何も置かれていない場合は、
            # 人間が別カウンターへ誤って置いてしまった、この注文向けの食材が
            # どこかに無いか先に探す。見つかればそこを正式な置き場として引き継ぐ
            # (人間がどこに置いても次のパスで追従できるようにするため)。
            # 既に部分的にでも正しい食材が乗っている(incomplete)割当は、
            # 進行中の状態を捨てないよう対象外にする。
            should_try_retarget = (
                merge_point_needed
                and (assigned_counter is None or state.get('status') == 'unassigned')
            )
            if should_try_retarget:
                candidate = self._find_unclaimed_matching_counter(
                    env, order_uid, ings_lower, reserved_counters, used_counters, order_idx
                )
                if candidate is not None:
                    assigned_counter = candidate
                    self._set_assigned_counter(order_uid, assigned_counter)
                    if assigned_counter not in used_counters:
                        used_counters.append(assigned_counter)
                    self._log_counter_policy(order_uid, "retarget", assigned_counter, "reason=misplaced_ingredient_found")

            if assigned_counter is None and merge_point_needed:
                assigned_counter = self._calculate_dynamic_merge_point(env, ings_lower, order_idx, pot_locs, used_counters, reserved_counters=reserved_counters)
                if assigned_counter is not None:
                    self._set_assigned_counter(order_uid, assigned_counter)
                    if assigned_counter not in used_counters:
                        used_counters.append(assigned_counter)
                    self._log_counter_policy(order_uid, "assign", assigned_counter, "reason=new_order")

            self._debug_order_counter_state(env, current_orders, order_idx=order_idx, order_uid=order_uid, assigned_counter=assigned_counter, reason='after_assignment_logic')

            assigned_counters_display_map[display_order + 1] = assigned_counter

            # dish_name / sorted_ings / assembly_needed は、置き場が要るかを決めるために
            # このループの先頭(割り当て判定の前)で算出済み。
            tasks = []

            for ing in ings_cap:
                if not assembly_needed:
                    continue
                reserved_other_counters = {
                    counter for counter in used_counters
                    if counter is not None and counter != assigned_counter
                }
                if consume_chopped(ing, assigned_counter, reserved_other_counters, order_uid=order_uid):
                    continue
                dur = self._task_duration_frames(env, 'chop', ing.lower(), order_idx, assigned_counter)
                if dur is None:
                    continue
                # 既に切られた物が別のテーブルにあるなら、切らずに運ぶだけでよい
                carry_from = carry_sources.pop(ing, None)
                tasks.append({
                    'carry_from': carry_from,
                    'id': ('chop', ing.lower(), order_uid),
                    'verb': 'chop', 'obj': ing.lower(), 'order': order_uid,
                    'slot_idx': order_idx,
                    'display_order': display_order,
                    'dur': dur,
                    'res_candidates': [('cutboard', r) for r in resources['cutboards']],
                    'assigned_counter': assigned_counter
                })

            if mix_needed:
                # ジュース: 刻んだフルーツをミキサーへ入れて混ぜる。
                # 鍋(cook)と同じ位置づけの工程だが、待ち時間ではなく
                # インタラクト回数で進む。
                dur = self._task_duration_frames(env, 'mix', dish_name, order_idx, assigned_counter)
                if dur is not None:
                    tasks.append({
                        'id': ('mix', dish_name, order_uid),
                        'verb': 'mix', 'obj': dish_name, 'order': order_uid,
                        'slot_idx': order_idx,
                        'display_order': display_order,
                        'dur': dur,
                        'res_candidates': [('blender', r) for r in resources['blenders']],
                        'assigned_counter': assigned_counter
                    })

            if cook_needed:
                dur = self._task_duration_frames(env, 'cook', dish_name, order_idx)
                if dur is not None:
                    tasks.append({
                        'id': ('cook', dish_name, order_uid),
                        'verb': 'cook', 'obj': dish_name, 'order': order_uid,
                        'slot_idx': order_idx,
                        'display_order': display_order,
                        'dur': dur,
                        'res_candidates': [('pot', r) for r in resources['pots']],
                        'assigned_counter': assigned_counter
                    })

            # サラダは鍋を経由せず、刻んだ食材をそのまま皿に乗せて提供する。
            serve_verb = ('serve_juice' if is_juice
                          else 'serve_salad' if is_salad else 'serve')
            dur = self._task_duration_frames(env, serve_verb, dish_name, order_idx, assigned_counter)

            if dur is None and self._map_is_partitioned(env):
                # 提供までを1人でこなせない = 仕切りの向こう側に提供口がある。
                # 「自分ができず相手ができる」ので、受け渡しを挟んで2つに割る。
                #   handover           : 鍋から盛って受け渡し台に置く(こちら側)
                #   serve_from_counter : 受け渡し台から取って提供する(向こう側)
                dish_kind = dish_kind_of(dish_name)
                handover_counter = (
                    self._counter_holding_dish(env, dish_name)
                    or self._find_shared_counter(
                        env, resources['pots'][0] if resources['pots'] else None))
                dur_h = self._task_duration_frames(
                    env, 'handover', dish_name, order_idx, handover_counter)
                dur_s = self._task_duration_frames(
                    env, 'serve_from_counter', dish_name, order_idx, handover_counter)
                already_handed_over = self._counter_holding_dish(env, dish_name) is not None
                if already_handed_over and dur_s is not None:
                    # 受け渡し台にもう置いてある。渡す工程は済んでいるので、
                    # 作り直させず、受け取る工程だけを残す。
                    tasks.append({
                        'id': ('serve_from_counter', dish_name, order_uid),
                        'verb': 'serve_from_counter', 'obj': dish_name, 'order': order_uid,
                        'dish_kind': dish_kind,
                        'slot_idx': order_idx,
                        'display_order': display_order,
                        'dur': dur_s,
                        'res_candidates': [],
                        'assigned_counter': handover_counter,
                    })
                elif dur_h is not None and dur_s is not None:
                    tasks.append({
                        'id': ('handover', dish_name, order_uid),
                        'verb': 'handover', 'obj': dish_name, 'order': order_uid,
                        # 受け渡し系は「何料理か」が分からないと完成品を判別できない。
                        'dish_kind': dish_kind,
                        'slot_idx': order_idx,
                        'display_order': display_order,
                        'dur': dur_h,
                        'res_candidates': [('pot', r) for r in resources['pots']],
                        'assigned_counter': handover_counter,
                    })
                    tasks.append({
                        'id': ('serve_from_counter', dish_name, order_uid),
                        'verb': 'serve_from_counter', 'obj': dish_name, 'order': order_uid,
                        # 受け渡し系は「何料理か」が分からないと完成品を判別できない。
                        'dish_kind': dish_kind,
                        'slot_idx': order_idx,
                        'display_order': display_order,
                        'dur': dur_s,
                        'res_candidates': [],
                        'assigned_counter': handover_counter,
                    })

            if dur is not None:
                tasks.append({
                    'id': (serve_verb, dish_name, order_uid),
                    'verb': serve_verb, 'obj': dish_name, 'order': order_uid,
                    'slot_idx': order_idx,
                    'display_order': display_order,
                    'dur': dur,
                    'res_candidates': [],
                    'assigned_counter': assigned_counter
                })

            orders.append({'order': order_uid, 'display_order': display_order, 'name': dish_name, 'ingredients': ings_lower, 'tasks': tasks})

        self.assigned_counters_display_map = assigned_counters_display_map
        return orders

    def _apply_instruction_deadline_constraints(self, model, tasks, starts_by_idx, env):
        """保留指示に基づく締切制約を CP-SAT モデルへ反映する。再スケジューリングごとに経過時間を反映する。"""
        try:
            pending_instr = list(getattr(env, '_pending_instructions', []))
            agent_pending = getattr(self, '_pending_instructions', [])
            if agent_pending:
                for pending in agent_pending:
                    if not any(existing.get('id') == pending.get('id') for existing in pending_instr):
                        pending_instr.append(pending)
            # ループ本体はそのまま実行し、不要な診断出力だけをコメント化する。
            task_index_by_fixed_id = {}
            for idx, t in enumerate(tasks):
                fixed_task_id = t.get('fixed_task_id')
                if fixed_task_id is None:
                    fixed_task_id = self._make_fixed_task_id(
                        t.get('verb', ''),
                        t.get('obj', ''),
                        t.get('order', 0),
                    )
                    t['fixed_task_id'] = fixed_task_id
                task_index_by_fixed_id[fixed_task_id] = idx

            active_task_ids = set(self._get_active_task_ids()) if hasattr(self, '_get_active_task_ids') else set()

            for pending in list(pending_instr):
                try:
                    status = pending.get('status', 'pending')
                    if status in {'done', 'canceled'}:
                        continue

                    selected_task = pending.get('task')
                    payload = None
                    if isinstance(selected_task, (list, tuple)) and len(selected_task) >= 2:
                        payload = selected_task[1]
                    elif isinstance(selected_task, dict):
                        payload = selected_task
                    else:
                        payload = selected_task

                    fixed_task_id = None
                    if isinstance(payload, dict):
                        fixed_task_id = payload.get('fixed_task_id')
                    elif isinstance(payload, (list, tuple)) and len(payload) >= 1:
                        fixed_task_id = payload[0]
                    else:
                        fixed_task_id = payload

                    if fixed_task_id is None:
                        # print(f"[CSPAgent] 指示の固定 ID がないためスキップ: {payload}")
                        pending['status'] = 'canceled'
                        pending['deadline_constraint_applied'] = True
                        continue

                    matched_idx = task_index_by_fixed_id.get(fixed_task_id)
                    if matched_idx is None:
                        # print(f"[CSPAgent] 対応するタスクがなくなったため完了扱い: fixed_id={fixed_task_id}")
                        pending['status'] = 'done'
                        pending['deadline_constraint_applied'] = True
                        continue

                    target_task = tasks[matched_idx]
                    target_tid = target_task.get('id')
                    if target_tid in self.completed_task_ids:
                        # print(f"[CSPAgent] 指示対象タスクは完了済み: fixed_id={fixed_task_id} tid={target_tid}")
                        pending['status'] = 'done'
                        pending['deadline_constraint_applied'] = True
                        continue

                    # execution_logged はイベントログのフラグであって、対象タスクの実際の進行状態を表すものではない。
                    # そのため、execution_logged が立っていても、実際にタスクが開始/完了していなければ保留状態のまま扱う。

                    accepted_env_time = pending.get('accepted_env_time', None)
                    current_env_time = None
                    if hasattr(env, 'time'):
                        current_env_time = getattr(env, 'time')
                    elif hasattr(env, 'current_time'):
                        current_env_time = getattr(env, 'current_time')
                    if current_env_time is None:
                        current_env_time = accepted_env_time if accepted_env_time is not None else 0.0
                    if accepted_env_time is None:
                        # print(f"[CSPAgent] 指示の環境時刻情報がないためスキップ: {pending}")
                        pending['status'] = 'canceled'
                        pending['deadline_constraint_applied'] = True
                        continue

                    elapsed_seconds = max(0.0, float(current_env_time) - float(accepted_env_time))
                    deadline_seconds = self.deadline_seconds
                    if deadline_seconds is None:
                        deadline_seconds = self.deadline_frames / float(self.fps)
                    deadline_info = self._classify_instruction_deadline(pending, current_env_time, deadline_seconds)
                    remaining_deadline_seconds = deadline_info.get('remaining_seconds', 0.0)

                    accepted_frame = int(float(accepted_env_time) * self.fps)
                    start_var = starts_by_idx.get(matched_idx)
                    if start_var is None:
                        pending['deadline_constraint_applied'] = True
                        continue

                    if deadline_info.get('mode') == 'urgent':
                        urgent_frame = int(float(current_env_time) * self.fps)
                        model.Add(start_var <= urgent_frame + 1)
                        # print(f"[CSPAgent] 指示を最優先で実行: fixed_id={fixed_task_id} start <= {urgent_frame + 1} (env_time={current_env_time})")
                    else:
                        remaining_frames = int(remaining_deadline_seconds * self.fps)
                        bound = accepted_frame + remaining_frames
                        model.Add(start_var <= bound)
                        # print(f"[CSPAgent] 期限制約を追加: fixed_id={fixed_task_id} start <= {bound} (env_time={current_env_time}, remaining_seconds={remaining_deadline_seconds:.3f})")
                    pending['deadline_constraint_applied'] = True
                except Exception as e:
                    # print(f"[CSPAgent] 保留指示反映中に例外: {e}")
                    pass
        except Exception as e:
            # print(f"[CSPAgent] 指示による締切制約の追加で失敗: {e}")
            pass

    def solve_csp_scheduling(self, env, orders):
        """
        OR-Tools CP-SAT を用いたスケジューリング（移動コスト込み）。
        Circuit制約を用いて順序依存のセットアップ時間（移動時間）を正確にモデル化する。
        """
        self._emit_counter_debug(f"[CSPAgent] CSPスケジューリング開始 ({len(orders)} 注文)...")
        previous_schedule = getattr(self, 'schedule', None)
        previous_schedule_per_agent = getattr(self, 'schedule_per_agent', None)
        model = cp_model.CpModel()
        
        # 1. タスクのリスト化とリソース位置の固定
        tasks = []
        for o in orders:
            for t in o['tasks']:
                t['order_obj'] = o # 親注文への参照（便利のため）
                tasks.append(t)
        
        num_tasks = len(tasks)
        # 直近の解の評価値。時間損失量 L(d) の計算
        # (estimate_instruction_time_loss)が「指示制約あり/なし」を解き比べる
        # ために読む。早期 return の前に必ず作り直すこと。前回の値が残ると
        # 2回目が解けなかったときに L=0 と誤認してしまう。
        self._last_solve_metrics = {
            'status': 'no_tasks',
            'num_tasks': num_tasks,
            'makespan_frames': None,
            'objective': None,
        }
        if num_tasks == 0:
            self._emit_counter_debug("[CSPAgent] スケジュール対象タスクがありません。")
            return []

        # 現在のエージェント位置（初期位置）
        if hasattr(env, 'sim_agents'):
            agent_pos = env.sim_agents[0].location
            agent1_pos = env.sim_agents[1].location if len(env.sim_agents) > 1 else agent_pos
        elif hasattr(env, 'agents'):
            agent_pos = env.agents[0].location
            agent1_pos = env.agents[1].location if len(env.agents) > 1 else agent_pos
        else:
            # Fallback (should not happen in standard gym_cooking envs)
            agent_pos = (0, 0)
            agent1_pos = (0, 0)

        # 人間の実座標は「いま何に手をつけているか」の推測にだけ使う。
        # 距離行列には使わない(下のコメント参照)。
        human_real_pos = agent1_pos if self.own_agent_idx == 0 else agent_pos

        if self.human_counterpart_mode:
            # human_counterpart_mode では「もう一方」は実際には CSP が指示できない人間で、
            # その計画(schedule_per_agent の反対側)は実行されない仮想的な what-if に過ぎない。
            # それにもかかわらずその人間の実座標を毎回ここに使うと、人間がランダムに
            # 歩き回るたびに距離行列がわずかに変化し、"どちらの仮想エージェントが
            # どのタスクを担当するか"という(実行結果に直結する)決定が同点でなくなって
            # 毎回入れ替わってしまう(食材を置く→拾うを延々繰り返す等の揺れの原因)。
            # 人間の実座標を計画に反映させず、常に自分(実際に動かす側)と同じ地点から
            # 出発すると仮定することで、この人間の動きに起因する揺れを断つ。
            # ただし仕切りのあるマップでは、この置き換えをしてはいけない。
            # 相手の出発地点を自分と同じ側にしてしまうと、相手側にしかない
            # タスクへの距離が「到達不能」となり、既定値の 1000 フレーム
            # (=100秒)が入って計画が壊れる。両者が同じ連結成分にいるとき
            # だけ置き換える。
            same_region = (self._agent_component(env, 0) == self._agent_component(env, 1))
            if same_region:
                if self.own_agent_idx == 0:
                    agent1_pos = agent_pos
                else:
                    agent_pos = agent1_pos
        
        # リソース位置の特定と固定 (Fixed Position)
        resources = self._get_resources(env)
        self._annotate_task_geometry(env, tasks, agent_pos)

        # 2. 距離行列の作成 (A* distance)
        node_num = num_tasks + 2 if self.sc_2agent else num_tasks + 1
        all_nodes = list(range(node_num))
        start_node = num_tasks # ダミーノード
        agent1_start_node = num_tasks + 1 if self.sc_2agent else None
        
        dist_matrix = {} # (from_idx, to_idx) -> distance
        self._emit_counter_debug("[CSPAgent] 距離行列を計算中...")
        
        # キャッシュ付きA*
        dist_cache = {}
        def get_dist(p1, p2):
            if (p1, p2) in dist_cache: return dist_cache[(p1, p2)]
            d = self.astar_distance(env, p1, p2)
            if d is None: d = 1000
            dist_cache[(p1, p2)] = d
            return d

        for i in all_nodes:
            for j in all_nodes:
                if i == j:
                    dist_matrix[(i,j)] = 0
                    continue
                # From Node
                if i == start_node:
                    pos_i = agent_pos
                elif self.sc_2agent and i == agent1_start_node:
                    pos_i = agent1_pos
                else:
                    pos_i = tasks[i]['end_pos']
                # To Node
                if j == start_node or (self.sc_2agent and j == agent1_start_node):
                    dist_matrix[(i,j)] = 0
                else:
                    pos_j = tasks[j]['start_pos']
                    dist = get_dist(pos_i, pos_j)
                    dist_matrix[(i,j)] = dist

        # Debug: Check distances between task types
        self._emit_counter_debug("--- 距離行列サンプル ---")
        sample_chop = next((i for i, t in enumerate(tasks) if t['verb'] == 'chop'), None)
        sample_cook = next((i for i, t in enumerate(tasks) if t['verb'] == 'cook'), None)
        if sample_chop is not None and sample_cook is not None:
            d1 = dist_matrix.get((sample_chop, sample_cook), -1)
            d2 = dist_matrix.get((sample_cook, sample_chop), -1)
            p1 = tasks[sample_chop]['end_pos']
            p2 = tasks[sample_cook]['start_pos']
            self._emit_counter_debug(f"Chop({sample_chop} @ {p1}) -> Cook({sample_cook} @ {p2}): {d1}")
            self._emit_counter_debug(f"Cook({sample_cook} @ {p2}) -> Chop({sample_chop} @ {p1}): {d2}")
        self._emit_counter_debug("------------------------")

        # 3. 変数と制約の定義
        horizon = 10000 

        starts = {}
        ends = {}
        intervals = {}
        
        # タスクごとのInterval作成
        for i in range(num_tasks):
            t = tasks[i]
            dur = int(t['dur']) # これは作業自体の正味時間（移動含まない）
            
            s_var = model.NewIntVar(0, horizon, f'start_{t["id"]}')
            e_var = model.NewIntVar(0, horizon, f'end_{t["id"]}')
            interval = model.NewIntervalVar(s_var, dur, e_var, f'interval_{t["id"]}')
            
            starts[i] = s_var
            ends[i] = e_var
            intervals[i] = interval

        
        # Startノード用のダミー変数（Circuit用）
        starts[start_node] = model.NewIntVar(0, 0, 'start_dummy')
        ends[start_node] = model.NewIntVar(0, 0, 'end_dummy')
        if self.sc_2agent:
            starts[agent1_start_node] = model.NewIntVar(0, 0, 'start_dummy_1')
            ends[agent1_start_node] = model.NewIntVar(0, 0, 'end_dummy_1')

        # =====================================================
        # エージェント割り当て変数 (JobShop型)
        # AddCircuitによる順序固定をやめ、各タスクにどちらのエージェントを
        # 割り当てるかを決定変数として扱う。
        # これにより依存待ち時間中に別タスクを実行できる解が見つかるようになる。
        # =====================================================
        is_a1 = None  # 2エージェントモード時にのみ設定される割り当てBoolVarリスト
        switch_penalty_terms = []  # 前回と担当エージェントが変わるタスクへの弱いペナルティ

        if self.sc_2agent:
            if True:  # 2エージェント割り当て(人間スロットは推測タスク1件のみ)
                # is_a1[i]: True = タスクiをAI1が担当、False = AI0が担当
                is_a1 = [model.NewBoolVar(f'is_a1_{i}') for i in range(num_tasks)]

                if self.human_counterpart_mode:
                    # human_counterpart_mode では「もう一方」は CSP が指示できない人間。
                    # 半分を相手に任せる前提で組むと、AI は自分の担当外のタスクの完了を
                    # 永久に待ち続けて停止する(例: serve が人間スロットに入ると鍋が
                    # 空かず、食材を全部持ったまま「鍋が空くまで待機」で固まる)。
                    #
                    # そこで人間スロットに置くのは「いま人間が手をつけていると推測した
                    # 1つだけ」に限り、残りは全部 AI が単独でこなす前提で組む。
                    # 人間スロットのタスクはそれ1つなので、必然的に人間の最初のタスクになる。
                    # 推測が外れても、人間が実際にやった分は世界の状態に現れて次回の
                    # 再スケジューリングでタスク一覧から消えるため自動的に補正される。
                    own_is_a1 = 1 if self.own_agent_idx == 1 else 0
                    human_is_a1 = 1 - own_is_a1
                    human_task = (
                        self._predict_human_current_task(env, tasks, human_real_pos)
                        if self.use_predicted_human_model else None
                    )
                    human_task_idx = None
                    if human_task is not None:
                        for i in range(num_tasks):
                            if tasks[i]['id'] == human_task['id']:
                                human_task_idx = i
                                break
                    partitioned = self._map_is_partitioned(env)
                    for i in range(num_tasks):
                        forced = human_is_a1 if i == human_task_idx else own_is_a1
                        if partitioned:
                            # 仕切りのあるマップでは、AI が物理的に行けないタスクがある。
                            # それを AI に割り当てると永久に実行できず全体が止まるので、
                            # 到達できる側のエージェントへ回す。
                            allowed = self._assignable_agents(env, tasks[i])
                            if len(allowed) == 1:
                                forced = next(iter(allowed))
                        model.Add(is_a1[i] == forced)
                    self.predicted_human_tasks = (
                        [{'id': human_task['id'], 'task': human_task}]
                        if human_task_idx is not None else []
                    )


                elif self._map_is_partitioned(env):
                    # 両方AIのモードでも、行ける側にしか割り当てないよう縛る。
                    for i in range(num_tasks):
                        allowed = self._assignable_agents(env, tasks[i])
                        if len(allowed) == 1:
                            model.Add(is_a1[i] == next(iter(allowed)))

                # エージェント出発位置からの最低到達時間
                for i in range(num_tasks):
                    dist_from_a0 = int(dist_matrix.get((start_node, i), 0))
                    dist_from_a1 = int(dist_matrix.get((agent1_start_node, i), 0))
                    model.Add(starts[i] >= dist_from_a0).OnlyEnforceIf(is_a1[i].Not())
                    model.Add(starts[i] >= dist_from_a1).OnlyEnforceIf(is_a1[i])
                
                for i in range(num_tasks):
                    for j in range(i + 1, num_tasks):
                        order_ij = model.NewBoolVar(f'order_{i}_{j}')
                        dij = int(dist_matrix.get((i, j), 0))
                        dji = int(dist_matrix.get((j, i), 0))
                        model.Add(starts[j] >= ends[i] + dij).OnlyEnforceIf(
                            [is_a1[i].Not(), is_a1[j].Not(), order_ij])
                        model.Add(starts[i] >= ends[j] + dji).OnlyEnforceIf(
                            [is_a1[i].Not(), is_a1[j].Not(), order_ij.Not()])
                        model.Add(starts[j] >= ends[i] + dij).OnlyEnforceIf(
                            [is_a1[i], is_a1[j], order_ij])
                        model.Add(starts[i] >= ends[j] + dji).OnlyEnforceIf(
                            [is_a1[i], is_a1[j], order_ij.Not()])

                # makespan と end_sum が同点になるタスク割り当てが複数存在する場合、
                # ソルバーはその中から任意の1つを選ぶため、世界の状態がわずかに
                # 変化するたびに「どちらのエージェントがどのタスクを担当するか」が
                # 入れ替わってしまうことがある(例: 同じ食材を置く/持つを繰り返す)。
                # 前回のスケジュールで各タスクがどちらのエージェント担当だったかを
                # 覚えておき、目的関数の最下位優先度としてそれを維持する方を弱く
                # 優先することで、makespan/end_sum を悪化させない範囲でのみ
                # 担当エージェントの入れ替わりを抑制する。
                prev_agent_by_task_id = {}
                if previous_schedule_per_agent:
                    for prev_agent_idx in (0, 1):
                        for prev_task in previous_schedule_per_agent.get(prev_agent_idx, []):
                            prev_agent_by_task_id[prev_task['id']] = prev_agent_idx
                for i in range(num_tasks):
                    prev_agent = prev_agent_by_task_id.get(tasks[i]['id'])
                    if prev_agent is None:
                        continue
                    switch_penalty_terms.append(is_a1[i] if prev_agent == 0 else (1 - is_a1[i]))
        else:
            # 1エージェント: 従来の circuit 方式
            arcs = []
            lit_map = {}
            for i in all_nodes:
                for j in all_nodes:
                    if i == j: continue
                    lit = model.NewBoolVar(f'arc_{i}_{j}')
                    arcs.append((i, j, lit))
                    lit_map[(i, j)] = lit
                    if j != start_node:
                        dist = dist_matrix[(i, j)]
                        model.Add(starts[j] >= ends[i] + dist).OnlyEnforceIf(lit)
                    if i == start_node:
                        dist = dist_matrix[(i, j)]
                        model.Add(starts[j] >= dist).OnlyEnforceIf(lit)
            model.AddCircuit(arcs)

        # ====================================================
        self._emit_counter_debug(f"[CSPAgent] スケジュール対象タスク数: {num_tasks}")
        
        # Helper: Group vars by Order ID / TID
        vars_by_order = {} 
        vars_by_tid = {}   
        
        for i in range(num_tasks):
            t = tasks[i]
            tid = t['id']
            order_uid = t['order']
            if order_uid not in vars_by_order: vars_by_order[order_uid] = []
            
            v_obj = {'start': starts[i], 'end': ends[i], 'task': t, 'interval': intervals[i]}
            vars_by_order[order_uid].append(v_obj)
            vars_by_tid[tid] = v_obj

        # 標準的な順序制約 (Chop -> Cook -> Serve)
        cooking_frames = int(COOKING_TIME_SECONDS * self.fps)  # config.pyから取得
        for i in range(num_tasks):
            t = tasks[i]
            verb = t['verb']
            if verb == 'cook':
                order_vars = vars_by_order.get(t['order'], [])
                chops = [v for v in order_vars if v['task']['verb'] == 'chop']
                for c in chops:
                    model.Add(starts[i] >= c['end'])
            elif verb == 'mix':
                # ミキサーに入れられるのは、フルーツを全部刻んでから。cook と同じ形。
                order_vars = vars_by_order.get(t['order'], [])
                chops = [v for v in order_vars if v['task']['verb'] == 'chop']
                for c in chops:
                    model.Add(starts[i] >= c['end'])
            elif verb == 'serve_juice':
                # 混ぜ終わってからでないと注げない。混ぜるのは待ち時間ではなく
                # インタラクトなので、cook のような待機分の加算は不要。
                order_vars = vars_by_order.get(t['order'], [])
                mixes = [v for v in order_vars if v['task']['verb'] == 'mix']
                for m in mixes:
                    model.Add(starts[i] >= m['end'])
            elif verb == 'serve_salad':
                # サラダは調理を挟まないので、chop が全部終われば提供できる。
                order_vars = vars_by_order.get(t['order'], [])
                chops = [v for v in order_vars if v['task']['verb'] == 'chop']
                for c in chops:
                    model.Add(starts[i] >= c['end'])
            elif verb == 'serve':
                order_vars = vars_by_order.get(t['order'], [])
                cooks = [v for v in order_vars if v['task']['verb'] == 'cook']
                for c in cooks:
                    # cook終了 + 実際の調理時間(config.pyのCOOKING_TIME_SECONDS)が経過してからserve可能
                    model.Add(starts[i] >= c['end'] + cooking_frames)
            elif verb == 'handover':
                order_vars = vars_by_order.get(t['order'], [])
                cooks = [v for v in order_vars if v['task']['verb'] == 'cook']
                if cooks:
                    # 渡せるのは煮上がってから。serve と同じ条件。
                    for c in cooks:
                        model.Add(starts[i] >= c['end'] + cooking_frames)
                else:
                    # 鍋を使わない料理は、下ごしらえが全部終われば渡せる。
                    for c in [v for v in order_vars if v['task']['verb'] == 'chop']:
                        model.Add(starts[i] >= c['end'])
            elif verb == 'serve_from_counter':
                # 受け渡し台に置かれてからでないと取れない。
                order_vars = vars_by_order.get(t['order'], [])
                handovers = [v for v in order_vars if v['task']['verb'] == 'handover']
                for h in handovers:
                    model.Add(starts[i] >= h['end'])

        # 鍋の占有制約 (Pot Usage Constraint)
        pot_usage_intervals = {}
        for order_idx, tasks_list in vars_by_order.items():
            cooks = [v for v in tasks_list if v['task']['verb'] == 'cook']
            # 鍋が空くのは中身を取り出したとき。仕切りありのマップでは
            # serve ではなく handover がその役割を担う。
            serves = [v for v in tasks_list
                      if v['task']['verb'] in ('serve', 'handover')]
            if cooks and serves:
                cook_task = cooks[0]
                serve_task = serves[0]
                pot_res = cook_task['task'].get('fixed_res')
                if pot_res and pot_res[0] == 'pot':
                    pot_loc = pot_res[1]
                    if pot_loc not in pot_usage_intervals:
                        pot_usage_intervals[pot_loc] = []
                    p_start = cook_task['start']
                    p_end = serve_task['end']
                    p_size = model.NewIntVar(0, horizon, f'pot_usage_dur_{order_idx}')
                    model.Add(p_size == p_end - p_start)
                    p_interval = model.NewIntervalVar(p_start, p_size, p_end, f'pot_usage_{order_idx}')
                    pot_usage_intervals[pot_loc].append(p_interval)

        # ミキサーの占有制約。鍋とまったく同じ形で、
        # 「入れてから取り出すまで」を1区間として重複を禁じる。
        blender_usage_intervals = {}
        for order_idx, tasks_list in vars_by_order.items():
            mixes = [v for v in tasks_list if v['task']['verb'] == 'mix']
            takes = [v for v in tasks_list if v['task']['verb'] == 'serve_juice']
            if mixes and takes:
                res = mixes[0]['task'].get('fixed_res')
                if res and res[0] == 'blender':
                    loc = res[1]
                    blender_usage_intervals.setdefault(loc, [])
                    b_start, b_end = mixes[0]['start'], takes[0]['end']
                    b_size = model.NewIntVar(0, horizon, f'blender_usage_dur_{order_idx}')
                    model.Add(b_size == b_end - b_start)
                    blender_usage_intervals[loc].append(
                        model.NewIntervalVar(b_start, b_size, b_end, f'blender_usage_{order_idx}'))

        for b_loc, intervals_list in blender_usage_intervals.items():
            if len(intervals_list) > 1:
                model.AddNoOverlap(intervals_list)
                self._emit_counter_debug(
                    f"[CSPAgent] ミキサー {b_loc} の重複禁止制約を追加 ({len(intervals_list)} 注文)")

        for pot_loc, intervals_list in pot_usage_intervals.items():
            if len(intervals_list) > 1:
                model.AddNoOverlap(intervals_list)
                self._emit_counter_debug(f"[CSPAgent] 鍋 {pot_loc} の重複禁止制約を追加 ({len(intervals_list)} 注文)")

        # まな板の占有制約 (Cutboard Usage Constraint)
        cutboard_intervals = {}
        for i in range(num_tasks):
            t = tasks[i]
            if t['verb'] == 'chop':
                c_res = t.get('fixed_res')
                if c_res and c_res[0] == 'cutboard':
                    c_loc = c_res[1]
                    if c_loc not in cutboard_intervals:
                        cutboard_intervals[c_loc] = []
                    cutboard_intervals[c_loc].append(intervals[i])
        
        for c_loc, intervals_list in cutboard_intervals.items():
            if len(intervals_list) > 1:
                model.AddNoOverlap(intervals_list)
                self._emit_counter_debug(f"[CSPAgent] まな板 {c_loc} の重複禁止制約を追加 ({len(intervals_list)} タスク)")

        # 動的制約 (Dynamic Constraints)
        if hasattr(self, 'active_constraints') and self.active_constraints:
            self._emit_counter_debug(f"[CSPAgent] 動的制約を適用中: {len(self.active_constraints)}件")
            for constr in self.active_constraints:
                c_type = constr.get('type')
                if c_type == 'order_sequential':
                    mode = constr.get('mode', 'pipeline')
                    sorted_orders = sorted(vars_by_order.keys())
                    for k in range(len(sorted_orders) - 1):
                        curr_o = sorted_orders[k]
                        next_o = sorted_orders[k+1]
                        curr_tasks_v = vars_by_order[curr_o]
                        next_tasks_v = vars_by_order[next_o]
                        if mode == 'strict':
                            for v_next in next_tasks_v:
                                for v_curr in curr_tasks_v:
                                    model.Add(v_next['start'] >= v_curr['end'])
                        elif mode == 'pipeline':
                            for verb in ['chop', 'cook', 'serve', 'serve_salad']:
                                curr_verb_tasks = [v for v in curr_tasks_v if v['task']['verb'] == verb]
                                next_verb_tasks = [v for v in next_tasks_v if v['task']['verb'] == verb]
                                for v_next in next_verb_tasks:
                                    for v_curr in curr_verb_tasks:
                                        model.Add(v_next['start'] >= v_curr['end'])
                elif c_type == 'precedence':
                    before_subs = constr.get('before', [])
                    after_subs = constr.get('after', [])
                    before_ends = []
                    after_starts = []
                    for tid, v in vars_by_tid.items():
                        full_name = f"{tid[0]}_{tid[1]}"
                        for sub in before_subs:
                            if sub in full_name:
                                before_ends.append(v['end'])
                                break
                        for sub in after_subs:
                            if sub in full_name:
                                after_starts.append(v['start'])
                                break
                    for be in before_ends:
                        for ast in after_starts:
                            model.Add(ast >= be)

        # ============ 指示による制約 (skip_budget ベース) ============
        # 旧: 秒数ベース制約 (当面は呼び出さない — メソッドは残す)
        # self._apply_instruction_deadline_constraints(model, tasks, starts, env)
        if self.skip_budget is not None:
            self._apply_instruction_skip_budget_constraints(model, tasks, starts, env, is_a1=is_a1)

        # Makespan 最小化
        makespan = model.NewIntVar(0, horizon, 'makespan')
        task_ends = [ends[i] for i in range(num_tasks)]
        if task_ends:
            model.AddMaxEquality(makespan, task_ends)
        else:
            model.Add(makespan == 0)
        end_sum = sum(task_ends) if task_ends else 0
        weight_makespan = num_tasks * 1000
        if switch_penalty_terms:
            # switch_scale は switch_penalty が取り得る最大値より大きくし、
            # (makespan, end_sum) の優先順位を一切変えずに完全な同点のときだけ
            # 担当エージェント維持側を選ばせる。
            switch_scale = len(switch_penalty_terms) + 1
            switch_penalty = sum(switch_penalty_terms)
            model.Minimize((makespan * weight_makespan + end_sum) * switch_scale + switch_penalty)
        else:
            model.Minimize(makespan * weight_makespan + end_sum)

        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        status_name = solver.StatusName(status)
        self._emit_counter_debug(f"[CSPAgent] ソルバー状態: {status_name}")

        self._last_solve_metrics.update(status=status_name, num_tasks=num_tasks)

        schedule = []
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            actual_makespan = solver.Value(makespan)
            self._last_solve_metrics['makespan_frames'] = int(actual_makespan)
            self._last_solve_metrics['objective'] = solver.ObjectiveValue()
            self._emit_counter_debug(f"[CSPAgent] 最適Makespan(移動込み): {actual_makespan} (評価値: {solver.ObjectiveValue()})")
            
            if not self.sc_2agent:
                # 1エージェント: 従来通りlit_mapでルートをトレース
                schedule.sort(key=lambda x: x['start'])
                self._emit_counter_debug("--- 推定順序と移動時間 (詳細) ---")
                current_node = start_node
                visited_count = 0
                while visited_count < num_tasks:
                    found_next = False
                    for j in all_nodes:
                        if current_node == j: continue
                        lit = lit_map.get((current_node, j))
                        if lit is not None and solver.Value(lit) == 1:
                            if j != start_node:
                                t = tasks[j]
                                schedule.append({
                                    'id': t['id'],
                                    'start': solver.Value(starts[j]),
                                    'end': solver.Value(ends[j]),
                                    'res': t.get('fixed_res'),
                                    'assigned_counter': t.get('assigned_counter'),
                                    'display_order': t.get('display_order', t.get('slot_idx', t['order'])),
                                    'fixed_task_id': self._make_fixed_task_id(t['verb'], t['obj'], t['order'])
                                })
                                self._emit_counter_debug(f" -> {t['verb']} {t['obj']}")
                            current_node = j
                            found_next = True
                            visited_count += 1
                            break
                    if not found_next: break
                self._emit_counter_debug("---------------------------------")
            else:
                # 2エージェント: is_a1 変数から直接エージェント割り当てを読む
                schedule_per_agent = {0: [], 1: []}
                for i in range(num_tasks):
                    t = tasks[i]
                    agent_idx = 1 if solver.Value(is_a1[i]) else 0
                    schedule_per_agent[agent_idx].append({
                        'id': t['id'],
                        'start': solver.Value(starts[i]),
                        'end': solver.Value(ends[i]),
                        'res': t.get('fixed_res'),
                        'assigned_counter': t.get('assigned_counter'),
                        'display_order': t.get('display_order', t.get('slot_idx', t['order'])),
                        'agent_idx': agent_idx,
                        'fixed_task_id': self._make_fixed_task_id(t['verb'], t['obj'], t['order'])
                    })
                # 各エージェントのタスクを開始時刻順に並べる
                for agent_idx in [0, 1]:
                    schedule_per_agent[agent_idx].sort(key=lambda x: x['start'])
                
                self.schedule_per_agent = schedule_per_agent
                schedule = schedule_per_agent[0] + schedule_per_agent[1]
                schedule.sort(key=lambda x: x['start'])
                
        else:
            if status_name == 'INFEASIBLE':
                self._emit_counter_debug(f"[CSPAgent] ソルバー結果: INFEASIBLE (締切制約などにより解なし) — 直前のスケジュールへフォールバックします。")
                if self.sc_2agent and previous_schedule_per_agent:
                    self.schedule_per_agent = previous_schedule_per_agent
                    fallback_schedule = previous_schedule_per_agent[0] + previous_schedule_per_agent[1]
                    fallback_schedule.sort(key=lambda x: x['start'])
                    return fallback_schedule
                if previous_schedule:
                    self.schedule = previous_schedule
                    return previous_schedule
            else:
                self._emit_counter_debug(f"[CSPAgent] 解が見つかりませんでした。 状態={status_name}")
                pass
            
        return schedule

    # ------------------------------------------------------------------
    # 時間損失量 L(d)
    # ------------------------------------------------------------------
    def estimate_instruction_time_loss(self, env, pending, skip_budget=None):
        """指示による時間損失量 L(d) = f'(d) - f を求める。

        f      : その指示の制約を入れずに解いたときの最適 makespan
        f'(d)  : 「指示タスクより前に同じエージェントが実行してよい他タスクは
                 d 個まで」という制約を入れて解いたときの最適 makespan
        L(d)   : その差。大きいほど、最適な段取りから外れた指示だったことになる。

        L >= 0 が成り立つのは「同じタスク集合・同じ目的関数で、両方を最適まで
        解いたとき」だけなので、ここでは時間制限を掛けずに解く。呼び出し側は
        ゲーム進行を止めないよう別スレッドから呼ぶこと(1回 0.3〜2秒かかる)。

        戻り値は秒単位。解けなかった場合は loss_seconds を None にして status に
        理由を残す。
        """
        from copy import deepcopy as _dcopy

        if skip_budget is None:
            skip_budget = self.skip_budget
        result = {
            'loss_seconds': None,
            'baseline_seconds': None,      # f
            'constrained_seconds': None,   # f'(d)
            'skip_budget': skip_budget,
            'status': 'unknown',
        }
        if skip_budget is None:
            result['status'] = 'no_constraint'
            return result

        try:
            # 本番の状態を一切汚さないよう、評価用の複製の上だけで解く。
            # replay は巨大で複製する意味がないため一時的に外す。
            saved_replay = self.replay
            self.replay = None
            probe = None
            try:
                # AI スレッドが同時に内部の辞書を書き換えていると
                # deepcopy が "dictionary changed size during iteration" で
                # 失敗する。複製は一瞬なので、数回やり直せばまず通る。
                import time as _time
                last_err = None
                for _ in range(5):
                    try:
                        probe = _dcopy(self)
                        break
                    except RuntimeError as err:
                        last_err = err
                        _time.sleep(0.02)
                if probe is None:
                    raise last_err
            finally:
                self.replay = saved_replay
            probe.replay = None
            probe.debug_counter_trace = False

            env_probe = _dcopy(env)
            # 対象の指示だけが載った状態にする(A3: 指示は同時に1つだけ)。
            probe._pending_instructions = [_dcopy(pending)]
            env_probe._pending_instructions = [_dcopy(pending)]

            # タスク集合は f / f'(d) で完全に同じでなければ比較にならないので、
            # 1度だけ作って両方に渡す。
            orders = probe._build_order_tasks(env_probe)

            def solve_makespan_frames(budget):
                probe.skip_budget = budget
                # solve_csp_scheduling は orders 内のタスク辞書に order_obj を
                # 書き込むため、毎回作り直した複製を渡す。
                probe.solve_csp_scheduling(env_probe, orders=_dcopy(orders))
                return dict(getattr(probe, '_last_solve_metrics', {}) or {})

            # f: 指示制約なし (skip_budget=None だと制約自体が追加されない)
            base = solve_makespan_frames(None)
            # f'(d): 指示制約あり
            cons = solve_makespan_frames(skip_budget)

            if base.get('makespan_frames') is None:
                result['status'] = f"baseline_{base.get('status', 'failed')}"
                return result
            if cons.get('makespan_frames') is None:
                # d が厳しすぎて解が存在しない場合はここに来る。
                result['status'] = f"constrained_{cons.get('status', 'failed')}"
                result['baseline_seconds'] = base['makespan_frames'] / float(self.fps)
                return result

            f = base['makespan_frames'] / float(self.fps)
            f_prime = cons['makespan_frames'] / float(self.fps)
            result.update({
                'loss_seconds': round(f_prime - f, 3),
                'baseline_seconds': round(f, 3),
                'constrained_seconds': round(f_prime, 3),
                'num_tasks': base.get('num_tasks'),
                'baseline_status': base.get('status'),
                'constrained_status': cons.get('status'),
                'status': 'ok',
            })
        except Exception as e:
            result['status'] = f'error: {e}'
        return result


    def solve_csp_selection(self, env, orders=None):
        if orders is None:
            orders = self._build_order_tasks(env)
        
        self._emit_counter_debug("\n--- 生成タスク (環境状態でフィルタ済) ---")
        for o in orders:
            self._emit_counter_debug(f"注文 {o.get('display_order', o['order']) + 1} (食材: {o['ingredients']}):")
            if not o['tasks']:
                self._emit_counter_debug("  (タスク不要)")
            for t in o['tasks']:
                self._emit_counter_debug(f"  - {t['id']}: 所要={t['dur']}, 資源候補={t['res_candidates']}")
        self._emit_counter_debug("-------------------------------------------------------\n")

        budget = self.budget_frames
        res_timeline = {'cutboard':{}, 'pot':{}}
        schedule = []
        time_cursor = 0

        orders_sorted = orders

        for o in orders_sorted:
            prev_finish = time_cursor
            for t in [t for t in o['tasks'] if t['verb']=='chop']:
                if not t['res_candidates']:
                    continue
                best_id=None; earliest=0
                for _, rid in t['res_candidates']:
                    free = res_timeline['cutboard'].get(rid, 0)
                    if best_id is None or free < earliest:
                        best_id=rid; earliest=free
                start = max(prev_finish, earliest)
                end = start + t['dur']
                if end - time_cursor > budget:
                    break
                res_timeline['cutboard'][best_id] = end
                schedule.append({'id':t['id'],'start':start,'end':end,'res':('cutboard',best_id)})
                prev_finish = end
            cooks = [t for t in o['tasks'] if t['verb']=='cook']
            if cooks:
                t = cooks[0]
                if t['res_candidates']:
                    best_id=None; earliest=0
                    for _, rid in t['res_candidates']:
                        free = res_timeline['pot'].get(rid, 0)
                        if best_id is None or free < earliest:
                            best_id=rid; earliest=free
                    start = max(prev_finish, earliest)
                    end = start + t['dur']
                    if end - time_cursor <= budget:
                        res_timeline['pot'][best_id] = end 
                        schedule.append({'id':t['id'],'start':start,'end':end,'res':('pot',best_id)})
                        prev_finish = end
            serves = [t for t in o['tasks'] if t['verb'] in self.SERVE_VERBS]
            if serves:
                t = serves[0]
                start = prev_finish
                end = start + t['dur']
                if end - time_cursor <= budget:
                    schedule.append({'id':t['id'],'start':start,'end':end,'res':None})
                    prev_finish = end

            if prev_finish - time_cursor > budget:
                break
            else:
                budget -= (prev_finish - time_cursor)
                time_cursor = prev_finish

        return schedule

    def _print_schedule(self, schedule):
        self._emit_counter_debug("\n=== CSP スケジュール（フレーム単位） ===")
        total_frames = 0
        
        # Group by agent
        agents_sched = {}
        for item in schedule:
            aid = item.get('agent_idx', 0)
            if aid not in agents_sched:
                agents_sched[aid] = []
            agents_sched[aid].append(item)
            total_frames = max(total_frames, item.get('end', 0))

        for aid in sorted(agents_sched.keys()):
            self._emit_counter_debug(f"\nAI{aid}")
            for item in agents_sched[aid]:
                tid = item.get('id'); start = item.get('start'); end = item.get('end'); res = item.get('res')
                verb, obj, order = tid
                display_order = item.get('display_order', order)
                self._emit_counter_debug(f"{verb} {obj} (注文{display_order+1}) : 開始={start}, 終了={end}, 資源={res}")

        self._emit_counter_debug(f"\n総投入フレーム: {total_frames}")
        self._emit_counter_debug("===================================\n")

    # ------------------------------------------------------------------
    # 到達可能性(仕切りのあるマップ用)
    # ------------------------------------------------------------------
    def _walkable_components(self, env):
        """歩ける床マスを連結成分に分ける。

        仕切りで左右が分断されていれば2つ以上になる。1回のスケジューリング中に
        地形は変わらないので、env ごとにキャッシュする。
        """
        cached = getattr(self, '_component_cache', None)
        key = id(env)
        if cached is not None and cached[0] == key:
            return cached[1]

        width, height = env.world_width, env.world_height
        grid = env.to_grid
        seen = set()
        comps = []
        for sx in range(width):
            for sy in range(height):
                if grid[sx][sy] != 1 or (sx, sy) in seen:
                    continue
                comp = set()
                stack = [(sx, sy)]
                seen.add((sx, sy))
                while stack:
                    cx, cy = stack.pop()
                    comp.add((cx, cy))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = cx + dx, cy + dy
                        if (0 <= nx < width and 0 <= ny < height
                                and grid[nx][ny] == 1 and (nx, ny) not in seen):
                            seen.add((nx, ny))
                            stack.append((nx, ny))
                comps.append(comp)
        self._component_cache = (key, comps)
        return comps

    def _components_touching(self, env, pos):
        """その位置を使うために立てる床マスが属する連結成分の番号の集合。

        資材(まな板・鍋・カウンター等)は通れないマスなので、隣から使う。
        仕切りの上のカウンターは両側から使えるため、複数の番号を返す。
        """
        if pos is None:
            return set()
        comps = self._walkable_components(env)
        width, height = env.world_width, env.world_height
        cells = [tuple(pos)]
        x, y = pos
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and env.to_grid[nx][ny] == 1:
                cells.append((nx, ny))
        out = set()
        for i, comp in enumerate(comps):
            if any(c in comp for c in cells):
                out.add(i)
        return out

    def _agent_component(self, env, agent_idx):
        """そのエージェントがいる連結成分の番号。分からなければ None。"""
        agents = getattr(env, 'agents', None)
        if not agents or agent_idx >= len(agents):
            return None
        touching = self._components_touching(env, tuple(agents[agent_idx].location))
        return min(touching) if touching else None

    def _map_is_partitioned(self, env):
        """歩ける領域が2つ以上に分かれているか(仕切りのあるマップか)。"""
        return len(self._walkable_components(env)) > 1

    def _pick_plate(self, env, resources, near_pos):
        """near_pos と同じ側にある皿置き場を選ぶ。

        仕切りのあるマップでは皿置き場が両側にあり、代表1枚だけを見ると
        「向こう側の皿を取りに行く」経路になって到達不能と判定されてしまう。
        """
        plates = resources.get('plates') or []
        if not plates:
            return resources.get('plate')
        if near_pos is None or not self._map_is_partitioned(env):
            return plates[0]
        want = self._components_touching(env, near_pos)
        same_side = [p for p in plates if self._components_touching(env, p) & want]
        pool = same_side or plates
        return min(pool, key=lambda p: abs(p[0] - near_pos[0]) + abs(p[1] - near_pos[1]))

    def _pick_cup(self, env, resources, near_pos):
        """near_pos と同じ側にあるコップ置き場を選ぶ(皿の _pick_plate と同じ)。"""
        cups = resources.get('cups') or []
        if not cups:
            return resources.get('cup')
        if near_pos is None or not self._map_is_partitioned(env):
            return cups[0]
        want = self._components_touching(env, near_pos)
        same_side = [c for c in cups if self._components_touching(env, c) & want]
        pool = same_side or cups
        return min(pool, key=lambda c: abs(c[0] - near_pos[0]) + abs(c[1] - near_pos[1]))

    def _can_reach_delivery(self, env, agent_idx):
        """そのエージェントが提供口まで行けるか。"""
        if not self._map_is_partitioned(env):
            return True
        deliveries = env.get_pos_by_obj_gs(gs='Delivery')
        if not deliveries:
            return True
        mine = self._agent_component(env, agent_idx)
        return any(mine in self._components_touching(env, d) for d in deliveries)

    def _find_shared_counter(self, env, near_pos=None):
        """両側から使えるカウンター(受け渡し台)を1つ選ぶ。

        仕切りの上に並んだカウンターは、どちらの側からも手が届く。
        壁を1マスだけ開けるような専用の仕組みは作らず、既存のカウンターの
        うち「両側の連結成分から使えるもの」をそのまま受け渡し口として使う。
        """
        counters = env.get_pos_by_obj_gs(gs="Counter")
        shared = [c for c in counters if len(self._components_touching(env, c)) > 1]
        if not shared:
            return None

        # 既に何か載っている台を選ぶと、置こうとしても混ざらず何も起きないまま
        # 止まる(例: 別の注文の刻んだ材料が置いてある台にスープの皿を置こうと
        # する)。空いている台を優先する。
        pos_obj = getattr(env, 'pos_obj', {}) or {}
        empty = [c for c in shared if pos_obj.get(c) is None]
        pool = empty or shared

        if near_pos is None:
            return pool[0]
        return min(pool, key=lambda c: abs(c[0] - near_pos[0]) + abs(c[1] - near_pos[1]))

    def _task_components(self, env, task):
        """そのタスクを単独でこなせる連結成分の集合。

        タスクは start_pos -> (使う資材) -> end_pos という移動を含むので、
        それら全部が同じ連結成分から使えなければ、1人では完結できない。
        空集合を返したときは仕切りを跨いでいる = 受け渡しが必要ということ。
        """
        positions = []
        for key in ('start_pos', 'end_pos', 'assigned_counter'):
            if task.get(key) is not None:
                positions.append(tuple(task[key]))
        res = task.get('fixed_res')
        if res and len(res) > 1 and res[1] is not None:
            positions.append(tuple(res[1]))
        if not positions:
            return set(range(len(self._walkable_components(env))))

        comps = None
        for pos in positions:
            touching = self._components_touching(env, pos)
            comps = touching if comps is None else (comps & touching)
        return comps or set()

    def _task_allowed_agents(self, env, task):
        """そのタスクを実行できるエージェント番号の集合。"""
        comps = self._task_components(env, task)
        return {a for a in (0, 1) if self._agent_component(env, a) in comps}

    DONE_STATE_BY_KIND = {KIND_SOUP: 'Cooked', KIND_SALAD: 'Chopped', KIND_JUICE: 'Mixed'}

    def _counter_holding_dish(self, env, dish_name):
        """完成した料理が既に置かれているカウンターを探す。

        受け渡し台の割り当ては計画のたびに選び直されるが、一度置いた料理は
        動かない。置いてある場所を優先しないと、渡した側と受け取る側が別の
        テーブルを見つめたまま噛み合わなくなる。また、既に置いてあるものを
        「まだ作っていない」と誤認すると、渡す側が二度手間で止まる。
        """
        needed = set(dish_ingredients(dish_name))
        if not needed:
            return None
        done_state = self.DONE_STATE_BY_KIND.get(dish_kind_of(dish_name), 'Cooked')
        container = 'Cup' if dish_kind_of(dish_name) == KIND_JUICE else 'Plate'
        for pos in env.get_pos_by_obj_gs(gs='Counter'):
            obj = env.pos_obj.get(pos)
            name = getattr(obj, 'full_name', '') or ''
            if not name or container not in name:
                continue
            bases = set()
            ok = True
            for part in name.split('-'):
                if part in ('Plate', 'Cup'):
                    continue
                if not part.startswith(done_state):
                    ok = False
                    break
                bases.add(part[len(done_state):].lower())
            if ok and bases == needed:
                return pos
        return None

    def _agents_reaching(self, env, pos):
        """その位置を使える(隣に立てる)エージェント番号の集合。"""
        if pos is None:
            return set()
        comps = self._components_touching(env, tuple(pos))
        return {a for a in (0, 1) if self._agent_component(env, a) in comps}

    def _assignable_agents(self, env, task):
        """そのタスクを割り当ててよいエージェント番号の集合。

        1人で完結できるならその人。誰も完結できない(仕切りを跨ぐ)場合でも、
        「誰でもいい」にすると行けない側に割り当たって永久に止まる。最後の
        工程(end_pos)へ行ける側に寄せておけば、少なくとも仕上げはできる。
        """
        allowed = self._task_allowed_agents(env, task)
        if allowed:
            return allowed
        return self._agents_reaching(env, task.get('end_pos'))

    def astar_distance(self, env, start, goal):
        import heapq
        width = env.world_width
        height = env.world_height
        grid = env.to_grid

        def in_bounds(x, y):
            return 0 <= x < width and 0 <= y < height

        def walkable(x, y):
            return in_bounds(x, y) and grid[x][y] == 1

        # 出発点にはカウンターや調理器具のマスを渡されることがある。人はその
        # 隣に立って触るので、隣の床から測る。ここで諦めると「そのタスクは
        # 不可能」と誤判定され、工程がまるごと計画から消える。
        if not walkable(start[0], start[1]):
            neighbours = [(start[0]+dx, start[1]+dy) for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]
                          if walkable(start[0]+dx, start[1]+dy)]
            if not neighbours:
                return None
            start = min(neighbours, key=lambda p: abs(p[0]-goal[0]) + abs(p[1]-goal[1]))

        original_goal = goal
        if not walkable(goal[0], goal[1]):
            adjacents = []
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = goal[0]+dx, goal[1]+dy
                if walkable(nx, ny):
                    adjacents.append((nx, ny))
            
            if not adjacents:
                return None
            
            goal = min(adjacents, key=lambda p: abs(p[0]-start[0]) + abs(p[1]-start[1]))

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0, start))
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return g_score[current]
            cx, cy = current
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = cx+dx, cy+dy
                if not walkable(nx, ny):
                    continue
                neighbor = (nx, ny)
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    def print_task_costs(self, env):
        tasks_all = []
        for order_tuple in env.order.current_orders:
            goal_obj = order_tuple[0]
            name = getattr(goal_obj, 'full_name', '').lower()
            ingredients = []
            for ing in ALL_INGREDIENTS:
                if ing in name:
                    ingredients.append(ing)
            soup_name = '-'.join(ingredients) + ' soup' if ingredients else name
            order_tasks = []
            for ing in ingredients:
                order_tasks.append(('chop', ing))
            if ingredients:
                order_tasks.append(('cook', soup_name))
                order_tasks.append(('serve', soup_name))
            tasks_all.append(order_tasks)

        # print("=== 現在のレシピから生成されたタスク列 (CSPAgent) ===")
        # for i, tasks in enumerate(tasks_all):
        #     print(f"レシピ{i+1}:")
        #     for t in tasks:
        #         print("  ", t)
        # print("=====================================")

        def get_adjacent_walkables(pos_list):
            width = env.world_width
            height = env.world_height
            grid = env.to_grid
            free = []
            for x, y in pos_list:
                for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < width and 0 <= ny < height and grid[nx][ny] == 1:
                        free.append((nx, ny))
            return list(set(free))

        tile_map = {
            "lettuce": "FreshLettuceTile",
            "onion": "FreshOnionTile",
            "tomato": "FreshTomatoTile"
        }

        special_places = [(2,3), (2,4), (2,5)]
        pot_places = [(3,5), (4,5), (5,5)]
        plate_pos = (6,6)
        delivery_pos = (6,3)

        # print("=== タスクグラフ（ノード, コスト） (CSPAgent) ===")
        # for order_idx, tasks in enumerate(tasks_all):
        #     for verb, obj in tasks:
        #         if verb == 'chop':
        #             ing_pos = env.get_pos_by_obj_gs(gs=tile_map[obj])
        #             cutboard_pos = env.get_pos_by_obj_gs(gs="Cutboard")
        #             ing_adj = get_adjacent_walkables(ing_pos)
        #             cut_adj = get_adjacent_walkables(cutboard_pos)
        #             target_adj = get_adjacent_walkables([special_places[order_idx % len(special_places)]])

        #             min_total = None
        #             best = None
        #             for s in ing_adj:
        #                 for m in cut_adj:
        #                     for e in target_adj:
        #                         d1 = self.astar_distance(env, s, m)
        #                         d2 = self.astar_distance(env, m, e)
        #                         if d1 is None or d2 is None:
        #                             continue
        #                         total = d1 + d2
        #                         if (min_total is None) or (total < min_total):
        #                             min_total = total
        #                             best = (s, m, e)
        #             if min_total is None:
        #                 # print((verb, obj, order_idx), ": 経路なし")
        #                 pass
        #             else:
        #                 base = min_total + 8 + 1 + 1
        #                 cost = base * self.frames_per_action
        #                 # print((verb, obj, order_idx), ":", cost)
        #         elif verb == 'cook':
        #             sp = special_places[order_idx % len(special_places)]
        #             pot = pot_places[order_idx % len(pot_places)]
        #             d = self.astar_distance(env, sp, pot)
        #             if d is None:
        #                 # print((verb, obj, order_idx), ": 経路なし")
        #                 pass
        #             else:
        #                 base = d + 2
        #                 cost = base * self.frames_per_action
        #                 # print((verb, obj, order_idx), ":", cost)
        #         elif verb == 'serve':
        #             pot = pot_places[order_idx % len(pot_places)]
        #             d1 = self.astar_distance(env, plate_pos, pot)
        #             d2 = self.astar_distance(env, pot, delivery_pos)
        #             if d1 is None or d2 is None:
        #                 # print((verb, obj, order_idx), ": 経路なし")
        #                 pass
        #             else:
        #                 base = d1 + d2 + 3
        #                 cost = base * self.frames_per_action
        #                 # print((verb, obj, order_idx), ":", cost)
        # print("===============================")