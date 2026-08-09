import random
import time
from dataclasses import dataclass
from copy import deepcopy
from ortools.sat.python import cp_model
from .csp.model import CSPModel
from .csp.solver import solve as solve_csp
from .TaskAgent import TaskAgent
from gym_cooking.utils.config import COOKING_TIME_SECONDS


@dataclass
class VirtualHumanState:
    current_time: int
    current_pos: tuple[int, int]
    remaining_task_ids: set

class CSPAgent:
    """
    CSP(制約充足問題)ベースのエージェント
    """
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
        self.stall_threshold = 8
        self.stall_counts = {0: 0, 1: 0} if self.sc_2agent else 0
        self.active_order_entries = []
        self.next_order_uid = 0
        self.counter_policy_by_order = {}
        self.order_display_labels = []
        self.carry_task_by_agent = {0: None, 1: None} if self.sc_2agent else None
        
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
        self.use_predicted_human_model = False
        self.predicted_human_tasks = []
        self.human_counterpart_mode = False
        # CSP が実際に操作するプレイヤー番号 (0 or 1)。
        # sc_2agent=True かつ human_counterpart_mode=True のとき有効。
        # play_main.py が ai_idx を設定する。
        self.own_agent_idx = 0

        print("[CSPAgent] 初期化完了")

    def _mark_reschedule_needed(self, reason):
        if not self.no_reschedule:
            self.pending_reschedule_reason = reason

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

    def _find_schedule_index_by_fixed_id(self, schedule, fixed_task_id):
        if not schedule:
            return None
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
        verb_priority = {'chop': 0, 'cook': 1, 'serve': 2}
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
            print(f"[SkipBudget] 保留指示数: {len(pending_instr)}")

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

                    fixed_task_id = self._extract_instruction_fixed_task_id(pending)
                    if fixed_task_id is None:
                        continue
                    matched_idx = task_index_by_fixed_id.get(fixed_task_id)
                    if matched_idx is None:
                        pending['status'] = 'done'
                        continue
                    if tasks[matched_idx].get('id') in self.completed_task_ids:
                        pending['status'] = 'done'
                        continue

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

                    budget_bound = max(0, remaining)
                    overage = max(0, -remaining)
                    if counts_vars:
                        model.Add(sum(counts_vars) <= budget_bound)
                    if overage > 0:
                        print(f"[SkipBudget] 警告: fixed_id={fixed_task_id} 超過中(超過量={overage}), 最善優先化")
                    else:
                        print(f"[SkipBudget] 制約: fixed_id={fixed_task_id} 残りbudget={remaining}, 対象前タスク≤{budget_bound} (非依存{len(counts_vars)}個)")
                    pending['skip_budget_constraint_applied'] = True
                except Exception as e:
                    print(f"[SkipBudget] 指示制約で例外: {e}")
        except Exception as e:
            print(f"[SkipBudget] skip_budget制約の追加に失敗: {e}")

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
            verb_prio = {'chop': 0, 'cook': 1, 'serve': 2}
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
            print(
                f"[SkipBudget] 対象前完了: {completed_tid} ({completed_secs:.1f}s) "
                f"初期d={initial_budget} 残り={new_remaining} 超過={overage} 累計={len(log_list)}件"
            )
            if overage > 0:
                print("[SkipBudget] 警告: skip_budget超過 → 次回再計画で即時優先化")
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
            if fixed_task_id is None:
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
            if fixed_task_id is None:
                continue

            deadline_info = self._classify_instruction_deadline(pending, current_env_time, deadline_seconds)
            if self.skip_budget is not None:
                is_urgent = self._is_urgent_by_skip_budget(pending)
            else:
                is_urgent = (deadline_info.get('mode') == 'urgent')
            if not is_urgent:
                continue

            if not self.sc_2agent:
                target_idx = self._find_schedule_index_by_fixed_id(getattr(self, 'schedule', []), fixed_task_id)
                if target_idx is None:
                    continue
                if isinstance(self.current_task_idx, int) and self.current_task_idx == target_idx:
                    continue
                return pending, fixed_task_id, None, target_idx

            schedule_per_agent = getattr(self, 'schedule_per_agent', None)
            if not schedule_per_agent:
                continue

            for agent_idx in [0, 1]:
                schedule = schedule_per_agent.get(agent_idx, [])
                if not schedule:
                    continue
                target_idx = self._find_schedule_index_by_fixed_id(schedule, fixed_task_id)
                if target_idx is None:
                    continue
                return pending, fixed_task_id, agent_idx, target_idx

        return None

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

            if verb == 'cook':
                recipe_ings = obj.replace(' soup', '').split('-')
                if any(f"Chopped{ing.capitalize()}" in holding_name for ing in recipe_ings):
                    if any(tid[0] == 'chop' and tid[2] == order_uid for tid in added):
                        return True

        return False

        if current_task_ids and not self._get_active_task_ids():
            return "no_active_task_with_remaining_work"

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
            if not any(ing in name for ing in ['lettuce', 'onion', 'tomato']):
                continue

            rest_time = order_tuple[1] if len(order_tuple) > 1 else None
            time_limit = order_tuple[2] if len(order_tuple) > 2 else None

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
            else:
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
            elif has_plate(obj):
                inv_plates_ings.append(obj_foods)
                    
        remaining_tids = set()
        
        for order in current_orders:
            order_uid = order['order']
            order_name = order['name']
            if ' soup' not in order_name: continue
            
            raw_parts = order_name.replace(' soup', '').split('-')
            req_set = set(raw_parts)
            
            needs_chop = list(raw_parts)
            needs_cook = True
            needs_serve = True
            
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
            else:
                # 2. 鍋にあるか？
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
                
            soup_name = "-".join([i.replace('Chopped', '').lower() for i in raw_parts]) + " soup"
            if needs_cook:
                remaining_tids.add(('cook', soup_name, order_uid))
            if needs_serve:
                remaining_tids.add(('serve', soup_name, order_uid))
                
        return remaining_tids

    def _make_fixed_task_id(self, verb, obj, order_uid):
        return ("task", str(verb), str(obj), int(order_uid))

    def get_instruction_candidates(self, env):
        """現在の環境で未実行のタスク候補を固定 ID 付きで返す。"""
        current_orders = self._build_order_tasks(env)
        remaining_tids = self.get_remaining_tids(env, current_orders)

        verb_priority = {'chop': 0, 'cook': 1, 'serve': 2}
        sorted_tids = sorted(
            remaining_tids,
            key=lambda tid: (tid[2], verb_priority.get(tid[0], 9), tid[1])
        )

        candidates = []
        for verb, obj, order_uid in sorted_tids:
            display = f"{verb}_{obj.replace(' ', '').replace('-', '_')} (order:{order_uid})"
            fixed_task_id = self._make_fixed_task_id(verb, obj, order_uid)
            payload = {
                'fixed_task_id': fixed_task_id,
                'verb': verb,
                'obj': obj,
                'order_uid': order_uid,
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
        except Exception as e:
            print(f"[CSPAgent] high_level_infer error: {e}")

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
            elif verb == 'cook':
                for ing in obj.replace(' soup', '').split('-'):
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
        if not holding_name:
            if self.sc_2agent:
                self.carry_task_by_agent[agent_idx] = None
            else:
                self.carry_task_by_agent = None
            return scheduled_task

        if 'Plate' in holding_name and 'Cooked' in holding_name:
            cooked_parts = []
            for part in holding_name.split('-'):
                if part.startswith('Cooked'):
                    cooked_parts.append(part.replace('Cooked', '').lower())
            if cooked_parts:
                cooked_parts.sort()
                return {
                    'id': ('serve', f"{'-'.join(cooked_parts)} soup", -1),
                    'res': ('delivery', None),
                }

        chopped_combo_parts = []
        if 'Plate' not in holding_name and '-' in holding_name:
            parts = holding_name.split('-')
            if parts and all(part.startswith('Chopped') for part in parts):
                chopped_combo_parts = sorted(part.replace('Chopped', '').lower() for part in parts)

        carried_ing = None
        if '-' not in holding_name:
            if holding_name.startswith('Fresh'):
                carried_ing = holding_name.replace('Fresh', '').lower()
            elif holding_name.startswith('Chopped'):
                carried_ing = holding_name.replace('Chopped', '').lower()

        carry_task = self.carry_task_by_agent[agent_idx] if self.sc_2agent else self.carry_task_by_agent

        def matches_single_chopped(task):
            if carried_ing is None or not holding_name.startswith('Chopped') or task is None:
                return False
            verb, obj, _ = task['id']
            if verb != 'cook':
                return False
            needed_parts = obj.replace(' soup', '').split('-')
            return carried_ing in needed_parts

        if scheduled_task:
            verb, obj, _ = scheduled_task['id']
            if verb == 'cook' and chopped_combo_parts:
                scheduled_parts = sorted(obj.replace(' soup', '').split('-'))
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
            if verb == 'cook' and chopped_combo_parts:
                carry_parts = sorted(obj.replace(' soup', '').split('-'))
                if carry_parts == chopped_combo_parts:
                    return deepcopy(carry_task)
            if matches_single_chopped(carry_task):
                return deepcopy(carry_task)
            food_names = (f"Fresh{obj.capitalize()}", f"Chopped{obj.capitalize()}")
            if verb == 'chop' and any(food_name in holding_name for food_name in food_names):
                return deepcopy(carry_task)

        if chopped_combo_parts:
            assigned_counter = None
            if carry_task:
                assigned_counter = carry_task.get('assigned_counter')
            if assigned_counter is None and scheduled_task:
                assigned_counter = scheduled_task.get('assigned_counter')
            return {
                'id': ('cook', f"{'-'.join(chopped_combo_parts)} soup", -1),
                'res': ('pot', None),
                'assigned_counter': assigned_counter,
            }

        if carried_ing:
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

        if not hasattr(self, 'prev_task_ids'):
            self.prev_task_ids = set()

        added = current_task_ids - self.prev_task_ids
        removed = self.prev_task_ids - current_task_ids

        if self._should_defer_holding_reschedule(env, added, removed):
            added = set()
            removed = set()

        reschedule_reason = None if self.no_reschedule and self.initialized else self._get_reschedule_reason(current_task_ids, added, removed)

        # 必要なタイミングだけリスケジュールする
        if reschedule_reason is not None:
            # 削除された cook タスクのみ物理完了として扱う。
            # chop は一時的に食材が存在して current_task_ids から消えても、
            # 後で消費されて再度必要になるため completed_task_ids へ積まない。
            if removed and self.initialized:
                physically_done = {t for t in removed if t[0] == 'cook'}
                if physically_done:
                    print(f"  [完了検知] 物理完了タスク: {physically_done}")
                    self.completed_task_ids |= physically_done
                    self._mark_reschedule_needed("physical_completion")

            if self.initialized: # 初回以外なら差分を表示
                print(f"\n[タスク更新] 時間: {env.time}")
                print(f"  [再計算理由] {reschedule_reason}")
                if added:
                    print(f"  (+) 追加: {added}")
                if removed:
                    print(f"  (-) 削除: {removed}")
                print("  -> スケジュール再計算中...")

            # ① リスケジュール前に「現在実行中のタスクID」を保存する（Bug 1&5 対策）
            in_progress_tasks = {}  # agent_idx -> task dict
            if self.sc_2agent and hasattr(self, 'schedule_per_agent'):
                for aidx in [0, 1]:
                    sc = self.schedule_per_agent.get(aidx, [])
                    t_idx = self.current_task_idx.get(aidx, 0) if isinstance(self.current_task_idx, dict) else 0
                    if t_idx < len(sc):
                        in_progress_tasks[aidx] = deepcopy(sc[t_idx])
                        print(f"  [継続確認] AI{aidx} 実行中タスク: {sc[t_idx]['id']}")

            # 簡易CSPスケジューリング
            try:
                start_time = time.time()
                self.schedule = self.solve_csp_scheduling(env, orders=current_orders)
                elapsed_time = time.time() - start_time
                print(f"[CSPAgent] スケジューリング時間: {elapsed_time:.4f} 秒")

                self._print_schedule(self.schedule)

                # ② 新スケジュール内で「実行中タスク」を探してインデックスを復元する
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
                                    print(f"  [継続] AI{aidx}: {in_prog_tid} → 新スケジュール idx={i} から再開")
                                    found = True
                                    break
                            if not found:
                                holding = getattr(env.agents[aidx], 'holding', None)
                                holding_name = getattr(holding, 'full_name', None) if holding is not None else None
                                if in_prog_tid[0] == 'chop' and holding_name and f"{in_prog_tid[1].capitalize()}" in holding_name:
                                    self.carry_task_by_agent[aidx] = deepcopy(in_prog_task)
                                    print(f"  [保持継続] AI{aidx}: {in_prog_tid} を carry task として継続")
                                print(f"  [スキップ] AI{aidx}: {in_prog_tid} が新スケジュールに存在しない → idx=0から開始")
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
                                print(f"[CSPAgent] 2エージェント即時切替: fixed_id={fixed_task_id} agent={agent_idx} idx={target_idx}")
                else:
                    preempt_target = self._get_immediate_instruction_preempt(env)
                    if preempt_target is not None:
                        pending, fixed_task_id, target_idx = preempt_target
                        if self.current_task_idx != target_idx:
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
                            print(f"[CSPAgent] 期限0指示により即時切替: fixed_id={fixed_task_id} idx={target_idx}")

                self.pending_reschedule_reason = None
                if self.sc_2agent:
                    self.stall_counts = {0: 0, 1: 0}
                else:
                    self.stall_counts = 0

            except Exception as e:
                print(f"[CSPAgent] CSPスケジュール中に例外: {e}")
                import traceback
                traceback.print_exc()

            self.initialized = True

        self.prev_task_ids = current_task_ids

        # ====== スケジュール実行 ======
        if not self.sc_2agent:
            # 単一エージェントモード
            if not hasattr(self, 'schedule') or not self.schedule or self.current_task_idx >= len(self.schedule):
                return (0, 0), "タスクなし"

            scheduled_task = self.schedule[self.current_task_idx]
            task = self._get_carry_override_task(env, 0, scheduled_task)
            scheduled_tid = scheduled_task['id']
            tid = task['id']
            verb, obj, order_uid = tid
            res = task['res'] 

            task_name = None
            if verb == 'chop':
                task_name = f"chop_{obj}"
                self.task_agent.assigned_counter = task.get('assigned_counter')
            elif verb == 'cook':
                parts = obj.replace(' soup', '').split('-')
                task_name = f"cook_{'_'.join(parts)}"
                self.task_agent.assigned_counter = task.get('assigned_counter')
            elif verb == 'serve':
                parts = obj.replace(' soup', '').split('-')
                task_name = f"serve_{'_'.join(parts)}"
                self.task_agent.assigned_counter = None
            
            if task_name:
                self.task_agent.task_name = task_name
                action, reason = self.task_agent(env)
                hold_before = self._hold_before_for_log(env)
                hold_hint = self._hold_hint_for_log(hold_before, reason)
                print(
                    f"[ACTION] AI task={task_name} tid={tid} action={action} reason='{reason}' "
                    f"hold_before={hold_before} hold_hint={hold_hint} counter={self.task_agent.assigned_counter}"
                )
                
                if "Done" in reason or "done" in reason or "完了" in reason:
                    print(f"[CSPAgent] タスク {task_name} 完了。次へ移動。")
                    if verb == 'serve':
                        print(f"[SERVE] AI task={task_name} tid={tid} completed=True")
                    self.completed_task_ids.add(tid)
                    self._update_skip_budget_on_completion(tid, 0, scheduled_task.get('dur', 0))
                    if tid == scheduled_tid:
                        self.current_task_idx += 1
                    self._mark_reschedule_needed("task_completed_single")
                    self.carry_task_by_agent = None
                    self.task_agent.assigned_cutboard = None
                    self.task_agent.assigned_pot = None
                    self.task_agent.assigned_plate = None
                    self.task_agent.assigned_serve_loc = None
                    self.task_agent.assigned_counter = None
                
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
                    actions[f"ai_{agent_idx}"] = (0, 0)
                    reasons.append(f"AI{agent_idx}:Idle")
                    continue
                
                scheduled_task = sc[t_idx]
                task = self._get_carry_override_task(env, agent_idx, scheduled_task)
                scheduled_tid = scheduled_task['id']
                tid = task['id']
                verb, obj, order_uid = tid
                
                import copy
                e_agent = copy.copy(env)
                e_agent.agent_idx = agent_idx
                # 相手の現在位置を dynamic_obstacles として先に定義（can_start前に必要）
                other_pos = env.agents[1 - agent_idx].location
                dynamic_obstacles = {other_pos}

                # --- 先行する依存タスクが終わっているか（フライング実行エラーの防止） ---
                # スケジュール上に存在しないchopタスクは「すでに食材がある」ため完了済みとみなす
                all_scheduled_ids = {t['id'] for agent_sc in self.schedule_per_agent.values() for t in agent_sc}

                can_start = True
                if verb == 'cook':
                    parts = obj.replace(' soup', '').split('-')
                    for p in parts:
                        req_tid = ('chop', p.strip(), order_uid)
                        if req_tid not in self.completed_task_ids and req_tid in all_scheduled_ids:
                            can_start = False
                            break
                elif verb == 'serve':
                    # serve は皿の先取りができるので、cook 完了前でも TaskAgent に進める。
                    # 鍋前待機や実際の取得タイミングは process_serve_task 側で判定する。
                    can_start = True
                        
                if not can_start:
                    missing_deps = []
                    if verb == 'cook':
                        parts = obj.replace(' soup', '').split('-')
                        missing_deps = [('chop', p.strip(), order_uid) for p in parts if ('chop', p.strip(), order_uid) not in self.completed_task_ids]
                    elif verb == 'serve':
                        if ('cook', obj, order_uid) not in self.completed_task_ids:
                            missing_deps = [('cook', obj, order_uid)]
                    print(f"[DEBUG] AI{agent_idx} 待機中: {verb} '{obj}' の前提タスク未完了 -> {missing_deps}")
                    print(f"[DEBUG]   完了済み: {self.completed_task_ids}")
                    
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
                ta.assigned_counter = task.get('assigned_counter')
                
                if verb == 'chop':
                    task_name = f"chop_{obj}"
                elif verb == 'cook':
                    parts = obj.replace(' soup', '').split('-')
                    task_name = f"cook_{'_'.join(parts)}"
                elif verb == 'serve':
                    parts = obj.replace(' soup', '').split('-')
                    task_name = f"serve_{'_'.join(parts)}"
                


                if task_name:
                    ta.task_name = task_name
                    if verb == 'cook':
                        print(f"[DEBUG] AI{agent_idx} cook_task_start tid={tid} assigned_counter={ta.assigned_counter} completed={sorted(self.completed_task_ids)} hold={getattr(e_agent, 'hold', None)}")
                    
                    # ユーザーの要望「同時に動かすのではなく交互に」
                    if agent_idx == self.turn:
                        action, reason = ta(e_agent, dynamic_obstacles=dynamic_obstacles)
                    else:
                        action, reason = (0, 0), "待機(相手のターン)"

                    hold_before = self._hold_before_for_log(e_agent)
                    hold_hint = self._hold_hint_for_log(hold_before, reason)

                    print(
                        f"[ACTION] AI{agent_idx} task={task_name} tid={tid} action={action} reason='{reason}' "
                        f"hold_before={hold_before} hold_hint={hold_hint} counter={ta.assigned_counter} turn={self.turn}"
                    )
                    
                    if action == (0, 0) and reason not in ("待機(相手のターン)",):
                        print(f"[DEBUG] AI{agent_idx} 停止: task={task_name} reason='{reason}' counter={ta.assigned_counter} hold_before={hold_before} hold_hint={hold_hint}")

                    self._update_stall_state(agent_idx, action, reason)

                    
                    if reason.endswith("(Done)") or reason.endswith("(完了)"):
                        print(f"[CSPAgent] AI{agent_idx} タスク {task_name} 完了。")
                        if verb == 'serve':
                            print(f"[SERVE] AI{agent_idx} task={task_name} tid={tid} completed=True")
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
                        
                    actions[f"ai_{agent_idx}"] = action
                    reasons.append(reason)
                else:
                    actions[f"ai_{agent_idx}"] = (0, 0)
                    reasons.append("アイドル")

            # ターンを入れ替え（次フレームはもう一方を動かす）
            self.turn = 1 - self.turn
                
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
            benefits[name] = 100 if t['verb'] == 'serve' else 0

        model.add_linear_le(durations, self.budget_frames)

        name_by_task_id = {}
        for name, t in zip(var_names, tasks):
            name_by_task_id[id(t)] = name
        for o in orders:
            chops = [t for t in o['tasks'] if t['verb'] == 'chop']
            cooks = [t for t in o['tasks'] if t['verb'] == 'cook']
            serves = [t for t in o['tasks'] if t['verb'] == 'serve']

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

        model.maximize_linear(benefits)

        result = solve_csp(model, time_limit=5.0)

        selected = []
        if result.status_name in ("OPTIMAL", "FEASIBLE"):
            for name, t in zip(var_names, tasks):
                if result.solution.get(name, 0) == 1:
                    selected.append(t)
        return selected

    def _print_selection(self, selected_tasks):
        print("\n=== OR-Tools 選択結果（予算内最大化） ===")
        total = 0
        for t in selected_tasks:
            verb = t['verb']; obj = t['obj']; order = t.get('display_order', t.get('slot_idx', t['order']))
            dur = t['dur']
            total += dur
            print(f"選択: {verb} {obj} (注文{order+1}) 所要={dur}")
        print(f"合計投入フレーム(選択分): {total}")
        print("===================================\n")

    # ============ CSP（選択問題 A解釈） ============ 
    def _get_resources(self, env):
        cutboards = env.get_pos_by_obj_gs(gs="Cutboard")
        pots = env.get_pos_by_obj_gs(gs="Pot")
        deliveries = env.get_pos_by_obj_gs(gs="Delivery")
        plates = env.get_pos_by_obj_gs(gs="Plate") 
        if not plates:
            plates = env.get_pos_by_obj_gs(gs="PlateTile") 
        counters = env.get_pos_by_obj_gs(gs="Counter")

        return {
            'cutboards': cutboards,
            'pots': pots,
            'delivery': deliveries[0] if deliveries else (0,0),
            'plate': plates[0] if plates else (0,0),
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
            verb = t['verb']
            obj = t['obj']
            order_idx = t.get('slot_idx', t['order'])

            if verb == 'chop':
                tile_map = {"lettuce": "FreshLettuceTile", "onion": "FreshOnionTile", "tomato": "FreshTomatoTile"}
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
                needed_ings = obj.replace(' soup', '').split('-')
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

            elif verb == 'serve':
                pots = resources['pots']
                pot = pots[order_idx % len(pots)] if pots else default_start_pos
                plate = resources['plate']
                delivery = resources['delivery']

                t['start_pos'] = plate
                t['end_pos'] = delivery
                t['fixed_res'] = ('pot', pot)

    def _task_is_available_in_virtual_state(self, task, remaining_task_ids):
        verb, obj, order_uid = task['id']
        if verb == 'chop':
            return True
        if verb == 'cook':
            needed_ings = obj.replace(' soup', '').split('-')
            return all(('chop', ing, order_uid) not in remaining_task_ids for ing in needed_ings)
        if verb == 'serve':
            return ('cook', obj, order_uid) not in remaining_task_ids
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

    def _predict_human_greedy_tasks(self, env, tasks, human_start_pos, limit=None):
        tasks_by_id = {task['id']: task for task in tasks}
        state = VirtualHumanState(
            current_time=0,
            current_pos=human_start_pos,
            remaining_task_ids={task['id'] for task in tasks},
        )
        predicted = []
        verb_priority = {'chop': 0, 'cook': 1, 'serve': 2}

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
            tile_map = {"lettuce": "FreshLettuceTile", "onion": "FreshOnionTile", "tomato": "FreshTomatoTile"}
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

            needed_ings = obj.replace(' soup', '').split('-')
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

        elif verb == 'serve':
            plate_pos = resources['plate']
            pot_pos_list = resources['pots']
            delivery_pos = resources['delivery']
            
            if not pot_pos_list: return None
            pot_pos = pot_pos_list[order_idx % len(pot_pos_list)]
            
            d1 = self.astar_distance(env, plate_pos, pot_pos)
            d2 = self.astar_distance(env, pot_pos, delivery_pos)
            
            if d1 is None or d2 is None: return None
            return int(d1 + d2 + 3)
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
        self._get_counter_policy_entry(order_uid)['counter'] = counter_pos

    def _get_counter_food_names(self, env, counter_pos):
        counter_obj = env.pos_obj.get(counter_pos)
        if counter_obj is None or not hasattr(counter_obj, 'contents'):
            return set()

        return {
            getattr(food, 'full_name', getattr(food, 'name', ''))
            for food in counter_obj.contents
            if getattr(food, 'full_name', getattr(food, 'name', '')) != 'Plate'
        }

    def _log_counter_policy(self, order_uid, action, counter_pos, details=""):
        entry = self._get_counter_policy_entry(order_uid)
        state = (action, counter_pos, details)
        if entry['last_state'] == state:
            return
        entry['last_state'] = state

        suffix = f" {details}" if details else ""
        print(f"[CounterPolicy] order_uid={order_uid} action={action} counter={counter_pos}{suffix}")

    def _resolve_assigned_counter(self, env, order_uid):
        assigned_counter = self._get_assigned_counter(order_uid)
        if assigned_counter is None:
            return None, False

        counter_food_names = self._get_counter_food_names(env, assigned_counter)
        details = f"foods={sorted(counter_food_names)}" if counter_food_names else "reason=fixed"
        self._log_counter_policy(order_uid, "fixed", assigned_counter, details)
        return assigned_counter, False

    def _calculate_dynamic_merge_point(self, env, ings_lower, order_idx, pot_locs, used_counters):
        ing = ings_lower[0] if ings_lower else None
        tile_map = {"lettuce": "FreshLettuceTile", "onion": "FreshOnionTile", "tomato": "FreshTomatoTile"}
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
        empty_counters = []
        for c in counters:
            if env.pos_obj.get(c) is None and c not in used_counters:
                empty_counters.append(c)
                
        if not empty_counters:
            # 完全に空いているカウンターがない場合は使用済みでも候補にする
            for c in counters:
                if env.pos_obj.get(c) is None:
                    empty_counters.append(c)
            if not empty_counters:
                empty_counters = counters 
            
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
            best_counter = counters[order_idx % len(counters)] if counters else None
            
        return best_counter

    def _build_order_tasks(self, env):
        available_chopped = {}
        available_chopped_by_pos = {}
        pot_states = []
        held_object_ids = {
            id(getattr(agent, 'holding', None))
            for agent in getattr(env, 'agents', [])
            if getattr(agent, 'holding', None) is not None
        }

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

        def consume_chopped(ingredient_name, assigned_counter, reserved_counters):
            preferred_positions = []
            if assigned_counter is not None:
                preferred_positions.append(assigned_counter)
            else:
                preferred_positions.extend(
                    pos for pos in available_chopped_by_pos.keys()
                    if pos not in reserved_counters
                )

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
                return True

            return False

        def choose_counter_for_order(ingredient_names, current_counter=None):
            best_counter = None
            best_score = None

            for pos, pos_stock in available_chopped_by_pos.items():
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

        for obj in all_objects:
            if id(obj) in held_object_ids:
                continue
            if type(obj).__name__ == 'Object':
                if obj.location in pot_locs:
                    c_names = sorted([c.name for c in obj.contents])
                    pot_states.append({'names': c_names, 'obj': obj, 'used': False})
                else:
                    if obj.location not in cutboard_locs:
                        register_chopped_item(obj, obj.location)

        resources = self._get_resources(env)
        orders = []
        current_orders = env.order.current_orders if hasattr(env, 'order') and hasattr(env.order, 'current_orders') else []
        order_uids = self._refresh_active_order_uids(current_orders)
        self.order_display_labels = [
            (order_uid + 1) if order_uid is not None else (order_idx + 1)
            for order_idx, order_uid in enumerate(order_uids)
        ]

        used_counters = [
            entry['counter']
            for entry in self.counter_policy_by_order.values()
            if entry.get('counter') is not None
        ]
        assigned_counters_display_map = {}

        for order_idx, order_tuple in enumerate(current_orders):
            goal = order_tuple[0]
            name = getattr(goal, 'full_name', '').lower()
            ings_lower = [ing for ing in ['lettuce', 'onion', 'tomato'] if ing in name]
            if not ings_lower:
                continue

            order_uid = order_uids[order_idx]
            display_order = order_uid
            ings_cap = [ing.capitalize() for ing in ings_lower]

            assigned_counter, released_for_cook = self._resolve_assigned_counter(
                env,
                order_uid,
            )

            if assigned_counter is not None and env.pos_obj.get(assigned_counter) is None:
                reassigned_counter = choose_counter_for_order(ings_cap, current_counter=assigned_counter)
                if reassigned_counter is not None and reassigned_counter != assigned_counter:
                    self._set_assigned_counter(order_uid, reassigned_counter)
                    self._log_counter_policy(order_uid, "reassign", reassigned_counter, f"reason=counter_empty prev={assigned_counter}")
                    if assigned_counter in used_counters:
                        used_counters.remove(assigned_counter)
                    assigned_counter = reassigned_counter
                elif reassigned_counter is None:
                    if assigned_counter in used_counters:
                        used_counters.remove(assigned_counter)
                    self._set_assigned_counter(order_uid, None)
                    self._log_counter_policy(order_uid, "release", assigned_counter, "reason=counter_empty")
                    assigned_counter = None

            if assigned_counter is None and not released_for_cook:
                assigned_counter = self._calculate_dynamic_merge_point(env, ings_lower, order_idx, pot_locs, used_counters)
                if assigned_counter is not None:
                    self._set_assigned_counter(order_uid, assigned_counter)
                    if assigned_counter not in used_counters:
                        used_counters.append(assigned_counter)
                    self._log_counter_policy(order_uid, "assign", assigned_counter, "reason=new_order")

            assigned_counters_display_map[display_order + 1] = assigned_counter

            soup_name = '-'.join(ings_lower) + ' soup'
            tasks = []
            sorted_ings = sorted(ings_cap)
            cook_needed = True

            for ps in pot_states:
                if not ps['used'] and ps['names'] == sorted_ings:
                    ps['used'] = True
                    cook_needed = False
                    break

            for ing in ings_cap:
                if not cook_needed:
                    continue
                reserved_other_counters = {
                    counter for counter in used_counters
                    if counter is not None and counter != assigned_counter
                }
                if consume_chopped(ing, assigned_counter, reserved_other_counters):
                    continue
                dur = self._task_duration_frames(env, 'chop', ing.lower(), order_idx, assigned_counter)
                if dur is None:
                    continue
                tasks.append({
                    'id': ('chop', ing.lower(), order_uid),
                    'verb': 'chop', 'obj': ing.lower(), 'order': order_uid,
                    'slot_idx': order_idx,
                    'display_order': display_order,
                    'dur': dur,
                    'res_candidates': [('cutboard', r) for r in resources['cutboards']],
                    'assigned_counter': assigned_counter
                })

            if cook_needed:
                dur = self._task_duration_frames(env, 'cook', soup_name, order_idx)
                if dur is not None:
                    tasks.append({
                        'id': ('cook', soup_name, order_uid),
                        'verb': 'cook', 'obj': soup_name, 'order': order_uid,
                        'slot_idx': order_idx,
                        'display_order': display_order,
                        'dur': dur,
                        'res_candidates': [('pot', r) for r in resources['pots']],
                        'assigned_counter': assigned_counter
                    })

            dur = self._task_duration_frames(env, 'serve', soup_name, order_idx)
            if dur is not None:
                tasks.append({
                    'id': ('serve', soup_name, order_uid),
                    'verb': 'serve', 'obj': soup_name, 'order': order_uid,
                    'slot_idx': order_idx,
                    'display_order': display_order,
                    'dur': dur,
                    'res_candidates': [],
                    'assigned_counter': assigned_counter
                })

            orders.append({'order': order_uid, 'display_order': display_order, 'name': soup_name, 'ingredients': ings_lower, 'tasks': tasks})

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
            if pending_instr:
                print(f"[CSPAgent] 適用可能な保留指示数: {len(pending_instr)}")

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
                        print(f"[CSPAgent] 指示の固定 ID がないためスキップ: {payload}")
                        pending['status'] = 'canceled'
                        pending['deadline_constraint_applied'] = True
                        continue

                    matched_idx = task_index_by_fixed_id.get(fixed_task_id)
                    if matched_idx is None:
                        print(f"[CSPAgent] 対応するタスクがなくなったため完了扱い: fixed_id={fixed_task_id}")
                        pending['status'] = 'done'
                        pending['deadline_constraint_applied'] = True
                        continue

                    target_task = tasks[matched_idx]
                    target_tid = target_task.get('id')
                    if target_tid in self.completed_task_ids:
                        print(f"[CSPAgent] 指示対象タスクは完了済み: fixed_id={fixed_task_id} tid={target_tid}")
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
                        print(f"[CSPAgent] 指示の環境時刻情報がないためスキップ: {pending}")
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
                        print(f"[CSPAgent] 指示を最優先で実行: fixed_id={fixed_task_id} start <= {urgent_frame + 1} (env_time={current_env_time})")
                    else:
                        remaining_frames = int(remaining_deadline_seconds * self.fps)
                        bound = accepted_frame + remaining_frames
                        model.Add(start_var <= bound)
                        print(f"[CSPAgent] 期限制約を追加: fixed_id={fixed_task_id} start <= {bound} (env_time={current_env_time}, remaining_seconds={remaining_deadline_seconds:.3f})")
                    pending['deadline_constraint_applied'] = True
                except Exception as e:
                    print(f"[CSPAgent] 保留指示反映中に例外: {e}")
        except Exception as e:
            print(f"[CSPAgent] 指示による締切制約の追加で失敗: {e}")

    def solve_csp_scheduling(self, env, orders):
        """
        OR-Tools CP-SAT を用いたスケジューリング（移動コスト込み）。
        Circuit制約を用いて順序依存のセットアップ時間（移動時間）を正確にモデル化する。
        """
        print(f"[CSPAgent] CSPスケジューリング開始 ({len(orders)} 注文)...")
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
        if num_tasks == 0:
            print("[CSPAgent] スケジュール対象タスクがありません。")
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
        
        # リソース位置の特定と固定 (Fixed Position)
        resources = self._get_resources(env)
        self._annotate_task_geometry(env, tasks, agent_pos)

        predicted_human_windows = {}
        predicted_human_task_ids = set()
        if self.sc_2agent and self.use_predicted_human_model:
            self.predicted_human_tasks = self._predict_human_greedy_tasks(
                env,
                tasks,
                human_start_pos=agent_pos,
                limit=1,
            )
            predicted_human_windows = {
                entry['id']: (entry['start'], entry['end'])
                for entry in self.predicted_human_tasks
            }
            predicted_human_task_ids = set(predicted_human_windows)
            if self.predicted_human_tasks:
                first_task = self.predicted_human_tasks[0]
                print(
                    f"[HumanModel] 予測人間タスク: {first_task['id']} "
                    f"start={first_task['start']} end={first_task['end']}"
                )

        # 2. 距離行列の作成 (A* distance)
        node_num = num_tasks + 2 if self.sc_2agent else num_tasks + 1
        all_nodes = list(range(node_num))
        start_node = num_tasks # ダミーノード
        agent1_start_node = num_tasks + 1 if self.sc_2agent else None
        
        dist_matrix = {} # (from_idx, to_idx) -> distance
        print("[CSPAgent] 距離行列を計算中...")
        
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
        print("--- 距離行列サンプル ---")
        sample_chop = next((i for i, t in enumerate(tasks) if t['verb'] == 'chop'), None)
        sample_cook = next((i for i, t in enumerate(tasks) if t['verb'] == 'cook'), None)
        if sample_chop is not None and sample_cook is not None:
            d1 = dist_matrix.get((sample_chop, sample_cook), -1)
            d2 = dist_matrix.get((sample_cook, sample_chop), -1)
            p1 = tasks[sample_chop]['end_pos']
            p2 = tasks[sample_cook]['start_pos']
            print(f"Chop({sample_chop} @ {p1}) -> Cook({sample_cook} @ {p2}): {d1}")
            print(f"Cook({sample_cook} @ {p2}) -> Chop({sample_chop} @ {p1}): {d2}")
        print("------------------------")

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

            if t['id'] in predicted_human_windows:
                fixed_start, fixed_end = predicted_human_windows[t['id']]
                model.Add(s_var == fixed_start)
                model.Add(e_var == fixed_end)
        
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

        if self.sc_2agent:
            if predicted_human_task_ids:
                is_a1 = [None] * num_tasks
                ai_indices = []
                for i in range(num_tasks):
                    if tasks[i]['id'] in predicted_human_task_ids:
                        continue
                    ai_indices.append(i)
                    dist_from_a1 = int(dist_matrix.get((agent1_start_node, i), 0))
                    model.Add(starts[i] >= dist_from_a1)

                for idx_i, i in enumerate(ai_indices):
                    for j in ai_indices[idx_i + 1:]:
                        order_ij = model.NewBoolVar(f'order_{i}_{j}')
                        dij = int(dist_matrix.get((i, j), 0))
                        dji = int(dist_matrix.get((j, i), 0))
                        model.Add(starts[j] >= ends[i] + dij).OnlyEnforceIf(order_ij)
                        model.Add(starts[i] >= ends[j] + dji).OnlyEnforceIf(order_ij.Not())
            else:
                # is_a1[i]: True = タスクiをAI1が担当、False = AI0が担当
                is_a1 = [model.NewBoolVar(f'is_a1_{i}') for i in range(num_tasks)]
                
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
        print(f"[CSPAgent] スケジュール対象タスク数: {num_tasks}")
        
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
            elif verb == 'serve':
                order_vars = vars_by_order.get(t['order'], [])
                cooks = [v for v in order_vars if v['task']['verb'] == 'cook']
                for c in cooks:
                    # cook終了 + 実際の調理時間(config.pyのCOOKING_TIME_SECONDS)が経過してからserve可能
                    model.Add(starts[i] >= c['end'] + cooking_frames)

        # 鍋の占有制約 (Pot Usage Constraint)
        pot_usage_intervals = {}
        for order_idx, tasks_list in vars_by_order.items():
            cooks = [v for v in tasks_list if v['task']['verb'] == 'cook']
            serves = [v for v in tasks_list if v['task']['verb'] == 'serve']
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

        for pot_loc, intervals_list in pot_usage_intervals.items():
            if len(intervals_list) > 1:
                model.AddNoOverlap(intervals_list)
                print(f"[CSPAgent] 鍋 {pot_loc} の重複禁止制約を追加 ({len(intervals_list)} 注文)")

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
                print(f"[CSPAgent] まな板 {c_loc} の重複禁止制約を追加 ({len(intervals_list)} タスク)")

        # 動的制約 (Dynamic Constraints)
        if hasattr(self, 'active_constraints') and self.active_constraints:
            print(f"[CSPAgent] 動的制約を適用中: {len(self.active_constraints)}件")
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
                            for verb in ['chop', 'cook', 'serve']:
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
        model.Minimize(makespan * weight_makespan + end_sum)

        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        status_name = solver.StatusName(status)
        print(f"[CSPAgent] ソルバー状態: {status_name}")

        schedule = []
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            actual_makespan = solver.Value(makespan)
            print(f"[CSPAgent] 最適Makespan(移動込み): {actual_makespan} (評価値: {solver.ObjectiveValue()})")
            
            if not self.sc_2agent:
                # 1エージェント: 従来通りlit_mapでルートをトレース
                schedule.sort(key=lambda x: x['start'])
                print("--- 推定順序と移動時間 (詳細) ---")
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
                                print(f" -> {t['verb']} {t['obj']}")
                            current_node = j
                            found_next = True
                            visited_count += 1
                            break
                    if not found_next: break
                print("---------------------------------")
            else:
                # 2エージェント: is_a1 変数から直接エージェント割り当てを読む
                schedule_per_agent = {0: [], 1: []}
                for i in range(num_tasks):
                    t = tasks[i]
                    if predicted_human_task_ids:
                        agent_idx = 0 if t['id'] in predicted_human_task_ids else 1
                    else:
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
                print(f"[CSPAgent] ソルバー結果: INFEASIBLE (締切制約などにより解なし) — 直前のスケジュールへフォールバックします。")
                if self.sc_2agent and previous_schedule_per_agent:
                    self.schedule_per_agent = previous_schedule_per_agent
                    fallback_schedule = previous_schedule_per_agent[0] + previous_schedule_per_agent[1]
                    fallback_schedule.sort(key=lambda x: x['start'])
                    return fallback_schedule
                if previous_schedule:
                    self.schedule = previous_schedule
                    return previous_schedule
            else:
                print(f"[CSPAgent] 解が見つかりませんでした。 状態={status_name}")
            
        return schedule

    def solve_csp_selection(self, env, orders=None):
        if orders is None:
            orders = self._build_order_tasks(env)
        
        print("\n--- 生成タスク (環境状態でフィルタ済) ---")
        for o in orders:
            print(f"注文 {o.get('display_order', o['order']) + 1} (食材: {o['ingredients']}):")
            if not o['tasks']:
                print("  (タスク不要)")
            for t in o['tasks']:
                print(f"  - {t['id']}: 所要={t['dur']}, 資源候補={t['res_candidates']}")
        print("-------------------------------------------------------\n")

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
            serves = [t for t in o['tasks'] if t['verb']=='serve']
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
        print("\n=== CSP スケジュール（フレーム単位） ===")
        total_frames = 0
        
        # Group by agent
        agents_sched = {}
        for item in schedule:
            aid = item.get('agent_idx', 0)
            if aid not in agents_sched:
                agents_sched[aid] = []
            agents_sched[aid].append(item)
            total_frames = max(total_frames, item['end'])
            
        for aid in sorted(agents_sched.keys()):
            print(f"\nAI{aid}")
            for item in agents_sched[aid]:
                tid = item['id']; start=item['start']; end=item['end']; res=item['res']
                verb,obj,order = tid
                display_order = item.get('display_order', order)
                print(f"{verb} {obj} (注文{display_order+1}) : 開始={start}, 終了={end}, 資源={res}")
                
        print(f"\n総投入フレーム: {total_frames}")
        print("===================================\n")

    def astar_distance(self, env, start, goal):
        import heapq
        width = env.world_width
        height = env.world_height
        grid = env.to_grid

        def in_bounds(x, y):
            return 0 <= x < width and 0 <= y < height

        def walkable(x, y):
            return in_bounds(x, y) and grid[x][y] == 1

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
            for ing in ['lettuce', 'onion', 'tomato']:
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

        print("=== 現在のレシピから生成されたタスク列 (CSPAgent) ===")
        for i, tasks in enumerate(tasks_all):
            print(f"レシピ{i+1}:")
            for t in tasks:
                print("  ", t)
        print("=====================================")

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

        print("=== タスクグラフ（ノード, コスト） (CSPAgent) ===")
        for order_idx, tasks in enumerate(tasks_all):
            for verb, obj in tasks:
                if verb == 'chop':
                    ing_pos = env.get_pos_by_obj_gs(gs=tile_map[obj])
                    cutboard_pos = env.get_pos_by_obj_gs(gs="Cutboard")
                    ing_adj = get_adjacent_walkables(ing_pos)
                    cut_adj = get_adjacent_walkables(cutboard_pos)
                    target_adj = get_adjacent_walkables([special_places[order_idx % len(special_places)]])

                    min_total = None
                    best = None
                    for s in ing_adj:
                        for m in cut_adj:
                            for e in target_adj:
                                d1 = self.astar_distance(env, s, m)
                                d2 = self.astar_distance(env, m, e)
                                if d1 is None or d2 is None:
                                    continue
                                total = d1 + d2
                                if (min_total is None) or (total < min_total):
                                    min_total = total
                                    best = (s, m, e)
                    if min_total is None:
                        print((verb, obj, order_idx), ": 経路なし")
                    else:
                        base = min_total + 8 + 1 + 1
                        cost = base * self.frames_per_action
                        print((verb, obj, order_idx), ":", cost)
                elif verb == 'cook':
                    sp = special_places[order_idx % len(special_places)]
                    pot = pot_places[order_idx % len(pot_places)]
                    d = self.astar_distance(env, sp, pot)
                    if d is None:
                        print((verb, obj, order_idx), ": 経路なし")
                    else:
                        base = d + 2
                        cost = base * self.frames_per_action
                        print((verb, obj, order_idx), ":", cost)
                elif verb == 'serve':
                    pot = pot_places[order_idx % len(pot_places)]
                    d1 = self.astar_distance(env, plate_pos, pot)
                    d2 = self.astar_distance(env, pot, delivery_pos)
                    if d1 is None or d2 is None:
                        print((verb, obj, order_idx), ": 経路なし")
                    else:
                        base = d1 + d2 + 3
                        cost = base * self.frames_per_action
                        print((verb, obj, order_idx), ":", cost)
        print("===============================")