import random
import time
from ortools.sat.python import cp_model
from .csp.model import CSPModel
from .csp.solver import solve as solve_csp
from .TaskAgent import TaskAgent
from gym_cooking.utils.config import COOKING_TIME_SECONDS

class CSPAgent:
    """
    CSP(制約充足問題)ベースのエージェント
    """
    def __init__(self, speed=2.5, replay=None, no_reschedule=False, sc_2agent=False):
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
        # 期限
        self.deadline_frames = 75 * self.fps
        # 30秒の選択予算
        self.budget_frames = 30 * self.fps
        # 実行状態管理
        self.current_task_idx = {0: 0, 1: 0} if self.sc_2agent else 0
        self.holding_state = None 
        # 交代制のためのターン管理
        self.turn = 0
        self.completed_task_ids = set() # 追加：完了したタスクのID集合（同期用）
        
        self.task_agent = TaskAgent()
        if self.sc_2agent:
            self.task_agents = {0: TaskAgent(), 1: TaskAgent()}
        
        # 優先度重み（GUI等で設定）
        self.priority_weights = {}
        # 制約指示テキスト（GUI等で設定）
        self.gui_constraint_input = ""
        # 適用する動的制約リスト (JSON format)
        self.active_constraints = []

        print("[CSPAgent] 初期化完了 - 現在はランダム行動")

    def get_remaining_tids(self, env, current_orders):
        """現在の環境から残存タスクのID集合(tid)を抽出する（インベントリ照合）"""
        inv_chopped = []
        inv_pots_ings = []
        inv_plates_ings = []
        
        items = []
        for a in getattr(env, 'agents', []):
            if hasattr(a, 'holding') and a.holding is not None:
                items.append(a.holding)
                
        for pos, obj in env.pos_obj.items():
            if obj is not None:
                items.append(obj)
                
        for obj in items:
            name = getattr(obj, 'name', '')
            if name.startswith('Chopped'):
                inv_chopped.append(name)
            elif 'Pot' in name or name == 'Soup_cooked':
                ings = getattr(obj, 'ingredient_names', [])
                if ings:
                    inv_pots_ings.append(set(ings))
            elif name == 'Plate':
                ings = getattr(obj, 'ingredient_names', [])
                if ings:
                    inv_plates_ings.append(set(ings))
                    
        remaining_tids = set()
        
        for order_idx, order in enumerate(current_orders):
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
                remaining_tids.add(('chop', raw_ing, order_idx))
                
            soup_name = "-".join([i.replace('Chopped', '').lower() for i in raw_parts]) + " soup"
            if needs_cook:
                remaining_tids.add(('cook', soup_name, order_idx))
            if needs_serve:
                remaining_tids.add(('serve', soup_name, order_idx))
                
        return remaining_tids

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

        if not hasattr(self, 'prev_task_ids'):
            self.prev_task_ids = set()

        added = current_task_ids - self.prev_task_ids
        removed = self.prev_task_ids - current_task_ids

        # 変化があった場合、または初回の場合
        if added or removed or not self.initialized:
            # 削除されたタスク = 物理的に完了済み → completed_task_idsに追加
            # (ターン制で「Done」を返せなかった場合でも正しく完了を検知できる)
            if removed and self.initialized:
                physically_done = {t for t in removed if t[0] in ('chop', 'cook')}
                if physically_done:
                    print(f"  [完了検知] 物理完了タスク: {physically_done}")
                    self.completed_task_ids |= physically_done

            # no_rescheduleが有効で、既に初期化済みなら再計算しない
            if self.no_reschedule and self.initialized:
                pass
            else:
                if self.initialized: # 初回以外なら差分を表示
                    print(f"\n[タスク更新] 時間: {env.time}")
                    if added:
                        print(f"  (+) 追加: {added}")
                    if removed:
                        print(f"  (-) 削除: {removed}")
                    print("  -> スケジュール再計算中...")

                # ① リスケジュール前に「現在実行中のタスクID」を保存する（Bug 1&5 対策）
                in_progress_tasks = {}  # agent_idx -> task_id
                if self.sc_2agent and hasattr(self, 'schedule_per_agent'):
                    for aidx in [0, 1]:
                        sc = self.schedule_per_agent.get(aidx, [])
                        t_idx = self.current_task_idx.get(aidx, 0) if isinstance(self.current_task_idx, dict) else 0
                        if t_idx < len(sc):
                            in_progress_tasks[aidx] = sc[t_idx]['id']
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
                            in_prog_tid = in_progress_tasks.get(aidx)
                            if in_prog_tid:
                                new_sc = self.schedule_per_agent.get(aidx, [])
                                found = False
                                for i, t in enumerate(new_sc):
                                    if t['id'] == in_prog_tid:
                                        new_idx[aidx] = i
                                        print(f"  [継続] AI{aidx}: {in_prog_tid} → 新スケジュール idx={i} から再開")
                                        found = True
                                        break
                                if not found:
                                    # 新スケジュールに実行中タスクがない = 他エージェントが担当 or 不要になった
                                    # → in_prog_tid を completed とみなしてスキップ
                                    print(f"  [スキップ] AI{aidx}: {in_prog_tid} が新スケジュールに存在しない → idx=0から開始")
                                    new_idx[aidx] = 0
                        self.current_task_idx = new_idx
                    else:
                        self.current_task_idx = 0

                except Exception as e:
                    print(f"[CSPAgent] CSPスケジュール中に例外: {e}")
                    import traceback
                    traceback.print_exc()
                
                self.prev_task_ids = current_task_ids
                self.initialized = True


        # スケジュール実行
        if not self.sc_2agent:
            if not hasattr(self, 'schedule') or not self.schedule or self.current_task_idx >= len(self.schedule):
                return (0, 0), "タスクなし"

            task = self.schedule[self.current_task_idx]
            tid = task['id']
            verb, obj, order_idx = tid
            res = task['res'] 

            # Construct Task Name
            task_name = None
            if verb == 'chop':
                task_name = f"chop_{obj}"
                self.task_agent.assigned_counter = task.get('assigned_counter')
            elif verb == 'cook':
                parts = obj.replace(' soup', '').split('-')
                task_name = f"cook_{'_'.join(parts)}"
                self.task_agent.assigned_counter = None
            elif verb == 'serve':
                parts = obj.replace(' soup', '').split('-')
                task_name = f"serve_{'_'.join(parts)}"
                self.task_agent.assigned_counter = None
            
            if task_name:
                self.task_agent.task_name = task_name
                action, reason = self.task_agent(env)
                
                # Check completion
                if "Done" in reason or "done" in reason or "完了" in reason:
                    print(f"[CSPAgent] タスク {task_name} 完了。次へ移動。")
                    self.completed_task_ids.add(tid)
                    self.current_task_idx += 1
                    # Reset assignments
                    self.task_agent.assigned_cutboard = None
                    self.task_agent.assigned_pot = None
                    self.task_agent.assigned_plate = None
                    self.task_agent.assigned_serve_loc = None
                    self.task_agent.assigned_counter = None
                
                return action, reason

            return (0,0), "アイドル"
        else:
            if not hasattr(self, 'schedule_per_agent') or not self.schedule_per_agent:
                return {"ai_0": (0, 0), "ai_1": (0, 0)}, "タスクなし"
            
            actions = {}
            reasons = []
            for agent_idx in [0, 1]:
                sc = self.schedule_per_agent.get(agent_idx, [])
                t_idx = self.current_task_idx[agent_idx]
                if t_idx >= len(sc):
                    actions[f"ai_{agent_idx}"] = (0, 0)
                    reasons.append(f"AI{agent_idx}:Idle")
                    continue
                
                task = sc[t_idx]
                tid = task['id']
                verb, obj, order_idx = tid
                
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
                        req_tid = ('chop', p.strip(), order_idx)
                        if req_tid not in self.completed_task_ids and req_tid in all_scheduled_ids:
                            can_start = False
                            break
                elif verb == 'serve':
                    req_tid = ('cook', obj, order_idx)
                    if req_tid not in self.completed_task_ids and req_tid in all_scheduled_ids:
                        can_start = False
                        
                if not can_start:
                    missing_deps = []
                    if verb == 'cook':
                        parts = obj.replace(' soup', '').split('-')
                        missing_deps = [('chop', p.strip(), order_idx) for p in parts if ('chop', p.strip(), order_idx) not in self.completed_task_ids]
                    elif verb == 'serve':
                        if ('cook', obj, order_idx) not in self.completed_task_ids:
                            missing_deps = [('cook', obj, order_idx)]
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
                    
                    # ユーザーの要望「同時に動かすのではなく交互に」
                    if agent_idx == self.turn:
                        action, reason = ta(e_agent, dynamic_obstacles=dynamic_obstacles)
                    else:
                        action, reason = (0, 0), "待機(相手のターン)"
                    
                    if action == (0, 0) and reason not in ("待機(相手のターン)",):
                        print(f"[DEBUG] AI{agent_idx} 停止: task={task_name} reason='{reason}' counter={ta.assigned_counter} hold={getattr(e_agent, 'hold', None)}")

                    
                    if "Done" in reason or "done" in reason or "完了" in reason:
                        print(f"[CSPAgent] AI{agent_idx} タスク {task_name} 完了。")
                        self.completed_task_ids.add(tid)
                        self.current_task_idx[agent_idx] += 1
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
            verb = t['verb']; obj = t['obj']; order = t['order']
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

    def _task_duration_frames(self, env, verb, obj, order_idx, assigned_counter=None):
        resources = self._get_resources(env)
        
        def get_nearest(start_pos, candidates):
            if not candidates: return None
            if not start_pos: return candidates[0]
            return min(candidates, key=lambda p: abs(p[0]-start_pos[0]) + abs(p[1]-start_pos[1]))

        if verb == 'chop':
            tile_map = {"lettuce": "FreshLettuceTile", "onion": "FreshOnionTile", "tomato": "FreshTomatoTile"}
            ing_pos_list = env.get_pos_by_obj_gs(gs=tile_map.get(obj, ""))
            if not ing_pos_list: return None
            ing_pos = ing_pos_list[0] 

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
            return int(min_total + 8 + 1 + 1 + 1)

        elif verb == 'cook':
            pot_pos_list = resources['pots']
            if not pot_pos_list: return None
            
            pot_pos = pot_pos_list[order_idx % len(pot_pos_list)]
            
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
        pot_states = []

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
            if type(obj).__name__ == 'Object':
                if obj.location in pot_locs:
                    c_names = sorted([c.name for c in obj.contents])
                    pot_states.append({'names': c_names, 'obj': obj, 'used': False})
                else:
                    if obj.location not in cutboard_locs:
                        if hasattr(obj, 'is_chopped') and obj.is_chopped():
                            for food in obj.contents:
                                available_chopped[food.name] = available_chopped.get(food.name, 0) + 1

        resources = self._get_resources(env)
        orders = []
        order_idx = 0

        current_orders = env.order.current_orders if hasattr(env, 'order') and hasattr(env.order, 'current_orders') else []

        if not hasattr(self, 'assigned_counters_map'):
            self.assigned_counters_map = {}  # stable_key -> counter_pos

        # 完了した注文のキーを解放する
        active_keys = set()
        for idx, ot in enumerate(current_orders):
            goal = ot[0]
            n = getattr(goal, 'full_name', '').lower()
            if any(ing in n for ing in ['lettuce', 'onion', 'tomato']):
                active_keys.add(f"{n}_{idx}")
        for k in [k for k in list(self.assigned_counters_map) if k not in active_keys]:
            del self.assigned_counters_map[k]

        used_counters = list(self.assigned_counters_map.values())
        assigned_counters_display_map = {}

        for order_tuple in current_orders:
            goal = order_tuple[0]
            name = getattr(goal, 'full_name', '').lower()
            ings_lower = [ing for ing in ['lettuce', 'onion', 'tomato'] if ing in name]
            if not ings_lower:
                order_idx += 1
                continue

            stable_key = f"{name}_{order_idx}"
            ings_cap = [ing.capitalize() for ing in ings_lower]
            chopped_names = {f"Chopped{i}" for i in ings_cap}
            all_counters = resources.get('counters', [])

            current = self.assigned_counters_map.get(stable_key)
            # 他の注文に割り当て済みのカウンター（自分のは除く）
            other_orders_counters = {c for c in used_counters if c != current}

            # ルール3: 別のカウンターに必要な食材がある → そこを置き場にする
            # ただし他注文に割り当て済みのカウンターは除外
            food_counter = None
            for c in all_counters:
                if c in other_orders_counters:
                    continue  # 他注文の置き場は除外
                obj = env.pos_obj.get(c)
                if obj is not None and hasattr(obj, 'contents'):
                    if {f.name for f in obj.contents} & chopped_names:
                        food_counter = c
                        break

            if food_counter and food_counter != current:
                assigned_counter = food_counter
                self.assigned_counters_map[stable_key] = assigned_counter
                if current in used_counters:
                    used_counters.remove(current)
                if assigned_counter not in used_counters:
                    used_counters.append(assigned_counter)
            elif current is not None:
                cur_obj = env.pos_obj.get(current)
                if cur_obj is not None and hasattr(cur_obj, 'contents'):
                    if {f.name for f in cur_obj.contents} & chopped_names:
                        # ルール1: 適切な食材がある → そのまま
                        assigned_counter = current
                    else:
                        # ルール2: 不適切な食材がある → 再計算（他注文の場所も除外）
                        assigned_counter = self._calculate_dynamic_merge_point(env, ings_lower, order_idx, pot_locs, list(other_orders_counters))
                        if assigned_counter:
                            self.assigned_counters_map[stable_key] = assigned_counter
                            if current in used_counters:
                                used_counters.remove(current)
                            used_counters.append(assigned_counter)
                else:
                    # カウンターが空 → 確定済みをそのまま
                    assigned_counter = current
            else:
                # 未割り当て → 最適な場所を計算
                assigned_counter = self._calculate_dynamic_merge_point(env, ings_lower, order_idx, pot_locs, used_counters)
                if assigned_counter:
                    self.assigned_counters_map[stable_key] = assigned_counter
                    used_counters.append(assigned_counter)

            assigned_counters_display_map[order_idx] = assigned_counter

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
                if available_chopped.get(ing, 0) > 0:
                    available_chopped[ing] -= 1
                    continue
                dur = self._task_duration_frames(env, 'chop', ing.lower(), order_idx, assigned_counter)
                if dur is None:
                    continue
                tasks.append({
                    'id': ('chop', ing.lower(), order_idx),
                    'verb': 'chop', 'obj': ing.lower(), 'order': order_idx,
                    'dur': dur,
                    'res_candidates': [('cutboard', r) for r in resources['cutboards']],
                    'assigned_counter': assigned_counter
                })

            if cook_needed:
                dur = self._task_duration_frames(env, 'cook', soup_name, order_idx)
                if dur is not None:
                    tasks.append({
                        'id': ('cook', soup_name, order_idx),
                        'verb': 'cook', 'obj': soup_name, 'order': order_idx,
                        'dur': dur,
                        'res_candidates': [('pot', r) for r in resources['pots']],
                        'assigned_counter': assigned_counter
                    })

            dur = self._task_duration_frames(env, 'serve', soup_name, order_idx)
            if dur is not None:
                tasks.append({
                    'id': ('serve', soup_name, order_idx),
                    'verb': 'serve', 'obj': soup_name, 'order': order_idx,
                    'dur': dur,
                    'res_candidates': [],
                    'assigned_counter': assigned_counter
                })

            orders.append({'order': order_idx, 'ingredients': ings_lower, 'tasks': tasks})
            order_idx += 1

        self.assigned_counters_display_map = assigned_counters_display_map
        return orders

    def solve_csp_scheduling(self, env, orders):
        """
        OR-Tools CP-SAT を用いたスケジューリング（移動コスト込み）。
        Circuit制約を用いて順序依存のセットアップ時間（移動時間）を正確にモデル化する。
        """
        print(f"[CSPAgent] CSPスケジューリング開始 ({len(orders)} 注文)...")
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
        
        def get_nearest(start_pos, candidates):
            if not candidates: return start_pos # Fallback
            return min(candidates, key=lambda p: abs(p[0]-start_pos[0]) + abs(p[1]-start_pos[1]))

        for t in tasks:
            verb = t['verb']
            obj = t['obj']
            order_idx = t['order']
            
            if verb == 'chop':
                # 食材の位置から一番近いまな板を選ぶ
                tile_map = {"lettuce": "FreshLettuceTile", "onion": "FreshOnionTile", "tomato": "FreshTomatoTile"}
                ing_pos_list = env.get_pos_by_obj_gs(gs=tile_map.get(obj, ""))
                ing_pos = ing_pos_list[0] if ing_pos_list else agent_pos
                
                cutboards = resources['cutboards']
                best_cb = get_nearest(ing_pos, cutboards)
                
                t['start_pos'] = best_cb
                t['end_pos'] = best_cb
                t['fixed_res'] = ('cutboard', best_cb)
                
            elif verb == 'cook':
                pots = resources['pots']
                # 注文インデックスに基づく鍋割り当て（簡易）
                pot = pots[order_idx % len(pots)] if pots else agent_pos
                t['start_pos'] = pot
                t['end_pos'] = pot
                t['fixed_res'] = ('pot', pot)
                
            elif verb == 'serve':
                pots = resources['pots']
                pot = pots[order_idx % len(pots)] if pots else agent_pos
                delivery = resources['delivery']
                
                t['start_pos'] = pot
                t['end_pos'] = delivery
                t['fixed_res'] = ('pot', pot) # ServeもPotリソース扱いにしておく

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
        
        if self.sc_2agent:
            # is_a1[i]: True = タスクiをAI1が担当、False = AI0が担当
            is_a1 = [model.NewBoolVar(f'is_a1_{i}') for i in range(num_tasks)]
            
            # エージェント出発位置からの最低到達時間
            for i in range(num_tasks):
                dist_from_a0 = int(dist_matrix.get((start_node, i), 0))
                dist_from_a1 = int(dist_matrix.get((agent1_start_node, i), 0))
                model.Add(starts[i] >= dist_from_a0).OnlyEnforceIf(is_a1[i].Not())
                model.Add(starts[i] >= dist_from_a1).OnlyEnforceIf(is_a1[i])
            
            # 同一エージェントのタスクペア間: 先後関係 + 移動時間制約
            # OnlyEnforceIf([is_a1[i].Not(), is_a1[j].Not(), order_ij])
            # → AI0担当 かつ i→j順 ならば starts[j] >= ends[i] + dist(i,j)
            for i in range(num_tasks):
                for j in range(i + 1, num_tasks):
                    order_ij = model.NewBoolVar(f'order_{i}_{j}')
                    dij = int(dist_matrix.get((i, j), 0))
                    dji = int(dist_matrix.get((j, i), 0))
                    
                    # AI0 同士でiが先
                    model.Add(starts[j] >= ends[i] + dij).OnlyEnforceIf(
                        [is_a1[i].Not(), is_a1[j].Not(), order_ij])
                    # AI0 同士でjが先
                    model.Add(starts[i] >= ends[j] + dji).OnlyEnforceIf(
                        [is_a1[i].Not(), is_a1[j].Not(), order_ij.Not()])
                    # AI1 同士でiが先
                    model.Add(starts[j] >= ends[i] + dij).OnlyEnforceIf(
                        [is_a1[i], is_a1[j], order_ij])
                    # AI1 同士でjが先
                    model.Add(starts[i] >= ends[j] + dji).OnlyEnforceIf(
                        [is_a1[i], is_a1[j], order_ij.Not()])
                    # 異なるエージェント: 移動制約不要（並列実行可能）
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
            order_idx = t['order']
            if order_idx not in vars_by_order: vars_by_order[order_idx] = []
            
            v_obj = {'start': starts[i], 'end': ends[i], 'task': t, 'interval': intervals[i]}
            vars_by_order[order_idx].append(v_obj)
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
        print(f"[CSPAgent] ソルバー状態: {solver.StatusName(status)}")

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
                                    'assigned_counter': t.get('assigned_counter')
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
                    agent_idx = 1 if solver.Value(is_a1[i]) else 0
                    schedule_per_agent[agent_idx].append({
                        'id': t['id'],
                        'start': solver.Value(starts[i]),
                        'end': solver.Value(ends[i]),
                        'res': t.get('fixed_res'),
                        'assigned_counter': t.get('assigned_counter'),
                        'agent_idx': agent_idx
                    })
                # 各エージェントのタスクを開始時刻順に並べる
                for agent_idx in [0, 1]:
                    schedule_per_agent[agent_idx].sort(key=lambda x: x['start'])
                
                self.schedule_per_agent = schedule_per_agent
                schedule = schedule_per_agent[0] + schedule_per_agent[1]
                schedule.sort(key=lambda x: x['start'])
                
        else:
            print(f"[CSPAgent] 解が見つかりませんでした。")
            
        return schedule

    def solve_csp_selection(self, env, orders=None):
        if orders is None:
            orders = self._build_order_tasks(env)
        
        print("\n--- 生成タスク (環境状態でフィルタ済) ---")
        for o in orders:
            print(f"注文 {o['order']} (食材: {o['ingredients']}):")
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
                print(f"{verb} {obj} (注文{order+1}) : 開始={start}, 終了={end}, 資源={res}")
                
        print(f"\n総投入フレーム: {total_frames}")
        print("===================================\n")

    def get_assigned_counters(self):
        """
        デバッグ用：各注文ごとの割り当てカウンター座標を返す
        Returns: {order_idx: (x, y), ...}
        """
        if not hasattr(self, 'schedule') or not self.schedule:
            return {}
        
        assignments = {}
        for task in self.schedule:
            tid = task['id']
            # tid is (verb, obj, order_idx)
            if len(tid) == 3 and tid[0] == 'chop':
                order_idx = tid[2]
                counter = task.get('assigned_counter')
                if counter:
                    assignments[order_idx] = counter
        return assignments

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