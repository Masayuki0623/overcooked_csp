import random
import time
from ortools.sat.python import cp_model
from .csp.model import CSPModel
from .csp.solver import solve as solve_csp
from .TaskAgent import TaskAgent

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
        # タスク重み
        self.w_chop = 1
        self.w_cook = 2
        self.w_serve = 5
        
        # 実行状態管理
        self.current_task_idx = {0: 0, 1: 0} if self.sc_2agent else 0
        self.holding_state = None 
        
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

                # 簡易CSPスケジューリング（選択問題：A解釈）
                try:
                    start_time = time.time()
                    self.schedule = self.solve_csp_scheduling(env, orders=current_orders)
                    elapsed_time = time.time() - start_time
                    print(f"[CSPAgent] スケジューリング時間: {elapsed_time:.4f} 秒")
                    
                    self._print_schedule(self.schedule)
                    
                    # スケジュールが再生成されたのでインデックスをリセット
                    self.current_task_idx = {0: 0, 1: 0} if self.sc_2agent else 0
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
                
                ta = self.task_agents[agent_idx]
                task_name = None
                if verb == 'chop':
                    task_name = f"chop_{obj}"
                    ta.assigned_counter = task.get('assigned_counter')
                elif verb == 'cook':
                    parts = obj.replace(' soup', '').split('-')
                    task_name = f"cook_{'_'.join(parts)}"
                    ta.assigned_counter = None
                elif verb == 'serve':
                    parts = obj.replace(' soup', '').split('-')
                    task_name = f"serve_{'_'.join(parts)}"
                    ta.assigned_counter = None
                
                import copy
                e_agent = copy.copy(env)
                e_agent.agent_idx = agent_idx
                
                if task_name:
                    ta.task_name = task_name
                    action, reason = ta(e_agent)
                    
                    if "Done" in reason or "done" in reason or "完了" in reason:
                        print(f"[CSPAgent] AI{agent_idx} タスク {task_name} 完了。")
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
            benefits[name] = int(t['weight'] * t['dur'])

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
            dur = t['dur']; w = t['weight']
            total += dur
            print(f"選択: {verb} {obj} (注文{order+1}) 所要={dur}, 重み={w}")
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

    def _task_weight(self, verb):
        return { 'chop': self.w_chop, 'cook': self.w_cook, 'serve': self.w_serve }.get(verb, 1)

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
                if hasattr(obj, 'is_chopped') and obj.is_chopped() and len(obj.contents) == 1 and not obj.is_held:
                    if obj.location not in cutboard_locs:
                        food_name = obj.contents[0].name
                        available_chopped[food_name] = available_chopped.get(food_name, 0) + 1
                
                if obj.location in pot_locs:
                    c_names = sorted([c.name for c in obj.contents])
                    pot_states.append({'names': c_names, 'obj': obj, 'used': False})

        resources = self._get_resources(env)
        counters = resources.get('counters', [])
        orders = []
        order_idx = 0
        
        current_orders = env.order.current_orders if hasattr(env, 'order') and hasattr(env.order, 'current_orders') else []

        for order_tuple in current_orders:
            goal = order_tuple[0]
            name = getattr(goal, 'full_name', '').lower()
            ings_lower = [ing for ing in ['lettuce','onion','tomato'] if ing in name]
            if not ings_lower:
                order_idx += 1
                continue
            
            assigned_counter = None
            if counters:
                assigned_counter = counters[order_idx % len(counters)]

            ings_cap = [ing.capitalize() for ing in ings_lower]
            
            soup_name = '-'.join(ings_lower) + ' soup'
            tasks=[]
            
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
                if dur is None: continue
                tasks.append({
                    'id': ('chop', ing.lower(), order_idx),
                    'verb':'chop','obj':ing.lower(),'order':order_idx,
                    'dur':dur,'weight':self._task_weight('chop'),
                    'res_candidates': [('cutboard', r) for r in resources['cutboards']],
                    'assigned_counter': assigned_counter
                })
            if cook_needed:
                dur = self._task_duration_frames(env, 'cook', soup_name, order_idx)
                if dur is not None:
                    tasks.append({
                        'id': ('cook', soup_name, order_idx),
                        'verb':'cook','obj':soup_name,'order':order_idx,
                        'dur':dur,'weight':self._task_weight('cook'),
                        'res_candidates': [('pot', r) for r in resources['pots']],
                    })
            dur = self._task_duration_frames(env, 'serve', soup_name, order_idx)
            if dur is not None:
                tasks.append({
                    'id': ('serve', soup_name, order_idx),
                    'verb':'serve','obj':soup_name,'order':order_idx,
                    'dur':dur,'weight':self._task_weight('serve'),
                    'res_candidates': [],
                })
            orders.append({'order':order_idx,'ingredients':ings_lower,'tasks':tasks})
            order_idx += 1
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

        # Circuit用のArc変数
        arcs = []
        lit_map = {} # (i, j) -> literal
        
        for i in all_nodes:
            for j in all_nodes:
                if i == j: continue
                
                lit = model.NewBoolVar(f'arc_{i}_{j}')
                arcs.append((i, j, lit))
                lit_map[(i, j)] = lit
                
                if not self.sc_2agent:
                    # 遷移制約: i -> j ならば start[j] >= end[i] + distance
                    if j != start_node: # Startノードに戻るときは制約不要
                        dist = dist_matrix[(i, j)]
                        model.Add(starts[j] >= ends[i] + dist).OnlyEnforceIf(lit)
                    
                    # 最初のタスク（Start -> j）の場合
                    if i == start_node:
                        dist = dist_matrix[(i, j)]
                        model.Add(starts[j] >= dist).OnlyEnforceIf(lit)
                else:
                    if j < num_tasks:
                        dist = dist_matrix[(i, j)]
                        if i == start_node or i == agent1_start_node:
                            model.Add(starts[j] >= dist).OnlyEnforceIf(lit)
                        else:
                            model.Add(starts[j] >= ends[i] + dist).OnlyEnforceIf(lit)

        # 一筆書き制約 (TSP)
        model.AddCircuit(arcs)
        
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
            
            # Legacy compatibility (for dynamic constraints)
            # tasks_vars used to map tid -> dict
            # We can reuse vars_by_tid as tasks_vars equivalent if we rename it later or use it directly.

        # 標準的な順序制約 (Chop -> Cook -> Serve)
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
                    model.Add(starts[i] >= c['end'])
                    model.Add(starts[i] >= c['end'] + 150)




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
                    model.Add(starts[i] >= c['end'])
                    model.Add(starts[i] >= c['end'] + 150)

        # 鍋の占有制約 (Pot Usage Constraint)
        # Cook開始からServe完了まで、その鍋は使用中となる
        # 同じ鍋を使う注文同士は、この期間が重なってはならない
        pot_usage_intervals = {} # pot_loc -> list of interval_vars
        
        for order_idx, tasks_list in vars_by_order.items():
            cooks = [v for v in tasks_list if v['task']['verb'] == 'cook']
            serves = [v for v in tasks_list if v['task']['verb'] == 'serve']
            
            if cooks and serves:
                cook_task = cooks[0]
                serve_task = serves[0]
                
                # この注文が使う鍋の位置
                # fixed_res = ('pot', (x, y))
                pot_res = cook_task['task'].get('fixed_res')
                if pot_res and pot_res[0] == 'pot':
                    pot_loc = pot_res[1]
                    
                    if pot_loc not in pot_usage_intervals:
                        pot_usage_intervals[pot_loc] = []
                    
                    # 占有期間: Cook開始 -> Serve終了
                    # IntervalVarを作成するには、Start, Size, Endの変数が必要
                    # Sizeは定数ではない（Serve開始が遅れる可能性があるため）
                    # なので、StartとEndを指定してSizeを推論させる、あるいはStart, Size, Endの関係式を持つIntervalを作る
                    
                    p_start = cook_task['start']
                    p_end = serve_task['end']
                    p_size = model.NewIntVar(0, horizon, f'pot_usage_dur_{order_idx}')
                    
                    # IntervalVar(start, size, end, name)
                    # p_size = p_end - p_start
                    model.Add(p_size == p_end - p_start)
                    
                    p_interval = model.NewIntervalVar(p_start, p_size, p_end, f'pot_usage_{order_idx}')
                    pot_usage_intervals[pot_loc].append(p_interval)

        # 各鍋について、占有期間の重複を禁止
        for pot_loc, intervals_list in pot_usage_intervals.items():
            if len(intervals_list) > 1:
                model.AddNoOverlap(intervals_list)
                print(f"[CSPAgent] 鍋 {pot_loc} の重複禁止制約を追加 ({len(intervals_list)} 注文)")

        # まな板の占有制約 (Cutboard Usage Constraint)
        cutboard_intervals = {} # cutboard_loc -> list of interval_vars
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
            
        # 複数エージェントの場合、Makespanだけだと「余裕のある(Slack)」タスクが遅延して
        # まるでシーケンシャルに動いているように見えるため、全タスクの開始時間もペナルティとして最小化する
        start_sum = sum(starts[i] for i in range(num_tasks))
        weight_makespan = num_tasks * 100 # Makespanを最優先
        model.Minimize(makespan * weight_makespan + start_sum)

        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        print(f"[CSPAgent] ソルバー状態: {solver.StatusName(status)}")

        schedule = []
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            actual_makespan = solver.Value(makespan)
            print(f"[CSPAgent] 最適Makespan(移動込み): {actual_makespan} (評価値: {solver.ObjectiveValue()})")
            
            for i in range(num_tasks):
                t = tasks[i]
                start_val = solver.Value(starts[i])
                end_val = solver.Value(ends[i])
                
                schedule.append({
                    'id': t['id'],
                    'start': start_val,
                    'end': end_val,
                    'res': t.get('fixed_res'),
                    'assigned_counter': t.get('assigned_counter')
                })
            
            if not self.sc_2agent:
                schedule.sort(key=lambda x: x['start'])
                # デバッグ: 順序と移動の表示
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
                                task_name = f"{tasks[j]['verb']} {tasks[j]['obj']}"
                                print(f" -> {task_name}")
                            current_node = j
                            found_next = True
                            visited_count += 1
                            break
                    if not found_next: break
                print("---------------------------------")
            else:
                schedule_per_agent = {0: [], 1: []}
                # Trace Agent 0
                curr = start_node
                while True:
                    next_n = None
                    for j in all_nodes:
                        if curr == j: continue
                        lit = lit_map.get((curr, j))
                        if lit is not None and solver.Value(lit) == 1:
                            next_n = j
                            break
                    if next_n is None or next_n == agent1_start_node: break
                    if next_n < num_tasks:
                        t = tasks[next_n]
                        schedule_per_agent[0].append({
                            'id': t['id'], 'start': solver.Value(starts[next_n]), 'end': solver.Value(ends[next_n]),
                            'res': t.get('fixed_res'), 'assigned_counter': t.get('assigned_counter'),
                            'agent_idx': 0
                        })
                    curr = next_n
                # Trace Agent 1
                curr = agent1_start_node
                while True:
                    next_n = None
                    for j in all_nodes:
                        if curr == j: continue
                        lit = lit_map.get((curr, j))
                        if lit is not None and solver.Value(lit) == 1:
                            next_n = j
                            break
                    if next_n is None or next_n == start_node: break
                    if next_n < num_tasks:
                        t = tasks[next_n]
                        schedule_per_agent[1].append({
                            'id': t['id'], 'start': solver.Value(starts[next_n]), 'end': solver.Value(ends[next_n]),
                            'res': t.get('fixed_res'), 'assigned_counter': t.get('assigned_counter'),
                            'agent_idx': 1
                        })
                    curr = next_n
                
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

        orders_sorted = sorted(orders, key=lambda o: sum(t['weight']*t['dur'] for t in o['tasks'] if t['verb']=='serve'), reverse=True)

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