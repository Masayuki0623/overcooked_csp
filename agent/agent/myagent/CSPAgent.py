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
    def __init__(self, speed=2.5, replay=None, no_reschedule=False, num_agents=1):
        self.speed = speed
        self.replay = replay
        self.no_reschedule = no_reschedule
        self.initialized = False
        self.num_agents = num_agents
        
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
        self.current_task_idx = [0] * num_agents
        self.schedules = [[] for _ in range(num_agents)]
        
        # TaskAgentをエージェント人数分用意
        self.task_agents = [TaskAgent() for _ in range(num_agents)]
        # Legacy support (accessor for first agent)
        self.task_agent = self.task_agents[0]
        
        # 優先度重み（GUI等で設定）
        self.priority_weights = {}
        # 制約指示テキスト（GUI等で設定）
        self.gui_constraint_input = ""
        # 適用する動的制約リスト (JSON format)
        self.active_constraints = []
        # 進入禁止エリア
        self.forbidden_zones = []

        # --- Cache for Precomputed Map Data ---
        self.map_initialized = False
        self.dist_cache = {}    # (p1, p2) -> distance
        self.path_info = {}     # start -> {end: parent}
        self.tile_to_zone_cache = None
        self.zones_cache = None

        print(f"[CSPAgent] 初期化完了 - Agents: {num_agents}")

    def __call__(self, env):
        """
        環境から呼ばれるメイン関数
        """
        if not self.map_initialized:
            self._compute_static_map_data(env)
            self.map_initialized = True

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
        
        # Check if we should reschedule
        should_reschedule = False
        if not self.initialized:
            should_reschedule = True
        elif (added or removed) and not self.no_reschedule:
             should_reschedule = True
             print(f"\n[タスク更新] 時間: {env.time} Added:{len(added)} Removed:{len(removed)}")

        # Reschedule if needed
        if should_reschedule:
            try:
                start_time = time.time()
                # Solve for multiple agents
                full_schedule = self.solve_csp_scheduling(env, orders=current_orders)
                
                # Split schedule by agent
                if full_schedule and isinstance(full_schedule, list) and len(full_schedule) > 0 and isinstance(full_schedule[0], list):
                    self.schedules = full_schedule
                else:
                    self.schedules = [[] for _ in range(self.num_agents)]
                    if full_schedule: # Flat list fallback
                        for task in full_schedule:
                            agent_idx = task.get('agent_idx', 0)
                            if 0 <= agent_idx < self.num_agents:
                                self.schedules[agent_idx].append(task)
                
                elapsed_time = time.time() - start_time
                print(f"[CSPAgent] スケジューリング完了: {elapsed_time:.4f} 秒, Agents: {self.num_agents}")
                self._print_schedule_multi(self.schedules)
                
                # Reset indices
                self.current_task_idx = [0] * self.num_agents
                # Sync forbidden zones to all executors
                for ta in self.task_agents:
                    ta.forbidden_zones = self.forbidden_zones

            except Exception as e:
                print(f"[CSPAgent] CSPスケジュール中に例外: {e}")
                import traceback
                traceback.print_exc()
            
            self.prev_task_ids = current_task_ids
            self.initialized = True

        # Execute steps for ALL agents
        actions = []
        
        # env.agent_idx comes from GamePlay: list of AI indices e.g. [0, 1]
        # We need to map our local agent index 0..K to real world index
        real_indices = env.agent_idx if isinstance(env.agent_idx, list) else [env.agent_idx]
        
        # Ensure we don't exceed what we expected
        limit = min(self.num_agents, len(real_indices))
        
        from copy import copy
        
        for i in range(limit):
            real_idx = real_indices[i]
            
            # Create a view for this agent
            # Ideally we clone envstate to modify agent_idx
            # Since EnvState is simple, we can just tweak it if we accept shallow copy risks,
            # but safer to create new EnvState or copy.
            # EnvState is in agent.executor.low.
            # We can just change agent_idx on a shallow copy of the object wrapper.
            
            env_view = copy(env) 
            env_view.agent_idx = real_idx # Set integer ID for TaskAgent
            
            # Get current task for this agent
            my_sched = self.schedules[i] if i < len(self.schedules) else []
            curr_idx = self.current_task_idx[i]
            
            print(f"[DEBUG-CSP] Agent {i} (Real {real_idx}): Step {curr_idx}/{len(my_sched)}")

            action = (0,0)
            reason = "Idle"
            
            # Setup TaskAgent (Pre-fetch for debugging info)
            ta = self.task_agents[i]

            # DEBUG POS
            pos = ta.last_pos if hasattr(ta, 'last_pos') else "?"
            curr_pos = env_view.self_pos
            print(f"  [DEBUG-POS] Prev: {pos}, Curr: {curr_pos}")
            if hasattr(ta, 'last_pos') and ta.last_pos == curr_pos:
                print(f"  [DEBUG-POS] STUCK! Grid around {curr_pos}:")
                cx, cy = curr_pos
                w, h = env_view.world_width, env_view.world_height
                grid = env_view.to_grid
                for dy in [-1,0,1]:
                    row = []
                    for dx in [-1,0,1]:
                        nx, ny = cx+dx, cy+dy
                        val = grid[nx][ny] if 0<=nx<w and 0<=ny<h else 'X'
                        row.append(str(val))
                    print(f"    {' '.join(row)}")
            ta.last_pos = curr_pos
            
            if curr_idx < len(my_sched):
                task = my_sched[curr_idx]
                tid = task['id']
                verb, obj, order_idx = tid
                
                print(f"  [DEBUG-CSP] Task Info: {tid}, Res: {task.get('res')}, Start/End: {task.get('start_pos')}/{task.get('end_pos')}")

                res = task.get('res') # CSP assigned resource

                # Determine Task Name
                task_name = None
                if verb == 'chop':
                    task_name = f"chop_{obj}"
                    ta.assigned_counter = task.get('assigned_counter')
                    if res and res[0] == 'cutboard':
                        ta.assigned_cutboard = res[1]
                elif verb == 'cook':
                    parts = obj.replace(' soup', '').split('-')
                    task_name = f"cook_{'_'.join(parts)}"
                    ta.assigned_counter = None
                    if res and res[0] == 'pot':
                        ta.assigned_pot = res[1]
                elif verb == 'serve':
                    parts = obj.replace(' soup', '').split('-')
                    task_name = f"serve_{'_'.join(parts)}"
                    ta.assigned_counter = None
                    if res and res[0] == 'pot':
                        ta.assigned_pot = res[1]
                    ta.assigned_serve_loc = task.get('end_pos')
                    
                    # 皿の場所指定（もしTask情報になければリソースから取得して割り当てる）
                    resources = self._get_resources(env)
                    if resources.get('plate'):
                        # CSPモデル上は皿の位置を固定リソースとして扱っている場合が多い
                        # ここでは簡易的にリソースの最初の皿、またはTaskAgentの探索に任せるが
                        # 明示的に渡すことで安定させる
                        ta.assigned_plate = resources['plate']
                
                if task_name:
                    print(f"  [DEBUG-CSP] Calling TaskAgent: {task_name}")
                    print(f"    CB={ta.assigned_cutboard}, Pot={ta.assigned_pot}, Plate={ta.assigned_plate}, Serve={ta.assigned_serve_loc}")

                    ta.task_name = task_name
                    # Forbidden zones already synced
                    action, reason = ta(env_view)
                    
                    print(f"  [DEBUG-CSP] TaskAgent Result: {action}, Reason='{reason}'")

                    if "done" in reason.lower() or "完了" in reason:
                        print(f"[CSPAgent-{i}] タスク {task_name} 完了 -> 次へ")
                        self.current_task_idx[i] += 1
                        ta.assigned_cutboard = None
                        ta.assigned_pot = None
                        ta.assigned_plate = None
                        ta.assigned_serve_loc = None
                        ta.assigned_counter = None
            else:
                print(f"  [DEBUG-CSP] Schedule Finished or Empty")
            
            actions.append(action)
            
            actions.append(action)
            
        return actions, "Multi-Agent Step"

    def _print_schedule_multi(self, schedules):
        print("\n=== CSP Multi-Agent Schedule ===")
        for i, sched in enumerate(schedules):
            print(f"--- Agent {i} ---")
            for item in sched:
                 tid = item['id']; start=item['start']; end=item['end']
                 verb,obj,order = tid
                 print(f"  {verb} {obj} ({order+1}): {start}-{end} Res:{item.get('res')}")
        print("===============================\n")


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
                            if hasattr(self, 'forbidden_zones') and (nx, ny) in self.forbidden_zones:
                                continue
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
            
            # Allow interacting with pot even if standing in forbidden zone
            d = self.astar_distance(env, start_pos, pot_pos, allow_forbidden_adjacent=False)
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

    def _compute_static_map_data(self, env):
        """
        環境の静的な地図情報（距離テーブル、ボトルネック）を事前計算する。
        """
        print("[CSPAgent] マップ情報を事前計算中（APSP & Bottlenecks）...")
        start_t = time.time()
        
        width, height = env.world_width, env.world_height
        grid = env.to_grid

        # 1. 有効な座標のリストアップ
        valid_points = []
        for x in range(width):
            for y in range(height):
                if grid[x][y] == 1:
                    valid_points.append((x, y))

        # 2. 全点対最短経路 (BFS from each point)
        # マップサイズが小さい(Overcookedは通常10x10程度)ので全点BFSで十分高速
        # 距離だけでなく、パス復元用のparent情報や、経路上のボトルネック通過情報もキャッシュ可能だが
        # ここでは距離テーブルをメインに作成
        
        self.dist_cache = {}
        # 今回はCSP内で使う「距離」だけあればよく、パス本体が必要なのはボトルネック制約のため。
        # ボトルネック制約を高速に適用するためには、(i, j)移動時に通過するZoneのリストがあればベスト。
        
        # まずボトルネックを特定
        self.tile_to_zone_cache, self.zones_cache = self._detect_bottlenecks(env)
        
        # パス構築用のヘルパー
        def get_neighbors(p):
            x, y = p
            ns = []
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < width and 0 <= ny < height and grid[nx][ny] == 1:
                    ns.append((nx, ny))
            return ns

        # 全点BFS
        # dist_matrix: self.dist_cache[(start, end)] = distance
        # path_zones: self.path_zones_cache[(start, end)] = list of zone_ids passed
        
        self.path_zones_cache = {} 

        for start_node in valid_points:
            # BFS Initialization
            q = [start_node]
            visited = {start_node: 0}
            parent = {start_node: None}
            
            while q:
                curr = q.pop(0)
                d = visited[curr]
                
                # 自分までの距離を保存
                self.dist_cache[(start_node, curr)] = d
                
                if curr != start_node:
                    # パス復元して通過ゾーンを記録（※メモリ節約のため、ゾーン通過時のみ記録しても良い）
                    # ここでは一旦、実際にパスを逆走してゾーンを回収する
                    # 頻出する重要な場所（鍋、まな板）へのパスだけキャッシュするのが賢いが、
                    # 全点これだと重いかもしれない。しかしマップは狭いので試行。
                    
                    if curr in valid_points: # 常にTrue
                        # Reconstruct path
                        path = []
                        temp = curr
                        while temp is not None:
                            path.append(temp)
                            temp = parent[temp]
                        path.reverse() # start -> end
                        
                        # Extract zones
                        zones_in_path = []
                        last_zone = None
                        zone_intervals = {} # zone_id -> (start_idx, end_idx) relative to path start time?
                                            # No, simple list of zones for now to check overlap?
                                            # CSP needs precise timing.
                                            # We need: for this (Start, End) pair, which bottlenecks are used at what distance offset?
                        
                        # キャッシュ構造: (start, end) -> [(zone_id, dist_from_start, duration), ...]
                        # duration は 連続してそのゾーンにいる長さ
                        
                        usage_list = []
                        # path[0] is start (t=0)
                        # path[k] is at t=k
                        
                        current_z = None
                        z_start_k = -1
                        
                        for k, pos in enumerate(path):
                            z = self.tile_to_zone_cache.get(pos)
                            
                            if z != current_z:
                                # Switch occurred
                                if current_z is not None:
                                    # Zone ended at k-1
                                    duration = (k - 1) - z_start_k + 1
                                    usage_list.append((current_z, z_start_k, duration))
                                
                                current_z = z
                                z_start_k = k
                        
                        # Loop finish check
                        if current_z is not None:
                            duration = (len(path) - 1) - z_start_k + 1
                            usage_list.append((current_z, z_start_k, duration))
                            
                        if usage_list:
                            self.path_zones_cache[(start_node, curr)] = usage_list

                # Neighbors
                for n in get_neighbors(curr):
                    if n not in visited:
                        visited[n] = d + 1
                        parent[n] = curr
                        q.append(n)
        
        elapsed = time.time() - start_t
        print(f"[CSPAgent] 静的データ計算完了: {elapsed:.4f}秒, 地点数={len(valid_points)}")


    def _detect_bottlenecks(self, env):
        width, height = env.world_width, env.world_height
        grid = env.to_grid
        bottlenecks = set()
        
        # 1. Identify narrow passage tiles
        for x in range(width):
            for y in range(height):
                if grid[x][y] == 0: continue # Wall
                
                # Check neighbors (Wall=0, Walkable=1)
                n_u = grid[x][y-1] == 0 if y > 0 else True
                n_d = grid[x][y+1] == 0 if y < height-1 else True
                n_l = grid[x-1][y] == 0 if x > 0 else True
                n_r = grid[x+1][y] == 0 if x < width-1 else True
                
                is_narrow_h = n_u and n_d # Walls above and below -> Horizontal Passage
                is_narrow_v = n_l and n_r # Walls left and right -> Vertical Passage
                
                if is_narrow_h or is_narrow_v:
                    bottlenecks.add((x, y))

        # 2. Group into zones
        visited = set()
        zones = {}
        zone_id = 0
        
        for bn in bottlenecks:
            if bn in visited: continue
            q = [bn]
            visited.add(bn)
            current_zone = []
            while q:
                curr = q.pop(0)
                current_zone.append(curr)
                cx, cy = curr
                for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nx, ny = cx+dx, cy+dy
                    neigh = (nx, ny)
                    if neigh in bottlenecks and neigh not in visited:
                        visited.add(neigh)
                        q.append(neigh)
            
            # Filter? Maybe keep only if length > 1? Single tile door is also a bottleneck.
            zones[zone_id] = current_zone
            zone_id += 1
            
        tile_to_zone = {}
        for zid, tiles in zones.items():
            for t in tiles:
                tile_to_zone[t] = zid
                
        return tile_to_zone, zones

    def astar_path(self, env, start, goal, allow_forbidden_adjacent=False):
        import heapq
        width = env.world_width
        height = env.world_height
        grid = env.to_grid

        def in_bounds(x, y):
            return 0 <= x < width and 0 <= y < height
        def is_forbidden(x, y):
            if hasattr(self, 'forbidden_zones'): return (x, y) in self.forbidden_zones
            return False
        def walkable_primitive(x, y):
            return in_bounds(x, y) and grid[x][y] == 1

        # Check Goal
        final_goal = goal
        if not walkable_primitive(goal[0], goal[1]):
            adjacents = []
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = goal[0]+dx, goal[1]+dy
                if walkable_primitive(nx, ny):
                    if not is_forbidden(nx, ny): adjacents.append((nx, ny))
                    elif allow_forbidden_adjacent: adjacents.append((nx, ny))
            if not adjacents: return None, None
            final_goal = min(adjacents, key=lambda p: abs(p[0]-start[0]) + abs(p[1]-start[1]))
        else:
             if is_forbidden(goal[0], goal[1]) and not allow_forbidden_adjacent: return None, None
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0, start))
        g_score = {start: 0}
        f_score = {start: heuristic(start, final_goal)}
        came_from = {}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == final_goal:
                path = []
                curr = current
                while curr in came_from:
                    path.append(curr)
                    curr = came_from[curr]
                path.append(start)
                path.reverse()
                return g_score[current], path

            cx, cy = current
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = cx+dx, cy+dy
                if not walkable_primitive(nx, ny): continue
                if is_forbidden(nx, ny):
                    is_dest = (nx, ny) == final_goal
                    if not (is_dest and allow_forbidden_adjacent): continue
                
                neighbor = (nx, ny)
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, final_goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None, None

    def solve_csp_scheduling(self, env, orders):
        """
        OR-Tools CP-SAT を用いたスケジューリング（Multi-Agent TSP対応）。
        Circuit制約を用いて複数エージェントの経路とタスク順序を同時最適化する。
        Bottleneck (Narrow Passage) Detection included.
        """
        num_agents = getattr(self, 'num_agents', 1)
        print(f"[CSPAgent] CSPスケジューリング開始 ({len(orders)} 注文, {num_agents} エージェント)...")
        model = cp_model.CpModel()
        
        # 0. Bottleneck Detection (Use Cached)
        # self.tile_to_zone_cache, self.zones_cache は _compute_static_map_data で計算済
        # 初回呼び出し前に必ず _compute_static_map_data が呼ばれている前提
        if not self.map_initialized:
             self._compute_static_map_data(env)
             self.map_initialized = True
             
        tile_to_zone = self.tile_to_zone_cache
        zones = self.zones_cache
        # print(f"[CSPAgent] 検出されたボトルネックゾーン数: {len(zones)}")

        # 1. タスクのリスト化
        tasks = []
        for o in orders:
            for t in o['tasks']:
                t['order_obj'] = o
                tasks.append(t)
        
        num_tasks = len(tasks)
        if num_tasks == 0:
            print("[CSPAgent] スケジュール対象タスクがありません。")
            return []

        # エージェント位置の取得 (Multiple Agents)
        agent_positions = []
        if hasattr(env, 'sim_agents'):
            agent_positions = [a.location for a in env.sim_agents]
        elif hasattr(env, 'agents'):
            agent_positions = [a.location for a in env.agents]
        
        # 位置情報不足時の補完
        while len(agent_positions) < num_agents:
            agent_positions.append((0, 0))
        agent_positions = agent_positions[:num_agents]

        # リソース位置の特定
        resources = self._get_resources(env)
        
        def get_nearest(start_pos_list, candidates):
            ref_pos = start_pos_list[0] 
            if not candidates: return ref_pos
            return min(candidates, key=lambda p: abs(p[0]-ref_pos[0]) + abs(p[1]-ref_pos[1]))

        # タスクのリソース割り当て (Location Fixing)
        for t in tasks:
            verb = t['verb']
            obj = t['obj']
            order_idx = t['order']
            
            if verb == 'chop':
                tile_map = {"lettuce": "FreshLettuceTile", "onion": "FreshOnionTile", "tomato": "FreshTomatoTile"}
                ing_pos_list = env.get_pos_by_obj_gs(gs=tile_map.get(obj, ""))
                ing_pos = ing_pos_list[0] if ing_pos_list else agent_positions[0]
                
                cutboards = resources['cutboards']
                best_cb = get_nearest([ing_pos], cutboards)
                
                t['start_pos'] = best_cb
                t['end_pos'] = best_cb
                t['fixed_res'] = ('cutboard', best_cb)
                
            elif verb == 'cook':
                pots = resources['pots']
                pot = pots[order_idx % len(pots)] if pots else agent_positions[0]
                t['start_pos'] = pot
                t['end_pos'] = pot
                t['fixed_res'] = ('pot', pot)
                
            elif verb == 'serve':
                pots = resources['pots']
                pot = pots[order_idx % len(pots)] if pots else agent_positions[0]
                delivery = resources['delivery']
                
                t['start_pos'] = pot
                t['end_pos'] = delivery
                t['fixed_res'] = ('pot', pot)

        # 2. 距離行列とパスの計算 (Use Cache)
        real_nodes = list(range(num_tasks))
        depot_nodes = list(range(num_tasks, num_tasks + num_agents))
        all_nodes = real_nodes + depot_nodes
        
        dist_matrix = {} 
        path_zone_usage = {} # (i,j) -> list of usage tuples

        # print("[CSPAgent] 距離行列とパスを計算中...")
        for i in all_nodes:
            for j in all_nodes:
                if i == j:
                    dist_matrix[(i,j)] = 0
                    continue
                
                if i in depot_nodes:
                    agent_idx = i - num_tasks
                    pos_i = agent_positions[agent_idx]
                else:
                    pos_i = tasks[i]['end_pos']
                
                if j in depot_nodes:
                    # Closing loop: cost 0
                    dist_matrix[(i,j)] = 0
                else:
                    pos_j = tasks[j]['start_pos']
                    
                    # Use Cache
                    if (pos_i, pos_j) in self.dist_cache:
                        dist = self.dist_cache[(pos_i, pos_j)]
                        dist_matrix[(i,j)] = dist
                        
                        # Retrieve cached zone usage
                        if (pos_i, pos_j) in self.path_zones_cache:
                             path_zone_usage[(i,j)] = self.path_zones_cache[(pos_i, pos_j)]
                    else:
                        # Fallback (e.g. initial pos_i might be (0,0) wall or off-grid?)
                        # Or if forbidden zones blocked BFS?
                        # For safety, use dynamic A* or Manhattan default
                        manhattan = abs(pos_i[0]-pos_j[0]) + abs(pos_i[1]-pos_j[1])
                        dist_matrix[(i,j)] = manhattan * 2 # Penalty

        # 3. 変数と制約
        horizon = 10000 
        starts = {}
        ends = {}
        
        # Real Tasks
        for i in real_nodes:
            t = tasks[i]
            dur = int(t['dur'])
            starts[i] = model.NewIntVar(0, horizon, f'start_{i}')
            ends[i] = model.NewIntVar(0, horizon, f'end_{i}')
            model.NewIntervalVar(starts[i], dur, ends[i], f'interval_{i}')

        # Depot Nodes
        for i in depot_nodes:
            starts[i] = model.NewIntVar(0, 0, f'start_depot_{i}')
            ends[i] = model.NewIntVar(0, 0, f'end_depot_{i}')

        # Arcs for Circuit
        arcs = []
        lit_map = {}
        
        # Bottleneck Usage Intervals
        # zone_id -> list of interval vars
        bottleneck_usage = {z: [] for z in zones}

        for i in all_nodes:
            for j in all_nodes:
                if i == j: continue
                
                if i in depot_nodes and j in depot_nodes:
                    current_agent = i - num_tasks
                    next_agent_target = (current_agent + 1) % num_agents
                    expected_next_depot = num_tasks + next_agent_target
                    if j != expected_next_depot:
                        continue 
                
                lit = model.NewBoolVar(f'arc_{i}_{j}')
                arcs.append((i, j, lit))
                lit_map[(i, j)] = lit
                
                if j not in depot_nodes:
                    dist = dist_matrix[(i, j)]
                    model.Add(starts[j] >= ends[i] + dist).OnlyEnforceIf(lit)
                    
                    # Passage Resource Constraints (From Cache)
                    if (i, j) in path_zone_usage:
                         usages = path_zone_usage[(i, j)]
                         for zone_id, start_offset, duration in usages:
                             
                            start_time_offset = start_offset * self.frames_per_action
                            dur_frames = duration * self.frames_per_action
                            
                            start_iv = model.NewIntVar(0, horizon, f'bn_start_{i}_{j}_{zone_id}_{start_offset}')
                            end_iv = model.NewIntVar(0, horizon, f'bn_end_{i}_{j}_{zone_id}_{start_offset}')
                            
                            # start_iv = ends[i] + start_time_offset
                            model.Add(start_iv == ends[i] + start_time_offset).OnlyEnforceIf(lit)
                            model.Add(end_iv == start_iv + dur_frames).OnlyEnforceIf(lit)
                            
                            iv = model.NewOptionalIntervalVar(start_iv, dur_frames, end_iv, lit, f'bn_iv_{i}_{j}_{zone_id}_{start_offset}')
                            bottleneck_usage[zone_id].append(iv)
                            
        model.AddCircuit(arcs)
        
        # Add NoOverlap for Bottlenecks
        for zid, iv_list in bottleneck_usage.items():
            if len(iv_list) > 1:
                model.AddNoOverlap(iv_list)

        # Resource Constraints (Overlap) - Cooking/Cutting
        vars_by_order = {} 
        for i in real_nodes:
            t = tasks[i]
            if t['order'] not in vars_by_order: vars_by_order[t['order']] = []
            vars_by_order[t['order']].append({'start': starts[i], 'end': ends[i], 'task': t})
            
        for i in real_nodes:
            t = tasks[i]
            verb = t['verb']
            if verb == 'cook':
                chops = [v for v in vars_by_order.get(t['order'], []) if v['task']['verb'] == 'chop']
                for c in chops:
                    model.Add(starts[i] >= c['end'])
            elif verb == 'serve':
                cooks = [v for v in vars_by_order.get(t['order'], []) if v['task']['verb'] == 'cook']
                for c in cooks:
                    model.Add(starts[i] >= c['end'])
                    model.Add(starts[i] >= c['end'] + 150) 

        pot_usage_intervals = {}
        for order_idx, v_list in vars_by_order.items():
            cooks = [v for v in v_list if v['task']['verb'] == 'cook']
            serves = [v for v in v_list if v['task']['verb'] == 'serve']
            if cooks and serves:
                c, s = cooks[0], serves[0]
                pot_res = c['task'].get('fixed_res')
                if pot_res and pot_res[0] == 'pot':
                    pot_loc = pot_res[1]
                    if pot_loc not in pot_usage_intervals: pot_usage_intervals[pot_loc] = []
                    
                    p_size = model.NewIntVar(0, horizon, f'size_pot_{order_idx}')
                    model.Add(p_size == s['end'] - c['start'])
                    p_iv = model.NewIntervalVar(c['start'], p_size, s['end'], f'iv_pot_{order_idx}')
                    pot_usage_intervals[pot_loc].append(p_iv)

        for intervals_list in pot_usage_intervals.values():
            if len(intervals_list) > 1:
                model.AddNoOverlap(intervals_list)

        makespan = model.NewIntVar(0, horizon, 'makespan')
        if real_nodes:
            model.AddMaxEquality(makespan, [ends[i] for i in real_nodes])
        model.Minimize(makespan)

        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        schedule = [[] for _ in range(num_agents)] 
        
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print(f"[CSPAgent] 最適Makespan: {solver.ObjectiveValue()}")
            for agent_idx in range(num_agents):
                curr = num_tasks + agent_idx 
                while True:
                    next_node = None
                    for j in all_nodes:
                        if curr == j: continue
                        if solver.Value(lit_map[(curr, j)]) == 1:
                            next_node = j
                            break
                    if next_node is None: break 
                    if next_node in depot_nodes: break 
                    
                    t = tasks[next_node]
                    schedule[agent_idx].append({
                        'id': t['id'],
                        'start': solver.Value(starts[next_node]),
                        'end': solver.Value(ends[next_node]),
                        'res': t.get('fixed_res'),
                        'start_pos': t.get('start_pos'),
                        'end_pos': t.get('end_pos'),
                        'agent_idx': agent_idx
                    })
                    curr = next_node

            for s in schedule:
                s.sort(key=lambda x: x['start'])
                
        else:
            print("[CSPAgent] 解が見つかりませんでした (Infeasible/Timeout)")
        
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
        total_weighted_unselected = 0
        for item in schedule:
            tid = item['id']; start=item['start']; end=item['end']; res=item['res']
            verb,obj,order = tid
            print(f"{verb} {obj} (注文{order+1}) : 開始={start}, 終了={end}, 資源={res}")
            total_frames = max(total_frames, end)
        print(f"総投入フレーム: {total_frames}")
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

    def astar_distance(self, env, start, goal, allow_forbidden_adjacent=False):
        import heapq
        width = env.world_width
        height = env.world_height
        grid = env.to_grid

        def in_bounds(x, y):
            return 0 <= x < width and 0 <= y < height

        def is_forbidden(x, y):
            if hasattr(self, 'forbidden_zones'):
                return (x, y) in self.forbidden_zones
            return False

        def walkable_primitive(x, y):
            return in_bounds(x, y) and grid[x][y] == 1

        original_goal = goal
        # ゴール地点の決定（立ち位置）
        if not walkable_primitive(goal[0], goal[1]):
            adjacents = []
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = goal[0]+dx, goal[1]+dy
                if walkable_primitive(nx, ny):
                     # Forbidden Zone チェック
                    if not is_forbidden(nx, ny):
                        adjacents.append((nx, ny))
                    elif allow_forbidden_adjacent:
                         # 特例: 立ち入り禁止だが目的地としてならOK
                         adjacents.append((nx, ny))
            
            if not adjacents:
                return None
            
            goal = min(adjacents, key=lambda p: abs(p[0]-start[0]) + abs(p[1]-start[1]))
        else:
             # Goal自体がWalkableなら、そこがForbiddenでないかチェック
             if is_forbidden(goal[0], goal[1]) and not allow_forbidden_adjacent:
                 return None

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
                
                if not walkable_primitive(nx, ny):
                    continue

                # Forbidden Check for Path
                if is_forbidden(nx, ny):
                    is_dest = (nx, ny) == goal
                    if not (is_dest and allow_forbidden_adjacent):
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