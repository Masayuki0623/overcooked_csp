import random
from ortools.sat.python import cp_model
from .csp.model import CSPModel
from .csp.solver import solve as solve_csp
from .TaskAgent import TaskAgent

class CSPAgent:
    """
    CSP(制約充足問題)ベースのエージェント
    """
    def __init__(self, speed=2.5, replay=None):
        self.speed = speed
        self.replay = replay
        self.initialized = False
        
        # CSP関連の変数（後で実装）
        self.variables = []  # CSPの変数
        self.domains = {}    # 各変数のドメイン
        self.constraints = [] # 制約のリスト
        # 入力フレーム間隔（1アクションに必要なフレーム数）
        # 環境側でフレームスキップがある場合にコストへ反映するための係数
        self.frames_per_action = 1  # デフォルト: 毎フレーム入力可能

        # FPS（フレーム→秒換算用）。環境側は10fps
        self.fps = 10
        # 期限（MAX_ORDER_LENGTH_SECONDS）をフレームへ
        self.deadline_frames = 75 * self.fps
        # 30秒の選択予算（A解釈）
        self.budget_frames = 30 * self.fps
        # タスク重み（serve重め）
        self.w_chop = 1
        self.w_cook = 2
        self.w_serve = 5
        
        # 実行状態管理
        self.current_task_idx = 0
        self.holding_state = None # 以前の持ち物状態（変化検知用）
        
        self.task_agent = TaskAgent()
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
            if self.initialized: # 初回以外なら差分を表示
                print(f"\n[Task Update] Time: {env.time}")
                if added:
                    print(f"  (+) Added: {added}")
                if removed:
                    print(f"  (-) Removed: {removed}")
                print("  -> Re-calculating Schedule...")

            # 簡易CSPスケジューリング（選択問題：A解釈）
            try:
                # solve_csp_selection内でタスクリスト全体も表示される
                # self.schedule = self.solve_csp_selection(env, orders=current_orders)
                
                # 新しいスケジューリングメソッドを使用
                self.schedule = self.solve_csp_scheduling(env, orders=current_orders)
                
                self._print_schedule(self.schedule)
            except Exception as e:
                print(f"[CSPAgent] CSPスケジュール中に例外: {e}")
                import traceback
                traceback.print_exc()
            
            self.prev_task_ids = current_task_ids
            self.initialized = True

        # スケジュール実行
        if not hasattr(self, 'schedule') or not self.schedule or self.current_task_idx >= len(self.schedule):
            return (0, 0), "No Task"

        task = self.schedule[self.current_task_idx]
        tid = task['id']
        verb, obj, order_idx = tid
        res = task['res'] # e.g. ('cutboard', (x,y)) or ('pot', (x,y))

        # Construct Task Name
        task_name = None
        if verb == 'chop':
            task_name = f"chop_{obj}"
            # self.task_agent.assigned_cutboard = res[1]
            # self.task_agent.assigned_pot = None
        elif verb == 'cook':
            # obj is soup name e.g. "tomato-onion soup"
            parts = obj.replace(' soup', '').split('-')
            task_name = f"cook_{'_'.join(parts)}"
            # self.task_agent.assigned_pot = res[1]
            # self.task_agent.assigned_cutboard = None
        elif verb == 'serve':
            parts = obj.replace(' soup', '').split('-')
            task_name = f"serve_{'_'.join(parts)}"
            # self.task_agent.assigned_pot = res[1]
            # self.task_agent.assigned_cutboard = None
            # serve needs plate and delivery? CSP schedule might not have them all in 'res'.
            # Assuming default plate/delivery logic in TaskAgent unless specified.
        
        if task_name:
            self.task_agent.task_name = task_name
            # Debug: Print what we are delegating
            # print(f"[CSPAgent] Delegating {task_name} to TaskAgent. Res: {res}")
            
            action, reason = self.task_agent(env)
            
            # Debug: Print result
            # print(f"  -> TaskAgent returned: {reason}")

            # Check completion
            if "Done" in reason or "done" in reason:
                print(f"[CSPAgent] Task {task_name} DONE. Moving to next.")
                self.current_task_idx += 1
                # Reset assignments
                self.task_agent.assigned_cutboard = None
                self.task_agent.assigned_pot = None
                self.task_agent.assigned_plate = None
                self.task_agent.assigned_serve_loc = None
            
            return action, reason

        return (0,0), "Idle"

    # ============ OR-Tools: 0-1選択問題（予算内で重み最大化） ============
    def solve_csp_knapsack_with_ortools(self, env):

        """
        0-1選択問題（Knapsack with precedence）：
        - 各タスクに Bool 変数 x_t を割当
        - 予算制約: sum(dur_t * x_t) <= budget
        - 前後関係: serve <= cook, cook <= chop_i（各注文）
        - 目的: sum(weight_t * dur_t * x_t) を最大化
        戻り値: 選択されたタスクリスト（開始時刻は付与しない簡易版）
        """
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

        # 予算制約
        model.add_linear_le(durations, self.budget_frames)

        # 前後関係（各注文に対して）
        # cook は全chopが選ばれている必要がある: x_cook <= x_chop_i
        # serve は cook が選ばれている必要がある: x_serve <= x_cook
        # NewBoolVarはIntVar扱いなので線形比較で十分（0/1）
        # ここでは model.model 直接アクセスで関係式を追加（具体的な変数参照が必要なため）
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

        # 目的関数（重み*時間の合計を最大化）
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
            print(f"選択: {verb} {obj} (注文{order+1}) dur={dur}, weight={w}")
        print(f"合計投入フレーム(選択分): {total}")
        print("===================================\n")

    # ============ CSP（選択問題 A解釈） ============
    def _get_resources(self, env):
        """
        資源ID（座標）を取得。
        Cutboard複数、Pot複数に対応。戻り値は辞書。
        """
        cutboards = env.get_pos_by_obj_gs(gs="Cutboard")
        pots = env.get_pos_by_obj_gs(gs="Pot")
        deliveries = env.get_pos_by_obj_gs(gs="Delivery")
        plates = env.get_pos_by_obj_gs(gs="Plate") # Plate object
        if not plates:
            plates = env.get_pos_by_obj_gs(gs="PlateTile") # Dispenser

        return {
            'cutboards': cutboards,
            'pots': pots,
            'delivery': deliveries[0] if deliveries else (0,0),
            'plate': plates[0] if plates else (0,0),
        }

    def _task_duration_frames(self, env, verb, obj, order_idx):
        """
        タスク所要フレーム数（移動含む）を返す。serveは重めの重みで扱うためベースは同じでも目的関数で重み付け。
        """
        resources = self._get_resources(env)
        
        def get_nearest(start_pos, candidates):
            if not candidates: return None
            if not start_pos: return candidates[0]
            # Simple Manhattan distance for heuristic
            return min(candidates, key=lambda p: abs(p[0]-start_pos[0]) + abs(p[1]-start_pos[1]))

        if verb == 'chop':
            # 距離 + CHOP 8 + 置く1 + 取得1 + 置く1
            # 距離は食材→まな板→特定場所の最短合計
            tile_map = {"lettuce": "FreshLettuceTile", "onion": "FreshOnionTile", "tomato": "FreshTomatoTile"}
            ing_pos_list = env.get_pos_by_obj_gs(gs=tile_map.get(obj, ""))
            if not ing_pos_list: return None
            ing_pos = ing_pos_list[0] # Assume first dispenser

            cutboard_pos_list = resources['cutboards']
            if not cutboard_pos_list: return None
            
            # Find nearest cutboard to ingredient
            cutboard_pos = get_nearest(ing_pos, cutboard_pos_list)
            
            # Find nearest counter to cutboard (to place chopped item)
            counters = env.get_pos_by_obj_gs(gs="Counter")
            if not counters: return None
            target = get_nearest(cutboard_pos, counters)

            def adj(pos_list):
                width = env.world_width; height = env.world_height; grid = env.to_grid
                out=[]
                for x,y in pos_list:
                    for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nx,ny=x+dx,y+dy
                        if 0<=nx<width and 0<=ny<height and grid[nx][ny]==1:
                            out.append((nx,ny))
                return list(set(out))
            
            # Calculate path: Ing -> Cutboard -> Counter
            # Note: astar_distance takes (start, goal). We need adjacent cells.
            # Simplified: just use center-to-center astar for estimation, or use existing adj logic
            
            ing_adj=adj([ing_pos]); cut_adj=adj([cutboard_pos]); tgt_adj=adj([target])
            
            min_total=None
            # Try to find valid path
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
            # Counter (Chopped) -> Pot
            # We don't know exactly which counter, so assume nearest counter to Pot
            pot_pos_list = resources['pots']
            if not pot_pos_list: return None
            
            # Use order_idx to distribute pots if multiple available? 
            # For duration estimation, just pick one (e.g. first or nearest to center)
            pot_pos = pot_pos_list[order_idx % len(pot_pos_list)]
            
            counters = env.get_pos_by_obj_gs(gs="Counter")
            if not counters: return None
            start_pos = get_nearest(pot_pos, counters)
            
            d = self.astar_distance(env, start_pos, pot_pos)
            if d is None: return None
            return int(d + 2)

        elif verb == 'serve':
            # Plate -> Pot -> Delivery
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
        """
        注文ごとにタスク列を作成（chop群→cook→serve）。各タスクにID、所要時間、資源候補を付与。
        環境の状態（Chopped食材の有無、Potの中身）を考慮し、不要なタスクは生成しない。
        """
        # 1. 環境内のChopped食材をカウント
        available_chopped = {} # {'Tomato': count, ...}
        
        # 2. Potの中身を確認
        pot_states = [] # [{'names': ['Tomato', 'Lettuce'], 'obj': obj, 'used': False}, ...]

        # env.world.get_object_list() を使用 -> env.world_all (EnvState) or env.world.get_object_list() (Environment)
        if hasattr(env, 'world_all'):
            all_objects = env.world_all
        elif hasattr(env, 'world'):
            all_objects = env.world.get_object_list()
        else:
            all_objects = []

        # Potの場所を取得 (PotはGridSquare)
        pot_locs = []
        for o in all_objects:
             if getattr(o, 'name', '') == 'Pot':
                 pot_locs.append(o.location)
        
        # Cutboardの場所を取得
        cutboard_locs = env.get_pos_by_obj_gs(gs="Cutboard")

        for obj in all_objects:
            # Objectクラスのインスタンスかどうかを判定（簡易的）
            if type(obj).__name__ == 'Object':
                # Chopped check: is_chopped() がTrue かつ 単一の食材
                # 【修正】まな板の上にあるChopped食材は「完了」とみなさない（タスクを継続させてテーブルへ移動させるため）
                if hasattr(obj, 'is_chopped') and obj.is_chopped() and len(obj.contents) == 1 and not obj.is_held:
                    if obj.location not in cutboard_locs:
                        food_name = obj.contents[0].name
                        available_chopped[food_name] = available_chopped.get(food_name, 0) + 1
                
                # Pot check: Potの場所にあるObject
                if obj.location in pot_locs:
                    # Potの中身
                    c_names = sorted([c.name for c in obj.contents])
                    pot_states.append({'names': c_names, 'obj': obj, 'used': False})

        resources = self._get_resources(env)
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
            
            # 照合用にCapitalize
            ings_cap = [ing.capitalize() for ing in ings_lower]
            
            soup_name = '-'.join(ings_lower) + ' soup'
            tasks=[]
            
            # Cookタスクが必要か判定
            sorted_ings = sorted(ings_cap)
            cook_needed = True
            
            # Potの状態と照合
            for ps in pot_states:
                if not ps['used'] and ps['names'] == sorted_ings:
                    ps['used'] = True
                    cook_needed = False
                    break

            # chops
            for ing in ings_cap:
                # 既にPotに入っているならChopも不要
                if not cook_needed:
                    continue

                # 環境にChopped食材があるか確認
                if available_chopped.get(ing, 0) > 0:
                    available_chopped[ing] -= 1
                    continue # Skip chop task

                dur = self._task_duration_frames(env, 'chop', ing.lower(), order_idx)
                if dur is None: continue
                tasks.append({
                    'id': ('chop', ing.lower(), order_idx),
                    'verb':'chop','obj':ing.lower(),'order':order_idx,
                    'dur':dur,'weight':self._task_weight('chop'),
                    'res_candidates': [('cutboard', r) for r in resources['cutboards']],
                })
            # cook
            if cook_needed:
                dur = self._task_duration_frames(env, 'cook', soup_name, order_idx)
                if dur is not None:
                    tasks.append({
                        'id': ('cook', soup_name, order_idx),
                        'verb':'cook','obj':soup_name,'order':order_idx,
                        'dur':dur,'weight':self._task_weight('cook'),
                        'res_candidates': [('pot', r) for r in resources['pots']],
                    })
            # serve
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
        OR-Tools CP-SAT を用いたスケジューリング（Makespan最小化）。
        """
        print(f"[CSPAgent] Solving CSP Scheduling for {len(orders)} orders...")
        model = cp_model.CpModel()
        
        # 1. 変数定義
        tasks_vars = {}
        agent_intervals = []
        
        # リソース割り当て変数の保存用
        # task_id -> {loc: bool_var}
        chop_res_vars = {} 
        # order_idx -> {loc: bool_var}
        pot_res_vars = {}

        resources = self._get_resources(env)
        cutboard_locs = resources['cutboards']
        pot_locs = resources['pots']
        
        cutboard_intervals = {loc: [] for loc in cutboard_locs}
        pot_intervals = {loc: [] for loc in pot_locs}
        
        all_end_vars = []
        # Horizon (十分大きな値)
        # 150フレームの待機時間があるため、タスク数が多いと時間は伸びる。
        # 予算制約ではなく物理的な限界として大きめに設定する。
        horizon = 10000 
        
        total_tasks_count = 0

        for o in orders:
            chops = [t for t in o['tasks'] if t['verb'] == 'chop']
            cooks = [t for t in o['tasks'] if t['verb'] == 'cook']
            serves = [t for t in o['tasks'] if t['verb'] == 'serve']
            
            total_tasks_count += len(chops) + len(cooks) + len(serves)

            # --- Chop Tasks ---
            chop_ends = []
            for t in chops:
                dur = int(t['dur'])
                start_var = model.NewIntVar(0, horizon, f"start_{t['id']}")
                end_var = model.NewIntVar(0, horizon, f"end_{t['id']}")
                agent_interval = model.NewIntervalVar(start_var, dur, end_var, f"interval_{t['id']}")
                agent_intervals.append(agent_interval)
                all_end_vars.append(end_var)
                chop_ends.append(end_var)
                
                tasks_vars[t['id']] = {'start': start_var, 'end': end_var, 'task': t}
                
                opts = []
                chop_res_vars[t['id']] = {}
                for c_loc in cutboard_locs:
                    is_present = model.NewBoolVar(f"pres_{t['id']}_{c_loc}")
                    chop_res_vars[t['id']][c_loc] = is_present
                    
                    opt_interval = model.NewOptionalIntervalVar(start_var, dur, end_var, is_present, f"opt_{t['id']}_{c_loc}")
                    cutboard_intervals[c_loc].append(opt_interval)
                    opts.append(is_present)
                model.Add(sum(opts) == 1)

            # --- Cook Task ---
            cook_end = None
            cook_start = None
            if cooks:
                t = cooks[0]
                dur = int(t['dur'])
                start_var = model.NewIntVar(0, horizon, f"start_{t['id']}")
                end_var = model.NewIntVar(0, horizon, f"end_{t['id']}")
                agent_interval = model.NewIntervalVar(start_var, dur, end_var, f"interval_{t['id']}")
                agent_intervals.append(agent_interval)
                all_end_vars.append(end_var)
                
                cook_start = start_var
                cook_end = end_var
                tasks_vars[t['id']] = {'start': start_var, 'end': end_var, 'task': t}
                
                for ce in chop_ends:
                    model.Add(ce <= start_var)

            # --- Serve Task ---
            serve_start = None
            if serves:
                t = serves[0]
                dur = int(t['dur'])
                start_var = model.NewIntVar(0, horizon, f"start_{t['id']}")
                end_var = model.NewIntVar(0, horizon, f"end_{t['id']}")
                agent_interval = model.NewIntervalVar(start_var, dur, end_var, f"interval_{t['id']}")
                agent_intervals.append(agent_interval)
                all_end_vars.append(end_var)
                
                serve_start = start_var
                tasks_vars[t['id']] = {'start': start_var, 'end': end_var, 'task': t}
                
                if cook_end is not None:
                    model.Add(cook_end <= start_var)
                    model.Add(start_var >= cook_end + 150)

            # --- Pot Allocation & Usage ---
            if cooks and serves:
                pot_opts = []
                pot_res_vars[o['order']] = {}
                for p_loc in pot_locs:
                    is_present = model.NewBoolVar(f"pres_pot_{o['order']}_{p_loc}")
                    pot_res_vars[o['order']][p_loc] = is_present
                    pot_opts.append(is_present)
                    
                    # Pot占有区間: Cook終了 〜 Serve開始
                    # Cookタスク自体は「鍋に入れる」だけなので、その終了時刻から占有が始まる
                    # Serve開始時に鍋から取り出すので、Serve開始まで占有
                    
                    # duration = serve_start - cook_end
                    duration_var = model.NewIntVar(0, horizon, f"dur_pot_{o['order']}_{p_loc}")
                    model.Add(duration_var == serve_start - cook_end)
                    
                    # OptionalInterval
                    # start=cook_end, size=duration_var, end=serve_start
                    pot_interval = model.NewOptionalIntervalVar(
                        cook_end, duration_var, serve_start, 
                        is_present, 
                        f"opt_pot_{o['order']}_{p_loc}"
                    )
                    pot_intervals[p_loc].append(pot_interval)
                
                model.Add(sum(pot_opts) == 1)

        print(f"[CSPAgent] Total tasks to schedule: {total_tasks_count}")
        if total_tasks_count == 0:
            print("[CSPAgent] No tasks to schedule.")
            return []

        # 2. 資源制約 (NoOverlap)
        model.AddNoOverlap(agent_intervals)
        
        for loc, intervals in cutboard_intervals.items():
            model.AddNoOverlap(intervals)
            
        for loc, intervals in pot_intervals.items():
            model.AddNoOverlap(intervals)

        # 3. 目的関数: Makespan最小化
        makespan = model.NewIntVar(0, horizon, 'makespan')
        if all_end_vars:
            model.AddMaxEquality(makespan, all_end_vars)
        model.Minimize(makespan)

        # 4. 解く
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        print(f"[CSPAgent] Solver Status: {solver.StatusName(status)}")

        schedule = []
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print(f"[CSPAgent] Optimal Makespan: {solver.ObjectiveValue()}")
            for tid, v in tasks_vars.items():
                start_val = solver.Value(v['start'])
                end_val = solver.Value(v['end'])
                
                res = None
                t = v['task']
                verb = t['verb']
                order_idx = t['order']
                
                if verb == 'chop':
                    # 割り当てられたまな板を探す
                    for loc, var in chop_res_vars.get(tid, {}).items():
                        if solver.Value(var) == 1:
                            res = ('cutboard', loc)
                            break
                    if res is None: res = ('cutboard', '?')
                    
                elif verb == 'cook':
                    # 割り当てられた鍋を探す
                    for loc, var in pot_res_vars.get(order_idx, {}).items():
                        if solver.Value(var) == 1:
                            res = ('pot', loc)
                            break
                    if res is None: res = ('pot', '?')
                    
                elif verb == 'serve':
                    # 割り当てられた鍋を探す（cookと同じはず）
                    for loc, var in pot_res_vars.get(order_idx, {}).items():
                        if solver.Value(var) == 1:
                            res = ('pot', loc)
                            break
                    if res is None: res = ('pot', '?')
                
                schedule.append({
                    'id': tid,
                    'start': start_val,
                    'end': end_val,
                    'res': res
                })
            schedule.sort(key=lambda x: x['start'])
        else:
            print(f"[CSPAgent] No solution found. Status: {solver.StatusName(status)}")
            
        return schedule

    def solve_csp_selection(self, env, orders=None):
        """
        選択問題（A解釈）：予算300フレーム内でスケジュール可能なタスク集合を選び、未選択コスト（重み*dur）の合計を最小化。
        近似解法：順序制約と資源占有を守りつつ、serveを優先し、予算まで積み上げる。
        戻り値：スケジュールリスト [{id, start, end, res}]
        """
        if orders is None:
            orders = self._build_order_tasks(env)
        
        # タスク列の表示
        print("\n--- Generated Tasks (Filtered by Environment State) ---")
        for o in orders:
            print(f"Order {o['order']} (Ings: {o['ingredients']}):")
            if not o['tasks']:
                print("  (No tasks needed)")
            for t in o['tasks']:
                print(f"  - {t['id']}: dur={t['dur']}, res={t['res_candidates']}")
        print("-------------------------------------------------------\n")

        budget = self.budget_frames
        # 資源タイムライン（各IDごとに終了時刻）
        res_timeline = {'cutboard':{}, 'pot':{}}
        schedule = []
        time_cursor = 0

        # serve重視のため、注文をserve重みでソート（擬似的に）
        orders_sorted = sorted(orders, key=lambda o: sum(t['weight']*t['dur'] for t in o['tasks'] if t['verb']=='serve'), reverse=True)

        for o in orders_sorted:
            # 前段階の終了時刻を追跡
            prev_finish = time_cursor
            # まず全chop
            for t in [t for t in o['tasks'] if t['verb']=='chop']:
                # cutboard資源割当：最も早く空くIDを選ぶ
                if not t['res_candidates']:
                    continue
                # 候補の中で最小の空き時刻
                best_id=None; earliest=0
                for _, rid in t['res_candidates']:
                    free = res_timeline['cutboard'].get(rid, 0)
                    if best_id is None or free < earliest:
                        best_id=rid; earliest=free
                start = max(prev_finish, earliest)
                end = start + t['dur']
                if end - time_cursor > budget:
                    # 予算超過ならこの注文の残りはスキップ
                    break
                res_timeline['cutboard'][best_id] = end
                schedule.append({'id':t['id'],'start':start,'end':end,'res':('cutboard',best_id)})
                prev_finish = end
            # cook
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
                        res_timeline['pot'][best_id] = end  # 調理中占有
                        schedule.append({'id':t['id'],'start':start,'end':end,'res':('pot',best_id)})
                        prev_finish = end
            # serve
            serves = [t for t in o['tasks'] if t['verb']=='serve']
            if serves:
                t = serves[0]
                start = prev_finish
                end = start + t['dur']
                if end - time_cursor <= budget:
                    schedule.append({'id':t['id'],'start':start,'end':end,'res':None})
                    prev_finish = end

            # 予算更新（累積投入分）
            if prev_finish - time_cursor > budget:
                break
            else:
                budget -= (prev_finish - time_cursor)
                time_cursor = prev_finish

        return schedule

    def _print_schedule(self, schedule):
        print("\n=== CSP スケジュール（フレーム単位） ===")
        total_frames = 0
        total_weighted_unselected = 0  # 近似実装では未選択評価を省略（必要なら追加）
        for item in schedule:
            tid = item['id']; start=item['start']; end=item['end']; res=item['res']
            verb,obj,order = tid
            print(f"{verb} {obj} (注文{order+1}) : start={start}, end={end}, res={res}")
            total_frames = max(total_frames, end)
        print(f"総投入フレーム: {total_frames}")
        print("===================================\n")

    # ============ 距離計算: A* ============
    def astar_distance(self, env, start, goal):
        """
        A*アルゴリズムで2座標間の最短距離(歩数)を返す。
        引数: start=(x,y), goal=(x,y)
        戻り値: 距離(整数)。到達不能ならNone。
        """
        import heapq
        width = env.world_width
        height = env.world_height
        grid = env.to_grid

        def in_bounds(x, y):
            return 0 <= x < width and 0 <= y < height

        def walkable(x, y):
            # Relaxed walkable check: allow goal to be non-walkable (e.g. interaction target)
            # But A* usually finds path to adjacent.
            # Here we assume start/goal are walkable OR we want path to adjacent?
            # If goal is a counter/pot (grid=0), we can't step ON it.
            # So we should find path to adjacent of goal.
            return in_bounds(x, y) and grid[x][y] == 1

        # If goal is not walkable (e.g. object), we need to find path to adjacent
        original_goal = goal
        if not walkable(goal[0], goal[1]):
            # Find nearest walkable adjacent
            adjacents = []
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = goal[0]+dx, goal[1]+dy
                if walkable(nx, ny):
                    adjacents.append((nx, ny))
            
            if not adjacents:
                return None
            
            # Pick closest adjacent to start
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

    # ============ タスクコスト出力(簡易) ============
    def print_task_costs(self, env):
        """
        現在の注文からタスク列を生成し、各タスクのコストを簡易に出力する。
        TSPソルバーの出力形式を参考に再実装。
        """
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

        # タスクコスト計算(簡易版):
        # chop: 食材タイルの隣接 -> まな板隣接 -> 特定場所隣接 の最短合計移動 + 定数
        # cook: 特定場所 -> 鍋 の移動 + インタラクト定数
        # serve: 皿 -> 鍋 -> 配膳 の移動 + 取得/配膳定数
        # さらに、1アクションあたり必要フレーム数(self.frames_per_action)をコストへ乗算して反映
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

        # 固定座標(現行レベルに合わせた簡易設定)
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
                        # 定数コスト(簡易): chop動作8 + 置く1 + 取得1
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
                        # インタラクト2回を+2
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
                        # 皿取得1 + 料理取得1 + 配膳1 を+3
                        base = d1 + d2 + 3
                        cost = base * self.frames_per_action
                        print((verb, obj, order_idx), ":", cost)
        print("===============================")
