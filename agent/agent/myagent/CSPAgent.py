import random
from .csp.model import CSPModel
from .csp.solver import solve as solve_csp

class CSPAgent:
    """
    CSP(制約充足問題)ベースのエージェント
    現在は簡易的にランダムな行動を行う
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

        # ランナー方向（直進用）: None なら再計算
        self.run_direction = None  # (dx, dy)

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
        
        print("[CSPAgent] 初期化完了 - 現在はランダム行動")

    def act(self, observation):
        """
        ランダムな行動を返す（簡易実装）
        """
        actions = ['up', 'down', 'left', 'right', 'stay']
        return random.choice(actions)

    # 直進ランナー: 現在位置から最も長く進める方向を選ぶ
    def _longest_walkable_direction(self, env, loc):
        """
        locから上下左右へ、壁に当たるまでの連続歩数を数え、最大の方向を返す。
        同値なら優先順: up, right, down, left。
        戻り値: (dx, dy)
        """
        width = env.world_width
        height = env.world_height
        grid = env.to_grid

        def walkable(x, y):
            return 0 <= x < width and 0 <= y < height and grid[x][y] == 1

        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        best_dir = (0,0)
        best_len = -1
        for dx, dy in directions:
            x, y = loc
            length = 0
            while True:
                nx, ny = x + dx, y + dy
                if not walkable(nx, ny):
                    break
                length += 1
                x, y = nx, ny
            if length > best_len:
                best_len = length
                best_dir = (dx, dy)
        return best_dir

    def __call__(self, env):
        """
        環境から呼ばれるメイン関数
        """
        if not self.initialized:
            print("[CSPAgent] 初回起動 - CSP問題の構築準備")
            # 初回にタスクコストを出力
            try:
                self.print_task_costs(env)
            except Exception as e:
                print(f"[CSPAgent] タスクコスト出力中に例外: {e}")
            # 簡易CSPスケジューリング（選択問題：A解釈）
            try:
                self.schedule = self.solve_csp_selection(env)
                self._print_schedule(self.schedule)
            except Exception as e:
                print(f"[CSPAgent] CSPスケジュール中に例外: {e}")
            # OR-Tools による制約最適化（0-1選択の例）
            try:
                selected = self.solve_csp_knapsack_with_ortools(env)
                self._print_selection(selected)
            except Exception as e:
                print(f"[CSPAgent] OR-Tools選択最適化中に例外: {e}")
            # 将来ここでCSP問題を構築
            self.initialized = True

        # 直進ランナーの行動決定
        # 単一エージェントを想定（最初のsim_agent）。複数対応は拡張可能。
        # EnvState 仕様に合わせて現在エージェント位置を取得
        current_loc = env.self_pos

        # 直前の方向が塞がれる/Noneなら再計算
        if self.run_direction is None:
            self.run_direction = self._longest_walkable_direction(env, current_loc)
        else:
            width = env.world_width
            height = env.world_height
            grid = env.to_grid
            nx = current_loc[0] + self.run_direction[0]
            ny = current_loc[1] + self.run_direction[1]
            if not (0 <= nx < width and 0 <= ny < height and grid[nx][ny] == 1):
                self.run_direction = self._longest_walkable_direction(env, current_loc)

        action_vec = self.run_direction
        
        # 行動をタプル形式に変換
        move = action_vec
        chat = ""
        
        return move, chat

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
        pots = [(3,5), (4,5), (5,5)]  # 現行レベルの鍋座標（必要なら環境から取得へ）
        return {
            'cutboards': cutboards,
            'pots': pots,
            'delivery': (6,3),
            'plate': (6,6),
        }

    def _task_duration_frames(self, env, verb, obj, order_idx):
        """
        タスク所要フレーム数（移動含む）を返す。serveは重めの重みで扱うためベースは同じでも目的関数で重み付け。
        """
        if verb == 'chop':
            # 距離 + CHOP 8 + 置く1 + 取得1
            # 距離は食材→まな板→特定場所の最短合計
            tile_map = {"lettuce": "FreshLettuceTile", "onion": "FreshOnionTile", "tomato": "FreshTomatoTile"}
            ing_pos = env.get_pos_by_obj_gs(gs=tile_map[obj])
            cutboard_pos = env.get_pos_by_obj_gs(gs="Cutboard")
            special_places = [(2,3), (2,4), (2,5)]
            target = special_places[order_idx % len(special_places)]
            def adj(pos_list):
                width = env.world_width; height = env.world_height; grid = env.to_grid
                out=[]
                for x,y in pos_list:
                    for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nx,ny=x+dx,y+dy
                        if 0<=nx<width and 0<=ny<height and grid[nx][ny]==1:
                            out.append((nx,ny))
                return list(set(out))
            ing_adj=adj(ing_pos); cut_adj=adj(cutboard_pos); tgt_adj=adj([target])
            min_total=None
            for s in ing_adj:
                for m in cut_adj:
                    for e in tgt_adj:
                        d1=self.astar_distance(env,s,m); d2=self.astar_distance(env,m,e)
                        if d1 is None or d2 is None: continue
                        tot=d1+d2
                        if min_total is None or tot<min_total: min_total=tot
            if min_total is None:
                return None
            return int(min_total + 8 + 1 + 1)
        elif verb == 'cook':
            # 特定場所→鍋 距離 + 調理時間（秒→フレーム） + インタラクト2（~2フレーム）
            special_places=[(3,2),(4,2),(5,2)]
            pot_places=[(3,5),(4,5),(5,5)]
            d=self.astar_distance(env,special_places[order_idx%3], pot_places[order_idx%3])
            if d is None: return None
            cook_frames = 15 * self.fps
            return int(d + cook_frames + 2)
        elif verb == 'serve':
            plate=(6,6); pot_places=[(3,5),(4,5),(5,5)]; delivery=(6,3)
            d1=self.astar_distance(env, plate, pot_places[order_idx%3])
            d2=self.astar_distance(env, pot_places[order_idx%3], delivery)
            if d1 is None or d2 is None: return None
            return int(d1 + d2 + 3)
        else:
            return None

    def _task_weight(self, verb):
        return { 'chop': self.w_chop, 'cook': self.w_cook, 'serve': self.w_serve }.get(verb, 1)

    def _build_order_tasks(self, env):
        """
        注文ごとにタスク列を作成（chop群→cook→serve）。各タスクにID、所要時間、資源候補を付与。
        """
        resources = self._get_resources(env)
        orders = []
        order_idx = 0
        for order_tuple in env.order.current_orders:
            goal = order_tuple[0]
            name = getattr(goal, 'full_name', '').lower()
            ings = [ing for ing in ['lettuce','onion','tomato'] if ing in name]
            if not ings:
                order_idx += 1
                continue
            soup_name = '-'.join(ings) + ' soup'
            tasks=[]
            # chops
            for ing in ings:
                dur = self._task_duration_frames(env, 'chop', ing, order_idx)
                if dur is None: continue
                tasks.append({
                    'id': ('chop', ing, order_idx),
                    'verb':'chop','obj':ing,'order':order_idx,
                    'dur':dur,'weight':self._task_weight('chop'),
                    'res_candidates': [('cutboard', r) for r in resources['cutboards']],
                })
            # cook
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
            orders.append({'order':order_idx,'ingredients':ings,'tasks':tasks})
            order_idx += 1
        return orders

    def solve_csp_selection(self, env):
        """
        選択問題（A解釈）：予算300フレーム内でスケジュール可能なタスク集合を選び、未選択コスト（重み*dur）の合計を最小化。
        近似解法：順序制約と資源占有を守りつつ、serveを優先し、予算まで積み上げる。
        戻り値：スケジュールリスト [{id, start, end, res}]
        """
        orders = self._build_order_tasks(env)
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
            return in_bounds(x, y) and grid[x][y] == 1

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
