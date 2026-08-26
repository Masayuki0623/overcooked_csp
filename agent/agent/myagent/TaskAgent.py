import heapq
import random
from gym_cooking.utils.core import mergeable

class TaskAgent:
    def __init__(self, speed=2.5, replay=None, task_name=None):
        # 「切らずに運ぶだけ」の運び元カウンター(CSPAgent が毎フレーム設定する)
        self.carry_from = None
        self.speed = speed
        self.replay = replay
        self.task_name = task_name
        self.strict_counter_management = False
        
        # Assigned resources for CSP
        self.assigned_cutboard = None
        self.assigned_pot = None
        self.assigned_plate = None
        self.assigned_serve_loc = None
        # ジュース用。鍋・皿と同じ役割のミキサーとコップ。
        self.assigned_blender = None
        self.assigned_cup = None
        # 受け渡し系のタスクは、料理の種類が分からないと完成品を判別できない。
        self.dish_kind = None
        self.assigned_counter = None 
        self.assigned_task_id = None
        self.protected_counters = set()
        
        # 経路予約用（Cooperative A*）
        self.planned_path = []
        self.wait_count = 0  # 待機カウンターを追加

        # move_to() の振動防止用（前回選んだ目的地/進入マスを記憶する）
        self._last_move_target_pos = None
        self._last_adjacent_goal = None
        
        #print(f"[TaskAgent] タスクで初期化: {self.task_name}")

    def _is_available_object(self, obj):
        return obj is not None and not getattr(obj, 'is_held', False)

    def _filter_unheld_positions(self, env, positions):
        filtered = []
        for pos in positions:
            obj = env.pos_obj.get(pos)
            if obj is None or not getattr(obj, 'is_held', False):
                filtered.append(pos)
        return filtered

    def choose_random_chop_task_name(self, env):
        """現在の注文から chop 対象を 1 つランダムに選ぶ"""
        candidates = []
        for order_tuple in getattr(env.order, 'current_orders', []):
            goal_obj = order_tuple[0]
            name = getattr(goal_obj, 'full_name', '').lower()
            for ingredient in ('lettuce', 'onion', 'tomato'):
                if ingredient in name and ingredient not in candidates:
                    candidates.append(ingredient)

        if not candidates:
            return None

        return f"chop_{random.choice(candidates)}"

    def _get_counter_content_name(self, env, counter_pos):
        counter_obj = env.pos_obj.get(counter_pos)
        if counter_obj is None:
            return None
        return getattr(counter_obj, 'full_name', None)

    def _resolve_assigned_counter_target(self, env, holding, assigned_counter, blocked_reason):
        if not assigned_counter:
            return None, None

        counter_obj = env.pos_obj.get(assigned_counter)
        if counter_obj is None:
            return assigned_counter, None

        holding_name = getattr(holding, 'full_name', '') if holding is not None else ''
        counter_name = getattr(counter_obj, 'full_name', '') if counter_obj is not None else ''
        if not counter_name:
            return assigned_counter, None

        def extract_ingredients(name):
            if not name:
                return set()
            candidates = []
            for token in name.replace('-', ' ').replace('_', ' ').replace('/', ' ').split():
                normalized = token.strip().lower()
                for prefix in ('fresh', 'chopped', 'cooked', 'cooking', 'raw', 'cut'):
                    if normalized.startswith(prefix):
                        normalized = normalized[len(prefix):]
                        break
                if normalized in ('lettuce', 'onion', 'tomato'):
                    candidates.append(normalized)
            return set(candidates)

        holding_ings = extract_ingredients(holding_name)
        counter_ings = extract_ingredients(counter_name)

        # assigned counter はその注文専用の保持場所とみなすが、
        # 「必要な食材が既に counter 上にある」か「same ingredient overlap がある」場合は
        # 安全な partial state とみなし、再割り当てや待機を起こさない。
        if holding_ings or counter_ings:
            valid_ingredients = {'lettuce', 'onion', 'tomato'}
            if holding_ings <= valid_ingredients and counter_ings <= valid_ingredients:
                if holding_ings & counter_ings or holding_ings or counter_ings:
                    return assigned_counter, None

        if holding_ings and counter_ings:
            shared = holding_ings & counter_ings
            if shared:
                return assigned_counter, None

        if mergeable(holding, counter_obj):
            return assigned_counter, None

        content_name = self._get_counter_content_name(env, assigned_counter)
        return None, f"{blocked_reason}: {assigned_counter} content='{content_name}'"

    def _handle_counter_fallback(self, wait_reason, fallback_func):
        if self.strict_counter_management:
            return (0, 0), wait_reason
        return fallback_func()

    def _log_chop_debug(self, env, ing_name, holding_name, assigned_cutboard, assigned_counter, stage, target=None, reason=None):
        # 既定は OFF。1回の判断ごとに複数行を出力するため、常時ONだと実コンソールへの
        # 書き込みが支配的になり、AI が毎フレーム動けなくなる。
        if not getattr(self, 'debug_trace', False):
            return
        current_time = getattr(env, 'time', None)
        parts = [
            f"[TaskAgent][CHOP] time={current_time}",
            f"agent={getattr(env, 'agent_idx', None)}",
            f"ing={ing_name}",
            f"hold={holding_name}",
            f"cutboard={assigned_cutboard}",
            f"counter={assigned_counter}",
            f"stage={stage}",
        ]
        if target is not None:
            parts.append(f"target={target}")
        if reason:
            parts.append(f"reason={reason}")
        print(" ".join(parts))

    def astar_path(self, env, start, goal, dynamic_obstacles=None):
        if dynamic_obstacles is None:
            dynamic_obstacles = set()
            
        width = env.world_width
        height = env.world_height
        grid = env.to_grid_a

        def in_bounds(x, y):
            return 0 <= x < width and 0 <= y < height

        def walkable(x, y):
            return in_bounds(x, y) and grid[x][y] == 1 and (x, y) not in dynamic_obstacles

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            cx, cy = current
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = cx+dx, cy+dy
                if not walkable(nx, ny):
                    continue
                neighbor = (nx, ny)
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None
        
    def astar_path_cost(self, env, start, goal, dynamic_obstacles=None):
        if dynamic_obstacles is None:
            dynamic_obstacles = set()
            
        width = env.world_width
        height = env.world_height
        grid = env.to_grid_a

        def in_bounds(x, y):
            return 0 <= x < width and 0 <= y < height

        def walkable(x, y):
            # 床であれば歩けるが、後で障害物コストを付与する
            return in_bounds(x, y) and grid[x][y] == 1 

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path, g_score[goal]

            cx, cy = current
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = cx+dx, cy+dy
                if not walkable(nx, ny):
                    continue
                neighbor = (nx, ny)
                
                # 通常のコストは1、他のエージェントが居る場合はペナルティを与える（迂回を優先させる）
                step_cost = 1
                if neighbor in dynamic_obstacles:
                    step_cost = self.DYNAMIC_OBSTACLE_PENALTY  # ここを通るなら大きく遠回りしてでも避けるプランを採用する

                tentative_g = g_score[current] + step_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None, float('inf')

    # 動的障害物(相手プレイヤー)が経路上にいるときのペナルティコスト。
    DYNAMIC_OBSTACLE_PENALTY = 20

    # 隣接進入マスの選択を切り替えるために必要な最小コスト差。
    # 相手プレイヤーが自分の進行ルート上のたった1マスに一時的に重なっただけでも
    # DYNAMIC_OBSTACLE_PENALTY 分コストが跳ね上がるため、このマージンを
    # ペナルティ以上に設定しないと「相手が動くたびに毎フレーム左右のルートが
    # 入れ替わって揺れる」挙動を防げない。ペナルティ超過分だけは、
    # 実際に長time経路が塞がれた/明確に短いルートが空いた場合とみなし
    # 切り替えを許可する。
    ADJACENT_GOAL_STICKY_MARGIN = DYNAMIC_OBSTACLE_PENALTY + 1

    def _deadlock_escape_step(self, env, self_pos, dynamic_obstacles):
        """15フレーム以上待機した場合の退避ステップ(お見合い防止)。動けなければ None。"""
        import random
        escapes = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = self_pos[0] + dx, self_pos[1] + dy
            if 0 <= nx < env.world_width and 0 <= ny < env.world_height and env.to_grid_a[nx][ny] == 1:
                if (nx, ny) not in dynamic_obstacles:
                    escapes.append((dx, dy))
        if not escapes:
            return None
        self.wait_count = 0
        return random.choice(escapes)

    def move_to(self, env, target_pos, dynamic_obstacles=None):
        # 経路は毎フレーム、実際の現在地(self_pos)から作り直す。
        # このゲームは speed パラメータによる連続的な移動を扱っており、
        # 「1回のアクションで必ず1マス分ちょうど進む」とは限らない。
        # そのため経路を跨フレームでキャッシュして辿ろうとすると、
        # 想定した位置と実際の self_pos がズレたときに1マスを超える
        # 不正な移動ベクトルを返してしまう(過去に実際に発生した不具合)。
        # 経路そのものは常に再計算しつつ、「どちらの隣接マスから近づくか」
        # の選択だけを固執させることで、対称な地形(環状通路など)での
        # 左右往復(揺れ)を防止する。
        dynamic_obstacles = dynamic_obstacles or set()
        self.planned_path = [] # リセット
        self_pos = env.self_pos
        dist = abs(self_pos[0] - target_pos[0]) + abs(self_pos[1] - target_pos[1])
        if dist == 1:
            self.wait_count = 0
            self._last_move_target_pos = None
            self._last_adjacent_goal = None
            #print(f"  [MoveTo] ターゲット {target_pos} に隣接。インタラクトします。")
            return (target_pos[0] - self_pos[0], target_pos[1] - self_pos[1])

        adjacents = []
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = target_pos[0]+dx, target_pos[1]+dy
            if 0 <= nx < env.world_width and 0 <= ny < env.world_height and env.to_grid_a[nx][ny] == 1:
                adjacents.append((nx, ny))

        if not adjacents:
            #print(f"  [MoveTo] ターゲット {target_pos} の歩行可能な隣接セルがありません (to_grid_a で確認)")
            self._last_move_target_pos = None
            self._last_adjacent_goal = None
            return (0,0)

        # ターゲット自体が変わったときだけ、固執していた進入マスをリセットする
        if target_pos != self._last_move_target_pos:
            self._last_move_target_pos = target_pos
            self._last_adjacent_goal = None

        # ターゲットの隣接マスのうち、コスト(距離＋障害物ペナルティ)が最小のルート(プラン)を採用する
        costs = {}
        paths = {}
        for adj in adjacents:
            path, cost = self.astar_path_cost(env, self_pos, adj, dynamic_obstacles=dynamic_obstacles)
            if path:
                costs[adj] = cost
                paths[adj] = path

        if not costs:
            return (0, 0)

        best_adj = min(costs, key=costs.get)
        min_cost = costs[best_adj]

        # 振動防止: 前回選んだ進入マスがまだ有効(コスト差がわずか)なら、そちらを優先して維持する。
        # これにより、相手プレイヤーの移動でコストが僅かに変わるたびに
        # 進入マス(≒接近する向き)が入れ替わって前後に揺れる挙動を防ぐ。
        # 経路自体は毎回 self_pos から再計算するため、実際の移動量とのズレは生じない。
        chosen_adj = best_adj
        sticky_adj = self._last_adjacent_goal
        if sticky_adj in costs and costs[sticky_adj] <= min_cost + self.ADJACENT_GOAL_STICKY_MARGIN:
            chosen_adj = sticky_adj

        self._last_adjacent_goal = chosen_adj
        best_path = paths[chosen_adj]

        self.planned_path = best_path
        next_step = best_path[0]

        # もし次の一歩が他のエージェントの現在位置なら、通り過ぎるのを待機する
        if next_step in dynamic_obstacles:
            self.wait_count += 1
            # print(f"[{env.agent_idx}:{self.task_name}] 最短距離上の障害物を避ける迂回ルートがない(またはコスト高すぎる)と判断し待機 (wait={self.wait_count}, cost={min_cost})")

            # デッドロック（お互いにお見合いで同じ場所で立ち往生する）防止策
            if self.wait_count > 15:
                escape = self._deadlock_escape_step(env, self_pos, dynamic_obstacles)
                if escape is not None:
                    # print(f"[{env.agent_idx}:{self.task_name}] お見合いが長すぎたため退避します！")
                    return escape

            return (0, 0)

        self.wait_count = 0
        return (next_step[0] - self_pos[0], next_step[1] - self_pos[1])
        
        # 目的地が塞がれている場合：到達可能な範囲内で目的地に最も近い「空きマス（一時的な目的地）」を探す
        #print(f"  [MoveTo] {target_pos} への経路がないため、可能な限り近い場所へ一時退避・接近します")
        
        width = env.world_width
        height = env.world_height
        grid = env.to_grid_a
        def walkable(x, y):
            dynamic_obs = dynamic_obstacles if dynamic_obstacles else set()
            return 0 <= x < width and 0 <= y < height and grid[x][y] == 1 and (x, y) not in dynamic_obs

        from collections import deque
        queue = deque([self_pos])
        visited = {self_pos}
        came_from = {self_pos: None}

        while queue:
            curr = queue.popleft()
            cx, cy = curr
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = cx+dx, cy+dy
                if walkable(nx, ny) and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    came_from[(nx, ny)] = curr
                    queue.append((nx, ny))

        # 到達可能なマスのうち、本来の目的地に最も近いマス（マンハッタン距離）を選ぶ
        best_temp_pos = self_pos
        min_dist_to_target = abs(self_pos[0] - target_pos[0]) + abs(self_pos[1] - target_pos[1])

        for v in visited:
            d = abs(v[0] - target_pos[0]) + abs(v[1] - target_pos[1])
            if d < min_dist_to_target:
                min_dist_to_target = d
                best_temp_pos = v

        if best_temp_pos == self_pos:
            #print(f"  [MoveTo] 現在地 {self_pos} が最も目的に近い到達可能マスです。待機します。")
            return (0, 0)
        else:
            # 経路復元
            curr = best_temp_pos
            temp_path = []
            while came_from[curr] is not None:
                temp_path.append(curr)
                curr = came_from[curr]
            temp_path.reverse()
            
            self.planned_path = temp_path
            next_step = temp_path[0]
            #print(f"  [MoveTo] 一時目的地 {best_temp_pos} への向かいます。次のステップ: {next_step}")
            return (next_step[0] - self_pos[0], next_step[1] - self_pos[1])

    def move_to_safe_position(self, env, blocking_task, own_next_task=None, dynamic_obstacles=None):
        """
        依存待ち状態のときに他エージェントのじゃまにならない場所へ移動する。

        Args:
            blocking_task  : 他エージェントが現在実行中のタスク辞書
                             {'res': ('cutboard'/(pot, (x,y)), ...}
                             このリソースの隣接マスが「立ち入り禁止エリア」になる
            own_next_task  : このエージェントが次に実行するタスク辞書
                             できるだけこのリソースに近くで待機する
            dynamic_obstacles: 他エージェントの現在位置セット
        Returns:
            action (dx, dy)
        """
        if dynamic_obstacles is None:
            dynamic_obstacles = set()

        width  = env.world_width
        height = env.world_height
        grid   = env.to_grid_a
        self_pos = env.self_pos

        # ① 他エージェントが使用中のリソース位置とその隣接マスを「禁止エリア」にする
        prohibited = set()
        if blocking_task and blocking_task.get('res'):
            res_pos = blocking_task['res'][1]  # ('cutboard'/'pot', (x,y)) の座標部分
            prohibited.add(res_pos)
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                prohibited.add((res_pos[0]+dx, res_pos[1]+dy))

        # ② 全 walkable タイルから安全な候補を列挙
        candidates = []
        for x in range(width):
            for y in range(height):
                pos = (x, y)
                if grid[x][y] != 1:   # 壁・設備は除外
                    continue
                if pos in prohibited:  # 禁止エリアは除外
                    continue
                if pos in dynamic_obstacles:  # 他エージェントがいる場所も除外
                    continue
                candidates.append(pos)

        if not candidates:
            # 候補がない場合はその場で待機
            return (0, 0)

        # ③ 自分の次タスクのリソース位置に最も近い候補を選ぶ
        #    (次タスクの開始をできるだけ早くするため、なるべく近くで待機)

        # ③-a 既に安全な場所にいれば動かない（Bug 7: 振動防止）
        #      self_pos が候補リストに含まれる = 現在地は既に安全
        if self_pos in candidates:
            return (0, 0)

        ref_pos = None
        if own_next_task and own_next_task.get('res'):
            ref_pos = own_next_task['res'][1]

        if ref_pos:
            # A*距離ではなくマンハッタン距離で近さを評価（計算コスト節約）
            target = min(candidates, key=lambda p: abs(p[0]-ref_pos[0]) + abs(p[1]-ref_pos[1]))
        else:
            # 次タスクが不明なら現在地に最も近い候補（なるべく動かない）
            target = min(candidates, key=lambda p: abs(p[0]-self_pos[0]) + abs(p[1]-self_pos[1]))

        # print(f"  [SafeWait] {self_pos} → {target} (避けているリソース={blocking_task.get('res') if blocking_task else None})")
        return self.move_to(env, target, dynamic_obstacles=dynamic_obstacles)


    def __call__(self, env, dynamic_obstacles=None):

        if self.task_name.startswith('chop_'):
            ing_name = self.task_name.split('_')[1].capitalize()
            return self.process_chop_task(env, ing_name, assigned_cutboard=self.assigned_cutboard, assigned_counter=self.assigned_counter, dynamic_obstacles=dynamic_obstacles)
        elif self.task_name.startswith('cook'):
            parts = self.task_name.split('_')
            ingredients = []
            if len(parts) > 1:
                ingredients = [p.capitalize() for p in parts[1:]]
            return self.process_cook_task(env, ingredients, assigned_pot=self.assigned_pot, assigned_counter=self.assigned_counter, dynamic_obstacles=dynamic_obstacles)
        elif self.task_name.startswith('mix'):
            parts = self.task_name.split('_')
            ingredients = [p.capitalize() for p in parts[1:]]
            return self.process_mix_task(env, ingredients, assigned_blender=self.assigned_blender,
                                         assigned_counter=self.assigned_counter,
                                         dynamic_obstacles=dynamic_obstacles)
        elif self.task_name.startswith('serve_juice'):
            # 'serve' より先に判定する(ジュースは鍋ではなくミキサー、皿ではなくコップ)。
            parts = self.task_name.split('_')
            ingredients = [p.capitalize() for p in parts[2:]]
            return self.process_serve_juice_task(env, ingredients, assigned_cup=self.assigned_cup,
                                                 assigned_serve_loc=self.assigned_serve_loc,
                                                 assigned_blender=self.assigned_blender,
                                                 dynamic_obstacles=dynamic_obstacles)
        elif self.task_name.startswith('serve_from_counter'):
            parts = self.task_name.split('_')
            ingredients = [p.capitalize() for p in parts[3:]]
            return self.process_serve_from_counter_task(
                env, ingredients, assigned_counter=self.assigned_counter,
                assigned_serve_loc=self.assigned_serve_loc,
                dish_kind=self.dish_kind, dynamic_obstacles=dynamic_obstacles)
        elif self.task_name.startswith('handover'):
            parts = self.task_name.split('_')
            ingredients = [p.capitalize() for p in parts[1:]]
            return self.process_handover_task(env, ingredients,
                                              assigned_counter=self.assigned_counter,
                                              assigned_plate=self.assigned_plate,
                                              assigned_pot=self.assigned_pot,
                                              assigned_cup=self.assigned_cup,
                                              assigned_blender=self.assigned_blender,
                                              dish_kind=self.dish_kind,
                                              dynamic_obstacles=dynamic_obstacles)
        elif self.task_name.startswith('serve_salad'):
            # 'serve' より先に判定する。サラダは鍋を使わない別工程なので
            # process_serve_task (cook 専用の提供) に流してはいけない。
            parts = self.task_name.split('_')
            ingredients = [p.capitalize() for p in parts[2:]]
            return self.process_serve_salad_task(env, ingredients, assigned_counter=self.assigned_counter, assigned_serve_loc=self.assigned_serve_loc, dynamic_obstacles=dynamic_obstacles)
        elif self.task_name.startswith('serve'):
            parts = self.task_name.split('_')
            ingredients = []
            if len(parts) > 1:
                ingredients = [p.capitalize() for p in parts[1:]]
            return self.process_serve_task(env, ingredients, assigned_plate=self.assigned_plate, assigned_serve_loc=self.assigned_serve_loc, assigned_pot=self.assigned_pot, dynamic_obstacles=dynamic_obstacles)
        return (0,0), f"不明なタスク: {self.task_name}"

    @classmethod
    def reachable_positions(cls, env, positions):
        """自分が実際に使える位置だけに絞る。

        仕切りのあるマップでは、皿・コップ・調理器具・提供口が両側にある。
        マンハッタン距離が近い方を選ぶと壁の向こうの資材を選んでしまい、
        そこへ行けずに動けなくなる。EnvState の到達可能マップ(rch_map)で、
        隣に立てるものだけを残す。
        """
        if not positions:
            return []
        w, h = env.world_width, env.world_height
        # rch_map は EnvState 生成時の視点で作られているため、視点を差し替えた
        # コピーでは古い可能性がある。自分の現在地から数え直す。
        rch = cls._reach_from(env)

        def usable(pos):
            for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = pos[0] + dx, pos[1] + dy
                if 0 <= nx < w and 0 <= ny < h and rch[nx][ny]:
                    return True
            return False

        filtered = [p for p in positions if usable(p)]
        return filtered or list(positions)

    _reach_cache = {}

    @classmethod
    def _reach_from(cls, env):
        """自分の現在地から歩いて行ける床マス(幅優先)。1手番ぶんキャッシュする。"""
        # id() は使い回されるので、別の状態に同じ鍵が当たって古い地図を
        # 返すことがある。状態そのものに持たせれば、その状態と一緒に消える。
        key = tuple(env.self_pos)
        cache = getattr(env, '_reach_cache_by_pos', None)
        if cache is None:
            cache = {}
            try:
                env._reach_cache_by_pos = cache
            except Exception:
                cache = None
        if cache is not None and key in cache:
            return cache[key]
        w, h = env.world_width, env.world_height
        grid = env.to_grid
        rch = [[False] * h for _ in range(w)]
        start = tuple(env.self_pos)
        stack = [start]
        rch[start[0]][start[1]] = True
        while stack:
            cx, cy = stack.pop()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and grid[nx][ny] == 1 and not rch[nx][ny]:
                    rch[nx][ny] = True
                    stack.append((nx, ny))
        if cache is not None:
            cache[key] = rch
        return rch

    def process_mix_task(self, env, ingredients=None, assigned_blender=None,
                         assigned_counter=None, dynamic_obstacles=None):
        """ジュース: 刻んだフルーツをミキサーへ入れ、混ぜ終わるまで回す。

        材料を集めて器具へ入れるところまでは鍋(cook)と同じなので、そこは
        process_cook_task に任せる。違うのは投入後で、鍋のように放っておいても
        進まず、手ぶらで向かってインタラクトした回数だけ混ざる。
        """
        blenders = [assigned_blender] if assigned_blender else self.reachable_positions(env, env.get_pos_by_obj_gs(gs='Blender'))
        holding = env.hold

        for b_loc in blenders:
            obj = env.pos_obj.get(b_loc)
            if obj is None:
                continue
            if getattr(obj, 'is_mixed', lambda: False)():
                return (0, 0), "混ぜ完了 (Done)"
            if getattr(obj, 'is_mixing', lambda: False)():
                if holding is not None:
                    # 手が塞がっているとインタラクトが「入れる/取り出す」に
                    # 化けてしまうので、まず置きに行く。
                    return self.drop_unwanted_item(
                        env, holding,
                        reason=f"ミキサーを回すため {holding.full_name} を置く",
                        dynamic_obstacles=dynamic_obstacles,
                        allow_strict_override=True,
                    )
                return self.move_to(env, b_loc, dynamic_obstacles=dynamic_obstacles), "ミキサーを回す"

        return self.process_cook_task(
            env, ingredients, assigned_pot=assigned_blender,
            assigned_counter=assigned_counter, dynamic_obstacles=dynamic_obstacles,
            appliance='Blender', appliance_label='ミキサー')

    def process_serve_juice_task(self, env, ingredients=None, assigned_cup=None,
                                 assigned_serve_loc=None, assigned_blender=None,
                                 dynamic_obstacles=None):
        """ジュース: ミキサーの中身をコップに注いで提供する。

        鍋->皿の提供とまったく同じ形なので、完成状態の名前・容器・器具だけを
        差し替えて process_serve_task を使う。
        """
        return self.process_serve_task(
            env, ingredients, assigned_plate=assigned_cup,
            assigned_serve_loc=assigned_serve_loc, assigned_pot=assigned_blender,
            dynamic_obstacles=dynamic_obstacles,
            done_state='Mixed', mid_states=('Mixing',),
            container='Cup', container_tile='CupTile', appliance='Blender',
            appliance_label='ミキサー')

    def process_handover_task(self, env, ingredients=None, assigned_counter=None,
                              assigned_plate=None, assigned_pot=None,
                              assigned_cup=None, assigned_blender=None,
                              dish_kind=None, dynamic_obstacles=None):
        """仕切りの向こうへ渡す: 完成させて、提供口ではなく受け渡し台に置く。

        完成させるまでの手順は通常の提供とまったく同じで、違うのは行き先だけ。
        サラダは鍋を使わず、ジュースは鍋ではなくミキサーとコップを使うので、
        料理の種類ごとの提供処理へ、行き先を受け渡し台に差し替えて任せる。
        """
        if assigned_counter is None:
            return (0, 0), "受け渡し台が割り当てられていません"

        if dish_kind == 'juice':
            return self.process_serve_juice_task(
                env, ingredients, assigned_cup=assigned_cup,
                assigned_serve_loc=assigned_counter, assigned_blender=assigned_blender,
                dynamic_obstacles=dynamic_obstacles)

        if dish_kind == 'salad':
            return self.process_serve_salad_task(
                env, ingredients, assigned_counter=assigned_counter,
                assigned_serve_loc=assigned_counter, dynamic_obstacles=dynamic_obstacles)

        return self.process_serve_task(
            env, ingredients, assigned_plate=assigned_plate,
            assigned_serve_loc=assigned_counter, assigned_pot=assigned_pot,
            dynamic_obstacles=dynamic_obstacles)

    def process_serve_from_counter_task(self, env, ingredients=None, assigned_counter=None,
                                        assigned_serve_loc=None, dish_kind=None,
                                        dynamic_obstacles=None):
        """受け渡し台に置かれた完成品を取って提供口へ運ぶ。"""
        self_pos = env.self_pos
        holding = env.hold
        holding_name = holding.full_name if holding else None
        # 完成品の名前は料理の種類で変わる(サラダは刻んだまま、ジュースはコップ)。
        done_state, container = {'juice': ('Mixed', 'Cup'),
                                 'salad': ('Chopped', 'Plate')}.get(dish_kind, ('Cooked', 'Plate'))
        target = sorted([f"{done_state}{i}" for i in (ingredients or [])])

        def is_target(name):
            return bool(name) and container in name and target and all(t in name for t in target)

        if is_target(holding_name):
            deliveries = ([assigned_serve_loc] if assigned_serve_loc
                          else self.reachable_positions(env, env.get_pos_by_obj_gs(gs='Delivery')))
            if not deliveries:
                return (0, 0), "提供口が見つかりません"
            d = min(deliveries, key=lambda p: abs(p[0]-self_pos[0]) + abs(p[1]-self_pos[1]))
            dist = abs(self_pos[0]-d[0]) + abs(self_pos[1]-d[1])
            action = self.move_to(env, d, dynamic_obstacles=dynamic_obstacles)
            return action, "配膳 (完了)" if dist == 1 else "配膳"

        if holding is not None:
            return self.drop_unwanted_item(
                env, holding,
                reason=f"受け取りタスクですが {holding_name} を持っています",
                dynamic_obstacles=dynamic_obstacles, allow_strict_override=True)

        # 受け渡し台に完成品が置かれるのを待って、置かれたら取りに行く。
        # 指定の台を先に見るが、別の台に置かれていたらそちらへ取りに行く。
        # 完成品はどの台にあっても同じものなので、待ち続ける理由はない。
        candidates = [assigned_counter] if assigned_counter else []
        candidates += [c for c in self.reachable_positions(env, env.get_pos_by_obj_gs(gs='Counter'))
                       if c not in candidates]
        for c in candidates:
            if c is None:
                continue
            obj = env.pos_obj.get(c)
            if obj is not None and is_target(getattr(obj, 'full_name', '')):
                return self.move_to(env, c, dynamic_obstacles=dynamic_obstacles), "受け渡し台から取る"

        if assigned_counter:
            return self.move_to(env, assigned_counter, dynamic_obstacles=dynamic_obstacles),                 "受け渡し待ち"
        return (0, 0), "受け渡し台が割り当てられていません"

    def process_serve_task(self, env, ingredients=None, assigned_plate=None, assigned_serve_loc=None,
                           assigned_pot=None, dynamic_obstacles=None,
                           done_state='Cooked', mid_states=('Cooking', 'Charred'),
                           container='Plate', container_tile='PlateTile', appliance='Pot',
                           appliance_label='鍋'):
        """調理器具の中身を容器に移して提供する。

        スープ(鍋->皿)とジュース(ミキサー->コップ)は工程が同じ形なので、
        「完成状態の名前・容器・器具」だけを差し替えて共用する。
        """
        self_pos = env.self_pos
        holding = env.hold
        holding_name = holding.full_name if holding else None
        
        target_food_name = None
        if ingredients:
            ingredients.sort()
            target_food_name = "-".join([f"{done_state}{i}" for i in ingredients])
            #print(f"[TaskAgent] 配膳ターゲット: {target_food_name}")
        
        def is_target_food(name):
            if not name: return False
            if target_food_name:
                return name == target_food_name
            return done_state in name and '-' in name

        def has_target_recipe(name):
            if not name:
                return False
            normalized = name
            for mid in mid_states:
                normalized = normalized.replace(mid, done_state)
            if target_food_name:
                return normalized == target_food_name
            return done_state in normalized and '-' in normalized

        def is_target_plate_food(name):
            if not name: return False
            if container not in name: return False
            if target_food_name:
                parts = target_food_name.split('-')
                return all(part in name for part in parts)
            return done_state in name and '-' in name

        # 1. If holding Plate + Food -> Go to Delivery
        if is_target_plate_food(holding_name):
            if assigned_serve_loc:
                deliveries = [assigned_serve_loc]
            else:
                deliveries = self.reachable_positions(env, env.get_pos_by_obj_gs(gs='Delivery'))
            
            if deliveries:
                target = min(deliveries, key=lambda p: abs(p[0]-self_pos[0]) + abs(p[1]-self_pos[1]))
                #print(f"  -> {target} へ配膳中")
                dist = abs(self_pos[0]-target[0]) + abs(self_pos[1]-target[1])
                if dist == 1:
                    return self.move_to(env, target, dynamic_obstacles=dynamic_obstacles), "配膳 (完了)"
                return self.move_to(env, target, dynamic_obstacles=dynamic_obstacles), "配膳"
            return (0,0), "受取場所が見つかりません"

        # 2. If holding Plate -> Go to Pot with Cooked Food
        if holding_name == container:
            def find_matching_pots(pots):
                target_pot = None
                waiting_pot = None
                min_dist = float('inf')
                waiting_min_dist = float('inf')

                for p_loc in pots:
                    obj = env.pos_obj[p_loc]
                    if obj and is_target_food(obj.full_name):
                        dist = abs(self_pos[0]-p_loc[0]) + abs(self_pos[1]-p_loc[1])
                        if dist < min_dist:
                            min_dist = dist
                            target_pot = p_loc
                    elif obj and has_target_recipe(obj.full_name):
                        dist = abs(self_pos[0]-p_loc[0]) + abs(self_pos[1]-p_loc[1])
                        if dist < waiting_min_dist:
                            waiting_min_dist = dist
                            waiting_pot = p_loc

                return target_pot, waiting_pot

            all_pots = self.reachable_positions(env, env.get_pos_by_obj_gs(gs=appliance))
            preferred_pots = [assigned_pot] if assigned_pot else all_pots
            target_pot, waiting_pot = find_matching_pots(preferred_pots)
            if (target_pot is None and waiting_pot is None) and assigned_pot:
                target_pot, waiting_pot = find_matching_pots(all_pots)
            
            if target_pot:
                action = self.move_to(env, target_pot, dynamic_obstacles=dynamic_obstacles)
                if action == (0, 0) and dynamic_obstacles:
                    action = self.move_to(env, target_pot, dynamic_obstacles=None)
                return action, "調理済み料理の取得"

            if waiting_pot:
                action = self.move_to(env, waiting_pot, dynamic_obstacles=dynamic_obstacles)
                if action == (0, 0) and dynamic_obstacles:
                    action = self.move_to(env, waiting_pot, dynamic_obstacles=None)
                return action, "調理完了待ち"
            
            return (0,0), f"ターゲットの完成品が入った{appliance_label}が見つかりません"

        # 3. If holding nothing -> Get Plate
        if not holding:
            if assigned_plate:
                plate_locs = [assigned_plate]
            else:
                plate_locs = self._filter_unheld_positions(env, env.get_pos_by_obj_gs(obj=container))
                if not plate_locs:
                    plate_locs = self.reachable_positions(env, env.get_pos_by_obj_gs(gs=container_tile))
            
            if plate_locs:
                target = min(plate_locs, key=lambda p: abs(p[0]-self_pos[0]) + abs(p[1]-self_pos[1]))
                #print(f"  -> {target} から皿を取得しに行きます")
                return self.move_to(env, target, dynamic_obstacles=dynamic_obstacles), f"{container}の取得"
            
            return (0,0), f"{container}が見つかりません"
            
        # ここまで来たのは、配膳に使えないもの(食材など)を持っている場合。
        # chop/cook タスクには「不要な持ち物を空きカウンターに置く」経路があるが、
        # serve だけ無く (0,0) を返していたため、食材を持ったまま配膳タスクに
        # 切り替わると永久にその場から動けなくなっていた。同じ経路で置きに行く。
        return self.drop_unwanted_item(
            env,
            holding,
            reason=f"配膳タスクですが、{holding_name} を持っています",
            dynamic_obstacles=dynamic_obstacles,
            allow_strict_override=True,
        )

    def process_cook_task(self, env, ingredients=None, assigned_pot=None, assigned_counter=None,
                          dynamic_obstacles=None, appliance='Pot', appliance_label='鍋'):
        """刻んだ食材を集めて調理器具へ入れる。

        鍋(cook)とミキサー(mix)は「材料を集めて器具へ入れる」までが同じなので、
        器具の種類だけを差し替えて共用する。"""
        self_pos = env.self_pos
        holding = env.hold
        holding_name = holding.full_name if holding else None
        
        if not ingredients:
            return (0, 0), "調理する食材が指定されていません"
            
        target_ing_names = sorted([f"Chopped{i}" for i in ingredients])
        # print(f"[DEBUG] cook:start agent={env.agent_idx} task={self.task_name} pos={self_pos} holding={holding_name} target={target_ing_names} assigned_counter={assigned_counter} assigned_pot={assigned_pot}")
        
        pots = [assigned_pot] if assigned_pot else self.reachable_positions(env, env.get_pos_by_obj_gs(gs=appliance))
        if env.agent_idx == 1:
            pots = list(reversed(pots))
            
        target_pot_loc = None
        blocked_pot_loc = assigned_pot if assigned_pot else (pots[0] if pots else None)
        missing_ings = target_ing_names.copy()
        
        # 1. 鍋の特定と不足食材(missing_ings)の算出
        for p_loc in pots:
            obj = env.pos_obj.get(p_loc)
            if obj is None:
                if target_pot_loc is None:
                    target_pot_loc = p_loc
            else:
                obj_name = getattr(obj, 'full_name', '')
                parts = obj_name.replace('Cooking', 'Chopped').replace('Cooked', 'Chopped').replace('Charred', 'Chopped').replace('Mixing', 'Chopped').replace('Mixed', 'Chopped').split('-')
                is_subset = True
                curr_ings = []
                for p in parts:
                    if p not in target_ing_names:
                        is_subset = False
                        break
                    curr_ings.append(p)
                
                if is_subset:
                    remaining = target_ing_names.copy()
                    for p in curr_ings:
                        if p in remaining:
                            remaining.remove(p)
                    if not remaining:
                        # 必要な材料が全て入っている鍋(調理中/調理済み)
                        target_pot_loc = p_loc
                        missing_ings = remaining
                        break
                    # 中身が一部だけの鍋には、このゲームでは後から材料を追加できない。
                    # interact() は「空の鍋」にしか食材を投入せず、埋まっている鍋へ
                    # 不足分を持っていっても何も起きない(人間が単品を鍋に入れた場合に
                    # 発生する)。追加しに行くと置けないまま永久に固まるため、この鍋は
                    # 対象にせず、カウンター上で全部マージしてから空の鍋へ投入する。
                    continue

        # print(f"[DEBUG] cook:pot_state target_pot={target_pot_loc} missing={missing_ings}")
                    
        if not target_pot_loc:
            if blocked_pot_loc:
                # print(f"[DEBUG] cook:wait_for_pot blocked_pot={blocked_pot_loc}")
                return self.move_to(env, blocked_pot_loc, dynamic_obstacles=dynamic_obstacles), "鍋が空くまで待機中"
            return (0, 0), "利用可能な鍋がありません"
            
        if not missing_ings:
            return (0, 0), "調理完了 (Done)"
            
        # 2. 手持ちアイテムの解析
        held_ings = []
        if holding_name:
            holding_parts = holding_name.replace('Cooking', 'Chopped').replace('Cooked', 'Chopped').replace('Charred', 'Chopped').replace('Mixing', 'Chopped').replace('Mixed', 'Chopped').split('-')
            for p in holding_parts:
                if p in missing_ings:
                    held_ings.append(p)
                    
        if holding_name and len(held_ings) < len(holding_parts):
            # 不要なものを持っている場合は捨てる
            # print(f"[DEBUG] cook:drop_unwanted holding_parts={holding_parts} held_ings={held_ings} missing={missing_ings}")
            # 置き場の管理中でも、どの工程にも要らない物は手放せなければ
            # ならない。持ったまま固まると、その人は以後何もできなくなる。
            # (余った食材を持つのは初心者なら普通に起きる)
            return self.drop_unwanted_item(
                env, holding, reason=f"不要なもの({holding_name})を持っています",
                dynamic_obstacles=dynamic_obstacles, allow_strict_override=True)
            
        # 3. 必要な全てを持っていれば鍋へ
        if set(held_ings) == set(missing_ings):
            # print(f"[DEBUG] cook:to_pot held_ings={held_ings} missing={missing_ings} target_pot={target_pot_loc}")
            return self.move_to(env, target_pot_loc, dynamic_obstacles=dynamic_obstacles), "完成した食材を鍋に入れる"
            
        # 4. 手に一部の食材だけを持っている -> 他の未調理食材の場所に行き、それを置いてマージする！
        if holding_name:
            remaining_ings = list(set(missing_ings) - set(held_ings))
            target_merge_loc = None
            order_allowed_names = {f"Chopped{i.capitalize()}" for i in ingredients}

            # 指定テーブルに必要な分がもう全部揃っている場合、いま手に持って
            # いるものは余り。同じ食材は重ねられないので、置きに行っても何も
            # 起きず永久に固まる。手放してから、完成した山を取りに行く。
            counter_obj = env.pos_obj.get(assigned_counter) if assigned_counter else None
            counter_name = getattr(counter_obj, 'full_name', '') or ''
            if counter_name:
                counter_parts = (counter_name.replace('Cooking', 'Chopped')
                                 .replace('Cooked', 'Chopped').replace('Charred', 'Chopped')
                                 .replace('Mixing', 'Chopped').replace('Mixed', 'Chopped').split('-'))
                if all(i in counter_parts for i in missing_ings):
                    return self.drop_unwanted_item(
                        env, holding,
                        reason=f"指定テーブルに{appliance_label}の材料が揃っているため",
                        dynamic_obstacles=dynamic_obstacles, allow_strict_override=True)
            
            # 常にCSPで指定された特定のカウンター(assigned_counter) をマージ先とする
            target_merge_loc, blocked_details = self._resolve_assigned_counter_target(
                env,
                holding,
                assigned_counter,
                "指定テーブルが使用中のため待機中"
            )
                    
            if target_merge_loc:
                # print(f"[DEBUG] cook:merge_place holding={holding_name} remaining={remaining_ings} target_merge_loc={target_merge_loc}")
                action = self.move_to(env, target_merge_loc, dynamic_obstacles=dynamic_obstacles)
                return action, "指定テーブルにて食材をマージさせるために置く"
            elif assigned_counter:
                # 指定テーブルが使えない（他注文の食材が乗っている等）場合は待機
                # print(f"[DEBUG] cook: assigned_counter={assigned_counter} がブロック holding='{holding_name}'")
                return (0, 0), blocked_details or "指定テーブルが使用中のため待機中"
            else:
                # フォールバック: 指定テーブルがない場合は今まで通り一番近くに探す
                def fallback_func():
                    min_dist = float('inf')
                    local_target_merge_loc = None
                    for pos, obj in env.pos_obj.items():
                        if self._is_available_object(obj):
                            obj_name = getattr(obj, 'full_name', '')
                            parts = obj_name.replace('Cooking', 'Chopped').replace('Cooked', 'Chopped').replace('Charred', 'Chopped').replace('Mixing', 'Chopped').replace('Mixed', 'Chopped').split('-')

                            is_valid_target = False
                            has_unwanted = False
                            for p in parts:
                                if p in remaining_ings or p in order_allowed_names:
                                    is_valid_target = True
                                if p not in order_allowed_names:
                                    has_unwanted = True

                            if is_valid_target and not has_unwanted:
                                dist = abs(self_pos[0] - pos[0]) + abs(self_pos[1] - pos[1])
                                if dist < min_dist:
                                    min_dist = dist
                                    local_target_merge_loc = pos

                    if local_target_merge_loc:
                        return self.move_to(env, local_target_merge_loc, dynamic_obstacles=dynamic_obstacles), "離れた食材とマージさせるために置く"
                    return self.move_to(env, target_pot_loc, dynamic_obstacles=dynamic_obstacles), "マージ対象がないため今の分を鍋に入れる"

                return self._handle_counter_fallback("共有置き場ID未割当のため待機中", fallback_func)
                
        # 5. 手が空の場合 -> 足りない食材のいずれかを探すが、すでにマージが進んでいるものを優先する
        def find_best_ingredient_target(only_assigned_counter):
            return self._find_chopped_pickup_target(
                env, ingredients, missing_ings, assigned_counter, only_assigned_counter
            )

        target_ing_loc = None
        best_score = -float('inf')
        assigned_counter_candidate = None
        assigned_counter_score = -float('inf')

        if assigned_counter:
            target_ing_loc, best_score, assigned_counter_candidate, assigned_counter_score = find_best_ingredient_target(True)
            if target_ing_loc is None and assigned_counter_candidate is None:
                target_ing_loc, best_score, assigned_counter_candidate, assigned_counter_score = find_best_ingredient_target(False)
        else:
            target_ing_loc, best_score, assigned_counter_candidate, assigned_counter_score = find_best_ingredient_target(False)

        # print(f"[DEBUG] cook:candidates target_ing_loc={target_ing_loc} best_score={best_score} assigned_counter_candidate={assigned_counter_candidate} assigned_counter_score={assigned_counter_score} missing={missing_ings}")

        if target_ing_loc is None and assigned_counter_candidate is not None:
            target_ing_loc = assigned_counter_candidate
            # print(f"[DEBUG] cook:use_assigned_counter_candidate pos={target_ing_loc}")
                        
        if target_ing_loc:
            # print(f"[DEBUG] cook:pickup_target pos={target_ing_loc} obj={getattr(env.pos_obj.get(target_ing_loc), 'full_name', None)}")
            return self.move_to(env, target_ing_loc, dynamic_obstacles=dynamic_obstacles), "食材の取得"
            
        # print(f"[DEBUG] cook: 必要食材が見つからない missing={missing_ings} counter={assigned_counter}")
        # print(f"[DEBUG]   カウンター上: { {p: env.pos_obj[p].full_name for p in env.get_pos_by_obj_gs('Counter') if env.pos_obj.get(p)} }")
        return (0, 0), "必要な食材 (Chopped) を待機中"

    def _find_chopped_pickup_target(self, env, ingredients, missing_ings, assigned_counter, only_assigned_counter):
        """置かれている刻んだ食材のうち、次に取りに行くべき場所を探す。

        戻り値: (target_pos, best_score, assigned_counter_candidate, assigned_counter_score)

        assigned_counter に「一部だけ」集まっている場合は、そこから取ってしまうと
        せっかく進んだマージが巻き戻るため、通常候補とは分けて返す
        (呼び出し側が、他に何も見つからないときの最後の手段として使う)。
        """
        self_pos = env.self_pos
        order_allowed_names = {f"Chopped{i.capitalize()}" for i in ingredients}

        local_target_ing_loc = None
        local_best_score = -float('inf')
        local_assigned_counter_candidate = None
        local_assigned_counter_score = -float('inf')

        for pos, obj in env.pos_obj.items():
            if not self._is_available_object(obj):
                continue
            if only_assigned_counter and assigned_counter and pos != assigned_counter:
                continue

            obj_name = getattr(obj, 'full_name', '')
            parts = obj_name.replace('Cooking', 'Chopped').replace('Cooked', 'Chopped').replace('Charred', 'Chopped').split('-')

            valid_count = 0
            has_unwanted = False
            for p in parts:
                if p in order_allowed_names:
                    valid_count += 1
                else:
                    has_unwanted = True

            if valid_count > 0 and not has_unwanted:
                dist = abs(self_pos[0] - pos[0]) + abs(self_pos[1] - pos[1])
                score = (valid_count * 100) - dist
                if assigned_counter and pos == assigned_counter and 0 < valid_count < len(missing_ings):
                    if score > local_assigned_counter_score:
                        local_assigned_counter_score = score
                        local_assigned_counter_candidate = pos
                    continue

                if score > local_best_score:
                    local_best_score = score
                    local_target_ing_loc = pos

        return local_target_ing_loc, local_best_score, local_assigned_counter_candidate, local_assigned_counter_score

    def process_serve_salad_task(self, env, ingredients=None, assigned_counter=None,
                                 assigned_serve_loc=None, dynamic_obstacles=None):
        """サラダの提供タスク(鍋を使わない)。

        サラダは「刻む → 皿に乗せる → 提供」で完成する。process_serve_task が
        「鍋の調理済み料理を皿ですくって運ぶ」のに対し、こちらは
        置き場に集めた刻んだ食材をまとめて取り、皿タイルに触れて皿に乗せ
        (食材を持ったまま皿タイルに触れると、皿に乗った状態で手に持てる)、
        提供口へ運ぶ。
        """
        self_pos = env.self_pos
        holding = env.hold
        holding_name = holding.full_name if holding else None

        if not ingredients:
            return (0, 0), "提供する食材が指定されていません"

        target_ing_names = sorted([f"Chopped{i}" for i in ingredients])
        target_set = set(target_ing_names)

        holding_parts = holding_name.split('-') if holding_name else []
        has_plate = 'Plate' in holding_parts
        held_ings = [p for p in holding_parts if p in target_set]
        unwanted = [p for p in holding_parts if p != 'Plate' and p not in target_set]

        # 0. この注文に関係ないものを持っている -> 置きに行く
        if unwanted:
            return self.drop_unwanted_item(
                env,
                holding,
                reason=f"サラダの提供タスクですが、{holding_name} を持っています",
                dynamic_obstacles=dynamic_obstacles,
                allow_strict_override=True,
            )

        missing_ings = [n for n in target_ing_names if n not in held_ings]

        # 1. 皿の上に材料が全部そろっている(=サラダ完成) -> 提供口へ
        if has_plate and not missing_ings:
            if assigned_serve_loc:
                deliveries = [assigned_serve_loc]
            else:
                deliveries = self.reachable_positions(env, env.get_pos_by_obj_gs(gs='Delivery'))
            if not deliveries:
                return (0, 0), "受取場所が見つかりません"
            target = min(deliveries, key=lambda p: abs(p[0] - self_pos[0]) + abs(p[1] - self_pos[1]))
            dist = abs(self_pos[0] - target[0]) + abs(self_pos[1] - target[1])
            if dist == 1:
                return self.move_to(env, target, dynamic_obstacles=dynamic_obstacles), "サラダの配膳 (完了)"
            return self.move_to(env, target, dynamic_obstacles=dynamic_obstacles), "サラダの配膳"

        # 2. 皿なしで材料が全部そろっている -> 皿タイルへ行って皿に乗せる
        #    カウンター上に置かれた皿と合流させると、マージ結果がカウンター側に
        #    残って手放してしまうため、必ず皿タイル(無限に皿が出る供給口)を使う。
        if not has_plate and not missing_ings:
            plate_tiles = self.reachable_positions(env, env.get_pos_by_obj_gs(gs='PlateTile'))
            if not plate_tiles:
                return (0, 0), "皿タイルが見つかりません"
            target = min(plate_tiles, key=lambda p: abs(p[0] - self_pos[0]) + abs(p[1] - self_pos[1]))
            return self.move_to(env, target, dynamic_obstacles=dynamic_obstacles), "サラダを皿に乗せる"

        # ここから先は材料がまだ足りない。cook と同じく置き場で合流させる。
        # 3. 一部だけ持っている -> 指定テーブルへ運んでマージする
        #    (皿を持っていれば拾い上げてそのまま盛り付けになり、
        #     皿を持っていなければ一旦テーブルに置いて合流させる)
        if holding_name:
            target_merge_loc, blocked_details = self._resolve_assigned_counter_target(
                env,
                holding,
                assigned_counter,
                "指定テーブルが使用中のため待機中"
            )

            if target_merge_loc:
                return (self.move_to(env, target_merge_loc, dynamic_obstacles=dynamic_obstacles),
                        "指定テーブルにて食材をマージさせるために置く")
            if assigned_counter:
                return (0, 0), blocked_details or "指定テーブルが使用中のため待機中"

            def fallback_func():
                order_allowed_names = {f"Chopped{i.capitalize()}" for i in ingredients}
                min_dist = float('inf')
                local_target_merge_loc = None
                for pos, obj in env.pos_obj.items():
                    if not self._is_available_object(obj):
                        continue
                    obj_name = getattr(obj, 'full_name', '')
                    parts = obj_name.split('-')

                    is_valid_target = False
                    has_unwanted = False
                    for p in parts:
                        if p in missing_ings or p in order_allowed_names:
                            is_valid_target = True
                        if p not in order_allowed_names:
                            has_unwanted = True

                    if is_valid_target and not has_unwanted:
                        dist = abs(self_pos[0] - pos[0]) + abs(self_pos[1] - pos[1])
                        if dist < min_dist:
                            min_dist = dist
                            local_target_merge_loc = pos

                if local_target_merge_loc:
                    return (self.move_to(env, local_target_merge_loc, dynamic_obstacles=dynamic_obstacles),
                            "離れた食材とマージさせるために置く")
                return (0, 0), "マージ対象の食材を待機中"

            return self._handle_counter_fallback("共有置き場ID未割当のため待機中", fallback_func)

        # 4. 手が空 -> 足りない食材を探す(マージが進んでいるものを優先)
        target_ing_loc = None
        assigned_counter_candidate = None

        if assigned_counter:
            target_ing_loc, _, assigned_counter_candidate, _ = self._find_chopped_pickup_target(
                env, ingredients, missing_ings, assigned_counter, True)
            if target_ing_loc is None:
                # 指定テーブルに「一部だけ」乗っている状態で、それを拾ってしまうと
                # 次のフレームに同じ場所へ置き直すだけの往復になる。
                # 先に他の場所から不足分を運んできて合流させる方を必ず優先する。
                target_ing_loc, _, other_candidate, _ = self._find_chopped_pickup_target(
                    env, ingredients, missing_ings, assigned_counter, False)
                if assigned_counter_candidate is None:
                    assigned_counter_candidate = other_candidate
        else:
            target_ing_loc, _, assigned_counter_candidate, _ = self._find_chopped_pickup_target(
                env, ingredients, missing_ings, assigned_counter, False)

        if target_ing_loc is None and assigned_counter_candidate is not None:
            target_ing_loc = assigned_counter_candidate

        if target_ing_loc:
            return self.move_to(env, target_ing_loc, dynamic_obstacles=dynamic_obstacles), "食材の取得"

        return (0, 0), "必要な食材 (Chopped) を待機中"

    @staticmethod
    def _blocked_cutboards(env, cutboard_locs, target_ing_name, chopping_ing_name):
        """自分の作業と関係ない物が載ったままのまな板。"""
        blocked = []
        for loc in cutboard_locs:
            obj = env.pos_obj.get(loc)
            name = getattr(obj, 'full_name', '') if obj is not None else ''
            if not name:
                continue
            if target_ing_name in name or chopping_ing_name in name:
                continue
            blocked.append(loc)
        return blocked

    def drop_unwanted_item(self, env, holding, reason="", dynamic_obstacles=None, allow_strict_override=False):
        '''手に持っている不要なアイテムを最寄りの空きカウンターに置く'''
        holding_name = getattr(holding, 'full_name', None)
        if holding_name == 'Plate':
            plate_tiles = self.reachable_positions(env, env.get_pos_by_obj_gs(gs='PlateTile'))
            if plate_tiles:
                target = min(
                    plate_tiles,
                    key=lambda p: abs(env.self_pos[0] - p[0]) + abs(env.self_pos[1] - p[1])
                )
                return self.move_to(env, target, dynamic_obstacles=dynamic_obstacles), f"不要な皿を戻す: {reason}"

        if self.strict_counter_management and not allow_strict_override:
            return (0, 0), f"共有置き場管理中のため待機: {reason}"

        # 通路が分断された地図では、近くても反対側のカウンターには置けない。
        counters = self.reachable_positions(env, env.get_pos_by_obj_gs(gs='Counter'))
        best_dist = float('inf')
        best_c = None

        for c_pos in counters:
            if env.pos_obj.get(c_pos) is None:  # 空いているカウンター
                dist = abs(env.self_pos[0] - c_pos[0]) + abs(env.self_pos[1] - c_pos[1])
                if dist < best_dist:
                    best_dist = dist
                    best_c = c_pos
                    
        if best_c:
            return self.move_to(env, best_c, dynamic_obstacles=dynamic_obstacles), f"不要アイテム放棄: {reason}"
            
        # 全てのカウンターが塞がっている場合はとりあえず待機
        return (0, 0), "空きカウンターがありません"

    def process_chop_task(self, env, ing_name, assigned_cutboard=None, assigned_counter=None, dynamic_obstacles=None):
        self_pos = env.self_pos
        holding = env.hold
        holding_name = holding.full_name if holding else None

        self._log_chop_debug(env, ing_name, holding_name, assigned_cutboard, assigned_counter, "start")
        
        target_ing_name = f"Fresh{ing_name}"
        chopping_ing_name = f"Chopping{ing_name}"
        chopped_ing_name = f"Chopped{ing_name}"
        
        # Check unwanted
        if holding_name:
            is_valid_holding = False
            if holding_name in [target_ing_name, chopping_ing_name]:
                is_valid_holding = True
            elif chopped_ing_name in holding_name:
                is_valid_holding = True
            
            if not is_valid_holding:
                 # ここに来る持ち物は CSPAgent 側で「どのタスク・注文にも紐づかない」と
                 # 判定済みの余剰品(例: 別経路で既に満たされた注文向けに切ってしまった食材)。
                 # strict_counter_management で待機させ続けると、行き場のない食材を
                 # 持ったまま永久に固まってしまうため、ここは例外的に空きカウンターへの
                 # 退避を許可する(process_chop_task 内の「共有置き場に既にある」ケースと同様)。
                 return self.drop_unwanted_item(
                     env, holding, reason=f"{ing_name} を切るタスクですが、{holding_name} を持っています",
                     allow_strict_override=True,
                 )

        # 0. If holding Chopped Ingredient -> Place on Table
        if holding_name and chopped_ing_name in holding_name:
            target_table = None
            counter_obj = env.pos_obj.get(assigned_counter) if assigned_counter else None
            
            target_table, blocked_details = self._resolve_assigned_counter_target(
                env,
                holding,
                assigned_counter,
                "指定テーブルが使用中(マージ不可)のため待機"
            )
            if assigned_counter and target_table is None:
                content_name = getattr(counter_obj, 'full_name', '') if counter_obj is not None else ''
                if chopped_ing_name in content_name:
                    self._log_chop_debug(env, ing_name, holding_name, assigned_cutboard, assigned_counter, "drop_unwanted", target=assigned_counter, reason="already_on_shared_counter")
                    return self.drop_unwanted_item(
                        env,
                        holding,
                        reason=f"{chopped_ing_name} は共有置き場に既にあるため退避",
                        dynamic_obstacles=dynamic_obstacles,
                        allow_strict_override=True,
                    )
                # print(f"[DEBUG] chop: assigned_counter={assigned_counter} がブロック holding='{holding_name}'")
                self._log_chop_debug(env, ing_name, holding_name, assigned_cutboard, assigned_counter, "wait_blocked", reason=blocked_details or "assigned_counter_blocked")
                return (0, 0), blocked_details or "指定テーブルが使用中(マージ不可)のため待機"
            
            if not target_table and not assigned_counter:
                def fallback_func():
                    counters = env.get_pos_by_obj_gs(gs='Counter')
                    if env.agent_idx == 1:
                        counters = list(reversed(counters))

                    best_dist = float('inf')
                    local_target_table = None
                    for c_pos in counters:
                        if env.pos_obj.get(c_pos) is None:
                            dist = abs(self_pos[0]-c_pos[0]) + abs(self_pos[1]-c_pos[1])
                            if dist < best_dist:
                                best_dist = dist
                                local_target_table = c_pos

                    if local_target_table:
                        dist = abs(self_pos[0]-local_target_table[0]) + abs(self_pos[1]-local_target_table[1])
                        if dist == 1:
                            self._log_chop_debug(env, ing_name, holding_name, assigned_cutboard, assigned_counter, "fallback_place_done", target=local_target_table)
                            return self.move_to(env, local_target_table, dynamic_obstacles=dynamic_obstacles), f"{chopped_ing_name} を置く (完了)"
                        self._log_chop_debug(env, ing_name, holding_name, assigned_cutboard, assigned_counter, "fallback_place_move", target=local_target_table)
                        return self.move_to(env, local_target_table, dynamic_obstacles=dynamic_obstacles), f"{chopped_ing_name} を置く"
                    self._log_chop_debug(env, ing_name, holding_name, assigned_cutboard, assigned_counter, "fallback_wait", reason="no_empty_counter")
                    return (0,0), "適切なテーブルが見つかりません"

                return self._handle_counter_fallback("共有置き場ID未割当のため待機中", fallback_func)
            
            if target_table:
                #print(f"  -> {chopped_ing_name} を {target_table} に置きます")
                dist = abs(self_pos[0]-target_table[0]) + abs(self_pos[1]-target_table[1])
                if dist == 1:
                    self._log_chop_debug(env, ing_name, holding_name, assigned_cutboard, assigned_counter, "place_done", target=target_table)
                    return self.move_to(env, target_table, dynamic_obstacles=dynamic_obstacles), f"{chopped_ing_name} を置く (完了)"
                self._log_chop_debug(env, ing_name, holding_name, assigned_cutboard, assigned_counter, "place_move", target=target_table)
                return self.move_to(env, target_table, dynamic_obstacles=dynamic_obstacles), f"{chopped_ing_name} を置く"
            else:
                self._log_chop_debug(env, ing_name, holding_name, assigned_cutboard, assigned_counter, "wait_no_table", reason="no_target_table")
                return (0,0), "適切なテーブルが見つかりません"

        # 0.5 既に切られた物が別のテーブルにあるなら、切らずにそれを取りに行くだけでよい。
        # (例: レタス+玉ねぎの置き場と、トマトだけの置き場が別々にある場合、
        #  トマトを切り直すのではなく運んで合流させる)
        # 運び先は必ず assigned_counter なので、運び元と運び先が入れ替わる往復は起きない。
        carry_from = getattr(self, 'carry_from', None)
        if carry_from is not None and carry_from != assigned_counter:
            source_obj = env.pos_obj.get(carry_from)
            source_name = getattr(source_obj, 'full_name', '') if source_obj is not None else ''
            if chopped_ing_name in source_name:
                self._log_chop_debug(env, ing_name, holding_name, assigned_cutboard,
                                     assigned_counter, "carry_pickup", target=carry_from)
                return (self.move_to(env, carry_from, dynamic_obstacles=dynamic_obstacles),
                        f"{chopped_ing_name} を取りに行く (切らずに運ぶ)")

        # 1. Check Cutboards
        all_cutboards = self.reachable_positions(env, env.get_pos_by_obj_gs(gs='Cutboard'))
        for loc in all_cutboards:
            obj = env.pos_obj[loc]
            if obj and chopped_ing_name in obj.full_name:
                #print(f"  [まな板確認] {loc} で {chopped_ing_name} を発見")
                if not holding:
                    self._log_chop_debug(env, ing_name, holding_name, assigned_cutboard, assigned_counter, "pickup_chopped", target=loc)
                    return self.move_to(env, loc, dynamic_obstacles=dynamic_obstacles), f"{chopped_ing_name} を拾う"

        if assigned_cutboard:
            cutboard_locs = [assigned_cutboard]
        else:
            cutboard_locs = all_cutboards
        
        for loc in cutboard_locs:
            obj = env.pos_obj[loc]
            if obj:
                if target_ing_name in obj.full_name or chopping_ing_name in obj.full_name:
                    #print(f"  [まな板確認] {loc} で {obj.full_name} を発見")
                    self._log_chop_debug(env, ing_name, holding_name, assigned_cutboard, assigned_counter, "move_to_cutboard", target=loc)
                    return self.move_to(env, loc, dynamic_obstacles=dynamic_obstacles), f"{ing_name} を切る"
        
        # 2. If holding Fresh Ingredient -> Place on Cutboard
        if holding_name and target_ing_name in holding_name:
            best_cb = None
            min_dist = float('inf')
            
            check_cbs = cutboard_locs
            if env.agent_idx == 1:
                check_cbs = list(reversed(cutboard_locs))

            for loc in check_cbs:
                if env.pos_obj[loc] is None:
                    dist = abs(self_pos[0]-loc[0]) + abs(self_pos[1]-loc[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_cb = loc
            
            if best_cb:
                #print(f"  -> {target_ing_name} をまな板 {best_cb} に置きます")
                self._log_chop_debug(env, ing_name, holding_name, assigned_cutboard, assigned_counter, "place_fresh", target=best_cb)
                return self.move_to(env, best_cb, dynamic_obstacles=dynamic_obstacles), f"{target_ing_name} を置く"
            else:
                # まな板が全部ふさがっている。関係ない物が置きっぱなしなら、
                # どかせば使える。まず手を空けてから片付けに向かう。
                if self._blocked_cutboards(env, check_cbs, target_ing_name, chopping_ing_name):
                    self._log_chop_debug(env, ing_name, holding_name, assigned_cutboard, assigned_counter, "free_hands_to_clear", reason="cutboard_blocked")
                    return self.drop_unwanted_item(
                        env, holding, reason="まな板を空けるために一度置く",
                        dynamic_obstacles=dynamic_obstacles, allow_strict_override=True)
                self._log_chop_debug(env, ing_name, holding_name, assigned_cutboard, assigned_counter, "wait_no_cutboard", reason="no_free_cutboard")
                return (0,0), "空いているまな板がありません"

        # 2.5 手ぶらで、まな板が関係ない物でふさがっている -> どかしに行く。
        #     置きっぱなしを片付けないと、その側では誰も何も切れなくなる。
        if not holding_name:
            blocked = self._blocked_cutboards(env, cutboard_locs, target_ing_name, chopping_ing_name)
            free = [loc for loc in cutboard_locs if env.pos_obj[loc] is None]
            if blocked and not free:
                self._log_chop_debug(env, ing_name, holding_name, assigned_cutboard, assigned_counter, "clear_cutboard", target=blocked[0])
                return (self.move_to(env, blocked[0], dynamic_obstacles=dynamic_obstacles),
                        "まな板に残った物をどかす")

        # 3. Fetch Fresh Ingredient
        target_loc = None
        min_dist = float('inf')
        
        for pos, obj in env.pos_obj.items():
            if self._is_available_object(obj) and target_ing_name in obj.full_name:
                dist = abs(self_pos[0]-pos[0]) + abs(self_pos[1]-pos[1])
                if dist < min_dist:
                    min_dist = dist
                    target_loc = pos
        
        if not target_loc:
            dispenser_name = f"{target_ing_name}Tile" 
            dispensers = env.get_pos_by_obj_gs(gs=dispenser_name)
            if dispensers:
                for d_pos in dispensers:
                    dist = abs(self_pos[0]-d_pos[0]) + abs(self_pos[1]-d_pos[1])
                    if dist < min_dist:
                        min_dist = dist
                        target_loc = d_pos

        if target_loc:
            #print(f"  -> {target_loc} から {target_ing_name} を取得しに行きます")
            action = self.move_to(env, target_loc, dynamic_obstacles=dynamic_obstacles)
            self._log_chop_debug(
                env, ing_name, holding_name, assigned_cutboard, assigned_counter, "fetch_fresh",
                target=target_loc,
                reason=f"self_pos={self_pos} action={action} planned_path={list(self.planned_path)}",
            )
            return action, f"{target_ing_name} の取得"

        self._log_chop_debug(env, ing_name, holding_name, assigned_cutboard, assigned_counter, "wait_no_fresh", reason=f"{target_ing_name}_not_found")
        return (0,0), f"{target_ing_name} が見つかりません"