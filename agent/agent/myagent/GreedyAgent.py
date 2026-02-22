from agent.TSP.pathfinder import print_player_positions
import gym_cooking.utils.config as config

class GreedyAgent:
    def __init__(self, speed=2.5, replay=None):
        self.speed = speed
        self.replay = replay
        self.initialized = False
        self.dist_matrix = None
        self.tasks = []
        self.current_plan = None
        self.current_step = 0
        self.current_task_name = ""
        self.current_ingredient = ""
        self.chop_count = 0

    def get_index(self, x, y, width):
        return width * y + x

    def is_walkable(self, grid, x, y, width, height):
        return 0 <= x < width and 0 <= y < height and grid[x][y] == 1

    def neighbors(self, grid, x, y, width, height):
        result = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if self.is_walkable(grid, nx, ny, width, height):
                result.append((nx, ny))
        return result

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def astar(self, grid, start, goal, width, height):
        import heapq
        open_set = []
        heapq.heappush(open_set, (0, start))
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                return g_score[current]
            for neighbor in self.neighbors(grid, *current, width, height):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    def astar_path(self, grid, start, goal, width, height):
        import heapq
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            for neighbor in self.neighbors(grid, *current, width, height):
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    def _compute_all_distances(self, env):
        width = env.world_width
        height = env.world_height
        grid = env.to_grid
        size = width * height
        self.dist_matrix = [[None for _ in range(size)] for _ in range(size)]
        for y1 in range(height):
            for x1 in range(width):
                if not self.is_walkable(grid, x1, y1, width, height):
                    continue
                idx1 = self.get_index(x1, y1, width)
                for y2 in range(height):
                    for x2 in range(width):
                        if not self.is_walkable(grid, x2, y2, width, height):
                            continue
                        idx2 = self.get_index(x2, y2, width)
                        if idx1 == idx2:
                            self.dist_matrix[idx1][idx2] = 0
                        else:
                            dist = self.astar(grid, (x1, y1), (x2, y2), width, height)
                            self.dist_matrix[idx1][idx2] = dist

    def extract_tasks_from_current_orders(self, env):
        tasks_all = []
        for order_tuple in env.order.current_orders:
            goal_obj = order_tuple[0]
            name = goal_obj.full_name.lower()
            ingredients = []
            for ing in ['lettuce', 'onion', 'tomato']:
                if ing in name:
                    ingredients.append(ing)
            soup_name = '-'.join(ingredients) + ' soup'
            tasks = []
            for ing in ingredients:
                tasks.append(('chop', ing))
            tasks.append(('cook', soup_name))
            tasks.append(('serve', soup_name))
            tasks_all.append(tasks)
        self.tasks = tasks_all
        return tasks_all

    def calc_chop_task_cost_from_pos(self, env, ingredient, order_idx, from_pos):
        width = env.world_width
        height = env.world_height
        grid = env.to_grid

        tile_map = {
            "lettuce": "FreshLettuceTile",
            "onion": "FreshOnionTile",
            "tomato": "FreshTomatoTile"
        }
        ingredient_tile_pos = env.get_pos_by_obj_gs(gs=tile_map[ingredient])
        cutboard_pos = env.get_pos_by_obj_gs(gs="Cutboard")
        special_places = [(0,1), (0,2), (0,3)]
        target_place = special_places[order_idx]

        def get_adjacent_free(pos_list):
            free = []
            for x, y in pos_list:
                for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < width and 0 <= ny < height and grid[nx][ny] == 1:
                        free.append((nx, ny))
            return list(set(free))

        ing_adj = get_adjacent_free(ingredient_tile_pos)
        cut_adj = get_adjacent_free(cutboard_pos)
        target_adj = get_adjacent_free([target_place])

        min_cost = float('inf')
        for start in ing_adj:
            for mid in cut_adj:
                for end in target_adj:
                    idx_from = self.get_index(*from_pos, width)
                    idx_start = self.get_index(*start, width)
                    idx_mid = self.get_index(*mid, width)
                    idx_end = self.get_index(*end, width)
                    move0 = self.dist_matrix[idx_from][idx_start] if self.dist_matrix[idx_from][idx_start] is not None else float('inf')
                    move1 = self.dist_matrix[idx_start][idx_mid] if self.dist_matrix[idx_start][idx_mid] is not None else float('inf')
                    move2 = self.dist_matrix[idx_mid][idx_end] if self.dist_matrix[idx_mid][idx_end] is not None else float('inf')
                    total = move0 + move1 + move2
                    if total < min_cost:
                        min_cost = total
        if min_cost == float('inf'):
            return None
        cost = min_cost + config.CHOPPING_NUM_STEPS + 1 + 1
        return cost

    def calc_cook_task_cost_from_pos(self, env, order_idx, from_pos):
        width = env.world_width
        height = env.world_height
        special_places = [(0,1), (0,2), (0,3)]
        pot_places = [(3,5), (4,5), (5,5)]
        idx_from = self.get_index(*from_pos, width)
        idx_special = self.get_index(*special_places[order_idx], width)
        idx_pot = self.get_index(*pot_places[order_idx], width)
        move1 = self.dist_matrix[idx_from][idx_special]
        move2 = self.dist_matrix[idx_special][idx_pot]
        if move1 is None or move2 is None:
            return None
        cost = move1 + 1 + move2 + 1
        return cost

    def calc_serve_task_cost_from_pos(self, env, order_idx, from_pos):
        width = env.world_width
        height = env.world_height
        plate_pos = (6,6)
        pot_places = [(3,5), (4,5), (5,5)]
        delivery_pos = (6,3)
        idx_from = self.get_index(*from_pos, width)
        idx_plate = self.get_index(*plate_pos, width)
        idx_pot = self.get_index(*pot_places[order_idx], width)
        idx_delivery = self.get_index(*delivery_pos, width)
        move1 = self.dist_matrix[idx_from][idx_plate]
        move2 = self.dist_matrix[idx_plate][idx_pot]
        move3 = self.dist_matrix[idx_pot][idx_delivery]
        if move1 is None or move2 is None or move3 is None:
            return None
        cost = move1 + 1 + move2 + 1 + move3 + 1 + 1
        return cost

    def plan_chop_steps(self, env, ingredient, order_idx):
        width = env.world_width
        tile_map = {
            "lettuce": "FreshLettuceTile",
            "onion": "FreshOnionTile",
            "tomato": "FreshTomatoTile"
        }
        ingredient_tile_pos = env.get_pos_by_obj_gs(gs=tile_map[ingredient])
        cutboard_pos = env.get_pos_by_obj_gs(gs="Cutboard")
        # 注文番号ごとのテーブル・置き場所
        move_places = [(0,1), (0,2), (0,3)]
        put_places = [(0,1), (0,2), (0,3)]
        move_place = move_places[order_idx]
        put_place = put_places[order_idx]

        def get_adjacent_free(pos_list):
            free = []
            width = env.world_width
            height = env.world_height
            grid = env.to_grid
            for x, y in pos_list:
                for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < width and 0 <= ny < height and grid[nx][ny] == 1:
                        free.append((nx, ny))
            return list(set(free))

        ing_adj = get_adjacent_free(ingredient_tile_pos)
        cut_adj = get_adjacent_free(cutboard_pos)
        move_adj = get_adjacent_free([move_place])
        put_adj = get_adjacent_free([put_place])

        steps = [
            {"action": "move_to_ingredient", "dests": ing_adj, "desc": f"1. move to ingredient {ingredient}"},
            {"action": "pickup_ingredient", "dests": [], "desc": "2. pick up ingredient"},
            {"action": "move_to_cutboard", "dests": cut_adj, "desc": "3. move to cutboard"},
        ]
        for i in range(config.CHOPPING_NUM_STEPS):
            steps.append({"action": "chop", "dests": [], "desc": f"4.{i+1} chop"})
        steps += [
            {"action": "move_to_table", "dests": move_adj, "desc": f"5. move to table {move_place}"},
            {"action": "put_down", "dests": put_adj, "desc": f"6. put down at {put_place}"}
        ]
        return steps

    def __call__(self, env):
        if not self.initialized:
            self._compute_all_distances(env)
            self.initialized = True

        agent_pos = env.self_pos
        width = env.world_width
        height = env.world_height
        grid = env.to_grid

        # レタスの位置を毎フレーム出力
        lettuce_tiles = env.get_pos_by_obj_gs(gs="FreshLettuceTile")
        print(f"[Debug] レタスの位置: {lettuce_tiles}")

        for ing in ['lettuce', 'onion', 'tomato']:
            tiles = env.get_pos_by_obj_gs(gs=f"Fresh{ing.capitalize()}Tile")
            print(f"[Debug] {ing} の位置: {tiles}")

        if self.current_plan is None:
            selectable_tasks = []
            tasks_all = self.extract_tasks_from_current_orders(env)
            for order_idx, tasks in enumerate(tasks_all):
                # 材料リストを取得
                ingredients = []
                for verb, obj in tasks:
                    if verb == "chop":
                        ingredients.append(obj)
                # chopタスク
                for ing in ingredients:
                    chopped_name = f"Chopped{ing.capitalize()}"
                    # 必要数をカウント
                    need_num = 1  # 1つの注文につき1つ必要と仮定
                    exist_num = len(env.get_pos_by_obj_gs(obj=chopped_name))
                    # まだ必要数に達していなければ選択可能
                    if exist_num < need_num:
                        selectable_tasks.append(('chop', ing, order_idx))
                # cookタスク
                # 例: choppedlettuce-choppedonion soup
                for verb, obj in tasks:
                    if verb == "cook":
                        # 必要なchopped材料名を抽出
                        soup_name = obj
                        # 例: "lettuce-onion soup" → ["lettuce", "onion"]
                        ings = [x for x in ["lettuce", "onion", "tomato"] if x in soup_name]
                        chopped_combo = "-".join([f"Chopped{x.capitalize()}" for x in ings])
                        # choppedX-choppedYが存在していればcook可能
                        exist_combo = len(env.get_pos_by_obj_gs(obj=chopped_combo)) > 0
                        if exist_combo:
                            selectable_tasks.append(('cook', soup_name, order_idx))
                # serveタスク
                for verb, obj in tasks:
                    if verb == "serve":
                        # 完成品が存在していればserve可能
                        cooked_name = obj.replace("soup", "Soup").replace("-", "").replace(" ", "")
                        exist_cooked = len(env.get_pos_by_obj_gs(obj=cooked_name)) > 0
                        if exist_cooked:
                            selectable_tasks.append(('serve', obj, order_idx))

            # コスト計算
            task_costs = []
            for task in selectable_tasks:
                verb, obj, order_idx = task
                if verb == "chop":
                    cost = self.calc_chop_task_cost_from_pos(env, obj, order_idx, agent_pos)
                elif verb == "cook":
                    cost = self.calc_cook_task_cost_from_pos(env, order_idx, agent_pos)
                elif verb == "serve":
                    cost = self.calc_serve_task_cost_from_pos(env, order_idx, agent_pos)
                else:
                    cost = None
                if cost is not None:
                    task_costs.append((cost, verb, obj, order_idx))

            # 最小コストのタスクを選択
            if task_costs:
                task_costs.sort()
                _, verb, obj, order_idx = task_costs[0]
                if verb == "chop":
                    self.current_plan = self.plan_chop_steps(env, obj, order_idx)
                    self.current_step = 0
                    self.current_task_name = f"Chop {obj} (Order {order_idx+1})"
                    self.current_ingredient = obj
                    self.current_order_idx = order_idx  # 追加
                    self.chop_count = 0
                # cook/serveのplanも同様に追加可能

        # ...（以降は既存のself.current_plan進行処理）...
        if self.current_plan:
            step = self.current_plan[self.current_step]
            action = step["action"]
            desc = step["desc"]
            dests = step["dests"]

            print(f"Task Name: {self.current_task_name}, Step: {desc}")

            # 材料取得ステップ
            if action == "pickup_ingredient":
                # 既に材料を持っていれば次のステップへ
                if env.hold is not None and self.current_ingredient in env.hold.full_name.lower():
                    print(f"  [Pickup] {self.current_ingredient} を所持しているので次のステップへ")
                    self.current_step += 1
                    if self.current_step >= len(self.current_plan):
                        self.current_plan = None
                        self.current_step = 0
                    print(f"[Debug] move: (0, 0)")
                    return (0, 0), ""
                # まだ持っていなければインタラクト方向を返す
                ingredient = None
                if "lettuce" in self.current_task_name.lower():
                    ingredient = "FreshLettuceTile"
                elif "onion" in self.current_task_name.lower():
                    ingredient = "FreshOnionTile"
                elif "tomato" in self.current_task_name.lower():
                    ingredient = "FreshTomatoTile"
                if ingredient:
                    tiles = env.get_pos_by_obj_gs(gs=ingredient)
                    for tx, ty in tiles:
                        dx = tx - agent_pos[0]
                        dy = ty - agent_pos[1]
                        if abs(dx) + abs(dy) == 1:
                            print(f"  [Interact] {ingredient} 方向にmove: ({dx}, {dy})")
                            print(f"[Debug] move: ({dx}, {dy})")
                            return (dx, dy), ""
                print("  [Interact] 隣接していないのでstay")
                print(f"[Debug] move: (0, 0)")
                return (0, 0), ""

            # chopステップ
            if action == "chop":
                cutboard_pos = env.get_pos_by_obj_gs(gs="Cutboard")
                for cx, cy in cutboard_pos:
                    dx = cx - agent_pos[0]
                    dy = cy - agent_pos[1]
                    if abs(dx) + abs(dy) == 1:
                        chopped_name = f"chopped{self.current_ingredient}"
                        # choppedXを持っていたら次のステップへ（1ステップだけ進める）
                        if env.hold is not None and chopped_name in env.hold.full_name.lower():
                            print(f"  [Chop] {chopped_name} を所持しているので次のステップへ")
                            self.current_step += 1
                            if self.current_step >= len(self.current_plan):
                                self.current_plan = None
                                self.current_step = 0
                            print(f"[Debug] move: (0, 0)")
                            return (0, 0), ""
                        # chop回数をカウントして必要回数終わったら進める
                        if not hasattr(self, "chop_count"):
                            self.chop_count = 0
                        self.chop_count += 1
                        print(f"  [Chop] Cutboard方向にmove: ({dx}, {dy}) ({self.chop_count}/{config.CHOPPING_NUM_STEPS})")
                        print(f"[Debug] move: ({dx}, {dy})")
                        if self.chop_count >= config.CHOPPING_NUM_STEPS:
                            self.current_step += 1
                            self.chop_count = 0
                            if self.current_step >= len(self.current_plan):
                                self.current_plan = None
                                self.current_step = 0
                        return (dx, dy), ""
                print("  [Chop] まな板に隣接していないのでstay")
                print(f"[Debug] move: (0, 0)")
                return (0, 0), ""

            # テーブルに移動
            if action == "move_to_table":
                if agent_pos in dests and len(dests) > 0:
                    print(f"  [Move] テーブル目的地 {agent_pos} に到達、次のステップへ")
                    self.current_step += 1
                    if self.current_step >= len(self.current_plan):
                        self.current_plan = None
                        self.current_step = 0
                    print(f"[Debug] move: (0, 0)")
                    return (0,0), ""
                min_path = None
                min_len = float('inf')
                for dest in dests:
                    path = self.astar_path(grid, agent_pos, dest, width, height)
                    if path and len(path) < min_len:
                        min_len = len(path)
                        min_path = path
                        best_dest = dest
                if min_path and len(min_path) > 1:
                    next_pos = min_path[1]
                    move = (next_pos[0] - agent_pos[0], next_pos[1] - agent_pos[1])
                    print(f"  [Move] {agent_pos}→{next_pos} (move: {move})")
                    print(f"[Debug] move: {move}")
                    return move, ""
                else:
                    print(f"  [Move] 到達不可（でもプランは維持）")
                    print(f"[Debug] move: (0, 0)")
                    return (0,0), ""

            # テーブルでput_down
            if action == "put_down":
                # 既に手持ちが空なら次のタスクへ
                if env.hold is None or env.hold.full_name.lower() == "nothing":
                    print(f"  [PutDown] 既に手持ちが空なので次のタスクへ")
                    self.current_plan = None
                    self.current_step = 0
                    print(f"[Debug] move: (0, 0)")
                    return (0, 0), ""
                # 置きたい場所（put_place）を取得
                # plan_chop_stepsでput_places = [(0,1), (0,2), (0,3)]としている
                order_idx = self.current_plan[self.current_step].get("order_idx", 0) if "order_idx" in self.current_plan[self.current_step] else self.current_order_idx
                put_places = [(0,1), (0,2), (0,3)]
                put_place = put_places[order_idx]
                # put_placeの隣接マスにいれば、その方向にmove
                dx = put_place[0] - agent_pos[0]
                dy = put_place[1] - agent_pos[1]
                if abs(dx) + abs(dy) == 1:
                    print(f"  [PutDown] テーブル({put_place})方向にmove: ({dx}, {dy})")
                    print(f"[Debug] move: ({dx}, {dy})")
                    return (dx, dy), ""
                # 隣接していなければA*でput_placeの隣接マスに移動
                put_adj = []
                for ddx, ddy in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nx, ny = put_place[0]+ddx, put_place[1]+ddy
                    if 0 <= nx < width and 0 <= ny < height and grid[nx][ny] == 1:
                        put_adj.append((nx, ny))
                min_path = None
                min_len = float('inf')
                for dest in put_adj:
                    path = self.astar_path(grid, agent_pos, dest, width, height)
                    if path and len(path) < min_len:
                        min_len = len(path)
                        min_path = path
                        best_dest = dest
                if min_path and len(min_path) > 1:
                    next_pos = min_path[1]
                    move = (next_pos[0] - agent_pos[0], next_pos[1] - agent_pos[1])
                    print(f"  [PutDown] put_place隣接に移動: {agent_pos}→{next_pos} (move: {move})")
                    print(f"[Debug] move: {move}")
                    return move, ""
                print("  [PutDown] テーブルに隣接していないのでstay")
                print(f"[Debug] move: (0, 0)")
                return (0, 0), ""

            # その他の移動系・アクション
            if "move_to" in action:
                if agent_pos in dests and len(dests) > 0:
                    print(f"  [Move] 目的地 {agent_pos} に到達、次のステップへ")
                    self.current_step += 1
                    if self.current_step >= len(self.current_plan):
                        self.current_plan = None
                        self.current_step = 0
                    print(f"[Debug] move: (0, 0)")
                    return (0,0), ""
                min_path = None
                min_len = float('inf')
                for dest in dests:
                    path = self.astar_path(grid, agent_pos, dest, width, height)
                    if path and len(path) < min_len:
                        min_len = len(path)
                        min_path = path
                        best_dest = dest
                if min_path and len(min_path) > 1:
                    next_pos = min_path[1]
                    move = (next_pos[0] - agent_pos[0], next_pos[1] - agent_pos[1])
                    print(f"  [Move] {agent_pos}→{next_pos} (move: {move})")
                    print(f"[Debug] move: {move}")
                    return move, ""
                else:
                    print(f"  [Move] 到達不可（でもプランは維持）")
                    print(f"[Debug] move: (0, 0)")
                    return (0,0), ""
            else:
                print(f"  [Action] {action} 実行")
                self.current_step += 1
                if self.current_step >= len(self.current_plan):
                    self.current_plan = None
                    self.current_step = 0
                return (0,0), ""
        print(f"[Debug] move: (0, 0)")
        return (0,0), ""