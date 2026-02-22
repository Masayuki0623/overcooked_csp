from agent.TSP.pathfinder import print_player_positions
import gym_cooking.utils.config as config

class TSPSolverAgent:
    def __init__(self, speed=2.5, replay=None):
        self.speed = speed
        self.replay = replay
        self.initialized = False
        self.dist_matrix = None

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
                        #print(f"Computed distance from ({x1},{y1}) to ({x2},{y2}): {self.dist_matrix[idx1][idx2]}")

    def act(self, observation):
        actions = ['up', 'down', 'left', 'right', 'stay']
        import random
        return random.choice(actions)

    def __call__(self, env):
        # 初回のみ距離計算
        if not self.initialized:
            self._compute_all_distances(env)
            self.initialized = True

        #print_player_positions(env)
        action = self.act(env)
        move_map = {
            'up': (0, 1),
            'down': (0, -1),
            'left': (-1, 0),
            'right': (1, 0),
            'stay': (0, 0)
        }
        move = move_map.get(action, (0, 0))
        chat = ""
        return move, chat

    def extract_tasks_from_current_orders(self, env):
        """
        EnvStateのenvから現在のレシピ（注文）を取得し、タスク列を出力する
        戻り値: タスクのリスト（例: [('chop', 'lettuce'), ...]）
        """
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
        self.tasks = tasks_all  # 必要ならメンバ変数に保存

        # プリントデバッグ追加
        # print("=== 現在のレシピから生成されたタスク列 ===")
        # for i, tasks in enumerate(tasks_all):
        #     print(f"レシピ{i+1}:")
        #     for t in tasks:
        #         print("  ", t)
        # print("=====================================")

        return tasks_all

    def calc_chop_task_cost(self, env, ingredient, order_idx):
        """
        ingredient: 'onion', 'lettuce', 'tomato'
        order_idx: 0,1,2 (注文番号)
        """
        width = env.world_width
        height = env.world_height
        grid = env.to_grid

        tile_map = {
            "lettuce": "FreshLettuceTile",
            "onion": "FreshOnionTile",
            "tomato": "FreshTomatoTile"
        }
        # 1. 食材タイルの座標リスト
        ingredient_tile_pos = env.get_pos_by_obj_gs(gs=tile_map[ingredient])
        # print(f"{ingredient} tile positions:", ingredient_tile_pos)
        # 2. まな板の座標リスト
        cutboard_pos = env.get_pos_by_obj_gs(gs="Cutboard")
        # print(f"cutboard positions: {cutboard_pos}")
        # 3. 特定の場所
        special_places = [(0,1), (0,2), (0,3)]
        target_place = special_places[order_idx]
        # print(f"target place for order {order_idx+1}: {target_place}")

        # 4. それぞれの前後左右の空きマス
        def get_adjacent_free(pos_list, label):
            free = []
            for x, y in pos_list:
                for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < width and 0 <= ny < height and grid[nx][ny] == 1:
                        free.append((nx, ny))
            # print(f"adjacent free for {label}: {free}")
            return list(set(free))

        ing_adj = get_adjacent_free(ingredient_tile_pos, f"{ingredient}_tile")
        cut_adj = get_adjacent_free(cutboard_pos, "cutboard")
        target_adj = get_adjacent_free([target_place], "target_place")

        # 5. 最短距離を探索
        min_cost = float('inf')
        best_path = None
        for start in ing_adj:
            for mid in cut_adj:
                for end in target_adj:
                    idx_start = self.get_index(*start, width)
                    idx_mid = self.get_index(*mid, width)
                    idx_end = self.get_index(*end, width)
                    move1 = self.dist_matrix[idx_start][idx_mid] if self.dist_matrix[idx_start][idx_mid] is not None else float('inf')
                    move2 = self.dist_matrix[idx_mid][idx_end] if self.dist_matrix[idx_mid][idx_end] is not None else float('inf')
                    total = move1 + move2
                    # print(f"  path: {start}->{mid}->{end} | move1: {move1}, move2: {move2}, total_move: {total}")
                    if total < min_cost:
                        min_cost = total
                        best_path = (start, mid, end)

        # 6. コスト合計
        if min_cost == float('inf'):
            # print(f"経路が見つかりません: {ingredient}, 注文{order_idx+1}")
            return None
        cost = min_cost + config.CHOPPING_NUM_STEPS + 1 + 1  # +chop +置く +食材取得
        # print(f"[DEBUG] chop {ingredient} (注文{order_idx+1}) 最短経路: {best_path}, 最短移動コスト: {min_cost}, chopping: {config.CHOPPING_NUM_STEPS}, 置く:1, 取得:1, 合計コスト: {cost}")
        return cost

    def get_chop_task_positions(self, env, ingredient, order_idx):
        """
        Chopタスクの開始位置（食材タイルadjacent）と終了位置（まな板adjacent）を返す
        """
        width = env.world_width
        height = env.world_height
        grid = env.to_grid

        tile_map = {
            "lettuce": "FreshLettuceTile",
            "onion": "FreshOnionTile",
            "tomato": "FreshTomatoTile"
        }
        # 食材タイルadjacent
        ingredient_tile_pos = env.get_pos_by_obj_gs(gs=tile_map[ingredient])
        def get_adjacent_free(pos_list):
            free = []
            for x, y in pos_list:
                for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < width and 0 <= ny < height and grid[nx][ny] == 1:
                        free.append((nx, ny))
            return list(set(free))
        ing_adj = get_adjacent_free(ingredient_tile_pos)
        # まな板adjacent
        cutboard_pos = env.get_pos_by_obj_gs(gs="Cutboard")
        cut_adj = get_adjacent_free(cutboard_pos)
        return ing_adj, cut_adj

    def calc_task_to_task_cost(self, env, from_task, to_task, from_order_idx, to_order_idx):
        """
        from_task, to_task: ('chop', 'onion') など
        order_idx: 注文番号
        """
        width = env.world_width
        # Chopタスク同士のみ対応
        if from_task[0] == "chop" and to_task[0] == "chop":
            _, from_ing = from_task
            _, to_ing = to_task
            # fromの終了位置（まな板adjacent）→toの開始位置（食材タイルadjacent）
            _, from_ends = self.get_chop_task_positions(env, from_ing, from_order_idx)
            to_starts, _ = self.get_chop_task_positions(env, to_ing, to_order_idx)
            min_cost = float('inf')
            for end in from_ends:
                for start in to_starts:
                    idx_end = self.get_index(*end, width)
                    idx_start = self.get_index(*start, width)
                    move = self.dist_matrix[idx_end][idx_start]
                    if move is not None and move < min_cost:
                        min_cost = move
            return min_cost if min_cost != float('inf') else None
        return None

    def print_task_transition_costs(self, env):
        """
        タスク間遷移コストを出力
        """
        tasks_all = self.extract_tasks_from_current_orders(env)
        # タスクリストをflatten
        flat_tasks = []
        for order_idx, tasks in enumerate(tasks_all):
            for t in tasks:
                flat_tasks.append((t, order_idx))
        # Chopタスク同士の遷移コストを出力
        for i, (from_task, from_idx) in enumerate(flat_tasks):
            if from_task[0] != "chop":
                continue
            for j, (to_task, to_idx) in enumerate(flat_tasks):
                if to_task[0] != "chop" or i == j:
                    continue
                cost = self.calc_task_to_task_cost(env, from_task, to_task, from_idx, to_idx)
                print(f"Chop {from_task[1].capitalize()} {from_idx+1} --> Chop {to_task[1].capitalize()} {to_idx+1} (Cost :{cost})")

    def calc_cook_task_cost(self, env, order_idx):
        """
        cookタスクのコスト計算
        order_idx: 0,1,2
        """
        width = env.world_width
        special_places = [(3,2), (4,2), (5,2)]
        pot_places = [(3,5), (4,5), (5,5)]
        from_pos = special_places[order_idx]
        to_pos = pot_places[order_idx]
        idx_from = self.get_index(*from_pos, width)
        idx_to = self.get_index(*to_pos, width)
        move = self.dist_matrix[idx_from][idx_to]
        # print(f"[DEBUG] cook order{order_idx+1}: 特定の場所{from_pos}→コンロ{to_pos}, 移動コスト:{move}")
        if move is None:
            # print(f"経路が見つかりません: cook, 注文{order_idx+1}")
            return None
        cost = move + 1 + 1  # インタラクト2回
        # print(f"[DEBUG] cook order{order_idx+1}: 合計コスト: 移動{move} + インタラクト2 = {cost}")
        return cost

    def calc_serve_task_cost(self, env, order_idx):
        """
        serveタスクのコスト計算
        order_idx: 0,1,2
        """
        width = env.world_width
        plate_pos = (6,6)
        pot_places = [(3,5), (4,5), (5,5)]
        delivery_pos = (6,3)
        idx_plate = self.get_index(*plate_pos, width)
        idx_pot = self.get_index(*pot_places[order_idx], width)
        idx_delivery = self.get_index(*delivery_pos, width)
        move1 = self.dist_matrix[idx_plate][idx_pot]
        move2 = self.dist_matrix[idx_pot][idx_delivery]
        # print(f"[DEBUG] serve order{order_idx+1}: 皿{plate_pos}→鍋{pot_places[order_idx]} 移動コスト:{move1}")
        # print(f"[DEBUG] serve order{order_idx+1}: 鍋{pot_places[order_idx]}→配膳{delivery_pos} 移動コスト:{move2}")
        if move1 is None or move2 is None:
            # print(f"経路が見つかりません: serve, 注文{order_idx+1}")
            return None
        cost = move1 + 1 + move2 + 1 + 1  # 皿取得+料理取得+配膳
        # print(f"[DEBUG] serve order{order_idx+1}: 合計コスト: 皿→鍋{move1} + 皿取得1 + 鍋→配膳{move2} + 料理取得1 + 配膳1 = {cost}")
        return cost

    def generate_task_graph(self, env):
        """
        タスクグラフ（コスト行列）を生成
        """
        tasks_all = self.extract_tasks_from_current_orders(env)
        graph = []
        for order_idx, tasks in enumerate(tasks_all):
            for verb, obj in tasks:
                if verb == "chop":
                    cost = self.calc_chop_task_cost(env, obj, order_idx)
                    graph.append(((verb, obj, order_idx), cost))
                elif verb == "cook":
                    cost = self.calc_cook_task_cost(env, order_idx)
                    graph.append(((verb, obj, order_idx), cost))
                elif verb == "serve":
                    cost = self.calc_serve_task_cost(env, order_idx)
                    graph.append(((verb, obj, order_idx), cost))
        # print("[DEBUG] タスクグラフ:", graph)
        return graph

# 使用例
# tasks = extract_tasks_from_current_orders(env)
# print(tasks)