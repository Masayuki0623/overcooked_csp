import heapq
import random
from gym_cooking.utils.core import mergeable

class TaskAgent:
    def __init__(self, speed=2.5, replay=None, task_name=None):
        self.speed = speed
        self.replay = replay
        self.task_name = task_name
        self.strict_counter_management = False
        
        # Assigned resources for CSP
        self.assigned_cutboard = None
        self.assigned_pot = None
        self.assigned_plate = None
        self.assigned_serve_loc = None
        self.assigned_counter = None 
        self.protected_counters = set()
        
        # 経路予約用（Cooperative A*）
        self.planned_path = []
        self.wait_count = 0  # 待機カウンターを追加
        
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
        if counter_obj is None or mergeable(holding, counter_obj):
            return assigned_counter, None

        content_name = self._get_counter_content_name(env, assigned_counter)
        return None, f"{blocked_reason}: {assigned_counter} content='{content_name}'"

    def _handle_counter_fallback(self, wait_reason, fallback_func):
        if self.strict_counter_management:
            return (0, 0), wait_reason
        return fallback_func()

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
                    step_cost = 20  # ここを通るなら20マス遠回りしてでも避けるプランを採用する

                tentative_g = g_score[current] + step_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None, float('inf')

    def move_to(self, env, target_pos, dynamic_obstacles=None):
        self.planned_path = [] # リセット
        self_pos = env.self_pos
        dist = abs(self_pos[0] - target_pos[0]) + abs(self_pos[1] - target_pos[1])
        if dist == 1:
            self.wait_count = 0
            #print(f"  [MoveTo] ターゲット {target_pos} に隣接。インタラクトします。")
            return (target_pos[0] - self_pos[0], target_pos[1] - self_pos[1])
        
        adjacents = []
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = target_pos[0]+dx, target_pos[1]+dy
            if 0 <= nx < env.world_width and 0 <= ny < env.world_height and env.to_grid_a[nx][ny] == 1:
                adjacents.append((nx, ny))
        
        if not adjacents:
            #print(f"  [MoveTo] ターゲット {target_pos} の歩行可能な隣接セルがありません (to_grid_a で確認)")
            return (0,0)

        best_path = None
        min_cost = float('inf')
        
        # ターゲットの隣接マスのうち、コスト(距離＋障害物ペナルティ)が最小のルート(プラン)を採用する
        for adj in adjacents:
            path, cost = self.astar_path_cost(env, self_pos, adj, dynamic_obstacles=dynamic_obstacles)
            if path and cost < min_cost:
                min_cost = cost
                best_path = path

        if not best_path:
            return (0, 0)

        self.planned_path = best_path
        next_step = best_path[0]
        
        # もし次の一歩が他のエージェントの現在位置なら、通り過ぎるのを待機する
        if next_step in (dynamic_obstacles or set()):
            self.wait_count += 1
            print(f"[{env.agent_idx}:{self.task_name}] 最短距離上の障害物を避ける迂回ルートがない(またはコスト高すぎる)と判断し待機 (wait={self.wait_count}, cost={min_cost})")
            
            # デッドロック（お互いにお見合いで同じ場所で立ち往生する）防止策
            if self.wait_count > 15:
                import random
                escapes = []
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nx, ny = self_pos[0]+dx, self_pos[1]+dy
                    if 0 <= nx < env.world_width and 0 <= ny < env.world_height and env.to_grid_a[nx][ny] == 1:
                        if (nx, ny) not in (dynamic_obstacles or set()):
                            escapes.append((dx, dy))
                if escapes:
                    self.wait_count = 0
                    print(f"[{env.agent_idx}:{self.task_name}] お見合いが長すぎたため退避します！")
                    return random.choice(escapes)

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

        print(f"  [SafeWait] {self_pos} → {target} (避けているリソース={blocking_task.get('res') if blocking_task else None})")
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
        elif self.task_name.startswith('serve'):
            parts = self.task_name.split('_')
            ingredients = []
            if len(parts) > 1:
                ingredients = [p.capitalize() for p in parts[1:]]
            return self.process_serve_task(env, ingredients, assigned_plate=self.assigned_plate, assigned_serve_loc=self.assigned_serve_loc, assigned_pot=self.assigned_pot, dynamic_obstacles=dynamic_obstacles)
        return (0,0), f"不明なタスク: {self.task_name}"

    def process_serve_task(self, env, ingredients=None, assigned_plate=None, assigned_serve_loc=None, assigned_pot=None, dynamic_obstacles=None):
        self_pos = env.self_pos
        holding = env.hold
        holding_name = holding.full_name if holding else None
        
        target_food_name = None
        if ingredients:
            ingredients.sort()
            target_food_name = "-".join([f"Cooked{i}" for i in ingredients])
            #print(f"[TaskAgent] 配膳ターゲット: {target_food_name}")
        
        def is_target_food(name):
            if not name: return False
            if target_food_name:
                return name == target_food_name
            return 'Cooked' in name and '-' in name

        def has_target_recipe(name):
            if not name:
                return False
            normalized = name.replace('Cooking', 'Cooked').replace('Charred', 'Cooked')
            if target_food_name:
                return normalized == target_food_name
            return 'Cooked' in normalized and '-' in normalized

        def is_target_plate_food(name):
            if not name: return False
            if 'Plate' not in name: return False
            if target_food_name:
                parts = target_food_name.split('-')
                return all(part in name for part in parts)
            return 'Cooked' in name and '-' in name

        # 1. If holding Plate + Food -> Go to Delivery
        if is_target_plate_food(holding_name):
            if assigned_serve_loc:
                deliveries = [assigned_serve_loc]
            else:
                deliveries = env.get_pos_by_obj_gs(gs='Delivery')
            
            if deliveries:
                target = min(deliveries, key=lambda p: abs(p[0]-self_pos[0]) + abs(p[1]-self_pos[1]))
                #print(f"  -> {target} へ配膳中")
                dist = abs(self_pos[0]-target[0]) + abs(self_pos[1]-target[1])
                if dist == 1:
                    return self.move_to(env, target, dynamic_obstacles=dynamic_obstacles), "配膳 (完了)"
                return self.move_to(env, target, dynamic_obstacles=dynamic_obstacles), "配膳"
            return (0,0), "受取場所が見つかりません"

        # 2. If holding Plate -> Go to Pot with Cooked Food
        if holding_name == 'Plate':
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

            all_pots = env.get_pos_by_obj_gs(gs='Pot')
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
            
            return (0,0), "ターゲットの調理済み料理が入った鍋が見つかりません"

        # 3. If holding nothing -> Get Plate
        if not holding:
            if assigned_plate:
                plate_locs = [assigned_plate]
            else:
                plate_locs = self._filter_unheld_positions(env, env.get_pos_by_obj_gs(obj='Plate'))
                if not plate_locs:
                    plate_locs = env.get_pos_by_obj_gs(gs='PlateTile')
            
            if plate_locs:
                target = min(plate_locs, key=lambda p: abs(p[0]-self_pos[0]) + abs(p[1]-self_pos[1]))
                #print(f"  -> {target} から皿を取得しに行きます")
                return self.move_to(env, target, dynamic_obstacles=dynamic_obstacles), "皿の取得"
            
            return (0,0), "皿が見つかりません"
            
        return (0,0), f"{holding_name} を持っていますが、配膳タスクで何をすべきかわかりません"

    def process_cook_task(self, env, ingredients=None, assigned_pot=None, assigned_counter=None, dynamic_obstacles=None):
        self_pos = env.self_pos
        holding = env.hold
        holding_name = holding.full_name if holding else None
        
        if not ingredients:
            return (0, 0), "調理する食材が指定されていません"
            
        target_ing_names = sorted([f"Chopped{i}" for i in ingredients])
        print(f"[DEBUG] cook:start agent={env.agent_idx} task={self.task_name} pos={self_pos} holding={holding_name} target={target_ing_names} assigned_counter={assigned_counter} assigned_pot={assigned_pot}")
        
        pots = [assigned_pot] if assigned_pot else env.get_pos_by_obj_gs(gs='Pot')
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
                parts = obj_name.replace('Cooking', 'Chopped').replace('Cooked', 'Chopped').replace('Charred', 'Chopped').split('-')
                is_subset = True
                curr_ings = []
                for p in parts:
                    if p not in target_ing_names:
                        is_subset = False
                        break
                    curr_ings.append(p)
                
                if is_subset:
                    target_pot_loc = p_loc
                    missing_ings = target_ing_names.copy()
                    for p in curr_ings:
                        if p in missing_ings:
                            missing_ings.remove(p)
                    break

        print(f"[DEBUG] cook:pot_state target_pot={target_pot_loc} missing={missing_ings}")
                    
        if not target_pot_loc:
            if blocked_pot_loc:
                print(f"[DEBUG] cook:wait_for_pot blocked_pot={blocked_pot_loc}")
                return self.move_to(env, blocked_pot_loc, dynamic_obstacles=dynamic_obstacles), "鍋が空くまで待機中"
            return (0, 0), "利用可能な鍋がありません"
            
        if not missing_ings:
            return (0, 0), "調理完了 (Done)"
            
        # 2. 手持ちアイテムの解析
        held_ings = []
        if holding_name:
            holding_parts = holding_name.replace('Cooking', 'Chopped').replace('Cooked', 'Chopped').replace('Charred', 'Chopped').split('-')
            for p in holding_parts:
                if p in missing_ings:
                    held_ings.append(p)
                    
        if holding_name and len(held_ings) < len(holding_parts):
            # 不要なものを持っている場合は捨てる
            print(f"[DEBUG] cook:drop_unwanted holding_parts={holding_parts} held_ings={held_ings} missing={missing_ings}")
            return self.drop_unwanted_item(env, holding, reason=f"不要なもの({holding_name})を持っています", dynamic_obstacles=dynamic_obstacles)
            
        # 3. 必要な全てを持っていれば鍋へ
        if set(held_ings) == set(missing_ings):
            print(f"[DEBUG] cook:to_pot held_ings={held_ings} missing={missing_ings} target_pot={target_pot_loc}")
            return self.move_to(env, target_pot_loc, dynamic_obstacles=dynamic_obstacles), "完成した食材を鍋に入れる"
            
        # 4. 手に一部の食材だけを持っている -> 他の未調理食材の場所に行き、それを置いてマージする！
        if holding_name:
            remaining_ings = list(set(missing_ings) - set(held_ings))
            target_merge_loc = None
            
            # 常にCSPで指定された特定のカウンター(assigned_counter) をマージ先とする
            target_merge_loc, blocked_details = self._resolve_assigned_counter_target(
                env,
                holding,
                assigned_counter,
                "指定テーブルが使用中のため待機中"
            )
                    
            if target_merge_loc:
                print(f"[DEBUG] cook:merge_place holding={holding_name} remaining={remaining_ings} target_merge_loc={target_merge_loc}")
                action = self.move_to(env, target_merge_loc, dynamic_obstacles=dynamic_obstacles)
                return action, "指定テーブルにて食材をマージさせるために置く"
            elif assigned_counter:
                # 指定テーブルが使えない（他注文の食材が乗っている等）場合は待機
                print(f"[DEBUG] cook: assigned_counter={assigned_counter} がブロック holding='{holding_name}'")
                return (0, 0), blocked_details or "指定テーブルが使用中のため待機中"
            else:
                # フォールバック: 指定テーブルがない場合は今まで通り一番近くに探す
                def fallback_func():
                    min_dist = float('inf')
                    local_target_merge_loc = None
                    for pos, obj in env.pos_obj.items():
                        if self._is_available_object(obj):
                            obj_name = getattr(obj, 'full_name', '')
                            parts = obj_name.replace('Cooking', 'Chopped').replace('Cooked', 'Chopped').replace('Charred', 'Chopped').split('-')

                            is_valid_target = False
                            has_unwanted = False
                            for p in parts:
                                if p in remaining_ings:
                                    is_valid_target = True
                                if p not in missing_ings or p in held_ings:
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
                    if p in missing_ings:
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

        print(f"[DEBUG] cook:candidates target_ing_loc={target_ing_loc} best_score={best_score} assigned_counter_candidate={assigned_counter_candidate} assigned_counter_score={assigned_counter_score} missing={missing_ings}")

        if target_ing_loc is None and assigned_counter_candidate is not None:
            target_ing_loc = assigned_counter_candidate
            print(f"[DEBUG] cook:use_assigned_counter_candidate pos={target_ing_loc}")
                        
        if target_ing_loc:
            print(f"[DEBUG] cook:pickup_target pos={target_ing_loc} obj={getattr(env.pos_obj.get(target_ing_loc), 'full_name', None)}")
            return self.move_to(env, target_ing_loc, dynamic_obstacles=dynamic_obstacles), "食材の取得"
            
        print(f"[DEBUG] cook: 必要食材が見つからない missing={missing_ings} counter={assigned_counter}")
        print(f"[DEBUG]   カウンター上: { {p: env.pos_obj[p].full_name for p in env.get_pos_by_obj_gs('Counter') if env.pos_obj.get(p)} }")
        return (0, 0), "必要な食材 (Chopped) を待機中"

    def drop_unwanted_item(self, env, holding, reason="", dynamic_obstacles=None, allow_strict_override=False):
        '''手に持っている不要なアイテムを最寄りの空きカウンターに置く'''
        holding_name = getattr(holding, 'full_name', None)
        if holding_name == 'Plate':
            plate_tiles = env.get_pos_by_obj_gs(gs='PlateTile')
            if plate_tiles:
                target = min(
                    plate_tiles,
                    key=lambda p: abs(env.self_pos[0] - p[0]) + abs(env.self_pos[1] - p[1])
                )
                return self.move_to(env, target, dynamic_obstacles=dynamic_obstacles), f"不要な皿を戻す: {reason}"

        if self.strict_counter_management and not allow_strict_override:
            return (0, 0), f"共有置き場管理中のため待機: {reason}"

        counters = env.get_pos_by_obj_gs(gs='Counter')
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
                 return self.drop_unwanted_item(env, holding, reason=f"{ing_name} を切るタスクですが、{holding_name} を持っています")

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
                    return self.drop_unwanted_item(
                        env,
                        holding,
                        reason=f"{chopped_ing_name} は共有置き場に既にあるため退避",
                        dynamic_obstacles=dynamic_obstacles,
                        allow_strict_override=True,
                    )
                print(f"[DEBUG] chop: assigned_counter={assigned_counter} がブロック holding='{holding_name}'")
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
                            return self.move_to(env, local_target_table, dynamic_obstacles=dynamic_obstacles), f"{chopped_ing_name} を置く (完了)"
                        return self.move_to(env, local_target_table, dynamic_obstacles=dynamic_obstacles), f"{chopped_ing_name} を置く"
                    return (0,0), "適切なテーブルが見つかりません"

                return self._handle_counter_fallback("共有置き場ID未割当のため待機中", fallback_func)
            
            if target_table:
                #print(f"  -> {chopped_ing_name} を {target_table} に置きます")
                dist = abs(self_pos[0]-target_table[0]) + abs(self_pos[1]-target_table[1])
                if dist == 1:
                    return self.move_to(env, target_table, dynamic_obstacles=dynamic_obstacles), f"{chopped_ing_name} を置く (完了)"
                return self.move_to(env, target_table, dynamic_obstacles=dynamic_obstacles), f"{chopped_ing_name} を置く"
            else:
                return (0,0), "適切なテーブルが見つかりません"

        # 1. Check Cutboards
        all_cutboards = env.get_pos_by_obj_gs(gs='Cutboard')
        for loc in all_cutboards:
            obj = env.pos_obj[loc]
            if obj and chopped_ing_name in obj.full_name:
                #print(f"  [まな板確認] {loc} で {chopped_ing_name} を発見")
                if not holding:
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
                return self.move_to(env, best_cb, dynamic_obstacles=dynamic_obstacles), f"{target_ing_name} を置く"
            else:
                return (0,0), "空いているまな板がありません"

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
            return self.move_to(env, target_loc, dynamic_obstacles=dynamic_obstacles), f"{target_ing_name} の取得"

        return (0,0), f"{target_ing_name} が見つかりません"