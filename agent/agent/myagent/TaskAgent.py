import heapq
from gym_cooking.utils.core import mergeable

class TaskAgent:
    def __init__(self, speed=2.5, replay=None, task_name=None):
        self.speed = speed
        self.replay = replay
        self.task_name = task_name
        
        # Assigned resources for CSP
        self.assigned_cutboard = None
        self.assigned_pot = None
        self.assigned_plate = None
        self.assigned_serve_loc = None
        self.assigned_counter = None 
        
        print(f"[TaskAgent] タスクで初期化: {self.task_name}")

    def astar_path(self, env, start, goal):
        width = env.world_width
        height = env.world_height
        grid = env.to_grid_a

        def in_bounds(x, y):
            return 0 <= x < width and 0 <= y < height

        def walkable(x, y):
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

    def move_to(self, env, target_pos):
        self_pos = env.self_pos
        dist = abs(self_pos[0] - target_pos[0]) + abs(self_pos[1] - target_pos[1])
        if dist == 1:
            print(f"  [MoveTo] ターゲット {target_pos} に隣接。インタラクトします。")
            return (target_pos[0] - self_pos[0], target_pos[1] - self_pos[1])
        
        adjacents = []
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = target_pos[0]+dx, target_pos[1]+dy
            if 0 <= nx < env.world_width and 0 <= ny < env.world_height and env.to_grid_a[nx][ny] == 1:
                adjacents.append((nx, ny))
        
        if not adjacents:
            print(f"  [MoveTo] ターゲット {target_pos} の歩行可能な隣接セルがありません (to_grid_a で確認)")
            return (0,0)

        best_path = None
        min_len = float('inf')
        
        for adj in adjacents:
            path = self.astar_path(env, self_pos, adj)
            if path and len(path) < min_len:
                min_len = len(path)
                best_path = path
        
        if best_path:
            next_step = best_path[0]
            print(f"  [MoveTo] 経路が見つかりました。次のステップ: {next_step}")
            return (next_step[0] - self_pos[0], next_step[1] - self_pos[1])
        
        print(f"  [MoveTo] {target_pos} への経路が見つかりません")
        return (0, 0)

    def __call__(self, env):
        if self.task_name.startswith('chop_'):
            ing_name = self.task_name.split('_')[1].capitalize()
            return self.process_chop_task(env, ing_name, assigned_cutboard=self.assigned_cutboard, assigned_counter=self.assigned_counter)
        elif self.task_name.startswith('cook'):
            parts = self.task_name.split('_')
            ingredients = []
            if len(parts) > 1:
                ingredients = [p.capitalize() for p in parts[1:]]
            return self.process_cook_task(env, ingredients, assigned_pot=self.assigned_pot)
        elif self.task_name.startswith('serve'):
            parts = self.task_name.split('_')
            ingredients = []
            if len(parts) > 1:
                ingredients = [p.capitalize() for p in parts[1:]]
            return self.process_serve_task(env, ingredients, assigned_plate=self.assigned_plate, assigned_serve_loc=self.assigned_serve_loc, assigned_pot=self.assigned_pot)
        return (0,0), f"不明なタスク: {self.task_name}"

    def process_serve_task(self, env, ingredients=None, assigned_plate=None, assigned_serve_loc=None, assigned_pot=None):
        self_pos = env.self_pos
        holding = env.hold
        holding_name = holding.full_name if holding else None
        
        target_food_name = None
        if ingredients:
            ingredients.sort()
            target_food_name = "-".join([f"Cooked{i}" for i in ingredients])
            print(f"[TaskAgent] 配膳ターゲット: {target_food_name}")
        
        def is_target_food(name):
            if not name: return False
            if target_food_name:
                return name == target_food_name
            return 'Cooked' in name and '-' in name

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
                print(f"  -> {target} へ配膳中")
                dist = abs(self_pos[0]-target[0]) + abs(self_pos[1]-target[1])
                if dist == 1:
                    return self.move_to(env, target), "配膳 (完了)"
                return self.move_to(env, target), "配膳"
            return (0,0), "受取場所が見つかりません"

        # 2. If holding Plate -> Go to Pot with Cooked Food
        if holding_name == 'Plate':
            if assigned_pot:
                pots = [assigned_pot]
            else:
                pots = env.get_pos_by_obj_gs(gs='Pot')
            target_pot = None
            min_dist = float('inf')
            
            for p_loc in pots:
                obj = env.pos_obj[p_loc]
                if obj and is_target_food(obj.full_name):
                    print(f"  [探索] 鍋 {p_loc} に調理済み料理 {obj.full_name} を発見")
                    dist = abs(self_pos[0]-p_loc[0]) + abs(self_pos[1]-p_loc[1])
                    if dist < min_dist:
                        min_dist = dist
                        target_pot = p_loc
            
            if target_pot:
                print(f"  -> 鍋 {target_pot} から調理済み料理を取りに行きます")
                return self.move_to(env, target_pot), "調理済み料理の取得"
            
            return (0,0), "ターゲットの調理済み料理が入った鍋が見つかりません"

        # 3. If holding nothing -> Get Plate
        if not holding:
            if assigned_plate:
                plate_locs = [assigned_plate]
            else:
                plate_locs = env.get_pos_by_obj_gs(obj='Plate')
                if not plate_locs:
                    plate_locs = env.get_pos_by_obj_gs(gs='PlateTile')
            
            if plate_locs:
                target = min(plate_locs, key=lambda p: abs(p[0]-self_pos[0]) + abs(p[1]-self_pos[1]))
                print(f"  -> {target} から皿を取得しに行きます")
                return self.move_to(env, target), "皿の取得"
            
            return (0,0), "皿が見つかりません"
            
        return (0,0), f"{holding_name} を持っていますが、配膳タスクで何をすべきかわかりません"

    def process_cook_task(self, env, ingredients=None, assigned_pot=None):
        self_pos = env.self_pos
        holding = env.hold
        holding_name = holding.full_name if holding else None
        
        target_name = None
        if ingredients:
            ingredients.sort()
            target_name = "-".join([f"Chopped{i}" for i in ingredients])
            print(f"[TaskAgent] 調理ターゲット: {target_name}")
        
        def is_target(name):
            if not name: return False
            if target_name:
                return name == target_name
            return 'Chopped' in name and '-' in name

        # 1. If holding target -> Go to Pot
        if is_target(holding_name):
            if assigned_pot:
                pots = [assigned_pot]
            else:
                pots = env.get_pos_by_obj_gs(gs='Pot')
            
            best_pot = None
            min_dist = float('inf')
            
            for p_loc in pots:
                pot_obj = env.pos_obj[p_loc]
                
                if pot_obj is None:
                    # Empty pot
                    dist = abs(self_pos[0]-p_loc[0]) + abs(self_pos[1]-p_loc[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_pot = p_loc
            
            if best_pot:
                print(f"  -> 鍋 {best_pot} へ移動中")
                dist = abs(self_pos[0]-best_pot[0]) + abs(self_pos[1]-best_pot[1])
                if dist == 1:
                    return self.move_to(env, best_pot), "鍋に食材を入れる (完了)"
                return self.move_to(env, best_pot), "鍋に食材を入れる"
            else:
                return (0,0), "空の鍋が見つかりません"

        # 2. Find target in environment
        target_loc = None
        min_dist = float('inf')
        
        for pos, obj in env.pos_obj.items():
            if obj:
                if is_target(obj.full_name):
                    print(f"  [探索] ターゲット {obj.full_name} を {pos} で発見")
                    dist = abs(self_pos[0]-pos[0]) + abs(self_pos[1]-pos[1])
                    if dist < min_dist:
                        min_dist = dist
                        target_loc = pos
        
        if target_loc:
            print(f"  -> {target_loc} からターゲットを取得しに行きます")
            return self.move_to(env, target_loc), "食材の取得"
            
        return (0,0), f"ターゲット {target_name if target_name else 'merged ingredients'} が見つかりません"

    def process_chop_task(self, env, ing_name, assigned_cutboard=None, assigned_counter=None):
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
            
            if assigned_counter:
                counter_obj = env.pos_obj.get(assigned_counter)
                can_place = False
                
                if counter_obj is None:
                    can_place = True
                elif mergeable(holding, counter_obj):
                    can_place = True
                    print(f"  [配置] 割り当てられたカウンター {assigned_counter} に {counter_obj.full_name} がありますが、マージ可能です。")
                
                if can_place:
                    target_table = assigned_counter
                    print(f"  [配置] 割り当てられたカウンターを使用: {target_table}")
                else:
                    print(f"  [配置] 割り当てられたカウンター {assigned_counter} は使用中/マージ不可です。近くを探します...")
                    counters = env.get_pos_by_obj_gs(gs='Counter')
                    best_dist = float('inf')
                    best_c = None
                    for c_pos in counters:
                        c_obj = env.pos_obj.get(c_pos)
                        is_valid = False
                        if c_obj is None:
                            is_valid = True
                        elif mergeable(holding, c_obj):
                            is_valid = True
                        
                        if is_valid:
                            dist = abs(assigned_counter[0]-c_pos[0]) + abs(assigned_counter[1]-c_pos[1])
                            if dist < best_dist:
                                best_dist = dist
                                best_c = c_pos
                    
                    if best_c:
                        target_table = best_c
                        print(f"  [配置] 近くの有効なカウンター {target_table} を発見 (距離 {best_dist})")
            
            if not target_table:
                counters = env.get_pos_by_obj_gs(gs='Counter')
                best_dist = float('inf')
                for c_pos in counters:
                    if env.pos_obj[c_pos] is None:
                        dist = abs(self_pos[0]-c_pos[0]) + abs(self_pos[1]-c_pos[1])
                        if dist < best_dist:
                            best_dist = dist
                            target_table = c_pos
            
            if target_table:
                print(f"  -> {chopped_ing_name} を {target_table} に置きます")
                dist = abs(self_pos[0]-target_table[0]) + abs(self_pos[1]-target_table[1])
                if dist == 1:
                    return self.move_to(env, target_table), f"{chopped_ing_name} を置く (完了)"
                return self.move_to(env, target_table), f"{chopped_ing_name} を置く"
            else:
                return (0,0), "適切なテーブルが見つかりません"

        # 1. Check Cutboards
        all_cutboards = env.get_pos_by_obj_gs(gs='Cutboard')
        for loc in all_cutboards:
            obj = env.pos_obj[loc]
            if obj and chopped_ing_name in obj.full_name:
                print(f"  [まな板確認] {loc} で {chopped_ing_name} を発見")
                if not holding:
                    return self.move_to(env, loc), f"{chopped_ing_name} を拾う"

        if assigned_cutboard:
            cutboard_locs = [assigned_cutboard]
        else:
            cutboard_locs = all_cutboards
        
        for loc in cutboard_locs:
            obj = env.pos_obj[loc]
            if obj:
                if target_ing_name in obj.full_name or chopping_ing_name in obj.full_name:
                    print(f"  [まな板確認] {loc} で {obj.full_name} を発見")
                    return self.move_to(env, loc), f"{ing_name} を切る"
        
        # 2. If holding Fresh Ingredient -> Place on Cutboard
        if holding_name and target_ing_name in holding_name:
            best_cb = None
            min_dist = float('inf')
            for loc in cutboard_locs:
                if env.pos_obj[loc] is None:
                    dist = abs(self_pos[0]-loc[0]) + abs(self_pos[1]-loc[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_cb = loc
            
            if best_cb:
                print(f"  -> {target_ing_name} をまな板 {best_cb} に置きます")
                return self.move_to(env, best_cb), f"{target_ing_name} を置く"
            else:
                return (0,0), "空いているまな板がありません"

        # 3. Fetch Fresh Ingredient
        target_loc = None
        min_dist = float('inf')
        
        for pos, obj in env.pos_obj.items():
            if obj and target_ing_name in obj.full_name:
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
            print(f"  -> {target_loc} から {target_ing_name} を取得しに行きます")
            return self.move_to(env, target_loc), f"{target_ing_name} の取得"

        return (0,0), f"{target_ing_name} が見つかりません"