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
        self.assigned_counter = None # New: for placing chopped ingredients
        self.target_ingredients = None # List of ingredient names (lowercase) for current task's order
        
        print(f"[TaskAgent] Initialized with task: {self.task_name}")

    def astar_path(self, env, start, goal):
        """
        A*探索で経路を求める。
        戻り値: [(x,y), ...] のリスト（startを含まず、goalを含む）。到達不能ならNone。
        """
        width = env.world_width
        height = env.world_height
        # Use to_grid_a to avoid other agents
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
            print(f"  [MoveTo] Adjacent to target {target_pos}. Interacting.")
            return (target_pos[0] - self_pos[0], target_pos[1] - self_pos[1])
        
        adjacents = []
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = target_pos[0]+dx, target_pos[1]+dy
            # Use to_grid_a to check if adjacent is occupied by agent
            if 0 <= nx < env.world_width and 0 <= ny < env.world_height and env.to_grid_a[nx][ny] == 1:
                adjacents.append((nx, ny))
        
        if not adjacents:
            print(f"  [MoveTo] No walkable adjacents for {target_pos} (checked to_grid_a)")
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
            print(f"  [MoveTo] Path found. Next step: {next_step}")
            return (next_step[0] - self_pos[0], next_step[1] - self_pos[1])
        
        print(f"  [MoveTo] No path found to {target_pos}")
        return (0, 0)

    def __call__(self, env):
        if self.task_name.startswith('chop_'):
            # Extract ingredient name from task_name (e.g. chop_tomato -> Tomato)
            ing_name = self.task_name.split('_')[1].capitalize()
            return self.process_chop_task(env, ing_name, assigned_cutboard=self.assigned_cutboard, assigned_counter=self.assigned_counter)
        elif self.task_name.startswith('cook'):
            parts = self.task_name.split('_')
            ingredients = []
            if len(parts) > 1:
                # e.g. cook_tomato_onion -> ['Tomato', 'Onion']
                ingredients = [p.capitalize() for p in parts[1:]]
            return self.process_cook_task(env, ingredients, assigned_pot=self.assigned_pot)
        elif self.task_name.startswith('serve'):
            parts = self.task_name.split('_')
            ingredients = []
            if len(parts) > 1:
                ingredients = [p.capitalize() for p in parts[1:]]
            return self.process_serve_task(env, ingredients, assigned_plate=self.assigned_plate, assigned_serve_loc=self.assigned_serve_loc, assigned_pot=self.assigned_pot)
        elif self.task_name.startswith('transfer_'):
            # e.g. transfer_tomato
            ing_name = self.task_name.split('_')[1].capitalize()
            return self.process_transfer_task(env, ing_name, assigned_counter=self.assigned_counter)
        return (0,0), f"Unknown Task: {self.task_name}"

    def process_serve_task(self, env, ingredients=None, assigned_plate=None, assigned_serve_loc=None, assigned_pot=None):
        self_pos = env.self_pos
        holding = env.hold
        holding_name = holding.full_name if holding else None
        
        # Target food name (without Plate)
        target_food_name = None
        if ingredients:
            ingredients.sort()
            target_food_name = "-".join([f"Cooked{i}" for i in ingredients])
            print(f"[TaskAgent] Serve Target: {target_food_name}")
        
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

        # Check unwanted
        is_valid_holding = False
        if not holding_name:
            is_valid_holding = True
        elif holding_name == 'Plate':
            is_valid_holding = True
        elif is_target_plate_food(holding_name):
            is_valid_holding = True
        
        if not is_valid_holding:
             return self.drop_unwanted_item(env, holding, reason=f"Serving {target_food_name}, unexpected item {holding_name}")

        # 1. If holding Plate + Food -> Go to Delivery
        if is_target_plate_food(holding_name):
            if assigned_serve_loc:
                deliveries = [assigned_serve_loc]
            else:
                deliveries = env.get_pos_by_obj_gs(gs='Delivery')
            
            if deliveries:
                target = min(deliveries, key=lambda p: abs(p[0]-self_pos[0]) + abs(p[1]-self_pos[1]))
                print(f"  -> Delivering to {target}")
                dist = abs(self_pos[0]-target[0]) + abs(self_pos[1]-target[1])
                if dist == 1:
                    return self.move_to(env, target), "Delivering (Done)"
                return self.move_to(env, target), "Delivering"
            return (0,0), "No Delivery found"

        # 2. If holding Plate -> Go to Pot with Cooked Food
        if holding_name == 'Plate':
            if assigned_pot:
                pots = [assigned_pot]
            else:
                pots = env.get_pos_by_obj_gs(gs='Pot')
            target_pot = None
            min_dist = float('inf')
            
            for p_loc in pots:
                obj = env.pos_obj.get(p_loc) # Pot object/contents
                # On pot location, there might be "CookedX-Y"
                if obj and is_target_food(obj.full_name):
                    print(f"  [Search] Found cooked food {obj.full_name} in Pot at {p_loc}")
                    dist = abs(self_pos[0]-p_loc[0]) + abs(self_pos[1]-p_loc[1])
                    if dist < min_dist:
                        min_dist = dist
                        target_pot = p_loc
            
            if target_pot:
                print(f"  -> Fetching cooked food from Pot at {target_pot}")
                return self.move_to(env, target_pot), "Fetching cooked food"
            
            return (0,0), "No Pot with target cooked food found"

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
                print(f"  -> Fetching Plate from {target}")
                return self.move_to(env, target), "Fetching Plate"
            
            return (0,0), "No Plate found"
            
        return (0,0), f"Holding {holding_name}, not sure what to do for serve"

    def drop_unwanted_item(self, env, holding, reason="Holding unwanted item"):
        """
        Drop currently held item to nearest empty counter.
        """
        print(f"  [Drop] {reason}: {holding.full_name}")
        counters = env.get_pos_by_obj_gs(gs='Counter')
        best_dist = float('inf')
        target_table = None
        self_pos = env.self_pos

        for c_pos in counters:
            if env.pos_obj[c_pos] is None:
                dist = abs(self_pos[0]-c_pos[0]) + abs(self_pos[1]-c_pos[1])
                if dist < best_dist:
                    best_dist = dist
                    target_table = c_pos
        
        if target_table:
            print(f"  -> Dropping at {target_table}")
            return self.move_to(env, target_table), f"Dropping {holding.full_name}"
        else:
            return (0,0), "No empty counter to drop item"

    def process_transfer_task(self, env, ing_name, assigned_counter=None):
        """
        Chopped食材を（主にまな板から）拾って、カウンターに置くタスク
        """
        self_pos = env.self_pos
        holding = env.hold
        holding_name = holding.full_name if holding else None
        
        chopped_ing_name = f"Chopped{ing_name}"
        
        # Check for unwanted item
        if holding_name and chopped_ing_name not in holding_name:
             return self.drop_unwanted_item(env, holding, reason=f"Wanted {chopped_ing_name}, but holding {holding_name}")

        # 1. 持っている場合 -> カウンターに置く
        if holding_name and chopped_ing_name in holding_name:
            target_table = None
            
            # まず指定されたカウンター
            if assigned_counter:
                counter_obj = env.pos_obj.get(assigned_counter)
                if counter_obj is None or mergeable(holding, counter_obj):
                    target_table = assigned_counter
                    print(f"  [Transfer] Placing on assigned counter: {target_table}")
            
            # 指定がない、または使えない場合は近くの空きカウンター
            if not target_table:
                counters = env.get_pos_by_obj_gs(gs='Counter')
                best_dist = float('inf')
                for c_pos in counters:
                    c_obj = env.pos_obj.get(c_pos)
                    if c_obj is None or mergeable(holding, c_obj):
                        dist = abs(self_pos[0]-c_pos[0]) + abs(self_pos[1]-c_pos[1])
                        if dist < best_dist:
                            best_dist = dist
                            target_table = c_pos
                
                if target_table:
                    print(f"  [Transfer] Placing on nearest counter: {target_table}")

            if target_table:
                dist = abs(self_pos[0]-target_table[0]) + abs(self_pos[1]-target_table[1])
                if dist == 1:
                    return self.move_to(env, target_table), f"Transferring {chopped_ing_name} (Done)"
                return self.move_to(env, target_table), f"Transferring {chopped_ing_name}"
            else:
                return (0,0), "No suitable table found for transfer"

        # 2. 持っていない場合 -> 拾いに行く
        # まな板の上にあるものを優先して探す
        all_cutboards = env.get_pos_by_obj_gs(gs='Cutboard')
        for loc in all_cutboards:
            obj = env.pos_obj[loc]
            if obj and chopped_ing_name in obj.full_name:
                print(f"  [Transfer] Found {chopped_ing_name} on Cutboard at {loc}")
                return self.move_to(env, loc), f"Picking up {chopped_ing_name} for transfer"
        
        # 万が一まな板になくても、どこかにあれば拾う（不整合防止）
        for pos, obj in env.pos_obj.items():
            if obj and chopped_ing_name in obj.full_name:
                print(f"  [Transfer] Found {chopped_ing_name} at {pos}")
                return self.move_to(env, pos), f"Picking up {chopped_ing_name} for transfer"

        return (0,0), f"Target {chopped_ing_name} not found for transfer"
        self_pos = env.self_pos
        holding = env.hold
        holding_name = holding.full_name if holding else None
        
        # Target food name (without Plate)
        target_food_name = None
        if ingredients:
            ingredients.sort()
            target_food_name = "-".join([f"Cooked{i}" for i in ingredients])
            print(f"[TaskAgent] Serve Target: {target_food_name}")
        
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
                print(f"  -> Delivering to {target}")
                dist = abs(self_pos[0]-target[0]) + abs(self_pos[1]-target[1])
                if dist == 1:
                    return self.move_to(env, target), "Delivering (Done)"
                return self.move_to(env, target), "Delivering"
            return (0,0), "No Delivery found"

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
                    print(f"  [Search] Found cooked food {obj.full_name} in Pot at {p_loc}")
                    dist = abs(self_pos[0]-p_loc[0]) + abs(self_pos[1]-p_loc[1])
                    if dist < min_dist:
                        min_dist = dist
                        target_pot = p_loc
            
            if target_pot:
                print(f"  -> Fetching cooked food from Pot at {target_pot}")
                return self.move_to(env, target_pot), "Fetching cooked food"
            
            return (0,0), "No Pot with target cooked food found"

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
                print(f"  -> Fetching Plate from {target}")
                return self.move_to(env, target), "Fetching Plate"
            
            return (0,0), "No Plate found"
            
        return (0,0), f"Holding {holding_name}, not sure what to do for serve"

    def process_cook_task(self, env, ingredients=None, assigned_pot=None):
        self_pos = env.self_pos
        holding = env.hold
        holding_name = holding.full_name if holding else None
        
        # 0. Check if ingredients are already in a Pot (Task Complete)
        # We check if ALL required ingredients are in ANY single pot.
        if ingredients:
            required_ings = sorted([i.lower() for i in ingredients])
            pots = env.get_pos_by_obj_gs(gs='Pot')
            for p_loc in pots:
                # Pot contents are in env.pos_obj[p_loc] if cooking? 
                # Actually Pot is a GridSquare. Items are ON it.
                # But if cooking, it might be a "Cooking..." object.
                # If just placed, they are separate items on the pot location.
                # However, gym-cooking logic merges them into a cooking object or keeps them.
                # Let's check the object at pot location.
                pot_obj = env.pos_obj.get(p_loc)
                if pot_obj:
                    # Check contents
                    if hasattr(pot_obj, 'contents'):
                        # e.g. names = ['Tomato', 'Onion']
                        # required = ['tomato', 'onion']
                        current_ings = sorted([c.name.lower() for c in pot_obj.contents])
                        if current_ings == required_ings:
                            return (0, 0), "Cook Task Done (Ingredients in Pot)"
                    # Also check full_name if it's already cooked/cooking object without contents attr (unlikely but safe)
                    # e.g. "CookedLettuce-Onion"
                    # But reliable way is contents.

        target_name = None
        if ingredients:
            ingredients.sort()
            target_name = "-".join([f"Chopped{i}" for i in ingredients])
            print(f"[TaskAgent] Cooking Target: {target_name}")
        
        def is_target(name):
            if not name: return False
            if target_name:
                return name == target_name
            # Default: any merged chopped thing
            return 'Chopped' in name and '-' in name

        # Check for unwanted item (Not target merged, and not one of the ingredients)
        # Ingredients check: e.g. holding ChoppedTomato is fine if we are making Tomato-Onion soup
        is_valid_holding = False
        if not holding_name:
            is_valid_holding = True
        elif is_target(holding_name):
            is_valid_holding = True
        elif ingredients:
             # Check if holding one of the raw chopped ingredients
             for i in ingredients:
                 if f"Chopped{i}" == holding_name:
                     is_valid_holding = True
                     break
        
        if not is_valid_holding and holding_name:
             return self.drop_unwanted_item(env, holding, reason=f"Cooking {target_name}, unexpected item {holding_name}")

        # 1. If holding target -> Go to Pot
        if is_target(holding_name):
            # Find empty or compatible Pot
            if assigned_pot:
                pots = [assigned_pot]
            else:
                pots = env.get_pos_by_obj_gs(gs='Pot')
            
            best_pot = None
            min_dist = float('inf')
            
            for p_loc in pots:
                # Check if pot is empty OR mergeable (can add ingredients)
                pot_obj = env.pos_obj.get(p_loc)
                is_valid_pot = False
                
                if pot_obj is None:
                    is_valid_pot = True
                elif holding and mergeable(holding, pot_obj):
                    is_valid_pot = True
                    print(f"  [Cook] Pot at {p_loc} is mergeable with {holding.full_name}")
                
                if is_valid_pot:
                    dist = abs(self_pos[0]-p_loc[0]) + abs(self_pos[1]-p_loc[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_pot = p_loc
            
            if best_pot:
                print(f"  -> Moving to Pot at {best_pot}")
                dist = abs(self_pos[0]-best_pot[0]) + abs(self_pos[1]-best_pot[1])
                if dist == 1:
                    # 鍋に隣接して入れる動作を行う瞬間にタスクを完了(Done)とする
                    return self.move_to(env, best_pot), "Putting ingredients in Pot (Done)"
                return self.move_to(env, best_pot), "Putting ingredients in Pot"
            else:
                return (0,0), "No empty Pot found"

        # 2. Find target in environment
        target_loc = None
        min_dist = float('inf')
        
        for pos, obj in env.pos_obj.items():
            if obj:
                if is_target(obj.full_name):
                    print(f"  [Search] Found target {obj.full_name} at {pos}")
                    dist = abs(self_pos[0]-pos[0]) + abs(self_pos[1]-pos[1])
                    if dist < min_dist:
                        min_dist = dist
                        target_loc = pos
        
        if target_loc:
            print(f"  -> Fetching target from {target_loc}")
            return self.move_to(env, target_loc), "Fetching ingredients"
            
        return (0,0), f"Target {target_name if target_name else 'merged ingredients'} not found"

    def process_chop_task(self, env, ing_name, assigned_cutboard=None, assigned_counter=None):
        self_pos = env.self_pos
        holding = env.hold
        holding_name = holding.full_name if holding else None
        
        target_ing_name = f"Fresh{ing_name}"
        chopping_ing_name = f"Chopping{ing_name}"
        chopped_ing_name = f"Chopped{ing_name}"
        
        # Check unwanted
        if holding_name:
            if holding_name not in [target_ing_name, chopping_ing_name, chopped_ing_name]:
                 return self.drop_unwanted_item(env, holding, reason=f"Chopping {ing_name}, but holding {holding_name}")

        # 0. If holding Chopped Ingredient -> Place on Table
        if holding_name and chopped_ing_name in holding_name:
            target_table = None
            
            # Priority 1: Check for Mergeable Counters (ANYWHERE)
            # This ensures we merge Tomato onto Onion if Onion is already there
            counters = env.get_pos_by_obj_gs(gs='Counter')
            best_dist = float('inf')
            
            for c_pos in counters:
                c_obj = env.pos_obj.get(c_pos)
                if c_obj and mergeable(holding, c_obj):
                    # Check if c_obj contents are valid for current order
                    is_valid_merge = True
                    if self.target_ingredients:
                        # Extract ingredients from c_obj
                        # c_obj might be a Plate or food. Food usually has contents.
                        contents = c_obj.contents if hasattr(c_obj, 'contents') else [c_obj]
                        for item in contents:
                            # Item name e.g. "ChoppedLettuce", "Lettuce"
                            # We want to match with ["lettuce", "onion"]
                            name_lower = item.name.lower()
                            # Remove prefixes
                            for prefix in ["chopped", "fresh", "cooked", "chopping"]:
                                if name_lower.startswith(prefix):
                                    name_lower = name_lower.replace(prefix, "")
                            
                            if name_lower not in self.target_ingredients:
                                is_valid_merge = False
                                break
                    
                    if is_valid_merge:
                        dist = abs(self_pos[0]-c_pos[0]) + abs(self_pos[1]-c_pos[1])
                        if dist < best_dist:
                            best_dist = dist
                            target_table = c_pos
            
            if target_table:
                print(f"  [Place] Found mergeable counter at {target_table}")
            
            # Priority 2: Assigned Counter (if empty)
            if not target_table and assigned_counter:
                counter_obj = env.pos_obj.get(assigned_counter)
                if counter_obj is None:
                    target_table = assigned_counter
                    print(f"  [Place] Using assigned counter (empty): {target_table}")

            # Priority 3: Nearest Empty Counter
            if not target_table:
                best_dist = float('inf')
                for c_pos in counters:
                    if env.pos_obj[c_pos] is None:
                        dist = abs(self_pos[0]-c_pos[0]) + abs(self_pos[1]-c_pos[1])
                        if dist < best_dist:
                            best_dist = dist
                            target_table = c_pos
            
            if target_table:
                print(f"  -> Placing {chopped_ing_name} at {target_table}")
                # Check if adjacent (Done)
                dist = abs(self_pos[0]-target_table[0]) + abs(self_pos[1]-target_table[1])
                if dist == 1:
                    return self.move_to(env, target_table), f"Placing {chopped_ing_name} (Done)"
                return self.move_to(env, target_table), f"Placing {chopped_ing_name}"
            else:
                return (0,0), "No suitable table found"

        # 1. Check Cutboards
        # Relaxed: Check ALL cutboards for the finished product (ChoppedX) to ensure pickup
        all_cutboards = env.get_pos_by_obj_gs(gs='Cutboard')
        for loc in all_cutboards:
            obj = env.pos_obj[loc]
            if obj and chopped_ing_name in obj.full_name:
                # Chopped ingredient on cutboard -> Pick it up
                print(f"  [Check Cutboard] Found {chopped_ing_name} at {loc}")
                if not holding:
                    return self.move_to(env, loc), f"Picking up {chopped_ing_name}"

        # For Chopping/Placing Fresh, respect assignment
        if assigned_cutboard:
            cutboard_locs = [assigned_cutboard]
        else:
            cutboard_locs = all_cutboards
        
        for loc in cutboard_locs:
            obj = env.pos_obj[loc]
            if obj:
                if target_ing_name in obj.full_name or chopping_ing_name in obj.full_name:
                    # Fresh or Chopping ingredient on cutboard -> Chop it
                    print(f"  [Check Cutboard] Found {obj.full_name} at {loc}")
                    return self.move_to(env, loc), f"Chopping {ing_name}"
        
        # 2. If holding Fresh Ingredient -> Place on Cutboard
        if holding_name and target_ing_name in holding_name:
            # Find empty cutboard
            best_cb = None
            min_dist = float('inf')
            for loc in cutboard_locs:
                if env.pos_obj[loc] is None:
                    dist = abs(self_pos[0]-loc[0]) + abs(self_pos[1]-loc[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_cb = loc
            
            if best_cb:
                print(f"  -> Placing {target_ing_name} on Cutboard {best_cb}")
                return self.move_to(env, best_cb), f"Placing {target_ing_name}"
            else:
                return (0,0), "No Empty Cutboard"

        # 3. Fetch Fresh Ingredient
        target_loc = None
        min_dist = float('inf')
        
        # Check all objects for Fresh Ingredient
        for pos, obj in env.pos_obj.items():
            if obj and target_ing_name in obj.full_name:
                dist = abs(self_pos[0]-pos[0]) + abs(self_pos[1]-pos[1])
                if dist < min_dist:
                    min_dist = dist
                    target_loc = pos
        
        if not target_loc:
            # Check dispensers
            dispenser_name = f"{target_ing_name}Tile" # e.g. FreshTomatoTile
            dispensers = env.get_pos_by_obj_gs(gs=dispenser_name)
            if dispensers:
                # Pick closest
                for d_pos in dispensers:
                    dist = abs(self_pos[0]-d_pos[0]) + abs(self_pos[1]-d_pos[1])
                    if dist < min_dist:
                        min_dist = dist
                        target_loc = d_pos

        if target_loc:
            print(f"  -> Fetching {target_ing_name} from {target_loc}")
            return self.move_to(env, target_loc), f"Fetching {target_ing_name}"

        return (0,0), f"No {target_ing_name} found"
