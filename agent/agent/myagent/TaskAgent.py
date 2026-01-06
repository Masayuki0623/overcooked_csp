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
                # Check if pot is empty or has same ingredients (cooking)
                # For simplicity, look for empty pot first
                # Pot is a GridSquare. Check if occupied.
                # env.pos_obj[p_loc] is the object ON the pot (cooking food)
                pot_obj = env.pos_obj[p_loc]
                
                if pot_obj is None:
                    # Empty pot
                    dist = abs(self_pos[0]-p_loc[0]) + abs(self_pos[1]-p_loc[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_pot = p_loc
                # If pot is not empty, we might be able to add to it, but task says "chopped X and chopped Y combined"
                # so we assume we have the full set.
            
            if best_pot:
                print(f"  -> Moving to Pot at {best_pot}")
                dist = abs(self_pos[0]-best_pot[0]) + abs(self_pos[1]-best_pot[1])
                if dist == 1:
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
            # Allow if it's Fresh, Chopping, OR if it contains the Chopped target (e.g. merged)
            is_valid_holding = False
            if holding_name in [target_ing_name, chopping_ing_name]:
                is_valid_holding = True
            elif chopped_ing_name in holding_name:
                is_valid_holding = True
            
            if not is_valid_holding:
                 return self.drop_unwanted_item(env, holding, reason=f"Chopping {ing_name}, but holding {holding_name}")

        # 0. If holding Chopped Ingredient -> Place on Table
        if holding_name and chopped_ing_name in holding_name:
            # Find target table
            target_table = None
            
            if assigned_counter:
                # Check if assigned counter is empty OR mergeable
                counter_obj = env.pos_obj.get(assigned_counter)
                can_place = False
                
                if counter_obj is None:
                    can_place = True
                elif mergeable(holding, counter_obj):
                    can_place = True
                    print(f"  [Place] Assigned counter {assigned_counter} has {counter_obj.full_name}, but is mergeable.")
                
                if can_place:
                    target_table = assigned_counter
                    print(f"  [Place] Using assigned counter: {target_table}")
                else:
                    print(f"  [Place] Assigned counter {assigned_counter} is occupied/not mergeable. Searching nearby...")
                    # Search for nearest empty counter to assigned_counter
                    counters = env.get_pos_by_obj_gs(gs='Counter')
                    best_dist = float('inf')
                    best_c = None
                    for c_pos in counters:
                        # Check if empty OR mergeable
                        c_obj = env.pos_obj.get(c_pos)
                        is_valid = False
                        if c_obj is None:
                            is_valid = True
                        elif mergeable(holding, c_obj):
                            is_valid = True
                        
                        if is_valid:
                            # Distance from assigned_counter
                            dist = abs(assigned_counter[0]-c_pos[0]) + abs(assigned_counter[1]-c_pos[1])
                            if dist < best_dist:
                                best_dist = dist
                                best_c = c_pos
                    
                    if best_c:
                        target_table = best_c
                        print(f"  [Place] Found nearby valid counter at {target_table} (dist {best_dist})")
            
            # Second priority: Empty Counter (if no assignment or assignment invalid/occupied?)
            # For now, if assigned, stick to it.
            
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
