import heapq

class TaskAgent:
    def __init__(self, speed=2.5, replay=None, task_name=None):
        self.speed = speed
        self.replay = replay
        self.task_name = task_name
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
        if self.task_name == 'chop_tomato':
            return self.chop_tomato(env)
        return (0,0), f"Unknown Task: {self.task_name}"

    def chop_tomato(self, env):
        self_pos = env.self_pos
        holding = env.hold
        holding_name = holding.full_name.lower() if holding else None
        
        print(f"[TaskAgent] Pos: {self_pos}, Holding: {holding_name}")

        # 1. Check for Tomato on Cutboard
        # Cutboard is a GridSquare, so use gs='Cutboard'
        cutboard_locs = env.get_pos_by_obj_gs(gs='Cutboard')
        target_cutboard = None
        
        for loc in cutboard_locs:
            # Check object at this location
            obj = env.pos_obj[loc]
            if obj:
                print(f"  [Check Cutboard] {loc}: Found {obj.full_name}")
                if 'tomato' in obj.full_name.lower():
                    # Found tomato on cutboard
                    if 'chopped' in obj.full_name.lower():
                        continue # Already chopped
                    target_cutboard = loc
                    break
        
        if target_cutboard:
            print(f"  -> Chopping at {target_cutboard}")
            # Go chop it
            return self.move_to(env, target_cutboard), "Chopping Tomato"

        # 2. If no tomato on cutboard, do we have one?
        if holding_name and 'tomato' in holding_name:
            # Find empty cutboard
            best_cb = None
            min_dist = float('inf')
            for loc in cutboard_locs:
                if env.pos_obj[loc] is None: # Empty
                    dist = abs(self_pos[0]-loc[0]) + abs(self_pos[1]-loc[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_cb = loc
            
            if best_cb:
                print(f"  -> Placing at {best_cb}")
                return self.move_to(env, best_cb), "Placing Tomato"
            else:
                print("  -> No Empty Cutboard")
                return (0,0), "No Empty Cutboard"

        # 3. Fetch Tomato
        # FreshTomato is an Object, so use obj='FreshTomato' (default positional)
        tomato_locs = env.get_pos_by_obj_gs(obj='FreshTomato')
        # Also check dispensers if no loose tomatoes
        if not tomato_locs:
            # FreshTomatoTile is a GridSquare, so use gs='FreshTomatoTile'
            tomato_locs = env.get_pos_by_obj_gs(gs='FreshTomatoTile')
            
        if tomato_locs:
            target = min(tomato_locs, key=lambda p: abs(p[0]-self_pos[0]) + abs(p[1]-self_pos[1]))
            print(f"  -> Fetching Tomato from {target}")
            return self.move_to(env, target), "Fetching Tomato"
            
        print("  -> No Tomato Found")
        return (0,0), "No Tomato Found"
