
from agent.TSP.TSPSolverAgent import TSPSolverAgent
from agent.executor.low import EnvState
import gym_cooking.utils.config as config

class HumanPredictor:
    def __init__(self, env):
        # 距離計算のためにTSPSolverAgentを利用
        # envはOvercookedEnvironmentなので、EnvStateに変換して渡す
        init_env_state = EnvState(env.world, env.sim_agents, 0, env.order_scheduler, [], env.chg_grid, env.current_time)
        
        self.solver = TSPSolverAgent()
        self.solver._compute_all_distances(init_env_state)
        self.width = init_env_state.world_width
        self.height = init_env_state.world_height

    def predict(self, env, agent_idx):
        agent = env.agents[agent_idx]
        
        # 1. 状態に基づく判定
        # 何か持っているか？
        if agent.holding:
            obj = agent.holding
            obj_name = obj.full_name
            
            # 生食材 -> Chop
            if "Fresh" in obj_name:
                # FreshOnion -> chop onion
                ing_name = obj_name.replace("Fresh", "").lower()
                return (f"chop {ing_name} (Holding)", 0)
            
            # 切った食材 -> Cook
            if "Chopped" in obj_name:
                # ChoppedOnion -> cook ...
                return (f"cook (Holding {obj_name})", 0)
                
            # 皿 or スープ -> Serve
            if "Plate" in obj_name or "Soup" in obj_name:
                return (f"serve (Holding {obj_name})", 0)

        # まな板で作業中か？
        # 現在地の隣接にまな板があり、そのまな板に食材が乗っている
        cutboard_pos = env.get_pos_by_obj_gs(gs="Cutboard")
        cut_adj = self._get_adjacent_free(env, cutboard_pos)
        if agent.location in cut_adj:
            # どのまな板の隣か特定
            for cb_pos in cutboard_pos:
                if self._get_dist(agent.location, cb_pos) == 1: # 隣接
                    # まな板の上のオブジェクト確認
                    for obj in env.all_obj:
                        if obj.location == cb_pos and "Fresh" in obj.full_name:
                             ing_name = obj.full_name.replace("Fresh", "").lower()
                             return (f"chop {ing_name} (Chopping)", 0)

        # 2. コスト最小化予測
        tasks_all = self.solver.extract_tasks_from_current_orders(env)
        min_cost = float('inf')
        best_task_info = ("None", 0)

        # print(f"[DEBUG] Tasks: {tasks_all}")

        for order_idx, tasks in enumerate(tasks_all):
            for task in tasks:
                verb, obj = task
                cost = float('inf')
                
                if verb == 'chop':
                    cost = self._calc_chop_cost_new(env, agent, obj)
                elif verb == 'cook':
                    cost = self._calc_cook_cost_new(env, agent, obj, order_idx)
                elif verb == 'serve':
                    cost = self._calc_serve_cost_new(env, agent, obj, order_idx)
                
                # print(f"[DEBUG] Task: {verb} {obj}, Cost: {cost}")

                if cost < min_cost:
                    min_cost = cost
                    best_task_info = (f"{verb} {obj} (Order {order_idx+1})", cost)
        
        return best_task_info

    def _get_dist(self, pos1, pos2):
        idx1 = self.solver.get_index(*pos1, self.width)
        idx2 = self.solver.get_index(*pos2, self.width)
        dist = self.solver.dist_matrix[idx1][idx2]
        return dist if dist is not None else float('inf')

    def _get_adjacent_free(self, env, pos_list):
        free = []
        grid = env.to_grid
        for x, y in pos_list:
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.width and 0 <= ny < self.height and grid[nx][ny] == 1:
                    free.append((nx, ny))
        return list(set(free))

    def _calc_chop_cost_new(self, env, agent, ingredient):
        # 環境中の生の材料があれば、その場所までの距離の最小化
        tile_map = {
            "lettuce": "FreshLettuceTile",
            "onion": "FreshOnionTile",
            "tomato": "FreshTomatoTile"
        }
        
        # Tileへの距離
        tile_pos_list = env.get_pos_by_obj_gs(gs=tile_map[ingredient])
        # print(f"[DEBUG] {ingredient} tile pos: {tile_pos_list}")
        tile_adj = self._get_adjacent_free(env, tile_pos_list)
        
        min_dist = float('inf')
        for pos in tile_adj:
            d = self._get_dist(agent.location, pos)
            if d < min_dist: min_dist = d
            
        # 落ちているFreshObjectへの距離も考慮（もしあれば）
        target_name = f"Fresh{ingredient.capitalize()}"
        for obj in env.all_obj:
            if obj.full_name == target_name:
                # オブジェクトの隣接へ
                obj_adj = self._get_adjacent_free(env, [obj.location])
                for pos in obj_adj:
                    d = self._get_dist(agent.location, pos)
                    if d < min_dist: min_dist = d
        
        # print(f"[DEBUG] Chop {ingredient} cost: {min_dist}")
        return min_dist

    def _calc_cook_cost_new(self, env, agent, soup_name, order_idx):
        # soup_name: "onion-tomato soup"
        ingredients = soup_name.replace(" soup", "").split("-")
        
        # 1. 既に混ざり合っている場合（鍋の中）
        pot_places = [(3,5), (4,5), (5,5)]
        
        for pot_loc in pot_places:
             pot = env.pos_obj[pot_loc]
             if pot and "Pot" in pot.full_name:
                 # 中身チェック
                 contents = [o.full_name for o in pot.contents] if hasattr(pot, 'contents') else []
                 # 必要な食材が全て入っているか（Chopped状態で）
                 # かつ、未調理であること
                 if hasattr(pot, 'is_cooked') and pot.is_cooked: continue
                 
                 match = True
                 for req in ingredients:
                     if not any(req.capitalize() in c for c in contents):
                         match = False
                         break
                 
                 if match:
                     # この鍋への距離
                     pot_adj = self._get_adjacent_free(env, [pot_loc])
                     min_d = float('inf')
                     for p in pot_adj:
                         d = self._get_dist(agent.location, p)
                         if d < min_d: min_d = d
                     return min_d

        # 2. 別々に存在する場合 (Chopped X, Chopped Y)
        # 環境中の ChoppedObject を探す
        locs = []
        for req in ingredients:
            req_name = f"Chopped{req.capitalize()}"
            req_locs = []
            for obj in env.all_obj:
                if obj.full_name == req_name:
                    req_locs.append(obj.location)
            if not req_locs:
                return float('inf') # 必要な食材がない
            locs.append(req_locs)
            
        if len(locs) == 1:
            # 1つの食材のみ
            min_d = float('inf')
            for l in locs[0]:
                adj = self._get_adjacent_free(env, [l])
                for p in adj:
                    d = self._get_dist(agent.location, p)
                    if d < min_d: min_d = d
            return min_d
            
        elif len(locs) == 2:
            # agent -> A -> B
            # agent -> B -> A
            min_total = float('inf')
            
            # A -> B
            for l1 in locs[0]:
                adj1 = self._get_adjacent_free(env, [l1])
                dist_to_1 = min([self._get_dist(agent.location, p) for p in adj1]) if adj1 else float('inf')
                
                if dist_to_1 == float('inf'): continue

                for l2 in locs[1]:
                    adj2 = self._get_adjacent_free(env, [l2])
                    dist_1_to_2 = float('inf')
                    for p1 in adj1:
                        for p2 in adj2:
                            d = self._get_dist(p1, p2)
                            if d < dist_1_to_2: dist_1_to_2 = d
                    
                    total = dist_to_1 + dist_1_to_2
                    if total < min_total: min_total = total

            # B -> A
            for l2 in locs[1]:
                adj2 = self._get_adjacent_free(env, [l2])
                dist_to_2 = min([self._get_dist(agent.location, p) for p in adj2]) if adj2 else float('inf')
                
                if dist_to_2 == float('inf'): continue

                for l1 in locs[0]:
                    adj1 = self._get_adjacent_free(env, [l1])
                    dist_2_to_1 = float('inf')
                    for p2 in adj2:
                        for p1 in adj1:
                            d = self._get_dist(p2, p1)
                            if d < dist_2_to_1: dist_2_to_1 = d
                    
                    total = dist_to_2 + dist_2_to_1
                    if total < min_total: min_total = total
            
            return min_total
            
        return float('inf')

    def _calc_serve_cost_new(self, env, agent, soup_name, order_idx):
        # 調理済みのXYsoupがあれば、そこまでの距離
        # 鍋を探す
        pot_places = [(3,5), (4,5), (5,5)]
        
        min_dist = float('inf')
        
        for pot_loc in pot_places:
             pot = env.pos_obj[pot_loc]
             if pot and "Pot" in pot.full_name:
                 # 調理済みか？
                 if hasattr(pot, 'is_cooked') and pot.is_cooked:
                     # 中身が合っているか？
                     ingredients = soup_name.replace(" soup", "").split("-")
                     contents = [o.full_name for o in pot.contents] if hasattr(pot, 'contents') else []
                     
                     match = True
                     for req in ingredients:
                         if not any(req.capitalize() in c for c in contents):
                             match = False
                             break
                     
                     if match:
                         # 距離計算
                         pot_adj = self._get_adjacent_free(env, [pot_loc])
                         for p in pot_adj:
                             d = self._get_dist(agent.location, p)
                             if d < min_dist: min_dist = d
                             
        return min_dist
