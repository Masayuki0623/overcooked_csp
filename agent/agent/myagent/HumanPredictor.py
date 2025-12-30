
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
                return (f"chop {ing_name} (Holding)", 0, [])
            
            # 切った食材 -> Cook
            if "Chopped" in obj_name:
                # ChoppedOnion -> cook ...
                return (f"cook (Holding {obj_name})", 0, [])
                
            # 皿 or スープ -> Serve
            if "Plate" in obj_name or "Soup" in obj_name:
                return (f"serve (Holding {obj_name})", 0, [])

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
                             return (f"chop {ing_name} (Chopping)", 0, [])

        # 2. コスト最小化予測
        tasks_all = self.solver.extract_tasks_from_current_orders(env)
        min_cost = float('inf')
        best_task_info = ("None", 0)
        
        all_costs = []

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
                
                task_str = f"{verb} {obj} (Order {order_idx+1})"
                all_costs.append((task_str, cost))

                if cost < min_cost:
                    min_cost = cost
                    best_task_info = (task_str, cost)
        
        return best_task_info[0], best_task_info[1], all_costs

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

    def _get_pos_by_name(self, env, name):
        locs = []
        for obj in env.world_all:
            # Try full_name (Object), then name (GridSquare)
            n = getattr(obj, 'full_name', getattr(obj, 'name', None))
            if n == name:
                locs.append(obj.location)
        return locs

    def _calc_chop_cost_new(self, env, agent, ingredient):
        # 環境中の生の材料があれば、その場所までの距離 + まな板までの距離
        tile_map = {
            "lettuce": "FreshLettuceTile",
            "onion": "FreshOnionTile",
            "tomato": "FreshTomatoTile"
        }
        
        sources = []
        # Tile
        sources.extend(env.get_pos_by_obj_gs(gs=tile_map[ingredient]))
        # Fresh Object
        target_name = f"Fresh{ingredient.capitalize()}"
        sources.extend(self._get_pos_by_name(env, target_name))
        
        if not sources: return float('inf')
        
        source_adj = self._get_adjacent_free(env, sources)
        
        # Agent -> Source
        dist_to_source = float('inf')
        best_source_pos = None
        for p in source_adj:
            d = self._get_dist(agent.location, p)
            if d < dist_to_source:
                dist_to_source = d
                best_source_pos = p
        
        if dist_to_source == float('inf'): return float('inf')
        
        # Source -> Cutboard
        cutboard_pos = env.get_pos_by_obj_gs(gs="Cutboard")
        
        # まな板が空いているか、または対象の食材が乗っている場合のみ有効
        valid_cutboards = []
        target_fresh_name = f"Fresh{ingredient.capitalize()}"
        
        for loc in cutboard_pos:
            obj = env.pos_obj.get(loc)
            if obj is None:
                valid_cutboards.append(loc)
            elif getattr(obj, 'full_name', '') == target_fresh_name:
                valid_cutboards.append(loc)
                
        if not valid_cutboards:
            return float('inf')

        cut_adj = self._get_adjacent_free(env, valid_cutboards)
        
        dist_source_to_cut = float('inf')
        for p in cut_adj:
            d = self._get_dist(best_source_pos, p)
            if d < dist_source_to_cut: dist_source_to_cut = d
            
        return dist_to_source + dist_source_to_cut

    def _get_ingredient_sources(self, env, req_base_name, valid_ingredients):
        # req_base_name: "onion" (lowercase)
        # valid_ingredients: ["onion", "tomato"] (lowercase)
        
        sources = []
        req_state = "Chopped"
        
        for loc, obj in env.pos_obj.items():
            if obj is None: continue
            
            # Check if obj contains the required ingredient in Chopped state
            has_req = False
            is_valid_obj = True
            
            if not hasattr(obj, 'contents'): continue
            
            for c in obj.contents:
                # Plate check: Plate objects do not have get_state()
                # If the object contains a Plate, it is likely a plated dish or just a plate.
                # We cannot use it as a source for raw/chopped ingredients to put into a pot.
                if not hasattr(c, 'get_state'):
                    is_valid_obj = False
                    break

                c_base = c.name.lower() # "onion"
                c_state = c.get_state() # "Chopped"
                
                if c_base == req_base_name and c_state == req_state:
                    has_req = True
                
                # Check if this content is allowed in the soup
                if c_base not in valid_ingredients:
                    is_valid_obj = False
                    break
                
                # Also must be Chopped
                if c_state != "Chopped":
                    is_valid_obj = False
                    break
            
            if has_req and is_valid_obj:
                sources.append(loc)
                
        return sources

    def _calc_cook_cost_new(self, env, agent, soup_name, order_idx):
        # soup_name: "onion-tomato soup"
        ingredients = soup_name.replace(" soup", "").split("-")
        
        # 鍋を探す
        pot_locs = env.get_pos_by_obj_gs(gs="Pot")
        min_total_cost = float('inf')

        for pot_loc in pot_locs:
            pot_gs = env.pos_gs[pot_loc]
            content_obj = env.pos_obj[pot_loc]
            
            # 既に調理済みならCookタスクの対象外（Serve対象）
            if hasattr(pot_gs, 'is_cooked') and pot_gs.is_cooked():
                continue
            
            # 現在の中身を確認
            current_contents = []
            if content_obj:
                # content_obj.contents は Food/Plate オブジェクトのリスト
                current_contents = [c.full_name for c in content_obj.contents]
            
            # この鍋がこのスープに使えるか判定
            # 鍋の中身が、求められている食材のサブセットであること
            is_valid_pot = True
            for c_name in current_contents:
                # c_name (例: ChoppedOnion) が ingredients (例: [onion, tomato]) のいずれかに該当するか
                # ChoppedOnion -> onion
                base_name = c_name.replace("Chopped", "").lower()
                if base_name not in ingredients:
                    is_valid_pot = False
                    break
            if not is_valid_pot:
                continue

            # 不足している食材を特定
            missing_ingredients = []
            for req in ingredients:
                # req (onion) が current_contents に含まれているか
                # current_contents には "ChoppedOnion" のように入っている
                found = False
                for c_name in current_contents:
                    if req.capitalize() in c_name:
                        found = True
                        break
                if not found:
                    missing_ingredients.append(req)
            
            # 必要な食材が全て揃っているか確認（持っているか、環境にあるか）
            # 1つでも欠けていれば、この鍋でのCookタスクは不可（Chopなどが先）
            all_req_exist = True
            for req in missing_ingredients:
                # 1. 持っているか確認
                is_holding = False
                if agent.holding:
                    h_name = agent.holding.full_name
                    if req.capitalize() in h_name and "Chopped" in h_name:
                        is_holding = True
                
                if is_holding: continue

                # 2. 環境にあるか確認
                if not self._get_ingredient_sources(env, req, ingredients):
                    all_req_exist = False
                    break
            
            if not all_req_exist:
                continue

            # print(f"[DEBUG] Pot at {pot_loc}, Missing: {missing_ingredients}")

            if not missing_ingredients:
                # 全て揃っている -> 鍋への距離 (調理開始)
                pot_adj = self._get_adjacent_free(env, [pot_loc])
                for p in pot_adj:
                    d = self._get_dist(agent.location, p)
                    if d < min_total_cost: min_total_cost = d
            else:
                # 不足食材を取りに行くコスト
                # 最も近い不足食材への距離 + その食材から鍋への距離（概算）
                for req in missing_ingredients:
                    # req_name = f"Chopped{req.capitalize()}"
                    # req_locs = self._get_pos_by_name(env, req_name)
                    
                    # 結合された食材（ChoppedLettuce-ChoppedOnionなど）も考慮して検索
                    req_locs = self._get_ingredient_sources(env, req, ingredients)
                    
                    # print(f"[DEBUG] Looking for {req} for {soup_name}: found at {req_locs}")

                    if not req_locs:
                        # 必要な食材が場にない -> この鍋では作れない(or まだ切ってない)
                        # ここでは無限大としておく（Chopタスクが先）
                        continue
                        
                    req_adj = self._get_adjacent_free(env, req_locs)
                    
                    # Agent -> Ingredient
                    dist_to_ing = float('inf')
                    best_ing_pos = None
                    for p in req_adj:
                        d = self._get_dist(agent.location, p)
                        if d < dist_to_ing: 
                            dist_to_ing = d
                            best_ing_pos = p # 隣接セル
                    
                    if dist_to_ing == float('inf'): continue

                    # Ingredient -> Pot
                    # best_ing_pos から pot_adj への距離
                    pot_adj = self._get_adjacent_free(env, [pot_loc])
                    dist_ing_to_pot = float('inf')
                    for p_pot in pot_adj:
                        d = self._get_dist(best_ing_pos, p_pot)
                        if d < dist_ing_to_pot: dist_ing_to_pot = d
                    
                    total = dist_to_ing + dist_ing_to_pot
                    # print(f"[DEBUG] Cost for {req_name}: {dist_to_ing} + {dist_ing_to_pot} = {total}")

                    if total < min_total_cost:
                        min_total_cost = total

        return min_total_cost

    def _calc_serve_cost_new(self, env, agent, soup_name, order_idx):
        # 調理済みのXYsoupがあれば、そこまでの距離
        # 鍋を探す
        pot_locs = env.get_pos_by_obj_gs(gs="Pot")
        min_dist = float('inf')
        
        for pot_loc in pot_locs:
            pot_gs = env.pos_gs[pot_loc]
            content_obj = env.pos_obj[pot_loc]
            
            # 調理済みか？
            if not (hasattr(pot_gs, 'is_cooked') and pot_gs.is_cooked()):
                continue

            # 中身が合っているか？
            ingredients = soup_name.replace(" soup", "").split("-")
            current_contents = []
            if content_obj:
                current_contents = [c.full_name for c in content_obj.contents]
            
            match = True
            for req in ingredients:
                # req (onion) が current_contents に含まれているか
                found = False
                for c_name in current_contents:
                    if req.capitalize() in c_name:
                        found = True
                        break
                if not found:
                    match = False
                    break
            
            if match:
                # 距離計算
                # 皿を持っているか？
                has_plate = False
                if agent.holding and "Plate" in agent.holding.full_name:
                    has_plate = True
                
                pot_adj = self._get_adjacent_free(env, [pot_loc])
                
                if has_plate:
                    # 鍋へ直行
                    for p in pot_adj:
                        d = self._get_dist(agent.location, p)
                        if d < min_dist: min_dist = d
                else:
                    # 皿を取りに行ってから鍋へ
                    # 皿の場所を探す (PlateTile or Plate object)
                    plate_locs = []
                    # PlateTile
                    plate_locs.extend(env.get_pos_by_obj_gs(gs="PlateTile"))
                    # Plate Object (on counter)
                    for obj in env.all_obj:
                        if obj.full_name == "Plate":
                            plate_locs.append(obj.location)
                    
                    if not plate_locs:
                        continue # 皿がない
                        
                    plate_adj = self._get_adjacent_free(env, plate_locs)
                    
                    # Agent -> Plate
                    dist_to_plate = float('inf')
                    best_plate_pos = None
                    for p in plate_adj:
                        d = self._get_dist(agent.location, p)
                        if d < dist_to_plate:
                            dist_to_plate = d
                            best_plate_pos = p
                    
                    if dist_to_plate == float('inf'): continue
                    
                    # Plate -> Pot
                    dist_plate_to_pot = float('inf')
                    for p_pot in pot_adj:
                        d = self._get_dist(best_plate_pos, p_pot)
                        if d < dist_plate_to_pot: dist_plate_to_pot = d
                        
                    total = dist_to_plate + dist_plate_to_pot
                    if total < min_dist: min_dist = total
                             
        return min_dist
