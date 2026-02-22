import sys
import time
from copy import deepcopy
from ortools.sat.python import cp_model
from gym_cooking.utils.config import COOKING_TIME_SECONDS, CHOPPING_NUM_STEPS

class CSP_A_Agent:
    def __init__(self, speed=10, replay=None):
        self.speed = speed
        self.replay = replay
        self.action_history = []
        self.initialized = False
        self.planned_actions = []
        
        # Directions: (dx, dy)
        self.moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)] # Up, Down, Right, Left, Stay

    def extract_leftmost_order(self, env):
        if env.order.current_orders:
            order = env.order.current_orders[0]
            goal_obj = order[0]
            name = goal_obj.full_name.lower()
            ingredients = [ing for ing in ['lettuce', 'onion', 'tomato'] if ing in name]
            return '-'.join(ingredients) + ' soup', ingredients
        return None, []

    def solve_path_csp(self, env, agent_start, target_pos):
        W = env.world_width
        H = env.world_height
        valid_floors = []
        for cx in range(W):
            for cy in range(H):
                if env.to_grid[cx][cy] == 1:
                    valid_floors.append((cx, cy))
                    
        min_t = abs(agent_start[0]-target_pos[0]) + abs(agent_start[1]-target_pos[1]) - 1
        min_t = max(1, min_t)
        
        for T in range(min_t, 60):
            model = cp_model.CpModel()
            
            x = [model.NewIntVar(0, W - 1, f'x_{t}') for t in range(T + 1)]
            y = [model.NewIntVar(0, H - 1, f'y_{t}') for t in range(T + 1)]
            a = [model.NewIntVar(0, 4, f'a_{t}') for t in range(T)]
            
            is_int = [model.NewIntVar(0, 1, f'is_int_{t}') for t in range(T)]
            ix = [model.NewIntVar(0, W - 1, f'ix_{t}') for t in range(T)]
            iy = [model.NewIntVar(0, H - 1, f'iy_{t}') for t in range(T)]
            
            model.Add(x[0] == agent_start[0])
            model.Add(y[0] == agent_start[1])
            
            # Goal: at time T-1, we perform an interaction with target_pos
            model.Add(is_int[T-1] == 1)
            model.Add(ix[T-1] == target_pos[0])
            model.Add(iy[T-1] == target_pos[1])
            
            transition_tuples = []
            for cx in range(W):
                for cy in range(H):
                    if (cx, cy) not in valid_floors:
                        continue
                    for ax, (ddx, ddy) in enumerate([(0,1), (0,-1), (1,0), (-1,0), (0,0)]):
                        nx, ny = cx + ddx, cy + ddy
                        if 0 <= nx < W and 0 <= ny < H:
                            if (nx, ny) in valid_floors:
                                transition_tuples.append((cx, cy, ax, nx, ny, 0, cx, cy))
                            else:
                                transition_tuples.append((cx, cy, ax, cx, cy, 1, nx, ny))
                        else:
                            transition_tuples.append((cx, cy, ax, cx, cy, 0, cx, cy))
                            
            for t in range(T):
                model.AddAllowedAssignments(
                    [x[t], y[t], a[t], x[t+1], y[t+1], is_int[t], ix[t], iy[t]],
                    transition_tuples
                )

            solver = cp_model.CpSolver()
            status = solver.Solve(model)
            if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                ans = []
                for t_idx in range(T):
                    ans.append(solver.Value(a[t_idx]))
                return ans, (solver.Value(x[T]), solver.Value(y[T]))
                
        return [], agent_start

    def precompute_plan(self, env):
        target_soup, target_ings = self.extract_leftmost_order(env)
        if not target_soup:
            return []
            
        agent_pos = env.self_pos
        
        cutboards = env.get_pos_by_obj_gs(gs="Cutboard")
        pots = env.get_pos_by_obj_gs(gs="Pot")
        deliveries = env.get_pos_by_obj_gs(gs="Delivery")
        plates = env.get_pos_by_obj_gs(obj="Plate") or env.get_pos_by_obj_gs(gs="PlateTile")
        counters = env.get_pos_by_obj_gs(gs="Counter")
        
        if not cutboards or not pots or not deliveries or not plates or not counters:
             return []
             
        cb = cutboards[0]
        pot = pots[0] 
        delivery = deliveries[0]
        plate = plates[0]
        merge_counter = counters[0] # Simply select the first counter for merging
        
        all_actions = []
        
        for ing in target_ings:
            tile_map = {"lettuce": "FreshLettuceTile", "onion": "FreshOnionTile", "tomato": "FreshTomatoTile"}
            ing_map = env.get_pos_by_obj_gs(gs=tile_map.get(ing))
            if not ing_map:
                obj_map = {"lettuce": "FreshLettuce", "onion": "FreshOnion", "tomato": "FreshTomato"}
                ing_map = env.get_pos_by_obj_gs(obj=obj_map.get(ing))
            
            if not ing_map:
                 continue
            ing_pos = ing_map[0]
            
            print(f"[CSP_A] 探索: {ing} の取得・切断・マージ...")
            
            # Navigate & Pick up Ingredient
            acts, agent_pos = self.solve_path_csp(env, agent_pos, ing_pos)
            all_actions.extend(acts)
            
            # Navigate & Drop on Cutboard
            acts, agent_pos = self.solve_path_csp(env, agent_pos, cb)
            all_actions.extend(acts)
            
            # Chop X times
            dx, dy = cb[0] - agent_pos[0], cb[1] - agent_pos[1]
            chop_act = self.moves.index((dx, dy))
            for _ in range(CHOPPING_NUM_STEPS):
                all_actions.append(chop_act)
                
            # Pick up chopped
            all_actions.append(chop_act)
            
            # Navigate & Drop on Merge Counter
            acts, agent_pos = self.solve_path_csp(env, agent_pos, merge_counter)
            all_actions.extend(acts)
            
        print(f"[CSP_A] 探索: マージ済み食材の鍋への投入・配膳...")
        
        # Pick up merged ingredients from counter
        acts, agent_pos = self.solve_path_csp(env, agent_pos, merge_counter)
        all_actions.extend(acts)
        
        # Navigate & Drop in Pot
        acts, agent_pos = self.solve_path_csp(env, agent_pos, pot)
        all_actions.extend(acts)
        
        time_start_cook = len(all_actions)
        
        # Get Plate while cooking
        acts, agent_pos = self.solve_path_csp(env, agent_pos, plate)
        all_actions.extend(acts)
        
        # Plate to Pot (wait if necessary, then fill plate)
        acts, agent_pos = self.solve_path_csp(env, agent_pos, pot)
        all_actions.extend(acts)
        
        time_arrive_pot = len(all_actions)
        time_spent_getting_plate = time_arrive_pot - time_start_cook
        
        # Pot needs time to cook
        cook_time_frames = COOKING_TIME_SECONDS * 10 + 2
        stay_idx = self.moves.index((0, 0))
        if time_spent_getting_plate < cook_time_frames:
            print(f"[CSP_A] 鍋の調理完了まで待機します ({cook_time_frames - time_spent_getting_plate} frames)...")
            for _ in range(cook_time_frames - time_spent_getting_plate):
                all_actions.append(stay_idx)
        
        # Need one action to properly interact/fill the plate after waiting
        all_actions.append(stay_idx)
        
        # Pot to Delivery
        acts, agent_pos = self.solve_path_csp(env, agent_pos, delivery)
        all_actions.extend(acts)
        
        return all_actions

    def high_level_infer(self, env, msg):
        pass

    def __call__(self, env):
        if not self.initialized:
            start_t = time.time()
            self.planned_actions = self.precompute_plan(env)
            elapsed = time.time() - start_t
            print(f"\n==============================================")
            print(f"[CSP_A] 事前探索完了!")
            print(f"[CSP_A] 計算時間: {elapsed:.3f}秒")
            print(f"[CSP_A] アクション数: {len(self.planned_actions)}")
            print(f"==============================================\n")
            self.initialized = True
            
        if self.planned_actions:
            move_idx = self.planned_actions.pop(0)
            move = self.moves[move_idx]
            chat = f"[CSP_A] 実行中... (残り手数: {len(self.planned_actions)})"
            return move, chat
        else:
            return (0, 0), "タスク完了 / 待機中"
