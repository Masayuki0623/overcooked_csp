import collections
import copy
from gym_cooking.utils.core import Object, ObjectRepr, Pot, GridSquare

class BreadthFirstSearchAgent:
    def __init__(self, speed=2.5, replay=None):
        self.speed = speed
        self.replay = replay
        self.current_plan = None
        self.current_step = 0
        self.goal_ingredients = frozenset(['ChoppedOnion', 'ChoppedLettuce'])

    def _get_initial_state(self, env):
        agent_pos = env.self_pos
        held_obj_repr = env.hold.get_repr() if env.hold else None

        world_objects = []
        for obj in env.world_all:
            if isinstance(obj, Object) and not obj.is_held:
                gs_at_loc = env.pos_gs.get(obj.location)
                if not isinstance(gs_at_loc, Pot):
                     world_objects.append(obj.get_repr())

        pot_states = []
        for obj in env.world_all:
            if isinstance(obj, Pot):
                content_set = frozenset()
                if obj.holding:
                    content_set = frozenset(c.full_name for c in obj.holding.contents)
                pot_states.append((obj.location, content_set))

        delivered_orders = []
        for obj in env.world_all:
            if obj.name == 'Delivery':
                delivered_orders.extend([o.full_name for o in obj.holding])

        state = (
            agent_pos,
            held_obj_repr,
            tuple(sorted(world_objects)),
            tuple(sorted(pot_states)),
            frozenset(delivered_orders)
        )
        return state

    def _is_goal_state(self, state):
        pot_states = state[3]
        for _, ingredients in pot_states:
            if self.goal_ingredients.issubset(ingredients):
                return True
        return False

    def _get_next_states(self, state, width, height, static_collidable_locs, static_objects_map):
        next_states = []
        agent_pos, held_obj_repr, world_objects_repr, pot_states, delivered_orders = state
        dynamic_collidable_locs = {repr.location for repr in world_objects_repr}

        possible_actions = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
        
        for action in possible_actions:
            if action == (0, 0):
                continue

            dx, dy = action
            target_pos = (agent_pos[0] + dx, agent_pos[1] + dy)

            if not (0 <= target_pos[0] < width and 0 <= target_pos[1] < height):
                continue

            is_interaction = target_pos in static_collidable_locs or target_pos in dynamic_collidable_locs

            if is_interaction:
                target_gs_type = static_objects_map.get(target_pos)
                target_dynamic_obj_repr = next((r for r in world_objects_repr if r.location == target_pos), None)

                if held_obj_repr:
                    if target_gs_type in ['Counter', 'Cutboard']:
                         if not target_dynamic_obj_repr:
                            # Freshなものをまな板に置く場合は、別ロジック(切る処理)で扱うためここでは除外
                            if target_gs_type == 'Cutboard' and held_obj_repr.name.startswith('Fresh'):
                                pass 
                            else:
                                new_list = list(world_objects_repr)
                                new_list.append(held_obj_repr._replace(location=target_pos, is_held=False))
                                new_state = (agent_pos, None, tuple(sorted(new_list)), pot_states, delivered_orders)
                                next_states.append((new_state, action))

                         elif target_gs_type == 'Counter':
                             # 既に物がある場合はマージを試みる
                             # 例: ChoppedOnion + ChoppedLettuce -> ChoppedLettuce-ChoppedOnion
                             # ここでは簡易的に、両方ともChoppedならマージ可能とする
                             if held_obj_repr.name.startswith('Chopped') and target_dynamic_obj_repr.name.startswith('Chopped'):
                                 # 名前を結合してソート
                                 parts = held_obj_repr.name.split('-') + target_dynamic_obj_repr.name.split('-')
                                 new_name = "-".join(sorted(parts))
                                 
                                 # 有効な組み合わせかチェック
                                 valid_combinations = {
                                     'ChoppedLettuce-ChoppedOnion', 
                                     'ChoppedLettuce-ChoppedTomato', 
                                     'ChoppedOnion-ChoppedTomato',
                                     'ChoppedLettuce-ChoppedOnion-ChoppedTomato'
                                 }
                                 
                                 if new_name in valid_combinations:
                                     # 既存のオブジェクトを削除し、新しいマージ済みオブジェクトを配置
                                     new_list = [r for r in world_objects_repr if r != target_dynamic_obj_repr]
                                     new_merged_obj = target_dynamic_obj_repr._replace(name=new_name, is_held=False)
                                     new_list.append(new_merged_obj)
                                     
                                     # エージェントは手ぶらになる
                                     new_state = (agent_pos, None, tuple(sorted(new_list)), pot_states, delivered_orders)
                                     next_states.append((new_state, action))

                    if target_gs_type == 'Cutboard' and held_obj_repr.name.startswith('Fresh'):
                        # まな板が空いている場合のみ、置いて切ることができる
                        if not target_dynamic_obj_repr:
                            # まな板に置くと同時に1段階刻まれる (Fresh -> Chopping)
                            # エージェントは手ぶらになり、まな板の上にChoppingアイテムが残る
                            if 'Fresh' in held_obj_repr.name:
                                new_held_name = held_obj_repr.name.replace('Fresh', 'Chopping')
                            else:
                                # 万が一Fresh以外でneeds_choppedなものが来た場合
                                new_held_name = held_obj_repr.name
                            
                            new_chopping_obj = held_obj_repr._replace(name=new_held_name, location=target_pos, is_held=False)
                            
                            new_list = list(world_objects_repr)
                            new_list.append(new_chopping_obj)
                            
                            new_state = (agent_pos, None, tuple(sorted(new_list)), pot_states, delivered_orders)
                            next_states.append((new_state, action))

                    elif target_gs_type == 'Pot' and held_obj_repr.name.startswith('Chopped'):
                        new_pot_states = list(pot_states)
                        for i, (pot_loc, ingredients) in enumerate(new_pot_states):
                            if pot_loc == target_pos:
                                # 環境の仕様上、鍋が空でないと追加投入できないため、空の場合のみ遷移を許可する
                                if len(ingredients) == 0:
                                    # 名前を分解してセットに追加
                                    item_names = set(held_obj_repr.name.split('-'))
                                    new_ingredients = ingredients.union(item_names)
                                    new_pot_states[i] = (pot_loc, new_ingredients)
                                    
                                    new_state = (agent_pos, None, world_objects_repr, tuple(sorted(new_pot_states)), delivered_orders)
                                    next_states.append((new_state, action))
                                else:
                                    # 既に具材が入っている場合はここでの投入は不可。
                                    # カウンターでマージしてから入れる戦略を強制する。
                                    pass 
                                break
                else: 
                    if target_dynamic_obj_repr and target_gs_type in ['Counter', 'Cutboard']:
                        
                        # まな板上の Chopping 状態のアイテムに対するインタラクト (Chopping -> Chopped)
                        if target_gs_type == 'Cutboard' and target_dynamic_obj_repr.name.startswith('Chopping'):
                            new_name = target_dynamic_obj_repr.name.replace('Chopping', 'Chopped')
                            new_chopped_obj = target_dynamic_obj_repr._replace(name=new_name)
                            
                            # エージェントはまだ手ぶら、アイテムはまな板の上で状態変化
                            new_list = [r for r in world_objects_repr if r != target_dynamic_obj_repr]
                            new_list.append(new_chopped_obj)
                            
                            new_state = (agent_pos, None, tuple(sorted(new_list)), pot_states, delivered_orders)
                            next_states.append((new_state, action))
                            
                        else:
                            # 拾う (Pick up) - 通常のアイテム、または Chopped になったアイテム、カウンターのアイテム
                            new_list = [r for r in world_objects_repr if r != target_dynamic_obj_repr]
                            new_held = target_dynamic_obj_repr._replace(location=agent_pos, is_held=True)
                            new_state = (agent_pos, new_held, tuple(sorted(new_list)), pot_states, delivered_orders)
                            next_states.append((new_state, action))

                    elif target_gs_type in ['FreshTomatoTile', 'FreshLettuceTile', 'FreshOnionTile', 'PlateTile']:
                        name_map = {'FreshTomatoTile': 'FreshTomato', 'FreshLettuceTile': 'FreshLettuce', 'FreshOnionTile': 'FreshOnion', 'PlateTile': 'Plate'}
                        new_name = name_map[target_gs_type]

                        # 必要な食材のみを取得するように制限 (枝刈り)
                        needed = False
                        if new_name == 'Plate':
                            # 今回のゴール(鍋に入れるだけ)には皿は不要と仮定
                            needed = False
                        else:
                            # FreshOnion -> Onion
                            base_name = new_name.replace('Fresh', '')
                            target_chopped = f'Chopped{base_name}'

                            if target_chopped in self.goal_ingredients:
                                # 既に場にある(カウンタ上 or 鍋の中)なら新たに取らない
                                existing_count = 0
                                # held_obj_repr is None here
                                for obj in world_objects_repr:
                                    if base_name in obj.name:
                                        existing_count += 1
                                for _, ingredients in pot_states:
                                    for ing in ingredients:
                                        if base_name in ing:
                                            existing_count += 1
                                
                                # まだ1つもなければ必要
                                if existing_count == 0:
                                    needed = True
                        
                        if needed:
                            new_held = ObjectRepr(name=new_name, location=agent_pos, is_held=True)
                            new_state = (agent_pos, new_held, world_objects_repr, pot_states, delivered_orders)
                            next_states.append((new_state, action))
            else: 
                new_pos = target_pos
                new_held = held_obj_repr._replace(location=new_pos) if held_obj_repr else None
                new_state = (new_pos, new_held, world_objects_repr, pot_states, delivered_orders)
                next_states.append((new_state, action))
        return next_states
    
    def bfs(self, env):
        import time
        start_time = time.time()
        initial_state = self._get_initial_state(env)
        
        width = env.world_width
        height = env.world_height
        static_collidable_locs = {o.location for o in env.world_all if o.collidable}
        static_objects_map = {o.location: o.name for o in env.world_all if isinstance(o, GridSquare)}

        queue = collections.deque([(initial_state, [], 0)])
        visited = {initial_state}
        max_depth = 400
        count = 0
        print_interval = 1

        while queue:
            count += 1
            current_state, action_history, depth = queue.popleft()

            if count % print_interval == 0:
                elapsed = time.time() - start_time
                agent_pos, held_obj_repr, _, pot_states, _ = current_state
                held_item = held_obj_repr.name if held_obj_repr else "Nothing"
                pot_summary = [p[1] for p in pot_states]
                print(f"探索中... [経過時間: {elapsed:.2f}s, ノード数: {count}, 現在の深さ: {depth}, キュー長: {len(queue)}, "
                      f"位置: {agent_pos}, 持ち物: {held_item}, 鍋: {pot_summary}]")

            if self._is_goal_state(current_state):
                elapsed = time.time() - start_time
                print(f"ゴールを発見! (経過時間: {elapsed:.2f}s, 深さ: {depth}, 探索ノード数: {count})")
                return action_history

            if depth >= max_depth:
                continue

            next_states_info = self._get_next_states(current_state, width, height, static_collidable_locs, static_objects_map)
            
            for next_state, action in next_states_info:
                if next_state not in visited:
                    visited.add(next_state)
                    new_action_history = action_history + [action]
                    queue.append((next_state, new_action_history, depth + 1))
        
        elapsed = time.time() - start_time
        print(f"探索が最大深度({max_depth})に達したか、キューが空になりましたがゴールは見つかりませんでした。(経過時間: {elapsed:.2f}s, 探索ノード数: {count})")
        return None

    def __call__(self, env):
        # This agent's planning is done entirely upfront in play_main.py
        # This __call__ method is only for executing the pre-computed plan.
        if self.current_plan is None:
             # This should not be reached if pre-planning in play_main.py is successful.
            print("警告: __call__がプランニング無しで呼び出されました。")
            return (0, 0), ""

        if self.current_step < len(self.current_plan):
            action = self.current_plan[self.current_step]
            self.current_step += 1
            return action, ""
        else:
            # Plan is complete
            return (0, 0), ""
