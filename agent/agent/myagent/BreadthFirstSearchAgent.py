import collections
import copy
from gym_cooking.utils.core import Object, ObjectRepr, Pot, GridSquare

class BreadthFirstSearchAgent:
    def __init__(self, speed=2.5, replay=None):
        self.speed = speed
        self.replay = replay
        self.current_plan = None
        self.current_step = 0

    def _get_initial_state(self, env):
        agent_pos = env.self_pos
        held_obj_repr = env.hold.get_repr() if env.hold else None

        world_objects = []
        # env.world_all contains all objects, including GridSquare
        for obj in env.world_all:
            if isinstance(obj, Object) and not obj.is_held:
                # Check if the object is on a Pot, if so, it's part of the pot's state
                gs_at_loc = env.pos_gs.get(obj.location)
                if not isinstance(gs_at_loc, Pot):
                     world_objects.append(obj.get_repr())

        pot_states = []
        for obj in env.world_all:
            if isinstance(obj, Pot):
                content_repr = None
                if obj.holding:
                    content_repr = obj.holding.full_name
                pot_states.append((obj.location, content_repr))

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

    def _is_goal_state(self, state, orders):
        delivered_orders_in_state = state[4]
        required_orders = {order[0].full_name for order in orders}
        return required_orders.issubset(delivered_orders_in_state)

    def _get_next_states(self, state, width, height, static_collidable_locs, static_objects_map, orders):
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
                    if target_gs_type in ['Counter', 'Cutboard'] and not target_dynamic_obj_repr:
                        new_list = list(world_objects_repr)
                        new_list.append(held_obj_repr._replace(location=target_pos, is_held=False))
                        new_state = (agent_pos, None, tuple(sorted(new_list)), pot_states, delivered_orders)
                        next_states.append((new_state, action))
                    elif target_gs_type == 'Delivery':
                        required = {order.full_name for order, _, _, _ in orders}
                        if held_obj_repr.name in required:
                            new_delivered = frozenset(list(delivered_orders) + [held_obj_repr.name])
                            new_state = (agent_pos, None, world_objects_repr, pot_states, new_delivered)
                            next_states.append((new_state, action))
                    elif target_gs_type == 'Cutboard' and held_obj_repr.name.startswith('Fresh'):
                        new_held_name = held_obj_repr.name.replace('Fresh', 'Chopped')
                        new_held_obj = held_obj_repr._replace(name=new_held_name)
                        new_state = (agent_pos, new_held_obj, world_objects_repr, pot_states, delivered_orders)
                        next_states.append((new_state, action))
                else: 
                    if target_dynamic_obj_repr and target_gs_type in ['Counter', 'Cutboard']:
                        new_list = [r for r in world_objects_repr if r != target_dynamic_obj_repr]
                        new_held = target_dynamic_obj_repr._replace(location=agent_pos, is_held=True)
                        new_state = (agent_pos, new_held, tuple(sorted(new_list)), pot_states, delivered_orders)
                        next_states.append((new_state, action))
                    elif target_gs_type in ['FreshTomatoTile', 'FreshLettuceTile', 'FreshOnionTile', 'PlateTile']:
                        name_map = {'FreshTomatoTile': 'FreshTomato', 'FreshLettuceTile': 'FreshLettuce', 'FreshOnionTile': 'FreshOnion', 'PlateTile': 'Plate'}
                        new_name = name_map[target_gs_type]
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
        initial_state = self._get_initial_state(env)
        orders = env.order.current_orders
        
        width = env.world_width
        height = env.world_height
        static_collidable_locs = {o.location for o in env.world_all if o.collidable}
        static_objects_map = {o.location: o.name for o in env.world_all if isinstance(o, GridSquare)}

        queue = collections.deque([(initial_state, [], 0)])
        visited = {initial_state}
        max_depth = 400
        count = 0
        print_interval = 2000  # Print progress every 2000 nodes

        while queue:
            count += 1
            current_state, action_history, depth = queue.popleft()

            # Print debug information at intervals
            if count % print_interval == 0:
                agent_pos, held_obj_repr, _, _, _ = current_state
                held_item = held_obj_repr.name if held_obj_repr else "Nothing"
                print(f"探索中... [ノード数: {count}, 現在の深さ: {depth}, キュー長: {len(queue)}, "
                      f"位置: {agent_pos}, 持ち物: {held_item}]")

            if self._is_goal_state(current_state, orders):
                print(f"ゴールを発見! (深さ: {depth}, 探索ノード数: {count})")
                return action_history

            if depth >= max_depth:
                continue

            next_states_info = self._get_next_states(current_state, width, height, static_collidable_locs, static_objects_map, orders)
            
            for next_state, action in next_states_info:
                if next_state not in visited:
                    visited.add(next_state)
                    new_action_history = action_history + [action]
                    queue.append((next_state, new_action_history, depth + 1))
        
        print(f"探索が最大深度({max_depth})に達したか、キューが空になりましたがゴールは見つかりませんでした。(探索ノード数: {count})")
        return None

    def __call__(self, env):
        if self.current_plan is None:
            print("BFSによるプランニングを開始します...")
            self.current_plan = self.bfs(env)
            
            if self.current_plan:
                print(f"プランが見つかりました。ステップ数: {len(self.current_plan)}")
                self.current_step = 0
            else:
                print("プランが見つかりませんでした。Stayします。")
                self.current_plan = []
        
        if self.current_step < len(self.current_plan):
            action = self.current_plan[self.current_step]
            self.current_step += 1
            return action, ""
        else:
            return (0, 0), ""
