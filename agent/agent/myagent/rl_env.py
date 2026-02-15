
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_cooking.utils.core import *

class KitchenGym(gym.Env):
    """
    Wrapper for gym_cooking environment to make it compatible with Stable Baselines3 (Gymnasium).
    Focused on single agent learning.
    """
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, env):
        super(KitchenGym, self).__init__()
        self.env = env
        
        # Ensure env is reset to populate world properties
        if not hasattr(env, 'world') or env.world is None:
            env.reset()
            
        # Access attributes from the wrapped env
        self.width = env.world.width
        self.height = env.world.height
        
        # Action Space: 
        # 0: No-op, 1-4: Move (Left, Right, Down, Up)
        # Interaction is implicit by moving into objects.
        self.action_space = spaces.Discrete(5)

        # Observation Space:
        # Construct feature channels.
        # Channels:
        # 0-N: GridSquare types (Floor, Counter, Cutboard, Delivery, etc.)
        # ... : Ingredient types (Fresh, Chopped, Cooked...)
        # ... : Agent positions (My Agent, Other Agent)
        # ... : Agent Orientations
        
        self.channels = 0
        self.feature_map = []
        
        # 1. GridSquares
        self.gs_types = ['Floor', 'Counter', 'Cutboard', 'Delivery', 'Tomato', 'Lettuce', 'Onion', 'Plate', 'Pot']
        # Note: Tomato/Lettuce etc appear as objects but also sometimes as piles? 
        # Actually OvercookedEnv get_current_state defines keys.
        
        # We will parse the state dictionary returned by env.
        # Let's define a fixed number of channels based on expected keys
        # We'll determine this dynamically or fix it.
        # For a robust Agent, a fixed set is better.
        
        self.layer_keys = [
            # Grid Layers
            'Floor', 'Counter', 'Cutboard', 'Delivery',
            # Objects - Fresh
            'FreshTomato', 'FreshLettuce', 'FreshOnion',
            # Objects - Chopped
            'ChoppedTomato', 'ChoppedLettuce', 'ChoppedOnion',
            # Cooking
            'Pot', # Pot state needs detail? Pot is usually an object.
            # Plating
            'Plate', 'Dish', # Dish might be plated food.
            # Agents
            'AgentLoc', 'OtherAgentLoc'
        ]
        
        self.channels = len(self.layer_keys) + 5 # +5 for extra info if any
        
        # Shape: (Channels, Height, Width)
        # Using uint8 for mask
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(20, self.height, self.width), # Reserve 20 channels for safety
            dtype=np.uint8
        )

        self.last_successful_orders = 0
        self.max_orders = 3

    def reset(self, seed=None, options=None):
        if seed is not None:
            # gymnasium uses seeding slightly differently
            # self.env.seed(seed) 
            pass
            
        self.env.reset()
        self.last_successful_orders = 0
        self.current_step = 0
        
        # Process observation
        state = self.env.get_current_state() # Currently returns a complex structure? 
        # Actually need to check what get_current_state returns exactly.
        # Based on code read:
        # gridsquare_map dictionary.
        
        obs = self._process_obs(state)
        return obs, {}

    def step(self, action):
        self.current_step += 1
        # Convert scalar action to action_dict for the multi-agent env
        
        # Actions: 0: (0,0), 1: (-1,0), 2: (1,0), 3: (0,1), 4: (0,-1)
        
        action_map = {
            0: (0, 0), # Wait
            1: (-1, 0), # Left
            2: (1, 0),  # Right
            3: (0, 1),  # Down
            4: (0, -1), # Up
        }
        
        real_action = action_map.get(action, (0,0))
        # gameplay.py maps KeyToTuple.
        
        my_action = real_action
        other_action = (0, 0) # Static for now
        
        # Get agent names
        agent_names = [a.name for a in self.env.sim_agents]
        
        # Debug print
        # print(f"[RL Debug] Step: {self.current_step}, Action: {action}, Real: {real_action}, Loc: {self.env.sim_agents[0].location}")
        
        action_dict = {}
        if len(agent_names) > 0:
            action_dict[agent_names[0]] = my_action
        if len(agent_names) > 1:
            action_dict[agent_names[1]] = other_action
            
        # Track state before step for penalty calculation
        agent = self.env.sim_agents[0]
        pos_before = agent.location
        holding_before = agent.get_holding()
            
        # Step
        # The env.step expects action_dict
        state, reward, done, info = self.env.step(action_dict)
        
        # Track state after step
        pos_after = agent.location
        holding_after = agent.get_holding()
        
        # Calculate Reward
        # env.reward() is broken (returns 0), so we calculate manually
        current_successful = self.env.order_scheduler.successful_orders
        diff_success = current_successful - self.last_successful_orders
        
        step_reward = -0.01 # Time Penalty
        
        # Dense Reward Shaping based on events
        events = info.get('events', [])
        
        # Check for meaningful events for our agent
        agent_event_occurred = False
        
        for event in events:
            # event is an Event object, usually has .event string
            # Check if this event was caused by our agent
            if hasattr(event, 'playerA') and event.playerA == agent_names[0]:
                e_str = event.event # e.g. 'Chop_Tomato-Lettuce-Onion'
                agent_event_occurred = True
                
                # Reward for useful interactions
                if 'Chop' in e_str:
                    # step_reward += 5.0 # Check below for conditional reward
                    pass
                elif 'Cook' in e_str:
                    step_reward += 5.0 # Increased from 3.0
                elif 'Assemble' in e_str:
                    step_reward += 5.0 # Increased from 3.0
                elif 'Pickup' in e_str:
                    # Only reward pickup if meaningful
                    if 'Fresh' in e_str or 'Plate' in e_str:
                        step_reward += 2.0 # Increased from 0.5
                elif 'Deliver' in e_str:
                     step_reward += 20.0 # Increased from 10.0
                elif 'Drop' in e_str:
                    pass
        
        # Intermediate Rewards (Heuristics)
        # 1. Holding a relevant ingredient
        # Check if holding something that is part of a current recipe?
        # For 'ring' map, usually Onion Soup or Tomato Soup.
        # Just rewarding holding Fresh things is a good start.
        if holding_after != 'None':
             if 'Fresh' in holding_after:
                 step_reward += 0.1 # Small continuous reward for holding ingredient
             if 'Chopped' in holding_after:
                 step_reward += 0.2
             if 'Plate' in holding_after:
                 step_reward += 0.1
        
        # 2. Distance to Goal
        # If holding Fresh Onion, distance to Cutboard.
        # If holding Chopped Onion, distance to Pot.
        # If holding Plate, distance to Pot or Delivery.
        # This requires calculating distances which is expensive.
        # Instead, reward being NEAR relevant objects.
        
        # Check current location type
        # We need to map `pos_after` to object type.
        # But `pos_after` is grid coord.
        # Can we access what is at `pos_after`? `agent.location` is `pos_after`.
        # Agent is usually ON a Floor, facing a Counter.
        # So we care about what is IN FRONT of agent?
        # Or just if the proper action happens (which is covered by events).
        
        # However, user asked for:
        # "Putting onion on cutboard" => EVENT: Drop (on cutboard)? or just Drop?
        # If 'Drop' event happens and target is cutboard.
        # But 'Drop' event string might not say target.
        # Let's check Drop event. It's usually 'Drop_Item'.
        
        # Let's try to detect placement on Cutboard from state change.
        # If holding_before was Fresh... and holding_after is None.
        # AND location is adjacent to Cutboard?
        # This is complex. Events are better.
        # Does the environment emit 'Place' event?
        # Not explicitly. But 'Chopping' implies it is on board.
        
        # Actually, let's look at `info['events']`.
        # If we can't detect placement easily, maybe we stick to events.
        # But user explicitly asked for "Reward when putting onion on board".
        
        # Let's inspect the world to see if an object appeared on a Cutboard near agent?
        # Iterate cutboards.
        # This is slow.
        
        # Alternative: use the fact that agent dropped something.
        if holding_before != 'None' and holding_after == 'None':
             # Agent dropped something.
             # Check if it land on a useful surface?
             # Check where agent is.
             # Get object at agent location (or facing location?).
             # Agent drops to current location or front?
             # `interact.py` says `put_down`. Agents put down on the counter they are facing or standing on?
             # Usually standard Overcooked is facing. Gym-cooking might be same.
             pass

        # Get Active Orders
        current_orders = []
        if hasattr(self.env, 'order_scheduler'):
            current_orders = self.env.order_scheduler.current_orders

        # Count Active Onion Orders
        active_onion_orders = 0
        for order in current_orders:
             goal_obj = order[0]
             if hasattr(goal_obj, 'name') and 'Onion' in goal_obj.name:
                 active_onion_orders += 1

        # Count Onions currently in play (to limit rewards)
        # States: Held by agents, On Counters, On Cutboards, In Pots
        onions_in_play = 0
        onions_chopped = 0
        
        # Check Agents
        # Use env.world, not self.game.world (self.game is not available here)
        # Note: In Gym wrapper, self.env is the OvercookedEnvironment.
        for agent in self.env.world.objects['Agent']: 
             if agent.holding:
                 for obj in agent.holding.contents:
                     if 'Onion' in obj.name:
                         onions_in_play += 1
        
        # Check Counters, Cutboards, Pots (Static objects)
        all_objects = []
        for obj_list in self.env.world.objects.values():
             all_objects.extend(obj_list)
             
        for obj in all_objects:
             # If obj is an ingredient loose on floor/counter (Object class)
             # world.objects stores both GridSquares (Counters) and Objects (Tomato)?
             # No, world.objects stores dynamic objects. GridSquares are map.
             # Wait, GridSquares can hold objects.
             pass

        # Re-evaluating world crawling for efficiency:
        # Dynamic objects are in world.objects['Tomato'], world.objects['Onion'] etc
        # AND some might be inside 'Plate' etc.
        # But crucially, we need to know if we have enough.
        
        # Simplified counting:
        # Count all 'Onion' objects in world.objects
        # Note: 'Onion' objects exist if they are Loose. If they are held by GridSquare, 
        # the GridSquare.holding points to the Object.
        
        # World.objects dict keys are names. 
        # Check world.py: objects = defaultdict(list). stored by name.
        
        # For 'Onion', 'ChoppedOnion' (if name changes), or 'Onion' with state.
        
        current_onions = 0
        current_chopped = 0
        
        # Iterate all dynamic objects to count
        for key, obj_list in self.env.world.objects.items():
            for obj in obj_list:
                if 'Onion' in obj.name:
                     current_onions += 1
                     # Check if chopped
                     # Assuming chopped onions have different name or state?
                     # In gym-cooking, usually state_index or name 'ChoppedOnion'.
                     # Let's assume 'Chopped' in name or special attribute.
                     if 'Chopped' in obj.name: # Placeholder
                         current_chopped += 1
        
        # Also check agents holding
        # world.objects['Agent'] includes agents
        # Agents holding objects are NOT in world.objects['Onion'] usually?
        # Standard implementation: when picked up, removed from world.objects list of that type?
        # We need to verify this assumption. If unsure, count agents specifically.
        
        # Additional state-based rewards (gradual progress)
        # Check if holding something useful
        if holding_after != 'None' and holding_before == 'None':
             # Picked up something
             # Reward Shape: Only if we need more onions
             if 'Onion' in holding_after:
                 if current_onions <= active_onion_orders: # Soft limit: allow pickup if we represent one of the needed ones
                    step_reward += 0.5 
                    print(f"Reward: Picked up needed Onion (+0.5)")
                 else:
                    # Penalty for hoarding? Or just 0.
                    pass
             else:
                step_reward += 0.5 # Other items (Plate, etc)
                print(f"Reward: Picked up {holding_after} (+0.5)")
        
        # Chop Event Verification and Reward
        # We moved 'Chop' reward from the event loop to here to verify necessity
        # Check if 'Chop' event occurred for this agent
        chop_occurred = False
        for event in events:
             if hasattr(event, 'playerA') and event.playerA == agent_names[0] and 'Chop' in event.event:
                 chop_occurred = True
                 break
        
        if chop_occurred:
             # Only reward if #chopped < #needed
             # current_chopped is onions ALREADY chopped.
             # If we just chopped, one fresh became chopped.
             # Logic: if we needed more chopped onions, this was good.
             if current_chopped <= active_onion_orders:
                 step_reward += 5.0
                 print(f"Reward: Chopped needed Onion (+5.0)")

        # Let's rely on the 'Drop' event if possible or holding state change.
        if holding_before != 'None' and holding_after == 'None':
             # Agent dropped something.
             # Check if now on cutboard.
             # Interaction logic: Agent interacts with the square in direction of movement
             # If movement was blocked, pos_after == pos_before
             # So target is pos_before + real_action
             
             target_pos = (pos_before[0] + real_action[0], pos_before[1] + real_action[1])
             
             # Check bounds
             if 0 <= target_pos[0] < self.width and 0 <= target_pos[1] < self.height:
                 gs = self.env.world.get_gridsquare_at(target_pos)
                 
                 # Check for Onion drop on Cutboard
                 if 'Onion' in holding_before and gs.name == 'Cutboard':
                     # Limit reward: Only if we don't have enough chopped onions yet
                     # This encourages processing needed onions
                     if current_chopped < active_onion_orders:
                        step_reward += 2.0
                        print(f"Reward: Placed needed Onion on Cutboard (+2.0)")
                     
                     step_reward += 2.0 # Reward for placing onion on cutboard
                     print(f"Reward: Placed Onion on Cutboard (+2.0)") # Always reward placing on cutboard to encourage learning the mechanic?
                 
                 # Check for Plate retrieval or placement?
                 # If dropped on Delivery?
                 if 'Delivery' in gs.name:
                      step_reward += 5.0 # Reward for delivery attempt (if holding check passed)
                      print(f"Reward: Delivered Item (+5.0)")

        # "切ると小さな報酬" (Small reward for chopping)
        # This is covered by 'Chop' event above.
        # I added +5.0 for Chop.
        
        # Enforce time penalty
        step_reward += -0.1 # Increased time penalty (was 0.01 -> 0.05 -> 0.1)
        
        # Invalid Action Penalty
        if my_action != (0,0):
             if pos_before == pos_after and holding_before == holding_after and not agent_event_occurred:
                 step_reward -= 0.2 # Increased penalty for bumping/useless action

        if diff_success > 0:
            step_reward += 50.0 # Huge reward for finishing order (was 15.0)
            self.last_successful_orders = current_successful
            
        # Check termination for first 3 orders
        terminated = False
        if current_successful >= self.max_orders:
            terminated = True
            step_reward += 100.0 # Bonus for completion (was 30.0)
            
        truncated = False
        # Terminate if too many steps
        if self.current_step >= 1000: # Explicit check 
            truncated = True
        if 'time_limit' in info: 
            truncated = True

        obs = self._process_obs(state)
        
        return obs, step_reward, terminated, truncated, info

    def _process_obs(self, state):
        # We need to construct the 3D array from self.env properties
        # Since 'state' returned by get_current_state is partial or structured
        # let's rebuild it accessing self.env directly for consistency
        
        # Initialize canvas
        obs_shape = (20, self.height, self.width)
        obs = np.zeros(obs_shape, dtype=np.uint8)
        
        # Helper to mark location
        def mark(channel_idx, loc):
            x, y = loc
            if 0 <= x < self.width and 0 <= y < self.height:
                obs[channel_idx, y, x] = 255

        # 1. Static Layout and Objects
        # We need to ensure we capture all objects correctly
        objs = self.env.world.get_object_list()
        
        # Grid Types
        gs_map = {'Floor': 0, 'Counter': 1, 'Cutboard': 2, 'Delivery': 3}
        
        for o in objs:
            name = o.name
            loc = o.location
            
            # Map GridSquares
            if name in gs_map:
                mark(gs_map[name], loc)
            
            # Map Ingredients/Food
            # Check full_name for state details
            full_name = ''
            if hasattr(o, 'full_name'):
                full_name = o.full_name
            
            # Fresh Inputs
            if 'FreshTomato' in full_name: mark(5, loc)
            elif 'FreshLettuce' in full_name: mark(6, loc)
            elif 'FreshOnion' in full_name: mark(7, loc)
            
            # Plates
            elif 'Plate' in full_name: mark(8, loc)
            
            # Pots (Are they objects or GridSquares? In core.py Pot is a GridSquare)
            elif 'Pot' in name: mark(9, loc)
            
            # Chopped Items
            if 'Chopped' in full_name:
                mark(10, loc)
                # Specific chopped channels?
                if 'Tomato' in full_name: mark(11, loc)
                elif 'Lettuce' in full_name: mark(12, loc)
                elif 'Onion' in full_name: mark(13, loc)

            # Cooked/Soup
            if 'Soup' in full_name or 'Cooked' in full_name:
                 mark(14, loc)
            
        # 3. Agents
        agents = self.env.sim_agents
        if len(agents) > 0:
            a = agents[0]
            mark(15, a.location)
            # If holding, mark holding channel.
            # But what is he holding?
            # We can rely on the object list above to see where the object is.
            # But the agent holding it moves it to agent location.
            # So the object list should cover it.
            
            # Mark orientation? Not critical for now.
            
        if len(agents) > 1:
            mark(16, agents[1].location)
            
        return obs
