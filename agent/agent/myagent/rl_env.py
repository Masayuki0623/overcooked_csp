
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
        
        # Fixed recipe list for RL training
        # For new1_rl.txt level file order: FullSoup, TomatoLettuceSoup, OnionTomatoSoup
        # Recipe indices after load: 0=FullSoup, 1=TomatoLettuceSoup, 2=OnionTomatoSoup
        self.fixed_recipe_list = [0, 1, 2] * 34  # Repeat to fill 100+ entries

    def reset(self, seed=None, options=None):
        if seed is not None:
            # gymnasium uses seeding slightly differently
            # self.env.seed(seed) 
            pass
            
        self.env.reset()
        
        # Apply fixed recipe list after reset
        if hasattr(self.env, 'order_scheduler') and self.fixed_recipe_list:
            self.env.order_scheduler.assign_rand_recipe_list(self.fixed_recipe_list)
        
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
        
        step_reward = -0.02  # Time Penalty (small but consistent)
        
        # Get Active Orders first (needed for reward calculations)
        current_orders = []
        if hasattr(self.env, 'order_scheduler'):
            current_orders = self.env.order_scheduler.current_orders

        # Count Active Orders by Ingredient Type
        active_onion_orders = 0
        active_lettuce_orders = 0
        active_tomato_orders = 0
        for order in current_orders:
             goal_obj = order[0]
             if hasattr(goal_obj, 'full_name'):
                 if 'Onion' in goal_obj.full_name:
                     active_onion_orders += 1
                 if 'Lettuce' in goal_obj.full_name:
                     active_lettuce_orders += 1
                 if 'Tomato' in goal_obj.full_name:
                     active_tomato_orders += 1

        # Count ingredients currently in play
        # current_X = total count of ingredient X in the world
        # processing_X = count of ingredients being chopped OR already chopped (on cutboard or ready)
        current_onions = 0
        processing_onions = 0  # Chopping + Chopped
        current_lettuce = 0
        processing_lettuce = 0
        current_tomato = 0
        processing_tomato = 0
        cooking_or_cooked_exists = False  # For Plate pickup condition
        
        # Track chopped ingredients on counters (not on cutboard, not merged)
        # For determining valid merge/placement decisions
        chopped_onion_on_counter = 0
        chopped_lettuce_on_counter = 0
        chopped_tomato_on_counter = 0
        
        # Track merged ingredient sets on counters (e.g., ChoppedOnion-ChoppedTomato)
        merged_sets_on_counter = []  # List of sets like {'Onion', 'Tomato'}
        
        # Track if FullSoup prep exists (all 3 ingredients merged)
        full_soup_prep_exists = False
        
        for key, obj_list in self.env.world.objects.items():
            for obj in obj_list:
                obj_full_name = obj.full_name if hasattr(obj, 'full_name') else obj.name
                # Count ingredients
                if 'Onion' in obj_full_name:
                    current_onions += 1
                    if 'Chopped' in obj_full_name or 'Chopping' in obj_full_name:
                        processing_onions += 1
                if 'Lettuce' in obj_full_name:
                    current_lettuce += 1
                    if 'Chopped' in obj_full_name or 'Chopping' in obj_full_name:
                        processing_lettuce += 1
                if 'Tomato' in obj_full_name:
                    current_tomato += 1
                    if 'Chopped' in obj_full_name or 'Chopping' in obj_full_name:
                        processing_tomato += 1
                # Check for cooking/cooked/chopped (ready for plating)
                if 'Cooking' in obj_full_name or 'Cooked' in obj_full_name or 'Chopped' in obj_full_name:
                    cooking_or_cooked_exists = True
                
                # Track chopped ingredients on counters
                # Check if object is on a Counter (not Cutboard, not Pot, not being held)
                if hasattr(obj, 'location') and obj.location is not None:
                    gs = self.env.world.get_gridsquare_at(obj.location)
                    if gs and gs.name == 'Counter' and 'Chopped' in obj_full_name:
                        # Count ingredients in this object
                        has_onion = 'Onion' in obj_full_name
                        has_lettuce = 'Lettuce' in obj_full_name
                        has_tomato = 'Tomato' in obj_full_name
                        
                        ingredient_count = sum([has_onion, has_lettuce, has_tomato])
                        
                        if ingredient_count == 1:
                            # Single chopped ingredient
                            if has_onion:
                                chopped_onion_on_counter += 1
                            elif has_lettuce:
                                chopped_lettuce_on_counter += 1
                            elif has_tomato:
                                chopped_tomato_on_counter += 1
                        elif ingredient_count >= 2:
                            # Merged ingredients
                            merged_set = set()
                            if has_onion:
                                merged_set.add('Onion')
                            if has_lettuce:
                                merged_set.add('Lettuce')
                            if has_tomato:
                                merged_set.add('Tomato')
                            merged_sets_on_counter.append(merged_set)
                            
                            if ingredient_count == 3:
                                full_soup_prep_exists = True

        # Define valid merge combinations based on current orders
        # Orders: FullSoup (T+L+O), TomatoLettuceSoup (T+L), OnionTomatoSoup (O+T)
        valid_2_ingredient_merges = [
            {'Onion', 'Tomato'},    # For OnionTomatoSoup or FullSoup
            {'Tomato', 'Lettuce'},  # For TomatoLettuceSoup or FullSoup
            {'Onion', 'Lettuce'},   # Only for FullSoup (as intermediate)
        ]

        # Dense Reward Shaping based on events
        events = info.get('events', [])
        
        # Check for meaningful events for our agent
        agent_event_occurred = False
        
        for event in events:
            # event is an Event object, usually has .event string
            # Check if this event was caused by our agent
            if hasattr(event, 'playerA') and event.playerA == agent_names[0]:
                e_str = event.event
                agent_event_occurred = True
                
                # === EVENT-BASED REWARDS ===
                # Chop: handled separately with ingredient check
                if 'Chop' in e_str:
                    pass  # Processed below with ingredient check
                    
                # Cook: +5.0 for starting cooking
                elif 'Cook' in e_str:
                    step_reward += 5.0
                    print(f"Reward: Started Cooking (+5.0)")
                    
                # Assemble/Merge: reward for combining ingredients
                elif 'Assemble' in e_str or 'Merge' in e_str:
                    # Exclude meaningless assembles like "Assemble_Plate"
                    if 'Assemble_Plate' == e_str or e_str == 'Merge_Plate':
                        # No reward for just picking up/interacting with empty plate
                        pass
                    elif 'Chopped' in e_str:
                        # Check if this merge is valid for current orders
                        # Parse merged ingredients from event string
                        merged_ingredients = set()
                        if 'Onion' in e_str:
                            merged_ingredients.add('Onion')
                        if 'Lettuce' in e_str:
                            merged_ingredients.add('Lettuce')
                        if 'Tomato' in e_str:
                            merged_ingredients.add('Tomato')
                        
                        # Check if FullSoup prep already exists
                        if full_soup_prep_exists and len(merged_ingredients) == 2:
                            # Penalty: unnecessary 2-ingredient merge when FullSoup prep exists
                            step_reward -= 1.0
                            print(f"Penalty: Unnecessary merge when FullSoup prep exists (-1.0)")
                        elif merged_ingredients in valid_2_ingredient_merges or len(merged_ingredients) == 3:
                            # Valid merge for current orders
                            step_reward += 2.0
                            print(f"Reward: Valid merge {merged_ingredients} (+2.0)")
                        else:
                            # Invalid merge (not needed for any order)
                            step_reward -= 1.0
                            print(f"Penalty: Invalid merge {merged_ingredients} (-1.0)")
                    elif 'Cooked' in e_str or 'Soup' in e_str or 'Salad' in e_str:
                        # Reward for assembling cooked food onto plate
                        step_reward += 3.0
                        print(f"Reward: Assembled food onto plate (+3.0)")
                    # else: no reward for other assembles
                    
                # Deliver: +15.0 (additional to order completion bonus)
                elif 'Deliver' in e_str:
                    step_reward += 15.0
                    print(f"Reward: Delivered dish (+15.0)")
                    
                # Pickup and Drop are handled below with state tracking
        
        # === PICKUP REWARDS/PENALTIES ===
        # Detect pickup: holding_before == 'None' and holding_after != 'None'
        if holding_before == 'None' and holding_after != 'None':
            # Determine what was picked up
            if 'Onion' in holding_after and 'Fresh' in holding_after:
                # Fresh Onion pickup - always reward (fixed items, no infinite supply)
                step_reward += 0.5
                print(f"Reward: Picked up FreshOnion (+0.5)")
                    
            elif 'Lettuce' in holding_after and 'Fresh' in holding_after:
                # Fresh Lettuce pickup - always reward (fixed items, no infinite supply)
                step_reward += 0.5
                print(f"Reward: Picked up FreshLettuce (+0.5)")
                    
            elif 'Tomato' in holding_after and 'Fresh' in holding_after:
                # Fresh Tomato pickup - always reward (fixed items, no infinite supply)
                step_reward += 0.5
                print(f"Reward: Picked up FreshTomato (+0.5)")
                    
            elif 'Chopped' in holding_after:
                # Chopped ingredient pickup - always useful for assembly
                step_reward += 0.5
                print(f"Reward: Picked up Chopped ingredient (+0.5)")
                
            elif 'Plate' in holding_after:
                # Plate pickup - only reward if cooking/cooked/chopped exists
                if cooking_or_cooked_exists:
                    step_reward += 0.5
                    print(f"Reward: Picked up Plate (ready to serve) (+0.5)")
                else:
                    step_reward -= 0.5
                    print(f"Penalty: Picked up Plate (nothing ready) (-0.5)")

        # === DROP REWARDS/PENALTIES ===
        # Detect drop: holding_before != 'None' and holding_after == 'None'
        if holding_before != 'None' and holding_after == 'None':
            # Determine where the item was dropped
            target_pos = (pos_before[0] + real_action[0], pos_before[1] + real_action[1])
            
            if 0 <= target_pos[0] < self.width and 0 <= target_pos[1] < self.height:
                gs = self.env.world.get_gridsquare_at(target_pos)
                is_fresh = 'Fresh' in holding_before
                is_chopped = 'Chopped' in holding_before
                is_plate = 'Plate' in holding_before
                is_cutboard = gs.name == 'Cutboard'
                is_delivery = 'Delivery' in gs.name
                is_pot = 'Pot' in gs.name
                
                # Fresh ingredient placement
                if is_fresh:
                    if is_cutboard:
                        # Only reward if we need more of this ingredient processed
                        needed = False
                        if 'Onion' in holding_before and processing_onions < active_onion_orders:
                            needed = True
                        elif 'Lettuce' in holding_before and processing_lettuce < active_lettuce_orders:
                            needed = True
                        elif 'Tomato' in holding_before and processing_tomato < active_tomato_orders:
                            needed = True
                        
                        if needed:
                            step_reward += 0.5
                            print(f"Reward: Placed needed {holding_before} on Cutboard (+0.5)")
                        else:
                            # Penalty for placing unneeded ingredient on cutboard
                            step_reward -= 0.5
                            print(f"Penalty: Placed unneeded {holding_before} on Cutboard (-0.5)")
                    else:
                        # Penalty for placing fresh ingredient elsewhere
                        step_reward -= 0.5
                        print(f"Penalty: Placed {holding_before} on non-Cutboard (-0.5)")
                
                # Chopped ingredient placement
                elif is_chopped:
                    if is_pot:
                        # Reward for putting chopped ingredient in pot
                        step_reward += 1.0
                        print(f"Reward: Put {holding_before} in Pot (+1.0)")
                    elif is_cutboard:
                        # Penalty for placing chopped ingredient on cutboard
                        step_reward -= 0.5
                        print(f"Penalty: Placed {holding_before} on Cutboard (-0.5)")
                    else:
                        # Placing chopped ingredient on counter
                        # Check what's already on the target counter
                        target_obj = gs.holding if hasattr(gs, 'holding') else None
                        
                        if target_obj is not None:
                            # Merging with another object - handled by Merge event above
                            pass
                        else:
                            # Placing on empty counter
                            # Determine held ingredient type
                            held_has_onion = 'Onion' in holding_before
                            held_has_lettuce = 'Lettuce' in holding_before
                            held_has_tomato = 'Tomato' in holding_before
                            
                            # Count how many ingredients are in what we're holding
                            held_ingredient_count = sum([held_has_onion, held_has_lettuce, held_has_tomato])
                            
                            if held_ingredient_count == 1:
                                # Single chopped ingredient being placed
                                # Check if there's already same type on counter that could be merged with something
                                # Or if placing separately is beneficial
                                
                                # Count existing single chopped items on counters
                                total_single_chopped = chopped_onion_on_counter + chopped_lettuce_on_counter + chopped_tomato_on_counter
                                
                                # If no other single chopped items exist, placing is good (first step)
                                # If other single chopped items exist, placing separately might be wasteful
                                # unless we're setting up for multiple merges
                                
                                # For current orders (FullSoup, TomatoLettuceSoup, OnionTomatoSoup):
                                # We need 3 separate final dishes, so some separation is needed
                                
                                # Check if placing here would allow a valid merge later
                                can_merge_later = False
                                if held_has_onion:
                                    # Onion can merge with Tomato or Lettuce
                                    if chopped_tomato_on_counter > 0 or chopped_lettuce_on_counter > 0:
                                        # Could pick this up and merge, but placing separately
                                        # Penalty: should merge instead
                                        step_reward -= 0.5
                                        print(f"Penalty: Placed ChoppedOnion on empty counter when merge available (-0.5)")
                                    else:
                                        # No merge available, placing is OK
                                        step_reward += 0.5
                                        print(f"Reward: Placed ChoppedOnion on counter (+0.5)")
                                elif held_has_lettuce:
                                    if chopped_tomato_on_counter > 0 or chopped_onion_on_counter > 0:
                                        step_reward -= 0.5
                                        print(f"Penalty: Placed ChoppedLettuce on empty counter when merge available (-0.5)")
                                    else:
                                        step_reward += 0.5
                                        print(f"Reward: Placed ChoppedLettuce on counter (+0.5)")
                                elif held_has_tomato:
                                    if chopped_onion_on_counter > 0 or chopped_lettuce_on_counter > 0:
                                        step_reward -= 0.5
                                        print(f"Penalty: Placed ChoppedTomato on empty counter when merge available (-0.5)")
                                    else:
                                        step_reward += 0.5
                                        print(f"Reward: Placed ChoppedTomato on counter (+0.5)")
                            else:
                                # Merged ingredients being placed - generally OK
                                step_reward += 0.5
                                print(f"Reward: Placed merged ingredients on counter (+0.5)")
                
                # Plate placement (with or without food)
                elif is_plate:
                    if is_delivery:
                        # Delivery handled by event, but add small bonus
                        step_reward += 5.0
                        print(f"Reward: Placed dish on Delivery (+5.0)")
                    elif not cooking_or_cooked_exists:
                        # Reward for putting down unneeded plate (cancels pickup penalty)
                        step_reward += 0.5
                        print(f"Reward: Put down unneeded Plate (+0.5)")
                    # Other plate placements (when food is ready) are neutral

        # === CHOPPING REWARDS ===
        # Chop Event: reward when chopping completes for needed ingredient
        chop_occurred = False
        chop_event_str = ''
        for event in events:
            if hasattr(event, 'playerA') and event.playerA == agent_names[0] and 'Chop' in event.event:
                chop_occurred = True
                chop_event_str = event.event
                break
        
        if chop_occurred:
            # Give reward only if this ingredient is needed for current orders
            if 'Onion' in chop_event_str:
                if processing_onions <= active_onion_orders:
                    step_reward += 5.0
                    print(f"Reward: Chopped needed Onion (+5.0)")
            elif 'Lettuce' in chop_event_str:
                if processing_lettuce <= active_lettuce_orders:
                    step_reward += 5.0
                    print(f"Reward: Chopped needed Lettuce (+5.0)")
            elif 'Tomato' in chop_event_str:
                if processing_tomato <= active_tomato_orders:
                    step_reward += 5.0
                    print(f"Reward: Chopped needed Tomato (+5.0)")

        # Reward for actively chopping on cutboard (each step of chopping)
        if holding_after == 'None' and real_action != (0, 0):
            target_pos = (pos_after[0] + real_action[0], pos_after[1] + real_action[1])
            if 0 <= target_pos[0] < self.width and 0 <= target_pos[1] < self.height:
                gs = self.env.world.get_gridsquare_at(target_pos)
                if gs.name == 'Cutboard' and gs.holding is not None:
                    obj_on_cutboard = gs.holding
                    obj_name = obj_on_cutboard.full_name if hasattr(obj_on_cutboard, 'full_name') else ''
                    if 'Chopping' in obj_name:
                        # Give reward for each chopping step on needed ingredient
                        if 'Onion' in obj_name and processing_onions <= active_onion_orders:
                            step_reward += 0.5
                            print(f"Reward: Chopping Onion (+0.5)")
                        elif 'Lettuce' in obj_name and processing_lettuce <= active_lettuce_orders:
                            step_reward += 0.5
                            print(f"Reward: Chopping Lettuce (+0.5)")
                        elif 'Tomato' in obj_name and processing_tomato <= active_tomato_orders:
                            step_reward += 0.5
                            print(f"Reward: Chopping Tomato (+0.5)")

        # === PENALTIES ===
        # Invalid Action Penalty (bumping into walls/objects)
        if my_action != (0, 0):
            if pos_before == pos_after and holding_before == holding_after and not agent_event_occurred:
                step_reward -= 0.1
                print(f"Penalty: Invalid action (-0.1)")

        # === ORDER COMPLETION REWARDS ===
        if diff_success > 0:
            step_reward += 20.0  # Reward for completing an order
            self.last_successful_orders = current_successful
            print(f"Reward: Order completed! (+20.0)")
            
        # Check termination for first 3 orders
        terminated = False
        if current_successful >= self.max_orders:
            terminated = True
            step_reward += 50.0  # Bonus for completing all orders
            print(f"Reward: All {self.max_orders} orders completed! (+50.0)")
            
        truncated = False
        if self.current_step >= 1000:
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
