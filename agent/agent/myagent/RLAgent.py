
import os
import sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from .rl_env import KitchenGym

class RLAgent:
    """
    Reinforcement Learning Agent using Stable Baselines3.
    Loads a pre-trained model to make decisions.
    """
    def __init__(self, speed=1, replay=None, model_path=None):
        self.speed = speed
        self.replay = replay
        self.model = None
        
        # Default model path if not provided
        if model_path is None:
            # RLAgent is in agent/agent/myagent/
            # Models are generally in agent/agent/myagent/rl_models/
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "rl_models", "final_model.zip")
            
        if os.path.exists(model_path):
            print(f"[RLAgent] Loading model from {model_path}")
            try:
                self.model = PPO.load(model_path)
            except Exception as e:
                print(f"[RLAgent] Failed to load model: {e}")
        else:
            print(f"[RLAgent] Model not found at {model_path}. Agent will act randomly.")

    def __call__(self, env_state):
        """
        Policy inference.
        """
        # Process observation from EnvState
        obs_array = self._process_obs(env_state)
        
        if self.model:
            action, _states = self.model.predict(obs_array, deterministic=True)
            action = int(action)
        else:
            action = np.random.randint(0, 5) # Random if no model
            
        # Map action index to (dx, dy)
        action_map = {
            0: (0, 0),
            1: (-1, 0),
            2: (1, 0),
            3: (0, 1),
            4: (0, -1),
        }
        
        move = action_map.get(action, (0,0))
        return move, "RL Action"

    def _process_obs(self, env_state):
        height = env_state.world_height
        width = env_state.world_width
        
        # Match rl_env.py observation shape (20 channels to be safe)
        obs_shape = (20, height, width)
        obs = np.zeros(obs_shape, dtype=np.uint8)
        
        def mark(channel_idx, loc):
            x, y = loc
            if 0 <= x < width and 0 <= y < height:
                obs[channel_idx, y, x] = 255
                
        # 1. Static Layout
        gs_map = {'Floor': 0, 'Counter': 1, 'Cutboard': 2, 'Delivery': 3, 'PlateTile': 4}
        
        objs = env_state.world_all
        for o in objs:
            name = o.name
            if name in gs_map:
                mark(gs_map[name], o.location)
            
            # 2. Dynamic Objects (using 'in' for flexibility)
            if 'Tomato' in name and 'Fresh' in name: mark(5, o.location)
            elif 'Lettuce' in name and 'Fresh' in name: mark(6, o.location)
            elif 'Onion' in name and 'Fresh' in name: mark(7, o.location)
            elif 'Plate' in name: mark(8, o.location)
            elif 'Pot' in name: mark(9, o.location)
            
            if 'Chopped' in name:
                mark(10, o.location)
            
            if 'Soup' in name:
                 mark(13, o.location)
                 
        # 3. Agents
        agents = env_state.agents 
        if len(agents) > 0:
            # Self (Agent 0)
            a = agents[0]
            mark(15, a.location)
            if a.holding:
                mark(17, a.location)
        
        if len(agents) > 1:
            # Other agent
            mark(16, agents[1].location)
            
        return obs
