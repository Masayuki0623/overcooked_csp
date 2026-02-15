
import sys
import os
from pathlib import Path

# Add project root to sys.path to ensure modules are found
current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, 'testbed-cooking'))
sys.path.append(os.path.join(current_dir, 'agent'))

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.play_test import MAP_SETTINGS
from agent.myagent.rl_env import KitchenGym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

# Configuration
# Focusing on 'ring' map as per context
MAP_NAME = 'ring'
MAP_CONFIG = MAP_SETTINGS[MAP_NAME]
TRAIN_STEPS = 200000
SAVE_DIR = "agent/agent/myagent/rl_models" # Adjusted path based on structure

def main():
    # Ensure save directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Setup environment
    print(f"Initializing Environment: {MAP_NAME}")
    arglist = MapSetting(**MAP_CONFIG)
    # Important: Enable training specific settings
    arglist.max_num_timesteps = 500 # Limit episode length
    arglist.max_num_orders = 3      # Goal: 3 orders
    
    # Gym Cooking Env
    # Note: Using single environment for now. Vectorized env requires pickle compatibility.
    raw_env = OvercookedEnvironment(arglist)
    env = KitchenGym(raw_env)
    
    # Init PPO
    # Using MlpPolicy because default CnnPolicy fails on small grids (8x8)
    print("Initializing PPO (MlpPolicy)...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=f"{SAVE_DIR}/logs",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.03 # Increased from 0.01 to encourage exploration
    )
    
    # Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=20000, 
        save_path=SAVE_DIR, 
        name_prefix='rl_model'
    )
    
    print(f"Start training for {TRAIN_STEPS} steps...")
    model.learn(total_timesteps=TRAIN_STEPS, callback=checkpoint_callback)
    
    final_path = os.path.join(SAVE_DIR, "final_model")
    model.save(final_path)
    print(f"Training finished. Model saved to {final_path}")

if __name__ == "__main__":
    main()
