
import sys
import os
import time
import pygame
from stable_baselines3.common.callbacks import BaseCallback

# Adjust paths
current_dir = os.getcwd()
# Ensure testbed-cooking is in python path
cooking_path = os.path.join(current_dir, 'testbed-cooking')
if cooking_path not in sys.path:
    sys.path.append(cooking_path)

# Ensure agent is in python path
agent_path = os.path.join(current_dir, 'agent')
if agent_path not in sys.path:
    sys.path.append(agent_path)

from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.play_test import MAP_SETTINGS
from gym_cooking.misc.game.game import Game
from agent.myagent.rl_env import KitchenGym

from stable_baselines3 import PPO

# Configuration
MAP_NAME = 'ring'
MAP_CONFIG = MAP_SETTINGS[MAP_NAME]
TRAIN_STEPS = 200000
SAVE_DIR = "agent/agent/myagent/rl_models"

class PygameRenderCallback(BaseCallback):
    def __init__(self, game_instance, fps=1000):
        super().__init__(verbose=0)
        self.game = game_instance
        self.fps = fps
        self.clock = pygame.time.Clock()
        self.ent_coef_start = 0.1
        self.ent_coef_end = 0.01

    def _on_step(self) -> bool:
        # Anneal ent_coef
        # progress_remaining is 1.0 at start, 0.0 at end
        # current_ent_coef = end + (start - end) * progress
        progress = self.locals.get('remaining_progress', 0) # SB3 logic might differ, let's check num_timesteps
        
        # SB3 callbacks have access to self.num_timesteps and self.total_timesteps (if passed in learn)
        # But we can use self.model._current_progress_remaining which is updated by PPO.learn
        
        current_progress = 1.0
        if hasattr(self.model, '_current_progress_remaining'):
            current_progress = self.model._current_progress_remaining
        
        current_ent_coef = self.ent_coef_end + (self.ent_coef_start - self.ent_coef_end) * current_progress
        self.model.ent_coef = current_ent_coef
        
        # Process Pygame events to keep window responsive and allow quitting
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        # Refresh Game State
        # OvercookedEnvironment.reset() creates NEW list objects for sim_agents and world object might change.
        # We must update game references manually.
        if hasattr(self.game.env, 'sim_agents'):
             self.game.sim_agents = self.game.env.sim_agents
             # Update current_agent reference if Game uses it (it's usually sim_agents[0])
             if hasattr(self.game, 'current_agent') and len(self.game.sim_agents) > 0:
                 self.game.current_agent = self.game.sim_agents[0]
                 
        if hasattr(self.game.env, 'world'):
             self.game.world = self.game.env.world
        if hasattr(self.game.env, 'order_scheduler'):
             self.game.order_scheduler = self.game.env.order_scheduler
        
        # Ensure recipes are refreshed if Game uses them?
        # Game doesn't seem to store recipes directly except via order_scheduler.

        
        # Call on_render. Based on grepping, it takes multiple arguments in some places.
        # But KitchenGym implementation of render is not present.
        # Game.on_render(self, paused=False, chat='', replay=False, debug_info=None)
        try:
             self.game.on_render() 
        except TypeError:
             self.game.on_render(paused=False, chat="", replay=False)
             
        pygame.display.flip()
        
        # Limit FPS
        if self.fps > 0:
            self.clock.tick(self.fps)
            
        return True

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    print(f"Initializing Environment: {MAP_NAME}")
    arglist = MapSetting(**MAP_CONFIG)
    arglist.max_num_timesteps = 1000 # Limit to 1000 frames
    arglist.max_num_orders = 3 # Reset after 3 orders
    
    # Init Pygame
    pygame.init()
    
    # Create Env
    raw_env = OvercookedEnvironment(arglist)
    env = KitchenGym(raw_env)
    
    # Create Game wrapper for rendering
    # Pass raw_env because Game expects OvercookedEnvironment
    # Game class signature: __init__(self, env, play=False)
    # Set play=True to force window creation in on_init
    game = Game(raw_env, play=True)
    game.on_init()
    
    print("Initializing PPO...")
    
    # Annealing schedule for entropy (exploration) is handled in PygameRenderCallback
    # Starts at 0.1 and decays to 0.01

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,  # Shorten rollout buffer for faster updates (Visualization purpose)
        batch_size=64,
        ent_coef=0.1, 
        tensorboard_log=f"{SAVE_DIR}/logs"
    )
    
    # Use a reasonable FPS cap
    # 0 is too fast and might make it flicker or skip visual updates if rendering logic lags behind logic
    # Set to 60 for smooth viewing
    callback = PygameRenderCallback(game, fps=60) 
    
    print("Starting training with visualization...")
    
    # We will reset env manually via PPO learning process.
    # However, if agent is completely stuck, maybe initial exploration is effectively zero?
    # No, ent_coef is 0.03.
    
    try:
        model.learn(total_timesteps=TRAIN_STEPS, callback=callback)
    except KeyboardInterrupt:
        print("Training interrupted.")
        
    final_path = os.path.join(SAVE_DIR, "visual_final_model")
    model.save(final_path)
    print(f"Model saved to {final_path}")
    
    pygame.quit()

if __name__ == "__main__":
    main()
