
import sys
import os
import time
import pygame
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

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
MAP_NAME = 'ring_rl'
MAP_CONFIG = MAP_SETTINGS[MAP_NAME]
TRAIN_STEPS = 200000
NUM_ENVS = 8  # Number of parallel environments
SAVE_DIR = "agent/agent/myagent/rl_models"

# Grid layout for 8 environments: 2 rows x 4 columns
GRID_COLS = 4
GRID_ROWS = 2

class MultiEnvRenderCallback(BaseCallback):
    """Callback to render all 8 environments in a grid layout."""
    
    def __init__(self, games, fps=0, render_every=10, vec_env=None):
        super().__init__(verbose=0)
        self.games = games  # List of Game instances, one per env
        self.fps = fps
        self.render_every = render_every
        self.vec_env = vec_env
        self.clock = pygame.time.Clock()
        self.ent_coef_start = 1.0   # 100% exploration at start
        self.ent_coef_end = 0.05    # 10% exploration at end
        self.step_count = 0
        self.total_steps = 0  # Total steps across all envs
        self.start_time = time.time()  # Track start time for ETA calculation
        self.last_eta_update = 0  # Last time ETA was updated
        self.cached_eta_str = "ETA: calculating..."  # Cached ETA string
        self.cached_elapsed_str = "Elapsed: 0m 0s"  # Cached elapsed string
        
        # Calculate window size based on single game size
        # Each game renders to its own surface
        self.single_width = games[0].screen.get_width() if games else 400
        self.single_height = games[0].screen.get_height() if games else 400
        
        # Reduce single height to show only game area (remove UI space)
        # Approximate game area height (world height * tile_size)
        if games:
            game_world_height = games[0].world.height * games[0].tile_size[1]
            self.single_height = game_world_height + 10  # Small margin
        
        self.total_width = self.single_width * GRID_COLS
        self.info_panel_height = 100  # Height for stats panel at bottom (increased)
        self.total_height = self.single_height * GRID_ROWS + self.info_panel_height
        
        # Create main display surface
        self.main_screen = pygame.display.set_mode((self.total_width, self.total_height))
        pygame.display.set_caption(f"Overcooked RL Training - {NUM_ENVS} Environments")
        
        # Fonts for info panel
        self.info_font = pygame.font.SysFont('Arial', 12)
        self.header_font = pygame.font.SysFont('Arial', 14, bold=True)

    def _on_step(self) -> bool:
        self.step_count += 1
        self.total_steps += NUM_ENVS  # Each step runs all envs
        
        # Anneal ent_coef
        current_progress = 1.0
        if hasattr(self.model, '_current_progress_remaining'):
            current_progress = self.model._current_progress_remaining
        
        current_ent_coef = self.ent_coef_end + (self.ent_coef_start - self.ent_coef_end) * current_progress
        self.model.ent_coef = current_ent_coef
        
        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        # Only render every N steps for speed
        if self.step_count % self.render_every != 0:
            return True
        
        # Clear screen
        self.main_screen.fill((30, 30, 30))
        
        # Collect stats from each environment
        env_stats = []
        
        # Sync and render each environment
        for i, game in enumerate(self.games):
            env_step = 0
            env_reward = 0.0
            env_penalties = 0
            
            if self.vec_env is not None:
                try:
                    # self.vec_env.envs[i] is KitchenGym wrapper
                    kitchen_gym = self.vec_env.envs[i]
                    actual_env = kitchen_gym.env  # Inner OvercookedEnvironment
                    
                    game.env.world = actual_env.world
                    game.env.sim_agents = actual_env.sim_agents
                    game.env.order_scheduler = actual_env.order_scheduler
                    game.env.t = actual_env.t  # Copy step counter for time display
                    
                    # Get stats from KitchenGym wrapper (not inner env)
                    env_step = kitchen_gym.current_step if hasattr(kitchen_gym, 'current_step') else 0
                    if hasattr(kitchen_gym, 'episode_stats'):
                        env_reward = kitchen_gym.episode_stats.get('total_reward', 0.0)
                        env_penalties = kitchen_gym.episode_stats.get('penalties', 0)
                    
                    # Update game references
                    game.world = actual_env.world
                    game.sim_agents = actual_env.sim_agents
                    game.order_scheduler = actual_env.order_scheduler
                    if len(game.sim_agents) > 0:
                        game.current_agent = game.sim_agents[0]
                except (AttributeError, IndexError):
                    pass
            
            env_stats.append({'step': env_step, 'reward': env_reward, 'penalties': env_penalties})
            
            # Render to game's own screen
            try:
                game.on_render()
            except TypeError:
                game.on_render(paused=False, chat="", replay=False)
            
            # Calculate position in grid
            col = i % GRID_COLS
            row = i // GRID_COLS
            x = col * self.single_width
            y = row * self.single_height
            
            # Blit only the game area (crop out UI area)
            game_area = pygame.Rect(0, 0, self.single_width, self.single_height - 10)
            self.main_screen.blit(game.screen, (x, y), game_area)
        
        # Draw info panel at bottom
        panel_y = self.single_height * GRID_ROWS
        pygame.draw.rect(self.main_screen, (50, 50, 50), 
                        (0, panel_y, self.total_width, self.info_panel_height))
        
        # Draw stats for each environment - layout matching 2x4 grid (4 cols, 2 rows)
        # Row 1: Env 0-3, Row 2: Env 4-7
        env_col_width = self.total_width // GRID_COLS  # 4 columns
        row_height = 35  # Height per row in info panel
        
        for i, stats in enumerate(env_stats):
            col = i % GRID_COLS  # 0-3
            row = i // GRID_COLS  # 0 or 1
            x = col * env_col_width + 5
            y = panel_y + row * row_height + 5
            
            # Env header and stats on same line
            header = self.info_font.render(f"Env{i}", True, (255, 255, 255))
            self.main_screen.blit(header, (x, y))
            
            # Step
            step_text = self.info_font.render(f"S:{stats['step']}", True, (200, 200, 200))
            self.main_screen.blit(step_text, (x + 40, y))
            
            # Reward
            reward_color = (100, 255, 100) if stats['reward'] >= 0 else (255, 100, 100)
            reward_text = self.info_font.render(f"R:{stats['reward']:.0f}", True, reward_color)
            self.main_screen.blit(reward_text, (x + 95, y))
            
            # Penalties
            penalty_text = self.info_font.render(f"P:{stats['penalties']}", True, (255, 150, 150))
            self.main_screen.blit(penalty_text, (x + 155, y))
        
        # Calculate progress and ETA (update every 1 second)
        progress_pct = (1.0 - current_progress) * 100  # current_progress goes from 1.0 to 0.0
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Update ETA only every 1 second
        if current_time - self.last_eta_update >= 1.0:
            self.last_eta_update = current_time
            
            if progress_pct > 0.1:  # Avoid division by zero / unstable early estimates
                total_time_estimate = elapsed_time / (progress_pct / 100)
                remaining_time = total_time_estimate - elapsed_time
                eta_mins = int(remaining_time // 60)
                eta_secs = int(remaining_time % 60)
                self.cached_eta_str = f"ETA: {eta_mins}m {eta_secs}s"
            else:
                self.cached_eta_str = "ETA: calculating..."
            
            # Format elapsed time
            elapsed_mins = int(elapsed_time // 60)
            elapsed_secs = int(elapsed_time % 60)
            self.cached_elapsed_str = f"Elapsed: {elapsed_mins}m {elapsed_secs}s"
        
        # Draw bottom row info
        bottom_y = panel_y + 2 * row_height + 5
        
        # Progress
        progress_text = self.header_font.render(f"Progress: {progress_pct:.1f}%", True, (100, 255, 100))
        self.main_screen.blit(progress_text, (10, bottom_y))
        
        # Explore rate
        explore_pct = current_ent_coef / 1.0 * 100  # ent_coef 1.0 = 100%
        explore_text = self.header_font.render(f"Explore: {explore_pct:.1f}%", True, (150, 200, 255))
        self.main_screen.blit(explore_text, (160, bottom_y))
        
        # Total steps
        total_text = self.header_font.render(f"Steps: {self.total_steps:,}", True, (255, 255, 0))
        self.main_screen.blit(total_text, (310, bottom_y))
        
        # Elapsed time (use cached value)
        elapsed_text = self.info_font.render(self.cached_elapsed_str, True, (200, 200, 200))
        self.main_screen.blit(elapsed_text, (450, bottom_y))
        
        # ETA (use cached value)
        eta_text = self.info_font.render(self.cached_eta_str, True, (255, 200, 100))
        self.main_screen.blit(eta_text, (580, bottom_y))
        
        pygame.display.flip()
        
        if self.fps > 0:
            self.clock.tick(self.fps)
            
        return True

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    print(f"Initializing Environment: {MAP_NAME} with {NUM_ENVS} parallel environments")
    
    # Init Pygame
    pygame.init()
    
    # Factory function to create environments
    def make_env():
        def _init():
            arglist = MapSetting(**MAP_CONFIG)
            arglist.max_num_timesteps = 1000
            arglist.max_num_orders = 3
            raw_env = OvercookedEnvironment(arglist)
            raw_env.reset()
            FIXED_RECIPE_LIST = [0, 1, 2] * 34
            raw_env.order_scheduler.assign_rand_recipe_list(FIXED_RECIPE_LIST)
            return KitchenGym(raw_env)
        return _init
    
    # Create vectorized environment with 8 parallel environments
    env = DummyVecEnv([make_env() for _ in range(NUM_ENVS)])
    
    # Create Game instances for each environment (for visualization)
    # Use play=False so each Game creates a Surface instead of display.set_mode
    games = []
    for i in range(NUM_ENVS):
        vis_arglist = MapSetting(**MAP_CONFIG)
        vis_arglist.max_num_timesteps = 1000
        vis_arglist.max_num_orders = 3
        vis_raw_env = OvercookedEnvironment(vis_arglist)
        vis_raw_env.reset()
        FIXED_RECIPE_LIST = [0, 1, 2] * 34
        vis_raw_env.order_scheduler.assign_rand_recipe_list(FIXED_RECIPE_LIST)
        
        # Create Game with play=False to use Surface instead of display
        game = Game(vis_raw_env, play=False)
        game.on_init()
        games.append(game)
    
    print("Initializing PPO...")
    
    # Annealing schedules: Start with high exploration, gradually decrease
    def linear_schedule(initial_value, final_value):
        """Linear interpolation between initial_value and final_value."""
        def func(progress_remaining):
            # progress_remaining goes from 1.0 to 0.0
            return final_value + progress_remaining * (initial_value - final_value)
        return func
    
    # Entropy: 1.0 (100% exploration) -> 0.05 (10% exploration)
    ent_schedule = linear_schedule(1.0, 0.05)
    # Learning rate: 3e-4 -> 3e-5
    lr_schedule = linear_schedule(3e-4, 3e-5)
    # Clip range: 0.3 -> 0.1
    clip_schedule = linear_schedule(0.3, 0.1)

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=lr_schedule,
        n_steps=1024,  # 1024 steps per env before update (1024 * 8 = 8192 total)
        batch_size=256,  # Larger batch for multi-env
        ent_coef=ent_schedule,
        clip_range=clip_schedule,
        tensorboard_log=f"{SAVE_DIR}/logs"
    )
    
    # Use multi-env render callback to show all 8 environments
    # render_every=1: Update display every step
    # fps=0: No FPS limit (maximum speed)
    callback = MultiEnvRenderCallback(games, fps=0, render_every=1, vec_env=env) 
    
    print(f"Starting training with {NUM_ENVS} parallel envs (2x4 grid), 1024 steps per update...")
    
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
