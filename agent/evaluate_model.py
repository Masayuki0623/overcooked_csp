"""
学習済みモデルを評価・実行するスクリプト
"""
import argparse
import pygame
import sys
import os

# パスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'testbed-cooking'))

from stable_baselines3 import PPO
from agent.myagent.rl_env import KitchenGym
from gym_cooking.envs.overcooked_environment import OvercookedEnvironment, MapSetting
from gym_cooking.misc.game.game import Game

# マップ設定
MAP_NAME = "new1_rl"
MAP_CONFIG = {
    "level": MAP_NAME,
    "num_agents": 1,
    "max_num_orders": 3,
    "max_num_timesteps": 1024,
}

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained RL model')
    parser.add_argument('--model', type=str, 
                        default='agent/agent/myagent/rl_models/visual_final_model.zip',
                        help='Path to the trained model')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frames per second for visualization')
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='Use deterministic actions (no exploration)')
    args = parser.parse_args()

    # pygame初期化
    pygame.init()
    
    # 環境を作成
    arglist = MapSetting(**MAP_CONFIG)
    raw_env = OvercookedEnvironment(arglist)
    raw_env.reset()
    
    # 固定レシピリスト
    FIXED_RECIPE_LIST = [0, 1, 2] * 34
    raw_env.order_scheduler.assign_rand_recipe_list(FIXED_RECIPE_LIST)
    
    # Gymラッパー
    env = KitchenGym(raw_env)
    
    # ゲーム画面を作成
    game = Game(raw_env, play=True)
    game.on_init()
    
    # モデルをロード
    print(f"Loading model from: {args.model}")
    model = PPO.load(args.model)
    print("Model loaded successfully!")
    print(f"\nRunning {args.episodes} episodes at {args.fps} FPS")
    print("Press ESC or close window to quit\n")
    
    clock = pygame.time.Clock()
    
    # 評価ループ
    for episode in range(args.episodes):
        print(f"{'='*50}")
        print(f"Episode {episode + 1}/{args.episodes}")
        print(f"{'='*50}")
        
        obs, info = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            # イベント処理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        return
            
            # モデルから行動を取得
            action, _ = model.predict(obs, deterministic=args.deterministic)
            
            # numpy配列から整数に変換
            if hasattr(action, 'item'):
                action = action.item()
            
            # 環境を1ステップ進める
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            
            # ゲーム画面を更新
            game.on_render()
            pygame.display.flip()
            
            # FPS制御
            clock.tick(args.fps)
        
        print(f"Episode {episode + 1} finished: {step} steps, reward: {total_reward:.1f}\n")
    
    pygame.quit()
    print("Evaluation complete!")

if __name__ == "__main__":
    main()
