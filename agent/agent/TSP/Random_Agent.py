import random
from agent.TSP.pathfinder import print_player_positions

class RandomAgent:
    def __init__(self, speed=2.5):
        self.speed = speed
        # 必要なら履歴や状態をここで初期化

    def act(self, observation):
        actions = ['up', 'down', 'left', 'right', 'stay']
        return random.choice(actions)

    def __call__(self, env):
        # envから必要な情報をobservationとして取得する場合はここで処理
        # ここではenvをそのままobservationとして扱う
        #print_player_positions(env)
        action = self.act(env)
        move_map = {
            'up': (0, 1),
            'down': (0, -1),
            'left': (-1, 0),
            'right': (1, 0),
            'stay': (0, 0)
        }
        move = move_map.get(action, (0, 0))
        chat = ""  # ランダムエージェントはチャットしない
        return move, chat