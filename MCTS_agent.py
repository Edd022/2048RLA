import numpy as np
import copy
from random import choice

class MCTSAgent:
    def __init__(self, action_space, simulations=50, depth=10):
        self.action_space = action_space
        self.simulations = simulations
        self.depth = depth
        self.action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

    def select_action(self, env_game):
        scores = np.zeros(len(self.action_space))
        counts = np.zeros(len(self.action_space))

        for _ in range(self.simulations):
            game_copy = copy.deepcopy(env_game)
            first_action = None

            for d in range(self.depth):
                valid_moves = [a for a in self.action_space if game_copy.can_move_in_direction(self.action_map[a])]
                if not valid_moves:
                    break
                move = choice(valid_moves)
                if d == 0:
                    first_action = move
                game_copy.move(self.action_map[move])

                if not game_copy.can_move():
                    break

            if first_action is not None:
                scores[first_action] += game_copy.get_score()
                counts[first_action] += 1

        avg_scores = scores / (counts + 1e-5)
        return int(np.argmax(avg_scores))
