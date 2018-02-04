import torch
from image_preprocessing import scale
from vizdoom.vizdoom import DoomGame


def create_actions():
    move_left =    [1, 0, 0, 0, 0, 0, 0]
    move_right =   [0, 1, 0, 0, 0, 0, 0]
    shoot =        [0, 0, 1, 0, 0, 0 ,0]
    back =         [0, 0, 0, 1, 0, 0 ,0]
    forward =      [0, 0, 0, 0, 1, 0, 0]
    turn_left =    [0, 0, 0, 0, 0, 1, 0]
    turn_right =   [0, 0, 0, 0, 0, 0, 1]

    return [move_left, move_right, shoot, back, forward, turn_left, turn_right]


class DoomTrainer:
    def __init__(self, params):
        self.game = DoomGame()
        self.game.load_config("../scenarios/" + params.scenario + ".cfg")
        self.actions = create_actions()

    def start_game(self):
        self.game.init()

    def set_seed(self, seed):
        self.game.set_seed(seed)

    def new_episode(self):
        self.game.new_episode()

    def get_screen(self):
        return torch.from_numpy(scale(self.game.get_state().screen_buffer, None, None, True))

    def make_action(self, action):
        reward = self.game.make_action(self.actions[action])
        done = self.game.is_episode_finished()

        return reward, done