import torch
from utils.image_preprocessing import scale
from vizdoom.vizdoom import DoomGame, Mode


def create_actions(scenario):
    if scenario == 'basic':
        move_left = [1, 0, 0]
        move_right = [0, 1, 0]
        shoot = [0, 0, 1]

        return [move_left, move_right, shoot]

    if scenario == 'deadly_corridor':
        move_left =    [1, 0, 0, 0, 0, 0, 0]
        move_right =   [0, 1, 0, 0, 0, 0, 0]
        shoot =        [0, 0, 1, 0, 0, 0 ,0]
        back =         [0, 0, 0, 1, 0, 0 ,0]
        forward =      [0, 0, 0, 0, 1, 0, 0]
        turn_left =    [0, 0, 0, 0, 0, 1, 0]
        turn_right =   [0, 0, 0, 0, 0, 0, 1]

        return [move_left, move_right, shoot, back, forward, turn_left, turn_right]

    if scenario == 'my_way_home':
        turn_left =  [1, 0, 0, 0, 0]
        turn_right = [0, 1, 0, 0, 0]
        forward =    [0, 0, 1, 0, 0]
        move_left =  [0, 0, 0, 1, 0]
        move_right = [0, 0, 0, 0, 1]

        return [turn_left, turn_right, forward, move_left, move_right]

    if scenario == 'defend_the_center':
        turn_left = [1, 0, 0]
        turn_right = [0, 1, 0]
        shoot = [0, 0, 1]

        return [turn_left, turn_right, shoot]


class DoomTrainer:
    def __init__(self, params):
        self.game = DoomGame()
        self.game.load_config("../scenarios/" + params.scenario + ".cfg")

        if params.model == 'human':
            self.game.set_mode(Mode.SPECTATOR)
        else:
            self.actions = create_actions(params.scenario)

    def play_human(self):
        episodes = 10
        for i in range(episodes):

            self.game.new_episode()
            while not self.game.is_episode_finished():
                self.game.get_state()
                self.game.advance_action()
                self.game.get_last_action()
                self.game.get_last_reward()

    def start_game(self):
        self.game.init()

    def set_seed(self, seed):
        self.game.set_seed(seed)

    def new_episode(self):
        self.game.new_episode()

    def get_screen(self):
        return torch.from_numpy(scale(self.game.get_state().screen_buffer, None, None, True))

    def get_screen_buffer(self):
        return self.game.get_state().screen_buffer

    def make_action(self, action):
        reward = self.game.make_action(self.actions[action])
        done = self.game.is_episode_finished()

        return reward, done

    def num_actions(self):
        return len(self.actions)