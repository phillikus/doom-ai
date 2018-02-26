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

    # speed = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # strafe = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # move_right = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # move_left = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # move_backward = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # move_forward = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # turn_right = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # turn_left = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # w1 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # w2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # w3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # w4 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    # w5 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # w6 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    # w_next = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    # w_prev = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    # look_up_down_delta = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    # turn_left_right_delta = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    # move_left_right_delta = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    # attack = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    #
    # return [speed, strafe, move_right, move_left, move_backward, move_forward, turn_right, turn_left, w1, w2, w3, w4,
    #            w5, w6, w_next, w_prev, look_up_down_delta, turn_left_right_delta, move_left_right_delta, attack]


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
            print("Episode #" + str(i + 1))

            self.game.new_episode()
            while not self.game.is_episode_finished():
                s = self.game.get_state()
                self.game.advance_action()
                a = self.game.get_last_action()
                r = self.game.get_last_reward()


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

    def num_actions(self):
        return len(self.actions)