import numpy as np
from collections import namedtuple, deque

Step = namedtuple('Step', ['state', 'action', 'reward', 'done'])


class NStepProgress:
    def __init__(self, trainer, ai, n_step):
        self.ai = ai
        self.rewards = []
        self.trainer = trainer
        self.n_step = n_step

    def __iter__(self):
        self.trainer.new_episode()
        state = self.trainer.get_screen()
        history = deque()
        reward = 0.0
        while True:
            action = self.ai(np.array(state.unsqueeze(0)))[0][0]
            r, is_done = self.trainer.make_action(action)

            reward += r
            history.append(Step(state=state, action=action, reward=r, done=is_done))

            while len(history) > self.n_step + 1:
                history.popleft()
            if len(history) == self.n_step + 1:
                yield tuple(history)

            if not is_done:
                state = self.trainer.get_screen()
            if is_done:
                if len(history) > self.n_step + 1:
                    history.popleft()
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()

                self.rewards.append(reward)
                reward = 0.0
                history.clear()
                self.trainer.new_episode()
                state = self.trainer.get_screen()

    def rewards_steps(self):
        rewards_steps = self.rewards
        self.rewards = []
        return rewards_steps
