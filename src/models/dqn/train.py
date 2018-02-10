import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from vizdoom import *
from models.dqn.NStepProgress import NStepProgress
from models.dqn.ReplayMemory import ReplayMemory


def eligibility_trace(batch, dqn):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state.numpy(), series[-1].state.numpy()], dtype=np.float32)).unsqueeze(1))
        output = dqn(input.cuda())
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()

        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward

        state = series[0].state.numpy()
        target = output[0].data
        target[series[0].action] = cumul_reward
        inputs.append(state)
        targets.append(target)

    return torch.from_numpy(np.array(inputs, dtype=np.float32)), torch.stack(targets)


def train(dqn, memory, n_steps):
    ma = MA(100)
    loss = nn.MSELoss()
    optimizer = optim.Adam(dqn.parameters(), lr=0.01)
    nb_epochs = 1000

    for epoch in range(1, nb_epochs + 1):
        memory.run_steps(200)
        for batch in memory.sample_batch(128):
            inputs, targets = eligibility_trace(batch, dqn)
            inputs, targets = Variable(inputs.unsqueeze(1)), Variable(targets.unsqueeze(1))
            predictions = dqn(inputs.cuda())
            loss_error = loss(predictions, targets)
            optimizer.zero_grad()
            loss_error.backward()
            optimizer.step()

        rewards_steps = n_steps.rewards_steps()
        ma.add(rewards_steps)
        avg_reward = ma.average()
        max_reward = max(ma.list_of_rewards) if len(ma.list_of_rewards) > 0 else 0

        print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward)))

        if avg_reward >= 1500 or max_reward >= 1500:
            print("Congratulations, your AI wins")


class MA:
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size

    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]

    def average(self):
        return np.mean(self.list_of_rewards)