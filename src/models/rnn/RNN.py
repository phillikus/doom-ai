import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class RNN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(RNN, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels=num_inputs, out_channels=32, kernel_size=5)
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.convolution4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2)
        self.fc1 = nn.Linear(in_features=self.count_neurons((1, 80, 80)), out_features=40)
        self.fc2 = nn.Linear(in_features=40, out_features=40)
        self.fc3 = nn.Linear(in_features=40, out_features=num_actions)