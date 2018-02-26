import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Initializing and setting the variance of a tensor of weights
def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, True).expand_as(out)) # thanks to this initialization, we have var(out) = std^2
    return out


# Initializing the weights of the neural network in an optimal way for the learning
def init_weights(m):
    class_name = m.__class__.__name__ # python trick that will look for the type of connection in the object "m" (convolution or full connection)
    if class_name.find('Conv') != -1: # if the connection is a convolution
        weight_shape = list(m.weight.data.size()) # list containing the shape of the weights in the object "m"
        fan_in = np.prod(weight_shape[1:4]) # dim1 * dim2 * dim3
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0] # dim0 * dim2 * dim3
        w_bound = np.sqrt(6. / (fan_in + fan_out)) # weight bound
        m.weight.data.uniform_(-w_bound, w_bound) # generating some random weights of order inversely proportional to the size of the tensor of weights
        m.bias.data.fill_(0) # initializing all the bias with zeros
    elif class_name.find('Linear') != -1: # if the connection is a full connection
        weight_shape = list(m.weight.data.size()) # list containing the shape of the weights in the object "m"
        fan_in = weight_shape[1] # dim1
        fan_out = weight_shape[0] # dim0
        w_bound = np.sqrt(6. / (fan_in + fan_out)) # weight bound
        m.weight.data.uniform_(-w_bound, w_bound) # generating some random weights of order inversely proportional to the size of the tensor of weights
        m.bias.data.fill_(0) # initializing all the bias with zeros


class A2C(torch.nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(A2C, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 64, 5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 2, stride=2, padding=1)

        self.lstm_size = 512
        self.lstm = nn.LSTMCell(10752, self.lstm_size)

        num_outputs = num_actions # getting the number of possible actions
        self.critic_linear = nn.Linear(self.lstm_size, 1) # full connection of the critic: output = V(S)
        self.actor_linear = nn.Linear(self.lstm_size, num_outputs) # full connection of the actor: output = Q(S,A)

        self.apply(init_weights) # initilizing the weights of the model with random weights

        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01) # setting the standard deviation of the actor tensor of weights to 0.01
        self.actor_linear.bias.data.fill_(0) # initializing the actor bias with zeros

        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1.0) # setting the standard deviation of the critic tensor of weights to 0.01
        self.critic_linear.bias.data.fill_(0) # initializing the critic bias with zeros

        self.lstm.bias_ih.data.fill_(0) # initializing the lstm bias with zeros
        self.lstm.bias_hh.data.fill_(0) # initializing the lstm bias with zeros

        self.train() # setting the module in "train" mode to activate the dropouts and batchnorms

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs.unsqueeze(0)))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(x.size(0), -1) # flattening the last convolutional layer into this 1D vector x
        hx, cx = self.lstm(x, (hx, cx)) # the LSTM takes as input x and the old hidden & cell states and ouputs the new hidden & cell states
        x = hx # getting the useful output, which are the hidden states (principle of the LSTM
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)