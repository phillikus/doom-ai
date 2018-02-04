import os
import torch
import torch.multiprocessing as _mp

from a2c.A2C import A2C
from a2c.test import test as test_a2c
from a2c.train import train as train_a2c

from a3c.A3C import A3C
from a3c import optimizers
from a3c.test import test as test_a3c
from a3c.train import train as train_a3c
from doom_trainer import DoomTrainer

mp = _mp.get_context('spawn')


class Params:
    def __init__(self):
        self.lr = 0.0001
        self.gamma = 0.99
        self.tau = 1
        self.seed = 1
        self.num_processes = 4
        self.num_steps = 20
        self.max_episode_length = 10000
        self.scenario = 'deadly_corridor'
        self.num_actions = 7
        self.model = 'a2c'
        self.num_updates = 100


def play(parameters=None):
    dtype = torch.cuda.FloatTensor
    parameters = Params() if parameters is None else parameters
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.manual_seed(parameters.seed)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if parameters.model == 'a3c':
        play_a3c(parameters)
    elif parameters.model == 'a2c':
        play_a2c(parameters)


def play_a2c(params):
    trainer = DoomTrainer(params)
    trainer.start_game()
    model = A2C(1, params.num_actions).cuda()

    counter = 0
    while True:
        if counter % 10 == 0:
            print("Iteration: ", counter)

        train_a2c(params, trainer, model)
        test_a2c(params, trainer, model)
        counter += 1


def play_a3c(params):
    shared_model = A3C(1, params.num_actions).cuda()  # shared_model is the model shared by the different agents (different threads in different cores)
    shared_model.share_memory()

    optimizer = optimizers.SharedAdam(shared_model.parameters(), lr=params.lr)  # the optimizer is also shared because it acts on the shared model
    optimizer.share_memory()

    processes = []

    process = mp.Process(target=test_a3c, args=(params.num_processes, params, shared_model))
    process.start()

    for rank in range(0, params.num_processes):  # making a loop to run all the other processes that will be trained by updating the shared model
        process = mp.Process(target=train_a3c, args=(rank, params, shared_model, optimizer))
        process.start()
        processes.append(process)

    for p in processes:  # creating a pointer that will allow to kill all the threads when at least one of the threads, or main.py will be killed, allowing to stop the program safely
        p.join()