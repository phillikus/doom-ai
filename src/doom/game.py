import os
import torch
import torch.multiprocessing as _mp

from models.a2c.A2C import A2C
from models.a2c.test import test as test_a2c
from models.a2c.train import train as train_a2c

from models.a3c.A3C import A3C
from models.a3c import optimizers
from models.a3c.test import test as test_a3c
from models.a3c.train import train as train_a3c

from models.dqn.DQN import DQN
from models.dqn.NStepProgress import NStepProgress
from models.dqn.SoftmaxBody import SoftmaxBody
from models.dqn.AI import AI
from models.dqn.ReplayMemory import ReplayMemory
from models.dqn.train import train as train_dqn
from models.dqn.train import eligibility_trace

from doom.doom_trainer import DoomTrainer

mp = _mp.get_context('spawn')


def play(parameters):
    dtype = torch.cuda.FloatTensor
    torch.manual_seed(parameters.seed)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if parameters.model == 'human':
        play_human(parameters)
    elif parameters.model == 'a3c':
        play_a3c(parameters)
    elif parameters.model == 'a2c':
        play_a2c(parameters)
    elif parameters.model == 'dqn':
        play_dqn(parameters)


def play_human(params):
    trainer = DoomTrainer(params)
    trainer.start_game()
    trainer.play_human()


def play_a2c(params):
    trainer = DoomTrainer(params)
    trainer.start_game()
    model = A2C(1, params.num_actions).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)

    counter = 0
    while True:
        if counter % 10 == 0:
            print("Iteration: ", counter)

        train_a2c(params, trainer, model, optimizer)
        test_a2c(params, trainer, model)
        counter += 1


def play_a3c(params):
    trainer = DoomTrainer(params)
    os.environ['OMP_NUM_THREADS'] = '1'
    shared_model = A3C(1, trainer.num_actions()).cuda()
    shared_model.share_memory()

    optimizer = optimizers.SharedAdam(shared_model.parameters(), lr=params.lr)
    optimizer.share_memory()

    processes = []

    process = mp.Process(target=test_a3c, args=(params.num_processes, params, shared_model))
    process.start()

    for rank in range(0, params.num_processes):
        process = mp.Process(target=train_a3c, args=(rank, params, shared_model, optimizer))
        process.start()
        processes.append(process)

    for p in processes:
        p.join()


def play_dqn(params):
    trainer = DoomTrainer(params)
    trainer.start_game()

    model = DQN(trainer.num_actions())
    softmax_body = SoftmaxBody(T=1)
    ai = AI(brain=model, body=softmax_body)

    n_steps = NStepProgress(trainer, ai, n_step=10)
    memory = ReplayMemory(n_steps=n_steps, capacity=10000)
    train_dqn(model, memory, n_steps)