# Training the AI

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from models.a3c.A3C import A3C
from doom.doom_trainer import DoomTrainer


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, params, shared_model, optimizer):
    torch.manual_seed(params.seed + rank)

    trainer = DoomTrainer(params)
    trainer.set_seed(params.seed + rank)
    trainer.start_game()

    model = A3C(1, trainer.num_actions()).cuda()

    trainer.new_episode()
    state = trainer.get_screen()

    done = True
    episode_length = 0

    while True:
        episode_length += 1
        model.load_state_dict(shared_model.state_dict())

        if done:
            cx = Variable(torch.zeros(1, 256)).cuda()
            hx = Variable(torch.zeros(1, 256)).cuda()
        else:
            cx = Variable(cx.data).cuda()
            hx = Variable(hx.data).cuda()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(params.num_steps):
            value, action_values, (hx, cx) = model((Variable(state.unsqueeze(0)).cuda(), (hx, cx)))
            prob = F.softmax(action_values)
            log_prob = F.log_softmax(action_values)

            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))

            values.append(value)
            log_probs.append(log_prob)

            reward, is_done = trainer.make_action(action[0][0])
            done = is_done or episode_length >= params.max_episode_length
            reward = max(min(reward, 1), -1)

            if done:
                episode_length = 0
                trainer.new_episode()

            state = trainer.get_screen()
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((Variable(state.unsqueeze(0)).cuda(), (hx, cx)))
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0

        R = Variable(R)
        gae = torch.zeros(1, 1)

        for i in reversed(range(len(rewards))):
            R = params.gamma * R + rewards[i]
            advantage = R - values[i]

            value_loss = value_loss + 0.5 * advantage.pow(2)
            TD = rewards[i] + params.gamma * values[i + 1].data - values[i].data
            gae = gae * params.gamma * params.tau + TD
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i]

        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        ensure_shared_grads(model, shared_model)

        optimizer.step()
