import torch
import torch.nn.functional as F
from torch.autograd import Variable
from models.a2c.A2C import A2C
from doom.doom_trainer import DoomTrainer

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def train(params, trainer, model, optimizer=None):
    trainer.set_seed(params.seed)
    torch.manual_seed(params.seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr) if optimizer is None else optimizer
    model.train()

    trainer.new_episode()
    state = trainer.get_screen()
    done = True

    episode_length = 0 # initializing the length of an episode to 0
    updates = 0

    while updates < params.num_updates:
        episode_length += 1

        if done:
            cx = Variable(torch.zeros(1, 512)).cuda()
            hx = Variable(torch.zeros(1, 512)).cuda()
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

            reward, is_done = trainer.make_action(action[0][0])
            done = is_done or episode_length >= params.max_episode_length
            reward = max(min(reward, 1), -1)

            if done:
                episode_length = 0
                trainer.new_episode()

            values.append(value)
            log_probs.append(log_prob)
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
            td = rewards[i] + params.gamma * values[i + 1].data - values[i].data
            gae = gae * params.gamma * params.tau + td
            policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i]

        optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss).backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), 40)

        optimizer.step()
        updates += 1
