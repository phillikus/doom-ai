from collections import deque

import torch
import torch.nn.functional as F
from models.a3c.A3C import A3C
from torch.autograd import Variable
import time
from doom.doom_trainer import DoomTrainer
from utils.logger import log_reward


torch.set_default_tensor_type('torch.cuda.FloatTensor')


def test(rank, params, shared_model):
    torch.manual_seed(params.seed + rank)

    trainer = DoomTrainer(params)
    trainer.set_seed(params.seed + rank)
    trainer.start_game()

    model = A3C(1, trainer.num_actions()).cuda()
    model.eval()

    trainer.new_episode()
    state = trainer.get_screen()

    reward_sum = 0
    done = True
    start_time = time.time()

    episode_length = 0
    actions = deque(maxlen=100)

    while True:
        episode_length += 1
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = Variable(torch.zeros(1, 256), volatile=True).cuda()
            hx = Variable(torch.zeros(1, 256), volatile=True).cuda()
        else:
            cx = Variable(cx.data, volatile=True).cuda()
            hx = Variable(hx.data, volatile=True).cuda()
        value, action_value, (hx, cx) = model((Variable(state.unsqueeze(0), volatile=True).cuda(), (hx, cx)))
        prob = F.softmax(action_value)
        action = prob.max(1)[1].cpu().data.numpy()

        reward, done = trainer.make_action(action[0])
        reward_sum += reward

        actions.append(action[0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            print("Time {}, episode reward {}, episode length {}".format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), reward_sum, episode_length))
            log_reward(reward_sum)
            reward_sum = 0
            episode_length = 0
            actions.clear()
            trainer.new_episode()
            time.sleep(15)
        state = trainer.get_screen()
