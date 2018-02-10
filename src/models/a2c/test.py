import time
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
from models.a2c.A2C import A2C
from doom.doom_trainer import DoomTrainer


def test(params, trainer, model):
    trainer = DoomTrainer(params)
    trainer.start_game()

    trainer.set_seed(params.seed)
    torch.manual_seed(params.seed)

    model.eval()

    trainer.new_episode()
    state = trainer.get_screen()

    reward_sum = 0
    done = True
    start_time = time.time()

    episode_length = 0
    actions = deque(maxlen=2100)

    while True:
        episode_length += 1
        if done:
            cx = Variable(torch.zeros(1, 512), volatile=True).cuda()
            hx = Variable(torch.zeros(1, 512), volatile=True).cuda()
        else:
            cx = Variable(cx.data, volatile=True).cuda()
            hx = Variable(hx.data, volatile=True).cuda()

        value, action_value, (hx, cx) = model((Variable(state.unsqueeze(0), volatile=True).cuda(), (hx, cx)))
        prob = F.softmax(action_value)
        action = prob.max(1)[1].cpu().data.numpy()

        reward, is_done = trainer.make_action(action[0])
        done = is_done or episode_length >= params.max_episode_length
        reward_sum += reward

        actions.append(action[0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            print("Time {}, episode reward {}, episode length {}".format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), reward_sum, episode_length))
            actions.clear()
            return

        state = trainer.get_screen()
