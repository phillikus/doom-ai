# Test Agent

import torch
import torch.nn.functional as F
from a3c.A3C import A3C
from torch.autograd import Variable
import time
from doom_trainer import DoomTrainer


torch.set_default_tensor_type('torch.cuda.FloatTensor')


# Making the test agent (won't update the model but will just use the shared model to explore)
def test(rank, params, shared_model):
    torch.manual_seed(params.seed + rank) # asynchronizing the test agent

    trainer = DoomTrainer(params)
    trainer.set_seed(params.seed + rank)
    trainer.start_game()

    model = A3C(1, params.num_actions).cuda()  # shared_model is the model shared by the different agents (different threads in different cores)
    model.eval()

    trainer.new_episode() # getting the input images as numpy arrays
    state = trainer.get_screen()

    reward_sum = 0 # initializing the sum of rewards to 0
    done = True # initializing done to True
    start_time = time.time() # getting the starting time to measure the computation time

    episode_length = 0 # initializing the episode length to 0

    while True: # repeat
        episode_length += 1 # incrementing the episode length by one
        if done: # synchronizing with the shared model (same as train.py)
            model.load_state_dict(shared_model.state_dict())
            cx = Variable(torch.zeros(1, 256), volatile=True).cuda()
            hx = Variable(torch.zeros(1, 256), volatile=True).cuda()
        else:
            cx = Variable(cx.data, volatile=True).cuda()
            hx = Variable(hx.data, volatile=True).cuda()
        value, action_value, (hx, cx) = model((Variable(state.unsqueeze(0), volatile=True).cuda(), (hx, cx)))
        prob = F.softmax(action_value)
        action = prob.max(1)[1].cpu().data.numpy() # the test agent does not explore, it directly plays the best action

        reward, done = trainer.make_action(action[0])
        reward_sum += reward

        if done: # printing the results at the end of each part
            print("Time {}, episode reward {}, episode length {}".format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), reward_sum, episode_length))
            reward_sum = 0 # reinitializing the sum of rewards
            episode_length = 0 # reinitializing the episode length
            trainer.new_episode() # reinitializing the environment
            time.sleep(3) # doing a one minute break to let the other agents practice (if the game is done)
        state = trainer.get_screen()
