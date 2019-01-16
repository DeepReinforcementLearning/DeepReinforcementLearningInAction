import torch
from torch.autograd import Variable
import numpy as np
from torch.nn import functional as F
from matplotlib import pyplot as plt
import sys
sys.path.append("../Environments/")
from Gridworld import Gridworld
import random
from tqdm import tqdm
import math
from copy import deepcopy

from argparse import ArgumentParser

from simulator import SimulatorState, SimulatorReward
from buffer import ExperienceReplay

action_set = {
    0: 'u',
    1: 'd',
    2: 'l',
    3: 'r',
}


def running_mean(x, N=500):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def encode_game_progress(reward):
    if reward == 0:
        return 0  # in progress
    elif reward == 1:
        return 1  # won
    else:
        return 2  # lost


def decode_game_progress(index):
    if index == 0:
        return 0  # in progress
    elif index == 1:
        return 1  # won
    else:
        return -1  # lost


def convert_to_state(state):
    s_ = state.reshape(1, 4, 16)
    s = s_.max(dim=2)
    output = torch.zeroes(1, 4, 16)
    output[0][s[0][0]] = 1
    output[0][s[0][1]] = 1
    output[0][s[0][2]] = 1
    output[0][s[0][3]] = 1
    return output


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument('--epochs', default=5000, type=int)
    parser.add_argument('--mode', default='rand', choice=['rand'])
    parser.add_argument('--warm_up_period', default=1000, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--max_steps', default=15, type=int)
    parser.add_argument('--lr', default=0.0015, type=float)

    args = parser.parse_args()
    device = torch.device('cuda' if args.use_cuda else 'cpu')

    simulator_s = SimulatorState().to(device)
    simulator_r = SimulatorReward().to(device)
    opt_s = torch.optim.Adam(simulator_s.parameters(), lr=args.lr)
    opt_r = torch.optim.Adam(simulator_r.parameters(), lr=args.lr)

    loss_fn_state = torch.nn.CrossEntropyLoss()
    loss_fn_reward = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1, 50, 50]))
    losses = []
    buffer = ExperienceReplay()

    progress = tqdm(range(args.epochs))
    for epoch_num in progress:
        game = Gridworld(mode=args.mode)
        z = 0
        for step_num in args.max_steps:
            # get starting state
            state = torch.from_numpy(game.board.render_np()).float().reshape(64, )
            # take random action
            action_ = np.random.randint(0, 4)
            action = action_set[action_]
            action_vec = torch.zeros(4, )
            action_vec[action_] = 1

            game.makeMove(action)
            next_state = torch.from_numpy(game.board.render_np()).float()
            reward_ = encode_game_progress(game.reward())
            buffer.add([(state, action_vec, next_state[0].argmax(), reward_, next_state)])

            if len(buffer) >= args.warm_up_period:
                minibatch = buffer.sample(args.batch_size)
                opt_s.zero_grad()
                opt_r.zero_grad()
                states, actions, next_states_i, rewards, next_states = zip(*minibatch)
                states = torch.stack(states).to(device)
                actions = torch.stack(actions).to(device)
                pred_states, _ = simulator_s(torch.cat((states, actions), dim=1)).to(device)
                next_states_i = torch.stack(next_states_i).to(device)
                pred_rewards = simulator_r(torch.stack(next_states)).to(device)
                loss_state = loss_fn_state(pred_states, next_states_i)
                loss_reward = loss_fn_reward(pred_rewards, torch.Tensor(rewards).long())
                overall_loss = loss_state + loss_reward * 0.5
                overall_loss.backward()
                opt_r.step()
                opt_s.step()
                progress.set_description(
                    "{:10.3f} {:10.3f}".format(loss_state.detach().numpy(), loss_reward.detach().numpy()))
                losses.append([loss_state.detach().numpy(), loss_reward.detach().numpy()])
            if game.reward() in [1, -1]:
                break
    losses = np.array(losses)