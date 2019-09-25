"""
python main.py
"""

from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import gym

class Agent(object):
    def __init__(self, env, state_space, action_space, weights=[], max_eps_length=500, trials=5):
        self.max_eps_length = max_eps_length
        self.trials = trials
        state_space = state_space[0] # add batch dimension
        self.state_space = state_space
        self.action_space = action_space

        self.weights = weights if weights else self._get_random_weights()
        self.fitness = self._get_fitness(env)

    def model(self, x):
        x = F.relu(torch.add(torch.mm(x, self.weights[0]),self.weights[1]))
        x = F.relu(torch.add(torch.mm(x, self.weights[2]), self.weights[3]))
        x = F.softmax(torch.add(torch.mm(x, self.weights[4]), self.weights[5]))
        return x

    def _get_random_weights(self):
        return [
            torch.rand(self.state_space, 10), # fc1 weights
            torch.rand(10),  # fc1 bias
            torch.rand(10, 10),  # fc2 weights
            torch.rand(10),  # fc2 bias
            torch.rand(10, self.action_space),  # fc3 weights
            torch.rand(self.action_space),  # fc3 bias
        ]

    def _get_fitness(self, env):
        total_reward = 0
        for _ in range(self.trials):
            observation = env.reset()
            for i in range(self.max_eps_length):
                action = self.get_action(observation)
                observation, reward, done, info = env.step(action)
                total_reward += reward
                if done: break
        return total_reward / self.trials

    def get_action(self, state):
        act_prob = self.model(torch.Tensor(state.reshape(1,-1))).detach().numpy()[0] # use predict api when merged
        action = np.random.choice(range(len(act_prob)), p=act_prob)
        return action

    def save(self, save_file):
        self.mod.save_params(save_file)


def cross(agent1, agent2, agent_config):
    num_params = len(agent1.weights)
    crossover_idx = np.random.randint(0, num_params)
    new_weights = agent1.weights[:crossover_idx] + agent2.weights[crossover_idx:]
    new_weights = mutate(new_weights)
    return Agent(weights=new_weights, **agent_config)


def mutate(new_weights):
    num_params_to_update = np.random.randint(0, num_params)  # num of params to change
    for i in range(num_params_to_update):
        n = np.random.randint(0, num_params)
        new_weights[n] = new_weights[n] + torch.rand(new_weights[n].size())
    return new_weights



def breed(agent1, agent2, agent_config, generation_size=10):
    next_generation = [agent1, agent2]

    for _ in range(generation_size - 2):
        next_generation.append(cross(agent1, agent2, agent_config))

    return next_generation

def reproduce(agents, agent_config, generation_size):
    top_agents = sorted(agents, reverse=True, key=lambda a: a.fitness)[:2]
    new_agents = breed(top_agents[0], top_agents[1], agent_config, generation_size)
    return new_agents


def run(n_generations, generation_size, agent_config, save_file=None, render=False):
    agents = [Agent(**agent_config), Agent(**agent_config)]
    max_fitness = 0
    for i in range(n_generations):
        next_generation = reproduce(agents, agent_config, generation_size)
        ranked_generation = sorted(next_generation, reverse=True, key=lambda a : a.fitness)
        avg_fitness = (ranked_generation[0].fitness + ranked_generation[1].fitness) / 2
        print(i, avg_fitness)
        agents = next_generation
        if ranked_generation[0].fitness > max_fitness:
            max_fitness = ranked_generation[0].fitness
            # ranked_generation[0].save(args.save_file)
            test_agent(ranked_generation[0], agent_config, render)


def test_agent(agent, agent_config, render):
    env = agent_config['env']
    obs = env.reset()
    total_reward = 0
    for i in range(agent_config['max_eps_length']):
        if render: env.render()
        action = agent.get_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done: break
    print('test', total_reward)
    env.close()



if __name__ == '__main__':
    env_names = [e.id for e in gym.envs.registry.all()]

    parser = ArgumentParser()
    parser.add_argument('--n_generations', default=10000)
    parser.add_argument('--render',  action='store_true')
    parser.add_argument('--generation_size', default=20)
    parser.add_argument('--max_eps_length', default=500)
    parser.add_argument('--trials', default=5)
    parser.add_argument('--env', default='CartPole-v1', choices=env_names)
    parser.add_argument('--save_file')

    args = parser.parse_args()
    env = gym.make(args.env)

    agent_config = {
        'state_space' : env.observation_space.shape,
        'action_space' : env.action_space.n,
        'max_eps_length' : args.max_eps_length,
        'trials' : args.trials,
        'env': env,
    }

    run(args.n_generations, args.generation_size, agent_config, args.save_file, args.render)










