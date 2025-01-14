import sys

import torch
import torch.nn as nn
import random
import numpy as np
from gymnasium.wrappers import TimeLimit
# from env_hiv import HIVPatient
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import random
from tqdm import tqdm


def greedy_action(network, state, device):
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()

# Define the QNetwork


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, h_dim=256, n_layers=6):
        super(QNetwork, self).__init__()

        # Ensure at least one hidden layer
        assert n_layers >= 1, "The network must have at least one layer!"

        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, h_dim))
        layers.append(nn.ReLU())  # Activation function

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(h_dim, h_dim))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(h_dim, output_dim))

        # Combine all layers into a Sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity)  # capacity of the buffer
        self.data = []
        self.index = 0  # index of the next cell to be filled
        self.device = device

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size, last_element=10000):
        # sample on the last elements
        batch = random.sample(self.data[-last_element:], batch_size)
        return list(map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))

    def __len__(self):
        return len(self.data)


class ProjectAgent:
    def __init__(self):
        config = {'nb_actions': 4,
                  'learning_rate': 0.001,
                  'gamma': 0.95,
                  'buffer_size': 100_000,
                  'epsilon_min': 0.05,
                  'epsilon_max': 1.,
                  'epsilon_decay_period': 500_000,
                  'epsilon_delay_decay': 10_000,
                  'batch_size': 1024,
                  'gradient_steps': 1,
                  'update_target_strategy': 'ema',
                  'update_target_freq': 50,
                  'update_target_tau': 0.005,
                  'criterion': torch.nn.SmoothL1Loss()}
        self.config = config

        if torch.backends.mps.is_available():
            device = "mps"
            print("Using MPS")
        elif torch.cuda.is_available():
            device = "cuda"
            print("Using CUDA")
        else:
            device = "cpu"
            print("Using CPU")
        self.device = device
        model = QNetwork(6, 4).to(device)
        # if apply_cuda:
        #     model = model.cuda()
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys(
        ) else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(
            1e5)
        self.memory = ReplayBuffer(buffer_size, device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys(
        ) else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys(
        ) else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys(
        ) else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys(
        ) else 20
        self.epsilon_step = (
            self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys(
        ) else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys(
        ) else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys(
        ) else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys(
        ) else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys(
        ) else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys(
        ) else 0.005

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def fill_replay_buffer(self, env):
        """Fill the replay buffer with a random policy."""
        s, _ = env.reset()
        for _ in tqdm(range(self.memory.capacity)):
            a = env.action_space.sample()
            s2, r, done, trunc, _ = env.step(a)
            self.memory.append(s, a, r, s2, done)
            if done or trunc:
                s, _ = env.reset()
            else:
                s = s2

    def train(self, env, max_episode, run_name):
        writer = SummaryWriter(log_dir="runs/"+run_name)
        writer.add_text("config", str(self.config))
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        max_episode_return = 0  # to keep track of the best return
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state, self.device)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + \
                        (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:
                episode += 1
                print("Episode ", '{:3d}'.format(episode),
                      ", epsilon ", '{:6.2f}'.format(epsilon),
                      ", batch size ", '{:5d}'.format(len(self.memory)),
                      ", episode return ", '{:4.1f}'.format(
                          episode_cum_reward),
                      ", max episode return ", '{:4.1f}'.format(
                    max_episode_return),
                    sep='')
                writer.add_scalar("return", episode_cum_reward, episode)
                writer.add_scalar("epsilon", epsilon, episode)
                writer.add_scalar("step", step, episode)
                writer.add_scalar("max_return", max_episode_return, episode)
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                if episode_cum_reward > max_episode_return:
                    max_episode_return = episode_cum_reward
                    self.save(f"model_{episode}.pth")
                episode_cum_reward = 0
            else:
                state = next_state
        # last_mean_return = np.mean(episode_return[-100:])
        writer.close()
        return episode_return

    def act(self, observation, use_random=False):
        return greedy_action(self.model, observation, self.device)

    def save(self, path="model.pth"):
        """Save the trained model."""
        torch.save(self.model.state_dict(), path)

    def load(self, path="model.pth"):
        """Load a trained model."""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


if __name__ == "__main__":
    import gymnasium as gym
    from fast_env import FastHIVPatient

    env = TimeLimit(env=FastHIVPatient(domain_randomization=True),
                    max_episode_steps=200)
    # env = gym.make('CartPole-v1')
    # Initialize the environment
    # action/observation space
    agent = ProjectAgent()
    agent.fill_replay_buffer(env)
    # # run_name from arguments
    run_name = sys.argv[1]
    scores = agent.train(env, 10_000, run_name)
    # # Save the model
    agent.save()
