import torch
import torch.nn as nn
import random
import numpy as np
from gymnasium.wrappers import TimeLimit
# from env_hiv import HIVPatient


def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()

# Define the replay buffer


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

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))

    def __len__(self):
        return len(self.data)


# Define the QNetwork
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the agent


class ProjectAgent:
    def __init__(self):
        config = {'nb_actions': 4,
                  'state_space': 6,
                  'learning_rate': 0.001,
                  'gamma': 0.95,
                  'buffer_size': 1000000,
                  'epsilon_min': 0.01,
                  'epsilon_max': 1.,
                  'epsilon_decay_period': 100000,
                  'epsilon_delay_decay': 20,
                  'batch_size': 20}

        self.model = QNetwork(config['state_space'], config['nb_actions'])
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.memory = ReplayBuffer(config['buffer_size'], device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (
            self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config['learning_rate'])
        self.max_episode_step = 200

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.model(Y).max(1)[0].detach()
            # update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def act(self, observation, use_random=False):
        return greedy_action(self.model, observation)

    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
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
                action = greedy_action(self.model, state)

            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            self.gradient_step()

            # next transition
            step += 1
            if done or trunc or step >= self.max_episode_step:
                episode += 1
                if episode % (max_episode//100) == 0:
                    print("Episode ", '{:3d}'.format(episode),
                          ", epsilon ", '{:6.2f}'.format(epsilon),
                          ", batch size ", '{:5d}'.format(len(self.memory)),
                          ", episode return ", '{:4.1f}'.format(
                        episode_cum_reward),
                        sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state

        return episode_return

    def save(self, path="model.pth"):
        """Save the trained model."""
        torch.save(self.model.state_dict(), path)

    def load(self, path="model.pth"):
        """Load a trained model."""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


if __name__ == "__main__":
    from fast_env import FastHIVPatient

    env = TimeLimit(env=FastHIVPatient(domain_randomization=False),
                    max_episode_steps=200)

    # Initialize the environment
    agent = ProjectAgent()
    scores = agent.train(env, 1000000)
    # Save the model
    agent.save()
