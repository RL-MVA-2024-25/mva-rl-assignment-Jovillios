import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from fast_env import FastHIVPatient

# Initialize the environment
env = TimeLimit(env=FastHIVPatient(domain_randomization=False),
                max_episode_steps=200)

# Define the Q-Network using PyTorch


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
    def __init__(self, state_size=6, action_size=4):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Replay buffer
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.learning_rate = 0.001
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def act(self, observation, use_random=False):
        """Choose an action based on the current observation."""
        if use_random or np.random.rand() <= self.epsilon:
            return random.randint(0, self.action_size - 1)

        observation = torch.FloatTensor(observation).unsqueeze(
            0).to(self.device)  # Add batch dimension
        with torch.no_grad():
            q_values = self.model(observation)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        """Train the model using experiences from the replay buffer."""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            done = torch.FloatTensor([done]).to(self.device)

            # Compute the target Q-value
            with torch.no_grad():
                target = reward
                if not done:
                    target += self.gamma * torch.max(self.model(next_state))

            # Compute the predicted Q-value
            q_values = self.model(state)
            target_f = q_values.clone()
            target_f[action] = target

            # Update the model
            self.optimizer.zero_grad()
            loss = self.criterion(q_values, target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path="model.pth"):
        """Save the trained model."""
        torch.save(self.model.state_dict(), path)

    def load(self, path="model.pth"):
        """Load a trained model."""
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


# Training the agent
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = ProjectAgent(state_size, action_size)
episodes = 500  # Number of episodes to train

for episode in range(episodes):
    state = env.reset()[0]  # Reset environment, extract initial state
    total_reward = 0
    for time in range(200):  # Max steps per episode
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            print(
                f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")
            break

    agent.replay()  # Train using replay buffer

# Save the trained model
agent.save("model.pth")
