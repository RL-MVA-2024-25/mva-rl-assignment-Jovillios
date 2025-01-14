from click import pass_context
import torch
import torch.nn as nn
import random
import numpy as np
from gymnasium.wrappers import TimeLimit
# from env_hiv import HIVPatient
from sklearn.ensemble import RandomForestRegressor
import pickle
import numpy as np
from tqdm import tqdm


class ProjectAgent:
    def __init__(self):
        self.Qfunction = None
        self.gamma = 0.95
        self.iterations = 10
        self.nb_actions = 4
        self.epsilon = 0.5

    def collect_samples(self, env, horizon, disable_tqdm=False, print_done_states=False, prior=None):
        s, _ = env.reset()
        # dataset = []
        S = []
        A = []
        R = []
        S2 = []
        D = []
        for _ in tqdm(range(horizon), disable=disable_tqdm):
            if prior:
                a = self.act(s, use_random=True)
            else:
                a = env.action_space.sample()
            s2, r, done, trunc, _ = env.step(a)
            # dataset.append((s,a,r,s2,done,trunc))
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)
            if done or trunc:
                s, _ = env.reset()
                if done and print_done_states:
                    print("done!")
            else:
                s = s2
        S = np.array(S)
        A = np.array(A).reshape((-1, 1))
        R = np.array(R)
        S2 = np.array(S2)
        D = np.array(D)
        return S, A, R, S2, D

    def act(self, observation, use_random=False):
        Qsa = []
        if use_random and random.random() < self.epsilon:
            return random.randint(0, 3)
        for a in range(4):
            sa = np.append(observation, a).reshape(1, -1)
            Qsa.append(self.Qfunction[-1].predict(sa))
        return np.argmax(Qsa)

    def rf_fqi(self, S, A, R, S2, D, iterations, nb_actions, gamma, disable_tqdm=False):
        nb_samples = S.shape[0]
        Qfunctions = []
        SA = np.append(S, A, axis=1)
        for iter in tqdm(range(iterations), disable=disable_tqdm):
            if iter == 0:
                value = R.copy()
            else:
                Q2 = np.zeros((nb_samples, nb_actions))
                for a2 in range(nb_actions):
                    A2 = a2*np.ones((S.shape[0], 1))
                    S2A2 = np.append(S2, A2, axis=1)
                    Q2[:, a2] = Qfunctions[-1].predict(S2A2)
                max_Q2 = np.max(Q2, axis=1)
                value = R + gamma*(1-D)*max_Q2
            Q = RandomForestRegressor()
            Q.fit(SA, value)
            Qfunctions.append(Q)
        return Qfunctions[-1]

    def train(self, env, horizon, n_epochs=6):
        for i in range(n_epochs):
            print("Epoch ", i+1)
            if i == 0:
                prior = False
            else:
                prior = True
            S, A, R, S2, D = self.collect_samples(env, horizon, prior=prior)
            self.Qfunction = self.rf_fqi(
                S, A, R, S2, D, self.iterations, self.nb_actions, self.gamma)

    def save(self, path="tree.pkl"):
        """Save the trained model."""
        with open(path, 'wb') as f:
            pickle.dump(self.Qfunction, f)

    def load(self, path="tree.pkl"):
        """Load a trained model."""
        with open(path, 'rb') as f:
            self.Qfunction = pickle.load(f)


if __name__ == "__main__":
    from fast_env import FastHIVPatient

    env = TimeLimit(env=FastHIVPatient(domain_randomization=False),
                    max_episode_steps=200)

    # Initialize the environment
    agent = ProjectAgent()
    agent.train(env, 10000)
    agent.save()
