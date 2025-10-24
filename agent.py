import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import List, Tuple

class QNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        # TODO: setup network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement
        return x


class QTrainer:
    def __init__(self, model: QNetwork, lr: float, gamma: float) -> None:
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state: np.ndarray, action: np.ndarray, reward: float,
                   next_state: np.ndarray, done: bool) -> None:
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # TODO: Q learning: Q_new = reward + gamma * max(Q(s'))


class Agent:
    def __init__(self, input_size: int = 12, hidden_size: int = 256, output_size: int = 3,
                 lr: float = 0.001, gamma: float = 0.9, epsilon_decay: float = 80,
                 memory_size: int = 100_000, batch_size: int = 64) -> None:
        self.n_games = 0
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

        self.model = QNetwork(input_size, hidden_size, output_size)
        self.trainer = QTrainer(self.model, lr=lr, gamma=gamma)

    def get_action(self, state: np.ndarray) -> int:
        # Exploration vs exploitation (epsilon-greedy)
        epsilon = max(0.0, 1.0 - self.n_games / self.epsilon_decay)
        if random.random() < epsilon:
            return random.randint(0, 2)

        # Predict action
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            return torch.argmax(prediction).item()

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool) -> None:
        self.memory.append((state, action, reward, next_state, done))

    def train(self) -> None:
        """Train on a random batch from experience replay memory."""
        mini_sample = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(np.array(states), np.array(actions),
                               np.array(rewards), np.array(next_states),
                               np.array(dones))

    def save_model(self, filename: str = 'model.pth') -> None:
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename: str = 'model.pth') -> None:
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()
