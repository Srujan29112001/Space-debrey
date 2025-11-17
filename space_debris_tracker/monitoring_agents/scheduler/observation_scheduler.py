"""
RL-based Observation Scheduler
Prioritizes observations based on collision risk and resource constraints
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


class DQN(nn.Module):
    """Deep Q-Network for observation scheduling"""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class ObservationScheduler:
    """
    RL-based observation scheduler
    Learns to prioritize observations for maximum collision detection
    """

    def __init__(
        self,
        satellite_id: int,
        state_dim: int = 20,
        action_dim: int = 10,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        """
        Initialize scheduler

        Args:
            satellite_id: Satellite NORAD ID
            state_dim: State dimension
            action_dim: Action dimension (observation priorities)
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate
        """
        self.satellite_id = satellite_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-network
        self.qnetwork = DQN(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.qnetwork.parameters(), lr=learning_rate)

        # Experience replay buffer
        self.experience_buffer = []
        self.buffer_size = 10000

    def get_action(self, state: np.ndarray) -> Dict:
        """
        Get observation action

        Args:
            state: Current state

        Returns:
            Observation plan
        """
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            # Random action (exploration)
            action = np.random.randint(self.action_dim)
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0)
                q_values = self.qnetwork(state_tensor)
                action = q_values.argmax().item()

        # Convert action to observation plan
        plan = self._action_to_plan(action)

        return plan

    def _action_to_plan(self, action: int) -> Dict:
        """Convert action index to observation plan"""
        # Map actions to observation strategies
        strategies = {
            0: {"priority": "critical", "frequency": "continuous"},
            1: {"priority": "high", "frequency": "every_minute"},
            2: {"priority": "medium", "frequency": "every_5_minutes"},
            3: {"priority": "low", "frequency": "every_15_minutes"},
            4: {"priority": "minimal", "frequency": "hourly"},
            5: {"priority": "critical", "frequency": "targeted"},
            6: {"priority": "high", "frequency": "sweep"},
            7: {"priority": "medium", "frequency": "periodic"},
            8: {"priority": "low", "frequency": "opportunistic"},
            9: {"priority": "adaptive", "frequency": "dynamic"},
        }

        return strategies.get(action, strategies[0])

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """
        Update Q-network

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # Add to experience buffer
        self.experience_buffer.append((state, action, reward, next_state))

        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)

        # Sample batch and update
        if len(self.experience_buffer) >= 32:
            self._update_network()

    def _update_network(self, batch_size: int = 32):
        """Update network using experience replay"""
        # Sample random batch
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]

        # Prepare tensors
        states = torch.tensor([b[0] for b in batch], dtype=torch.float32)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32)

        # Compute Q-values
        current_q = self.qnetwork(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q = self.qnetwork(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q

        # Loss and update
        loss = nn.functional.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    # Test scheduler
    scheduler = ObservationScheduler(satellite_id=25544)

    # Test state
    state = np.random.randn(20)

    # Get action
    plan = scheduler.get_action(state)

    print(f"Observation plan: {plan}")
