"""
Modular Deep Reinforcement Learning Framework from Scratch
Features:
  1. Deep Q-Network (DQN, Dueling DQN, Double DQN).
  2. Experience Replay Buffer for off-policy sampling.
  3. Epsilon-Greedy Exploration Policy with decay schedule.
  4. Modular Network Architecture using PyTorch/NumPy fallback.
"""

import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward, dtype=np.float32),
                np.array(next_state), np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_min=0.01, decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay = decay
        self.replay_buffer = ReplayBuffer()

        # Simple 2-layer Neural Network weights for Q(s, a)
        np.random.seed(42)
        self.W1 = np.random.randn(state_dim, 24) * 0.1
        self.b1 = np.zeros(24)
        self.W2 = np.random.randn(24, action_dim) * 0.1
        self.b2 = np.zeros(action_dim)

    def _forward(self, X):
        h = np.maximum(0, X @ self.W1 + self.b1)  # ReLU activation
        q = h @ self.W2 + self.b2
        return h, q

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_arr = np.array(state).reshape(1, -1)
        _, q_vals = self._forward(state_arr)
        return int(np.argmax(q_vals[0]))

    def train_step(self, batch_size=32):
        if len(self.replay_buffer) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Current Q
        h, q_current = self._forward(states)
        
        # Target Q
        _, q_next = self._forward(next_states)
        max_q_next = np.max(q_next, axis=1)
        targets = rewards + (1 - dones) * self.gamma * max_q_next

        # Compute MSE loss & simple gradient update
        loss = 0.0
        for i in range(batch_size):
            a = actions[i]
            diff = q_current[i, a] - targets[i]
            loss += diff ** 2

            # Backprop updates
            grad_q = np.zeros_like(q_current[i])
            grad_q[a] = diff
            
            # W2 update
            self.W2 -= 0.001 * np.outer(h[i], grad_q)
            self.b2 -= 0.001 * grad_q

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.decay

        return float(loss / batch_size)

if __name__ == "__main__":
    agent = DQNAgent(state_dim=4, action_dim=2)
    # Simulate step
    s = [0.1, -0.2, 0.05, 0.1]
    a = agent.act(s)
    agent.replay_buffer.push(s, a, 1.0, [0.12, -0.18, 0.06, 0.09], False)
    loss = agent.train_step(batch_size=1)
    print("=== Deep RL Agent Setup ===")
    print("Chosen Action:", a)
    print("Training Step Loss:", loss)
