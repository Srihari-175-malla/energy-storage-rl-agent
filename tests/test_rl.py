import unittest
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from deep_rl_framework import DQNAgent, ReplayBuffer

class TestDeepRLFramework(unittest.TestCase):
    def test_replay_buffer(self):
        buf = ReplayBuffer(capacity=10)
        buf.push([1, 2], 0, 1.0, [1.1, 2.1], False)
        self.assertEqual(len(buf), 1)

    def test_dqn_action_and_training(self):
        agent = DQNAgent(state_dim=4, action_dim=2)
        s = [0.1, 0.2, 0.3, 0.4]
        a = agent.act(s)
        self.assertIn(a, [0, 1])

        # Fill buffer & train step
        for _ in range(5):
            agent.replay_buffer.push(s, a, 1.0, s, False)

        loss = agent.train_step(batch_size=4)
        self.assertGreaterEqual(loss, 0.0)

if __name__ == '__main__':
    unittest.main()
