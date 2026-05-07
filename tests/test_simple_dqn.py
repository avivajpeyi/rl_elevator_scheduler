import numpy as np

from simple.dqn_agent import DQNAgent
from simple.env import SimpleElevatorEnv
from simple.replay_buffer import ReplayBuffer
from simple.train_dqn import TrainConfig, evaluate_dqn, train_dqn


def test_replay_buffer_push_and_sample():
    buffer = ReplayBuffer(capacity=10, seed=0)
    state = np.zeros(4, dtype=np.float32)
    next_state = np.ones(4, dtype=np.float32)

    for idx in range(5):
        buffer.push(state + idx, idx % 4, float(idx), next_state + idx, False)

    states, actions, rewards, next_states, dones = buffer.sample(3)

    assert states.shape == (3, 4)
    assert actions.shape == (3,)
    assert rewards.shape == (3,)
    assert next_states.shape == (3, 4)
    assert dones.shape == (3,)


def test_dqn_agent_selects_valid_action():
    env = SimpleElevatorEnv(num_floors=5, episode_steps=5, call_probability=0.0)
    observation, _ = env.reset(seed=0)
    agent = DQNAgent(env.observation_size, env.action_size, seed=0)

    action = agent.select_action(observation, epsilon=0.0)

    assert 0 <= action < env.action_size


def test_train_and_eval_progress_callbacks_are_called():
    calls = []

    def callback(phase, current, total, metrics):
        calls.append((phase, current, total, metrics))

    config = TrainConfig(
        num_floors=3,
        episode_steps=2,
        train_episodes=2,
        call_probability=0.0,
        batch_size=1,
        learning_starts=1,
        train_every=1,
        seed=0,
    )
    agent, _ = train_dqn(config, progress_callback=callback)
    evaluate_dqn(agent, config, episodes=2, progress_callback=callback)

    assert [call[:3] for call in calls] == [
        ("train", 1, 2),
        ("train", 2, 2),
        ("eval", 1, 2),
        ("eval", 2, 2),
    ]
