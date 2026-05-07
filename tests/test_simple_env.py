import numpy as np
import pytest

from simple.env import ACTION_MOVE_DOWN, ACTION_MOVE_UP, ACTION_OPEN_DOORS, SimpleElevatorEnv


def test_reset_is_deterministic_and_observation_shape():
    env = SimpleElevatorEnv(num_floors=5, episode_steps=10, seed=123)
    obs1, info1 = env.reset(seed=123)
    env.step(ACTION_MOVE_UP)
    obs2, info2 = env.reset(seed=123)

    assert obs1.shape == (env.observation_size,)
    assert np.array_equal(obs1, obs2)
    assert info1["elevator_floor"] == info2["elevator_floor"] == 0


def test_movement_respects_floor_boundaries():
    env = SimpleElevatorEnv(num_floors=3, episode_steps=10, call_probability=0.0)
    env.reset(seed=0)

    env.step(ACTION_MOVE_DOWN)
    assert env.elevator_floor == 0

    env.step(ACTION_MOVE_UP)
    env.step(ACTION_MOVE_UP)
    env.step(ACTION_MOVE_UP)
    assert env.elevator_floor == 2


def test_open_doors_boards_and_drops_off_passenger():
    env = SimpleElevatorEnv(num_floors=3, episode_steps=10, call_probability=0.0)
    env.reset(seed=0)
    env.add_passenger(origin=0, destination=2)

    _, reward, _, _, info = env.step(ACTION_OPEN_DOORS)
    assert reward > -0.1
    assert info["waiting_passengers"] == 0
    assert len(info["onboard"]) == 1

    env.step(ACTION_MOVE_UP)
    env.step(ACTION_MOVE_UP)
    _, reward, _, _, info = env.step(ACTION_OPEN_DOORS)
    assert reward > 0.0
    assert info["served_passengers"] == 1


def test_unnecessary_open_door_penalty():
    env = SimpleElevatorEnv(num_floors=3, episode_steps=10, call_probability=0.0)
    env.reset(seed=0)

    _, reward, _, _, _ = env.step(ACTION_OPEN_DOORS)

    assert reward == pytest.approx(-0.06)
