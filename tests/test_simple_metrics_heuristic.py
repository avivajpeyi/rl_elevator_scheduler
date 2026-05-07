from simple.env import ACTION_MOVE_UP, ACTION_OPEN_DOORS, ACTION_STAY, SimpleElevatorEnv
from simple.heuristic import NearestCallHeuristic
from simple.metrics import collect_metrics, run_episode


def test_collect_metrics_with_no_served_passengers():
    env = SimpleElevatorEnv(num_floors=3, episode_steps=5, call_probability=0.0)
    env.reset(seed=0)

    metrics = collect_metrics(env)

    assert metrics == {
        "mean_wait_time": 0.0,
        "mean_trip_time": 0.0,
        "served_passengers": 0.0,
        "delivery_rate_pct": 0.0,
        "total_reward": 0.0,
    }


def test_collect_metrics_includes_delivery_rate_percentage():
    env = SimpleElevatorEnv(num_floors=3, episode_steps=10, call_probability=0.0)
    env.reset(seed=0)
    env.add_passenger(origin=0, destination=2)
    env.add_passenger(origin=1, destination=2)

    env.step(ACTION_OPEN_DOORS)
    env.step(ACTION_MOVE_UP)
    env.step(ACTION_MOVE_UP)
    env.step(ACTION_OPEN_DOORS)
    metrics = collect_metrics(env)

    assert metrics["served_passengers"] == 1.0
    assert metrics["delivery_rate_pct"] == 50.0


def test_nearest_call_heuristic_moves_toward_nearest_call():
    env = SimpleElevatorEnv(num_floors=5, episode_steps=10, call_probability=0.0)
    env.reset(seed=0)
    env.add_passenger(origin=4, destination=0)
    env.add_passenger(origin=2, destination=4)

    action = NearestCallHeuristic().get_action(env)

    assert action == 1


def test_run_episode_returns_expected_metric_keys():
    env = SimpleElevatorEnv(num_floors=3, episode_steps=3, call_probability=0.0)

    metrics = run_episode(env, lambda _: ACTION_STAY, seed=0)

    assert set(metrics) == {
        "mean_wait_time",
        "mean_trip_time",
        "served_passengers",
        "delivery_rate_pct",
        "total_reward",
    }
