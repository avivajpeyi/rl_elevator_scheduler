from simple.plots import plot_comparison, plot_metrics, plot_training_history


def test_plot_helpers_write_pngs(tmp_path):
    heuristic_metrics = {
        "mean_wait_time": 1.0,
        "mean_trip_time": 2.0,
        "served_passengers": 3.0,
        "delivery_rate_pct": 75.0,
        "total_reward": 4.0,
    }
    dqn_metrics = {
        "mean_wait_time": 0.0,
        "mean_trip_time": 0.0,
        "served_passengers": 0.0,
        "delivery_rate_pct": 0.0,
        "total_reward": -5.0,
    }
    history = [
        {"episode": 0.0, "total_reward": -1.0, "mean_wait_time": 2.0, "delivery_rate_pct": 25.0},
        {"episode": 1.0, "total_reward": 1.0, "mean_wait_time": 1.0, "delivery_rate_pct": 50.0},
    ]

    metric_path = plot_metrics(heuristic_metrics, tmp_path / "metrics.png")
    history_path = plot_training_history(history, tmp_path / "history.png")
    comparison_path = plot_comparison(heuristic_metrics, dqn_metrics, tmp_path / "comparison.png")

    for path in (metric_path, history_path, comparison_path):
        assert path.exists()
        assert path.stat().st_size > 0
