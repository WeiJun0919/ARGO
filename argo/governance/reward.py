def compute_reward(
    old_val_score,
    new_val_score,
    action_name,
    processed_ratio=0.0,
    user_satisfaction=0.0,
    issues_resolved_ratio=None,
    operational_cost=None,
    lambda_performance=20.0,
    lambda_quality=0.5,
    lambda_cost=0.2,
    lambda_human=0.05,
):
    """Equation (5): downstream gain + quality gain - cost + feedback.

    ``issues_resolved_ratio`` is eta_t, normalized against the issue mass
    detected at the beginning of the trajectory. ``operational_cost`` may
    combine compute, review, budget, deletion, and ineffective no-op costs.
    """
    delta = float(new_val_score) - float(old_val_score)
    eta = float(processed_ratio if issues_resolved_ratio is None else issues_resolved_ratio)
    cost = float(processed_ratio if operational_cost is None else operational_cost)
    reward = (
        float(lambda_performance) * delta
        + float(lambda_quality) * eta
        - float(lambda_cost) * cost
        + float(lambda_human) * float(user_satisfaction)
    )
    if action_name == "noop":
        reward -= 0.03
    return float(reward)
