"""Wuji OpenPI training/checkpoint action contract (not a global eval restriction)."""
import numpy as np

ACTION_HORIZON = 50
ACTION_DIM = 54

def unpack(actions):
    actions = np.asarray(actions)
    if actions.shape != (ACTION_HORIZON, ACTION_DIM) or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite (50, 54) action chunk, got {actions.shape}")
    return [{"left_arm_joint_state": a[:7], "left_ee_joint_state": a[7:27],
             "right_arm_joint_state": a[27:34], "right_ee_joint_state": a[34:54]}
            for a in actions]
