"""RoboDojo deployment model; skips ALOHA gripper rescaling used in agilex_model.py."""

import os

import torch

from scripts.agilex_model import (
    AGILEX_STATE_INDICES,
    RoboticDiffusionTransformerModel as _BaseRoboticDiffusionTransformerModel,
)


class RoboticDiffusionTransformerModel(_BaseRoboticDiffusionTransformerModel):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.robot_indices = list(range(54)) if args.get("robot_action_dim") == 54 else AGILEX_STATE_INDICES

    def _format_joint_to_state(self, joints):
        B, N, _ = joints.shape
        state = torch.zeros(
            (B, N, self.args["model"]["state_token_dim"]),
            device=joints.device,
            dtype=joints.dtype,
        )
        state[:, :, self.robot_indices] = joints
        state_elem_mask = torch.zeros(
            (B, self.args["model"]["state_token_dim"]),
            device=joints.device,
            dtype=joints.dtype,
        )
        state_elem_mask[:, self.robot_indices] = 1
        return state, state_elem_mask

    def _unformat_action_to_joint(self, action):
        return action[:, :, self.robot_indices]


def create_model(args, **kwargs):
    model = RoboticDiffusionTransformerModel(args, **kwargs)
    pretrained = kwargs.get("pretrained", None)
    if pretrained is not None and os.path.isfile(pretrained):
        model.load_pretrained_weights(pretrained)
    return model
