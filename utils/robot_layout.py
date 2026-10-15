"""Robot field layouts shared by dataset converters and policy codecs.

Widths describe fields, never inferred from a padded model tensor. Hands are
continuous joint vectors; scalar-gripper postprocessing is a separate choice.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class RobotLayout:
    arm_dims: tuple
    hand_dims: tuple

    def __post_init__(self):
        if len(self.arm_dims) != len(self.hand_dims) or len(self.arm_dims) not in (1, 2):
            raise ValueError("Expected one or two matching arm/hand widths")
        if any(int(n) != n or n <= 0 for n in self.arm_dims + self.hand_dims):
            raise ValueError("Robot field widths must be positive integers")

    @classmethod
    def from_info(cls, info):
        return cls(tuple(info['arm_dim']), tuple(info['ee_dim']))

    @property
    def dim(self):
        return sum(self.arm_dims + self.hand_dims)

    @property
    def slices(self):
        out, offset = [], 0
        for arm, hand in zip(self.arm_dims, self.hand_dims):
            out.append((slice(offset, offset + arm), slice(offset + arm, offset + arm + hand)))
            offset += arm + hand
        return out

    def validate(self, value):
        value = np.asarray(value)
        if value.ndim == 0 or value.shape[-1] != self.dim or not np.isfinite(value).all():
            raise ValueError(f"Expected finite robot vector of width {self.dim}, got {value.shape}")
        return value

    def grouped(self, value, arm_slots=None):
        """Interleaved arms/hands -> all arms (optionally padded), all hands."""
        value = self.validate(value)
        slots = self.arm_dims if arm_slots is None else tuple(arm_slots)
        if len(slots) != len(self.arm_dims) or any(s < a for s, a in zip(slots, self.arm_dims)):
            raise ValueError("Arm slots cannot truncate robot joints")
        arms, hands = [], []
        for (a, h), width in zip(self.slices, slots):
            arm = np.zeros((*value.shape[:-1], width), dtype=value.dtype)
            arm[..., :a.stop-a.start] = value[..., a]
            arms.append(arm)
            hands.append(value[..., h])
        return np.concatenate(arms + hands, axis=-1)

    def interleaved(self, value, arm_slots=None):
        value = np.asarray(value)
        slots = self.arm_dims if arm_slots is None else tuple(arm_slots)
        if len(slots) != len(self.arm_dims) or any(s < a for s, a in zip(slots, self.arm_dims)):
            raise ValueError("Arm slots cannot truncate robot joints")
        expected = sum(slots) + sum(self.hand_dims)
        if value.shape[-1] != expected:
            raise ValueError(f"Expected grouped width {expected}, got {value.shape}")
        out, arm_offset, hand_offset = [], 0, sum(slots)
        for a, h, slot in zip(self.arm_dims, self.hand_dims, slots):
            out.extend((value[..., arm_offset:arm_offset+a], value[..., hand_offset:hand_offset+h]))
            arm_offset += slot
            hand_offset += h
        return self.validate(np.concatenate(out, axis=-1))


LEGACY_DUAL = RobotLayout((6, 6), (1, 1))
MARVIN = RobotLayout((7, 7), (20, 20))


def joint_layout_for_width(width):
    """Compatibility for standalone converters with a known packed schema."""
    for layout in (LEGACY_DUAL, MARVIN, RobotLayout((7, 7), (1, 1))):
        if layout.dim == width:
            return layout
    raise ValueError(f"No implicit joint layout for width {width}; provide robot metadata")
