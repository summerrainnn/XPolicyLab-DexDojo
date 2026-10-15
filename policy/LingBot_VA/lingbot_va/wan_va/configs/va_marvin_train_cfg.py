"""Marvin dense joint actions; statistics are generated from its dataset."""
import copy
import json
import os
from .va_robotwin30_train_cfg import va_robotwin30_train_cfg

va_marvin_train_cfg = copy.deepcopy(va_robotwin30_train_cfg)
va_marvin_train_cfg.action_dim = 54
va_marvin_train_cfg.used_action_channel_ids = list(range(54))
va_marvin_train_cfg.inverse_used_action_channel_ids = list(range(54))
# Training and inference use the same generated action quantiles.
_stats_path = os.environ.get("LINGBOT_VA_ACTION_CONFIG", os.path.join(va_marvin_train_cfg.dataset_path, "meta/action_norm.json"))
if os.path.isfile(_stats_path):
    with open(_stats_path) as handle:
        _stats = json.load(handle)
    va_marvin_train_cfg.norm_stat = _stats.get("norm_stat", _stats)
else:
    va_marvin_train_cfg.norm_stat = None
