"""Expand SmolVLA's robot projections; retain VLM/expert weights and processor files.

Executed once before native lerobot-train, never during evaluation. The expanded
projections are freshly initialized and need training on Marvin demonstrations.
"""
import argparse
import json
import shutil
from pathlib import Path


def prepare(source, destination):
    from safetensors.torch import load_file, save_file
    import torch

    if not Path(source).is_dir():
        from huggingface_hub import snapshot_download
        source = snapshot_download(source)
    source = Path(source)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Use a new initialization directory: {destination}")
    cfg = json.loads((source / 'config.json').read_text())
    weights = load_file(str(source / 'model.safetensors'))
    old_state, old_action = cfg['max_state_dim'], cfg['max_action_dim']
    for key, value in list(weights.items()):
        shape = list(value.shape)
        if key.endswith('state_proj.weight'):
            shape[-1] = 54
        elif key.endswith('action_in_proj.weight'):
            shape[-1] = 54
        elif key.endswith(('action_out_proj.weight', 'action_out_proj.bias')):
            shape[0] = 54
        else:
            continue
        if tuple(shape) != tuple(value.shape):
            tensor = torch.empty(shape, dtype=value.dtype)
            if tensor.ndim == 2:
                torch.nn.init.kaiming_uniform_(tensor, a=5 ** .5)
            else:
                torch.nn.init.zeros_(tensor)
            weights[key] = tensor
    cfg.update(max_state_dim=54, max_action_dim=54)
    if "adapt_to_pi_aloha" in cfg:
        cfg["adapt_to_pi_aloha"] = False
    for group, key in (("input_features", "observation.state"), ("output_features", "action")):
        if cfg.get(group) and key in cfg[group]:
            cfg[group][key]['shape'] = [54]
    destination.mkdir(parents=True)
    for file in source.iterdir():
        if file.is_file() and file.suffix in ('.json', '.safetensors') and file.name != 'model.safetensors':
            shutil.copy2(file, destination / file.name)
    (destination / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    save_file(weights, str(destination / 'model.safetensors'))
    (destination / 'robot_adaptation.json').write_text(json.dumps(dict(
        source=str(source), original_state_dim=old_state, original_action_dim=old_action,
        layout='left_arm7,left_hand20,right_arm7,right_hand20', trained=False), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('destination')
    args = parser.parse_args()
    prepare(args.source, args.destination)
