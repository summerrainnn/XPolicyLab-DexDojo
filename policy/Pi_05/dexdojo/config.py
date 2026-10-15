"""Task configuration built on RoboDojo's LeRobotWujiDataConfig."""

import argparse
import dataclasses
import json
from pathlib import Path

from flax import nnx
import numpy as np
import pyarrow.parquet as pq

from openpi.models.pi0_config import Pi0Config
from openpi import transforms
from openpi.shared import nnx_utils, normalize
from openpi.training import config as upstream
from openpi.training import optimizer, weight_loaders
from teleop.output_paths import external_output_path
from src.eval_client.protocol import CONTROL_HZ
from .protocol import ACTION_HORIZON

@dataclasses.dataclass(frozen=True)
class CompletionPrompt:
    def __call__(self, data):
        from teleop.collection.success import collection_instruction
        return {**data, "prompt": collection_instruction(data["prompt"])}


@dataclasses.dataclass(frozen=True)
class CompletionDataConfig(upstream.LeRobotWujiDataConfig):
    def create(self, assets_dirs, model_config):
        data = super().create(assets_dirs, model_config)
        return dataclasses.replace(data, data_transforms=transforms.Group(
            inputs=[CompletionPrompt(), *data.data_transforms.inputs],
            outputs=data.data_transforms.outputs))


ASSET_ID = "align_blocks_selected31"
SAVE_UPDATES = (0, 500, 1000, 2500, 5000)


def make_config(dataset, output, base_params, *, steps=5000, fsdp_devices=2, resume=False, model="pi05", seed=1000):
    if model not in ("pi0","pi05"): raise ValueError("Unsupported OpenPI variant")
    output = external_output_path(output)
    # Gemma stores the action expert as *_1 leaves in the joint llm module.
    freeze = nnx.Any(
        nnx_utils.PathRegex("PaliGemma/img/.*"),
        nnx.All(nnx_utils.PathRegex("PaliGemma/llm/.*"),
                nnx.Not(nnx_utils.PathRegex(".*_1(/.*)?"))),
    )
    info = json.loads((Path(dataset) / "meta/info.json").read_text())
    image_keys = info.get("features", {})
    base_image_key = next((key for key in ("observation.images.cam_head", "observation.images.cam_high")
                           if key in image_keys), None)
    if base_image_key is None:
        raise ValueError("Dataset must contain a head camera")
    return upstream.TrainConfig(
        name="dexdojo_align_blocks_openpi", exp_name="selected31_seed1000",
        model=Pi0Config(pi05=model=="pi05", action_dim=54, action_horizon=ACTION_HORIZON, max_token_len=256),
        data=CompletionDataConfig(
            repo_id=str(Path(dataset).resolve()),
            assets=upstream.AssetsConfig(assets_dir=str(output / "assets"), asset_id=ASSET_ID),
            base_config=upstream.DataConfig(prompt_from_task=True, video_backend="pyav"),
            base_image_key=base_image_key, extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.PartialCheckpointWeightLoader(str(Path(base_params).resolve())),
        freeze_filter=freeze, ema_decay=.99, seed=seed, batch_size=64, num_workers=8,
        num_train_steps=steps, fsdp_devices=fsdp_devices,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=167, peak_lr=2.5e-5, decay_steps=5000, decay_lr=2.5e-6),
        optimizer=optimizer.AdamW(), log_interval=25, save_interval=500,
        keep_period=500, checkpoint_dir_override=str(output / "checkpoints"),
        resume=resume, overwrite=False, wandb_enabled=False,
        policy_metadata={"control_hz":CONTROL_HZ,"action_dim":54,"source":"RoboDojo OpenPI"},
    )


def compute_norm_stats(dataset, output):
    """Exact numeric statistics over every retained frame, without video decoding."""
    dataset, output = Path(dataset), external_output_path(output)
    tables = [pq.read_table(p, columns=["observation.state", "action"])
              for p in sorted((dataset / "data").rglob("*.parquet"))]
    if not tables:
        raise ValueError("No numeric data found")
    info=json.loads((dataset/"meta/info.json").read_text())
    if info.get("fps") != CONTROL_HZ:
        raise ValueError("Dataset FPS must match the 30 Hz policy protocol; do not relabel timestamps")
    stats = {}
    for src, dst in (("observation.state", "state"), ("action", "actions")):
        values = np.concatenate([np.asarray(t[src].to_pylist(), dtype=np.float64) for t in tables])
        if values.ndim != 2 or values.shape[1] != 54 or not len(values) or not np.isfinite(values).all():
            raise ValueError(f"Unexpected {src} data: {values.shape}")
        stats[dst] = normalize.NormStats(mean=values.mean(0), std=values.std(0),
            q01=np.quantile(values,.01,axis=0), q99=np.quantile(values,.99,axis=0))
    normalize.save(output / "assets" / ASSET_ID, stats)
    output.mkdir(parents=True, exist_ok=True)
    (output / "normalization_manifest.json").write_text(json.dumps({
        "dataset":str(dataset.resolve()),"frames":len(values),"episodes":info["total_episodes"],
        "action_representation":"absolute_radians","quantile_method":"numpy_exact"},indent=2)+"\n")


def parse_config():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-params", required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--fsdp-devices", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model",choices=["pi0","pi05"],default="pi05")
    parser.add_argument("--prepare-only", action="store_true")
    args=parser.parse_args()
    compute_norm_stats(args.dataset,args.output)
    config=make_config(args.dataset,args.output,args.base_params,
        steps=args.steps,fsdp_devices=args.fsdp_devices,resume=args.resume,model=args.model,seed=args.seed)
    config.checkpoint_dir.parent.mkdir(parents=True,exist_ok=True)
    (config.checkpoint_dir.parent / "settings.json").write_text(json.dumps({
        "dataset":args.dataset,"base_params":args.base_params,"steps":args.steps,"model":args.model,
        "batch_size":64,"seed":args.seed,"save_updates":list(SAVE_UPDATES),
        "fsdp_devices":args.fsdp_devices,"action_horizon":ACTION_HORIZON,
        "freeze":"vision and language backbone; action expert/projections trainable",
        "peak_lr":2.5e-5,"warmup_updates":167,"decay_lr":2.5e-6},indent=2)+"\n")
    return config,args.prepare_only
