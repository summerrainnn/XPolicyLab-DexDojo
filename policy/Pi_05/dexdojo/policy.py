"""Wuji extension of RoboDojo's Pi_05 Model and websocket policy server."""
import argparse
from teleop.policy_transport import activate
activate()
import asyncio
import logging
from pathlib import Path

import jax
import numpy as np
from XPolicyLab.policy.Pi_05.model import Model, slice_stacked_obs
from client_server.ws.model_server import PolicyServer, PolicyServerConfig
from openpi.policies.policy_config import create_trained_policy
from XPolicyLab.policy.Pi_05.dexdojo.config import make_config
from .protocol import unpack


class WujiModel(Model):
    def __init__(self, model_cfg):
        self.case_seed=0
        super().__init__(model_cfg)

    def get_model(self, model_cfg):
        checkpoint=Path(model_cfg["checkpoint"])
        config=make_config(model_cfg["dataset"], checkpoint.parent.parent,
                           model_cfg["base_params"],model=model_cfg.get("model","pi05"))
        return create_trained_policy(config, checkpoint, sample_kwargs={"num_steps":10})

    def prepare_case(self, case_meta):
        meta=case_meta.get("case_meta",case_meta)
        self.case_seed=int(meta.get("inference_seed",0))
        return {"inference_seed":self.case_seed}

    def reset(self):
        super().reset()
        self.policy._rng=jax.random.key(self.case_seed)

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self.observation_window is None:
            raise RuntimeError("Observation required before inference")
        output=[]
        for index in range(len(env_idx_list or self._latest_env_idx_list)):
            obs=slice_stacked_obs(self.observation_window,index)
            data={"observation/state":obs["state"],"prompt":obs["prompt"],
                  "observation/image":obs["images"]["cam_high"],
                  "observation/left_wrist_image":obs["images"]["cam_left_wrist"],
                  "observation/right_wrist_image":obs["images"]["cam_right_wrist"]}
            actions=np.asarray(self.policy.infer(data)["actions"])
            output.append(unpack(actions))
        return output


def main():
    parser=argparse.ArgumentParser()
    for key in ("checkpoint","dataset","base-params"):
        parser.add_argument("--"+key,required=True)
    parser.add_argument("--port",type=int,required=True)
    parser.add_argument("--model",choices=["pi0","pi05"],default="pi05")
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    model=WujiModel({"task_name":"align_blocks","env_cfg_type":"marvin_wuji","action_type":"joint","checkpoint":args.checkpoint,
        "dataset":args.dataset,"base_params":args.base_params,"model":args.model})
    server=PolicyServer(model,PolicyServerConfig(host="127.0.0.1",port=args.port,
        ws_ping_interval_s=None,ws_ping_timeout_s=None))
    asyncio.run(server.serve_forever())


if __name__=="__main__":
    main()
