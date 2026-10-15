"""Eight-GPU DDP trainer; uses upstream DP model, normalization and EMA."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from XPolicyLab.policy.DP.dexdojo.data import Dataset, CAMERAS
from teleop.output_paths import external_output_path

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'XPolicyLab/policy/DP'))


def config():
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(ROOT/'XPolicyLab/policy/DP/diffusion_policy/config/robot_dp.yaml')
    task=OmegaConf.load(ROOT/'XPolicyLab/policy/DP/diffusion_policy/config/task/default_task.yaml')
    cfg.pop('defaults');cfg.task=task
    cfg.task.name='align_blocks'
    cfg.task.shape_meta={'action':{'shape':[54]},'obs':{**{key:{'shape':[3,240,320],'type':'rgb'} for key in CAMERAS},'agent_pos':{'shape':[54],'type':'low_dim'}}}
    cfg.horizon=52;cfg.n_obs_steps=3;cfg.n_action_steps=50
    return cfg


class Loss(torch.nn.Module):
    def __init__(self,policy): super().__init__();self.policy=policy
    def forward(self,batch): return self.policy.compute_loss(batch)


def main():
    import hydra
    from omegaconf import OmegaConf
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    from diffusion_policy.common.normalize_util import get_image_range_normalizer
    from diffusion_policy.model.diffusion.ema_model import EMAModel
    from diffusion_policy.model.common.lr_scheduler import get_scheduler
    from diffusion_policy.common.pytorch_util import dict_apply
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=5000);p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=64)
    a=p.parse_args();output=external_output_path(a.output)
    dist.init_process_group('nccl');rank=dist.get_rank();world=dist.get_world_size();local=int(os.environ['LOCAL_RANK'])
    if a.batch_size%world: raise ValueError('Batch must divide world size')
    torch.cuda.set_device(local);torch.set_num_threads(4)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    OmegaConf.register_new_resolver('eval',eval,replace=True)
    cfg=config();dataset=Dataset(a.data)
    policy=hydra.utils.instantiate(cfg.policy)
    normalizer=LinearNormalizer();normalizer.fit({'action':np.asarray(dataset.arrays['action']).copy(),'agent_pos':np.asarray(dataset.arrays['state']).copy()},last_n_dims=1,mode='limits')
    for name in CAMERAS: normalizer[name]=get_image_range_normalizer()
    policy.set_normalizer(normalizer);policy.cuda(local)
    ema_policy=copy.deepcopy(policy);ema=EMAModel(ema_policy,power=.75)
    model=DistributedDataParallel(Loss(policy),device_ids=[local],find_unused_parameters=True)
    torch.manual_seed(a.seed+rank)  # Different diffusion noise on each rank, shared initial weights.
    optimizer=hydra.utils.instantiate(cfg.optimizer,params=policy.parameters())
    schedule=get_scheduler('cosine',optimizer=optimizer,num_warmup_steps=500,num_training_steps=a.steps)
    sampler=DistributedSampler(dataset,shuffle=True,seed=a.seed)
    loader=DataLoader(dataset,batch_size=a.batch_size//world,sampler=sampler,num_workers=2,pin_memory=True,drop_last=True)
    if rank==0:
        output.mkdir(parents=True,exist_ok=False)
        (output/'settings.json').write_text(json.dumps(dict(steps=a.steps,seed=a.seed,batch_size=a.batch_size,world_size=world,
            data=a.data,action_horizon=50,internal_horizon=52,n_obs_steps=3,upstream='72b3a52f2b069220088ef824ade5890885858379'),indent=2))
        OmegaConf.save(cfg,output/'config.yaml')
    dist.barrier();step=0;epoch=0;started=time.time()
    while step<a.steps:
        sampler.set_epoch(epoch)
        for batch in loader:
            batch=dict_apply(batch,lambda x:x.cuda(local,non_blocking=True))
            optimizer.zero_grad(set_to_none=True);loss=model(batch)
            if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
            loss.backward();optimizer.step();schedule.step();ema.step(policy);step+=1
            if step%25==0 or step==1:
                metric=loss.detach().clone();dist.all_reduce(metric);metric/=world
                if rank==0:
                    report=dict(step=step,loss=metric.item(),elapsed_s=time.time()-started)
                    print(json.dumps(report),flush=True)
                    (output/'progress.json').write_text(json.dumps(report))
            if step in (500,1000,2500,a.steps):
                if rank==0:
                    checkpoint=output/f'{step}.pt'
                    torch.save(dict(config={"policy":OmegaConf.to_container(cfg.policy,resolve=True)},policy=policy.state_dict(),ema=ema_policy.state_dict(),optimizer=optimizer.state_dict(),step=step),checkpoint)
                dist.barrier()
            if step>=a.steps: break
        epoch+=1
    if rank==0: (output/'complete.json').write_text(json.dumps(dict(updates=step,world_size=world,elapsed_s=time.time()-started)))
    dist.destroy_process_group()


if __name__=='__main__':main()
