"""Auditable LeRobot v2.1 to memory-mapped DP data; no trajectory resampling."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from teleop.output_paths import external_output_path

CAMERAS = {'head_cam':'cam_head','left_cam':'cam_left_wrist','right_cam':'cam_right_wrist'}


def prepare(source, output):
    import av
    import cv2
    import pyarrow.parquet as pq
    source=Path(source);output=external_output_path(output)
    if output.exists(): raise FileExistsError(output)
    info=json.loads((source/'meta/info.json').read_text())
    if info['fps']!=30: raise ValueError('Require native 30 Hz demonstrations')
    output.mkdir(parents=True)
    n=info['total_frames']; arrays={}
    for name in ('state','action'):
        arrays[name]=np.lib.format.open_memmap(output/(name+'.npy'),mode='w+',dtype='float32',shape=(n,54))
    for name in CAMERAS:
        arrays[name]=np.lib.format.open_memmap(output/(name+'.npy'),mode='w+',dtype='uint8',shape=(n,3,240,320))
    ends=[];offset=0;hashes={}
    for episode in range(info['total_episodes']):
        p=source/info['data_path'].format(episode_chunk=episode//1000,episode_index=episode)
        table=pq.read_table(p);count=len(table)
        for key,column in [('state','observation.state'),('action','action')]:
            values=np.asarray(table[column].to_pylist(),dtype=np.float32)
            if values.shape!=(count,54) or not np.isfinite(values).all(): raise ValueError(column)
            arrays[key][offset:offset+count]=values
        hashes[str(p.relative_to(source))]=hashlib.sha256(p.read_bytes()).hexdigest()
        for name,camera in CAMERAS.items():
            video=source/info['video_path'].format(episode_chunk=episode//1000,episode_index=episode,video_key='observation.images.'+camera)
            decoded=0
            with av.open(str(video)) as container:
                for frame in container.decode(video=0):
                    if decoded>=count: raise ValueError('Extra video frames')
                    arrays[name][offset+decoded]=cv2.resize(frame.to_ndarray(format='rgb24'),(320,240),interpolation=cv2.INTER_AREA).transpose(2,0,1)
                    decoded+=1
            if decoded!=count: raise ValueError('Missing video frames')
            hashes[str(video.relative_to(source))]=hashlib.sha256(video.read_bytes()).hexdigest()
        offset+=count;ends.append(offset)
    if offset!=n: raise ValueError('Frame count mismatch')
    for value in arrays.values(): value.flush()
    np.save(output/'episode_ends.npy',ends)
    (output/'manifest.json').write_text(json.dumps(dict(source=str(source.resolve()),episodes=len(ends),frames=n,
        source_files=hashes,curation=json.loads((source/'meta/curation.json').read_text()),fps=30,action_dim=54),indent=2))


class Dataset:
    def __init__(self,path):
        path=Path(path)
        self.arrays={key:np.load(path/(key+'.npy'),mmap_mode='r') for key in ['state','action',*CAMERAS]}
        self.ends=np.load(path/'episode_ends.npy')
        self.starts=np.r_[0,self.ends[:-1]]
    def __len__(self): return int(self.ends[-1])
    def __getitem__(self,index):
        episode=np.searchsorted(self.ends,index,side='right');lo=self.starts[episode];hi=self.ends[episode]-1
        observations=np.clip(np.arange(index-2,index+1),lo,hi)
        actions=np.clip(np.arange(index-2,index+50),lo,hi)
        return {'obs':{**{k:self.arrays[k][observations].astype(np.float32)/255 for k in CAMERAS},
                       'agent_pos':self.arrays['state'][observations].copy()},'action':self.arrays['action'][actions].copy()}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();prepare(a.source,a.output)
