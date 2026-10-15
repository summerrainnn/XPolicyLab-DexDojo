"""Add ABot modality metadata to a prepared Marvin LeRobot dataset."""
import argparse
import json
from pathlib import Path

def prepare(path):
    path = Path(path)
    info = json.loads((path / "meta/info.json").read_text())
    for key in ("observation.state", "action"):
        if info["features"][key]["shape"] != [54]:
            raise ValueError(f"{key} must use Marvin's 54D interleaved joint layout")
    fields = (("left_joints", 0, 7), ("left_gripper", 7, 27),
              ("right_joints", 27, 34), ("right_gripper", 34, 54))
    meta = {}
    for kind, column in (("state", "observation.state"), ("action", "action")):
        meta[kind] = {name: {"start": start, "end": end, "original_key": column}
                      for name, start, end in fields}
    meta["video"] = {name: {"original_key": "observation.images." + name}
                     for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")}
    meta["annotation"] = {"human.action.task_description": {"original_key": "task_index"}}
    (path / "meta/modality.json").write_text(json.dumps(meta, indent=2) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    prepare(parser.parse_args().dataset)
