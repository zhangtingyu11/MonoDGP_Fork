import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import torch
import yaml
from torch.utils.data import DataLoader

from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.helpers.model_helper import build_model
import lib.models.monodgp.backbone as backbone_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/monodgp.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.load(handle, Loader=yaml.Loader)

    dataset_cfg = dict(cfg["dataset"])
    dataset_cfg["root_dir"] = args.data_root
    dataset_cfg["batch_size"] = 1
    dataset = KITTI_Dataset(split="val", cfg=dataset_cfg)
    loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    # The checkpoint contains all backbone parameters, so this contract must not
    # download an unrelated ImageNet initialization while constructing the model.
    backbone_module.is_main_process = lambda: False
    model, _ = build_model(cfg["model"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.cuda().eval()

    inputs, calibs, targets, info = next(iter(loader))
    with torch.inference_mode():
        outputs = model(
            inputs.cuda(non_blocking=False),
            calibs.cuda(non_blocking=False),
            targets,
            info["img_size"].cuda(non_blocking=False),
            dn_args=0,
        )
        torch.cuda.synchronize()

    tensor_outputs = {key: value for key, value in outputs.items() if torch.is_tensor(value)}
    bad = [key for key, value in tensor_outputs.items() if not torch.isfinite(value).all()]
    if bad:
        raise RuntimeError(f"non-finite model outputs: {bad}")

    print(f"checkpoint_epoch={checkpoint.get('epoch')}")
    print(f"image_id={int(info['img_id'][0])}")
    for key, value in tensor_outputs.items():
        print(f"{key}={tuple(value.shape)}")
    print("MODEL_FORWARD_OK")


if __name__ == "__main__":
    main()
