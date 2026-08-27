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
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger, set_random_seed
import lib.models.monodgp.backbone as backbone_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/monodgp.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Decode only the configured primary score and skip diagnostics.",
    )
    parser.add_argument(
        "--cuda-prefetch",
        action="store_true",
        help="Use pinned-memory one-batch CUDA evaluation prefetch.",
    )
    parser.add_argument(
        "--work-dir",
        default="/tmp/monodgp_cuda129_eval_contract",
        help="Temporary directory for the contract log; predictions stay in memory.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.load(handle, Loader=yaml.Loader)
    set_random_seed(cfg.get("random_seed", 444))

    dataset_cfg = dict(cfg["dataset"])
    dataset_cfg["root_dir"] = args.data_root
    dataset_cfg["batch_size"] = args.batch_size
    dataset = KITTI_Dataset(split="val", cfg=dataset_cfg)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        pin_memory=args.cuda_prefetch,
        drop_last=False,
    )

    backbone_module.is_main_process = lambda: False
    model, _ = build_model(cfg["model"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.cuda().eval()

    save_path = os.path.abspath(args.work_dir)
    model_name = "official_monodgp_epoch177"
    output_dir = os.path.join(save_path, model_name)
    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(os.path.join(output_dir, "eval.log"))

    train_cfg = {
        "save_path": save_path,
        "save_all": False,
        "use_cuda_eval_prefetch": args.cuda_prefetch,
    }
    tester_cfg = dict(cfg["tester"])
    tester_cfg["export_predictions"] = False
    tester = Tester(
        cfg=tester_cfg,
        model=model,
        dataloader=loader,
        logger=logger,
        train_cfg=train_cfg,
        model_name=model_name,
    )
    results = tester.inference(
        collect_diagnostics=not args.primary_only,
        primary_only=args.primary_only,
    )
    moderate_ap = tester.evaluate(results)
    print(f"checkpoint_epoch={checkpoint.get('epoch')}")
    print(f"car_moderate_3d_ap_r40={moderate_ap}")
    print("FULL_KITTI_EVAL_OK")


if __name__ == "__main__":
    main()
