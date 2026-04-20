import argparse
import os
import random
import sys

import numpy as np
import torch
from easydict import EasyDict as edict

# Get paths and validate
try:
    file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(file_dir)
    workspace_root = os.path.join("/workspace")  # Docker mount point

    paths_to_add = [
        project_root,
        workspace_root,
        os.path.join(workspace_root, "mad")
    ]

    # Add paths if they exist and aren't already in sys.path
    for path in paths_to_add:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            print(f"Added to Python path: {path}")
        else:
            print(f"Path does not exist or already in sys.path: {path}")

    print(f"Project root: {project_root}")
    print(f"Python path: {os.environ.get('PYTHONPATH', '')}")

except Exception as e:
    print(f"Failed to setup paths: {str(e)}")
    raise

def get_config(args):

    if args.debug:
        os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    config = edict(vars(args))

    if config.training_type == "MAD_training":
        config.use_lora = True
        config.train_scratch = False
    elif config.training_type == "MAD_training_only_header":
        config.use_lora = False
        config.train_scratch = False
    elif config.training_type == "MAD_training_scratch":
        config.use_lora = False
        config.train_scratch = True
    elif config.training_type == "test_clip":
        config.train_scratch = False
        config.use_lora = False
        if config.model_name == "MADation":
            config.use_lora = True

    dataset_paths = {
        "lfw": "/data/mcaldeir/exit_entry/lfw/MAD_crop/lfw.csv",
        "IJBC": "/data/mcaldeir/exit_entry/IJBC/MAD_crop/IJBC.csv",
        "IJBC_fpn-ps-224": "/data/mcaldeir/exit_entry/IJBC/MAD_crop/IJBC_fpn-ps-224.csv",
        "IJBC_segformer-ps-224": "/data/mcaldeir/exit_entry/IJBC/MAD_crop/IJBC_segformer-ps-224.csv",
        "IJBC_bisenet-ps":"/data/mcaldeir/exit_entry/IJBC/MAD_crop/IJBC_bisenet-ps.csv",
        "IJBC_danet-ps":"/data/mcaldeir/exit_entry/IJBC/MAD_crop/IJBC_danet-ps.csv",
        "IJBC_fastscnn-ps":"/data/mcaldeir/exit_entry/IJBC/MAD_crop/IJBC_fastscnn-ps.csv",
        "IJBC_fcn-ps":"/data/mcaldeir/exit_entry/IJBC/MAD_crop/IJBC_fcn-ps.csv",
        "IJBC_sam_full_no_ctr":"/data/mcaldeir/exit_entry/IJBC/MAD_crop/IJBC_sam_full_no_ctr.csv",
        "feret": "/data/mcaldeir/exit_entry/feret/MAD_crop/feret.csv",
        "feret_fpn-ps-224": "/data/mcaldeir/exit_entry/feret/MAD_crop/feret_fpn-ps-224.csv",
        "feret_segformer-ps-224": "/data/mcaldeir/exit_entry/feret/MAD_crop/feret_segformer-ps-224.csv",
        "feret_bisenet-ps": "/data/mcaldeir/exit_entry/feret/MAD_crop/feret_bisenet-ps.csv",
        "feret_danet-ps": "/data/mcaldeir/exit_entry/feret/MAD_crop/feret_danet-ps.csv",
        "feret_fastscnn-ps": "/data/mcaldeir/exit_entry/feret/MAD_crop/feret_fastscnn-ps.csv",
        "feret_fcn-ps": "/data/mcaldeir/exit_entry/feret/MAD_crop/feret_fcn-ps.csv",
        "feret_sam_full_no_ctr": "/data/mcaldeir/exit_entry/feret/MAD_crop/feret_sam_full_no_ctr.csv",
        "frgc": "/data/mcaldeir/exit_entry/frgc/MAD_crop/frgc.csv",
        "frgc_fpn-ps-224": "/data/mcaldeir/exit_entry/frgc/MAD_crop/frgc_fpn-ps-224.csv",
        "frgc_segformer-ps-224": "/data/mcaldeir/exit_entry/frgc/MAD_crop/frgc_segformer-ps-224.csv",
        "frgc_bisenet-ps": "/data/mcaldeir/exit_entry/frgc/MAD_crop/frgc_bisenet-ps.csv",
        "frgc_danet-ps": "/data/mcaldeir/exit_entry/frgc/MAD_crop/frgc_danet-ps.csv",
        "frgc_fastscnn-ps": "/data/mcaldeir/exit_entry/frgc/MAD_crop/frgc_fastscnn-ps.csv",
        "frgc_fcn-ps": "/data/mcaldeir/exit_entry/frgc/MAD_crop/frgc_fcn-ps.csv",
        "frgc_sam_full_no_ctr": "/data/mcaldeir/exit_entry/frgc/MAD_crop/frgc_sam_full_no_ctr.csv",
        "frgc_sam_bb_extra": "/data/mcaldeir/exit_entry/frgc/MAD_crop/frgc_sam_bb_extra.csv"
    }
    if config.dataset_name == "lfw":
        config.dataset_path = dataset_paths["lfw"]
        config.test_dataset_path = [dataset_paths["lfw"]]
        config.test_data = ["lfw"]
    elif config.dataset_name == "IJBC":
        config.dataset_path = dataset_paths["IJBC"]
        config.test_dataset_path = [dataset_paths["IJBC"]]
        config.test_data = ["IJBC"]
    elif config.dataset_name == "IJBC_fpn-ps-224":
        config.dataset_path = dataset_paths["IJBC_fpn-ps-224"]
        config.test_dataset_path = [dataset_paths["IJBC_fpn-ps-224"]]
        config.test_data = ["IJBC_fpn-ps-224"]
    elif config.dataset_name == "IJBC_segformer-ps-224":
        config.dataset_path = dataset_paths["IJBC_segformer-ps-224"]
        config.test_dataset_path = [dataset_paths["IJBC_segformer-ps-224"]]
        config.test_data = ["IJBC_segformer-ps-224"]
    elif config.dataset_name == "IJBC_bisenet-ps":
        config.dataset_path = dataset_paths["IJBC_bisenet-ps"]
        config.test_dataset_path = [dataset_paths["IJBC_bisenet-ps"]]
        config.test_data = ["IJBC_bisenet-ps"]
    elif config.dataset_name == "IJBC_danet-ps":
        config.dataset_path = dataset_paths["IJBC_danet-ps"]
        config.test_dataset_path = [dataset_paths["IJBC_danet-ps"]]
        config.test_data = ["IJBC_danet-ps"]
    elif config.dataset_name == "IJBC_fastscnn-ps":
        config.dataset_path = dataset_paths["IJBC_fastscnn-ps"]
        config.test_dataset_path = [dataset_paths["IJBC_fastscnn-ps"]]
        config.test_data = ["IJBC_fastscnn-ps"]
    elif config.dataset_name == "IJBC_fcn-ps":
        config.dataset_path = dataset_paths["IJBC_fcn-ps"]
        config.test_dataset_path = [dataset_paths["IJBC_fcn-ps"]]
        config.test_data = ["IJBC_fcn-ps"]
    elif config.dataset_name == "IJBC_sam_full_no_ctr":
        config.dataset_path = dataset_paths["IJBC_sam_full_no_ctr"]
        config.test_dataset_path = [dataset_paths["IJBC_sam_full_no_ctr"]]
        config.test_data = ["IJBC_sam_full_no_ctr"]
    elif config.dataset_name == "feret":
        config.dataset_path = dataset_paths["feret"]
        config.test_dataset_path = [dataset_paths["feret"]]
        config.test_data = ["feret"]
    elif config.dataset_name == "feret_fpn-ps-224":
        config.dataset_path = dataset_paths["feret_fpn-ps-224"]
        config.test_dataset_path = [dataset_paths["feret_fpn-ps-224"]]
        config.test_data = ["feret_fpn-ps-224"]
    elif config.dataset_name == "feret_segformer-ps-224":
        config.dataset_path = dataset_paths["feret_segformer-ps-224"]
        config.test_dataset_path = [dataset_paths["feret_segformer-ps-224"]]
        config.test_data = ["feret_segformer-ps-224"]
    elif config.dataset_name == "feret_bisenet-ps":
        config.dataset_path = dataset_paths["feret_bisenet-ps"]
        config.test_dataset_path = [dataset_paths["feret_bisenet-ps"]]
        config.test_data = ["feret_bisenet-ps"]
    elif config.dataset_name == "feret_danet-ps":
        config.dataset_path = dataset_paths["feret_danet-ps"]
        config.test_dataset_path = [dataset_paths["feret_danet-ps"]]
        config.test_data = ["feret_danet-ps"]
    elif config.dataset_name == "feret_fastscnn-ps":
        config.dataset_path = dataset_paths["feret_fastscnn-ps"]
        config.test_dataset_path = [dataset_paths["feret_fastscnn-ps"]]
        config.test_data = ["feret_fastscnn-ps"]
    elif config.dataset_name == "feret_fcn-ps":
        config.dataset_path = dataset_paths["feret_fcn-ps"]
        config.test_dataset_path = [dataset_paths["feret_fcn-ps"]]
        config.test_data = ["feret_fcn-ps"]
    elif config.dataset_name == "feret_sam_full_no_ctr":
        config.dataset_path = dataset_paths["feret_sam_full_no_ctr"]
        config.test_dataset_path = [dataset_paths["feret_sam_full_no_ctr"]]
        config.test_data = ["feret_sam_full_no_ctr"]
    elif config.dataset_name == "frgc":
        config.dataset_path = dataset_paths["frgc"]
        config.test_dataset_path = [dataset_paths["frgc"]]
        config.test_data = ["frgc"]
    elif config.dataset_name == "frgc_fpn-ps-224":
        config.dataset_path = dataset_paths["frgc_fpn-ps-224"]
        config.test_dataset_path = [dataset_paths["frgc_fpn-ps-224"]]
        config.test_data = ["frgc_fpn-ps-224"]
    elif config.dataset_name == "frgc_segformer-ps-224":
        config.dataset_path = dataset_paths["frgc_segformer-ps-224"]
        config.test_dataset_path = [dataset_paths["frgc_segformer-ps-224"]]
        config.test_data = ["frgc_segformer-ps-224"]
    elif config.dataset_name == "frgc_bisenet-ps":
        config.dataset_path = dataset_paths["frgc_bisenet-ps"]
        config.test_dataset_path = [dataset_paths["frgc_bisenet-ps"]]
        config.test_data = ["frgc_bisenet-ps"]
    elif config.dataset_name == "frgc_danet-ps":
        config.dataset_path = dataset_paths["frgc_danet-ps"]
        config.test_dataset_path = [dataset_paths["frgc_danet-ps"]]
        config.test_data = ["frgc_danet-ps"]
    elif config.dataset_name == "frgc_fastscnn-ps":
        config.dataset_path = dataset_paths["frgc_fastscnn-ps"]
        config.test_dataset_path = [dataset_paths["frgc_fastscnn-ps"]]
        config.test_data = ["frgc_fastscnn-ps"]
    elif config.dataset_name == "frgc_fcn-ps":
        config.dataset_path = dataset_paths["frgc_fcn-ps"]
        config.test_dataset_path = [dataset_paths["frgc_fcn-ps"]]
        config.test_data = ["frgc_fcn-ps"]
    elif config.dataset_name == "frgc_sam_full_no_ctr":
        config.dataset_path = dataset_paths["frgc_sam_full_no_ctr"]
        config.test_dataset_path = [dataset_paths["frgc_sam_full_no_ctr"]]
        config.test_data = ["frgc_sam_full_no_ctr"]
    elif config.dataset_name == "frgc_sam_bb_extra":
        config.dataset_path = dataset_paths["frgc_sam_bb_extra"]
        config.test_dataset_path = [dataset_paths["frgc_sam_bb_extra"]]
        config.test_data = ["frgc_sam_bb_extra"]

    config.num_classes = 2
    if config.backbone_size == "ViT-B/16" or config.backbone_size == "ViT-B/32":
        config.training_desc = f'ViT-B16/{config.training_type}'
    elif config.backbone_size == "ViT-L/14":
        config.training_desc = f'ViT-L14/{config.training_type}'
    elif config.backbone_size == "ViT-G/14":
        config.training_desc = f'ViT-G14/{config.training_type}'
    elif config.backbone_size == "base":
        config.training_desc = f'base/{config.training_type}'
    elif config.backbone_size == "large":
        config.training_desc = f'large/{config.training_type}'
    config.output_path = "/igd/a1/home/mcaldeir/MAD_exit_entry"

    if config.training_type == "MAD_training":
        config.output_path = (
            f"{config.output_path}/lrm{config.lr_model:.0e}_lrh{config.lr_header:.0e}_d{config.lora_dropout:.0e}_a{config.lora_a}_r{config.lora_r}"
        )
    elif config.training_type == "MAD_training_only_header":
        config.output_path = f"{config.output_path}/lrh{config.lr_header:.0e}"
    elif config.training_type == "MAD_training_scratch":
        config.output_path = (
            f"{config.output_path}/lrm{config.lr_model:.0e}_lrh{config.lr_header:.0e}"
        )
    elif config.training_type == "test_clip":
        config.output_path = "/igd/a1/home/mcaldeir/MAD_exit_entry"

    config.method = 'avg'

    return config

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.

    # torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    # cudnn.benchmark = True
    set_seed(seed=777)

    parser = argparse.ArgumentParser(description="Distributed training job")
    parser.add_argument("--local-rank", type=int, help="local_rank")
    parser.add_argument(
        "--mode",
        default="training",
        choices=["training", "evaluation"],
        help="train or eval mode",
    )
    parser.add_argument(
        "--debug", default=False, type=bool, help="Log additional debug informations"
    )

    parser.add_argument("--backbone_size", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--training_type", type=str, required=True)

    parser.add_argument("--num_epoch", type=int, default=40)
    parser.add_argument("--lr_model", type=float, default=1e-6)
    parser.add_argument("--lr_header", type=float, default=1e-6)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=2)
    parser.add_argument("--lora_a", type=int, default=2)
    parser.add_argument("--max_norm", type=float, default=5)
    parser.add_argument("--loss", type=str, default="BinaryCrossEntropy")
    parser.add_argument("--global_step", type=int, default=0)
    parser.add_argument("--scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup", type=bool, default=True)
    parser.add_argument("--num_warmup_epochs", type=int, default=5)
    parser.add_argument("--T_0", type=int, default=5)
    parser.add_argument("--T_mult", type=int, default=2)
    parser.add_argument("--model_name", type=str, default="clip")
    parser.add_argument("--lr_func_drop", type=int, nargs="+", default=[22, 30, 40])
    parser.add_argument("--batch_size", type=int, default=86)
    parser.add_argument("--lora_bias", type=str, default="none")
    parser.add_argument(
        "--lora_target_modules", type=str, nargs="+", default=["q", "v"]
    )
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--normalize_type", type=str, default="clip")
    parser.add_argument("--interpolation_type", type=str, default="bicubic")
    parser.add_argument(
        "--eval_path", type=str, default="/home/chettaou/data/validation"
    )
    parser.add_argument("--val_targets", type=str, nargs="+", default=[])
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=40)
    parser.add_argument("--batch_size_eval", type=int, default=16)
    parser.add_argument("--horizontal_flip", type=bool, default=True)
    parser.add_argument("--rand_augment", type=bool, default=True)
    parser.add_argument("--eval_method", type=str, default="eval")
    args = parser.parse_args()
    config = get_config(args)
    from src.train import main
    main(config)