import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel

from config.config import config as cfg
from utils.utils_callbacks import CallBackVerification
from utils.utils_logging import init_logging

from backbones.iresnet import iresnet100, iresnet50
from backbones import get_model
from model import build_model
    
class SwinFaceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        out = self.model(x)
        return out['Recognition']

class TransFaceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, tuple):
            return out[0]
        return out
    
class SwinFaceCfg:
    network = "swin_t"
    fam_kernel_size=3
    fam_in_chans=2112
    fam_conv_shared=False
    fam_conv_mode="split"
    fam_channel_attention="CBAM"
    fam_spatial_attention=None
    fam_pooling="max"
    fam_la_num_list=[2 for j in range(11)]
    fam_feature="all"
    fam = "3x3_2112_F_s_C_N_max"
    embedding_size = 512
    
def main(args):
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()

    total_step=0
    targets = args.dataset
    if args.segmentation_network != "none":
        targets = targets + "_" + args.segmentation_network
    
    rec = "/data/mcaldeir/exit_entry/" + args.dataset + "/aligned/" + targets
    print(rec)

    log_root = logging.getLogger()
    init_logging(log_root, rank, '../../../../data/mcaldeir/exit_entry/' + args.dataset, logfile='save_FR_embs_' + targets + '.log')

    callback_verification = CallBackVerification(total_step, rank, [targets], rec, method='save_embs')

    for backbone_name in args.FR_networks:
        if backbone_name == "ElasticCos" or backbone_name == "ms1mv3_arcface_r100_fp16":
            backbone = iresnet100(num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
        elif backbone_name == "CosFace" or backbone_name == "AdaFace" or backbone_name == "ArcFace":
            backbone = iresnet50(dropout=0.4, num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
        elif backbone_name == "TransFace-L":
            backbone = get_model("ViT-L", dropout=0, fp16=False).to(local_rank) # TODO: automate selection if we want to include other transformer sizes in the evaluation
        elif backbone_name == "SwinFace":
            backbone = build_model(SwinFaceCfg()).to(local_rank)
        else:
            backbone = None
            logging.info("load backbone failed!")
            exit()

        try:
            if backbone_name == "SwinFace":
                backbone_pth = os.path.join("FR_models/" + backbone_name + ".pt")
                dict_checkpoint = torch.load(backbone_pth, map_location=torch.device(local_rank))
                backbone.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
                backbone.fam.load_state_dict(dict_checkpoint["state_dict_fam"])
                backbone.tss.load_state_dict(dict_checkpoint["state_dict_tss"])
                backbone.om.load_state_dict(dict_checkpoint["state_dict_om"])
                backbone = SwinFaceWrapper(backbone)
            elif backbone_name == "TransFace-L":
                backbone_pth = os.path.join("FR_models/" + backbone_name + ".pt")
                backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))
                backbone = TransFaceWrapper(backbone)
            elif backbone_name == "ms1mv3_arcface_r100_fp16" or backbone_name == "ElasticCos":
                backbone_pth = os.path.join("FR_models/" + backbone_name + ".pth")
                backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))
            if rank == 0:
                logging.info("backbone resume loaded successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("load backbone resume init, failed!")

        for ps in backbone.parameters():
            dist.broadcast(ps, 0)

        backbone = DistributedDataParallel(
            module=backbone, broadcast_buffers=False, device_ids=[local_rank]
        )
        backbone.eval()

        emb_dir = "/data/mcaldeir/exit_entry/" + args.dataset + "/embeddings/" + backbone_name + "/" + targets
        if not os.path.exists(os.path.join(emb_dir)) and rank == 0:
            os.makedirs(os.path.join(emb_dir))
        else:
            time.sleep(2)

        logging.info("Saving embeddings at " + emb_dir + "...")
        callback_verification(total_step, backbone, method='save_embs', save_dir=emb_dir, finished_training=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch margin penalty loss  training')
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0, help='local_rank')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--segmentation_network', type=str)
    parser.add_argument('--FR_networks', nargs='+')
    args_ = parser.parse_args()
    main(args_)