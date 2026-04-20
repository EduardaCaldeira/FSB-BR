import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
from utils.utils_callbacks import CallBackVerification
from utils.utils_logging import init_logging

def main(args):
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()

    total_step = 0
    targets = args.dataset
    if args.segmentation_network != "none":
        targets = targets + "_" + args.segmentation_network
    rec = "/data/mcaldeir/exit_entry/" + args.dataset + "/embeddings/" + args.FR_network + "/" + targets
    
    log_root = logging.getLogger()
    init_logging(log_root, rank, '../../../../data/mcaldeir/exit_entry/' + args.dataset , logfile= args.method + '_' + targets + '.log')

    callback_verification = CallBackVerification(total_step, rank, [targets], rec, method=args.method)

    if args.method == 'eval':
        save_png = None
    else:
        dist_dir = '/data/mcaldeir/exit_entry/' + args.dataset + '/dist/' + args.FR_network
        os.makedirs(dist_dir, exist_ok=True)
        save_png = os.path.join(dist_dir, targets + ".png")

    logging.info("Performance assessment " + args.FR_network + "...")
    callback_verification(total_step, backbone=None, method=args.method, finished_training=True, save_png=save_png)

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch margin penalty loss training')
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0, help='local_rank')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--segmentation_network', type=str)
    parser.add_argument('--FR_network', type=str)
    parser.add_argument('--method', type=str)
    args_ = parser.parse_args()
    main(args_)