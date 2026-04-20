import os
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool 
import argparse
from PIL import Image
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

def get_bounding_box(each_line, in_dir, method):
    name_box_score = each_line.strip().split(',')
    img_name = os.path.join(in_dir, name_box_score[0])
    img = Image.open(img_name).convert("RGB") # SAM can only process RGB images
    img = np.array(img) 

    if in_dir.find("IJBC") != -1:
        lmk = np.array([float(x) for x in name_box_score[7:]], dtype=np.float32)
    else:
        lmk = np.array([float(x) for x in name_box_score[7:-2]], dtype=np.float32)
    lmk = lmk.reshape((5, 2))

    if method == 'bb' or method == 'bb_no_ctr':
        x1, y1, x2, y2 = np.array([float(x) for x in name_box_score[2:6]], dtype=np.float32)

        # MAD cropping
        n = 0.05 # add margin for cropping
        x1_min = 0; x2_max = img.shape[1] - 1
        y1_min = 0; y2_max = img.shape[0] - 1

        w = x2 - x1
        h = y2 - y1
        x1 = max(x1 - n*w, x1_min)
        y1 = max(y1 - n*h, y1_min)
        x2 = min(x2 + n*w, x2_max)
        y2 = min(y2 + n*h, y2_max)
    elif method == 'full' or method == 'full_no_ctr':
        x1 = 0
        y1 = 0
        y2 = img.shape[0] - 1
        x2 = img.shape[1] - 1
    elif method == "just_lmk":
        x1 = 0
        y1 = 0
        y2 = 0
        x2 = 0
    else:
        exit('Error in method selection...')

    return img, name_box_score[0], np.array([x1, y1, x2, y2]), lmk

""" Function run by each GPU process """
def segment_images(in_dir, gt_dir, boxes_path, method, out_seg_dir):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    torch.cuda.set_device(local_rank)

    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=local_rank 
        )

    device = f"cuda:{local_rank}"
    print(f"Rank {rank} (GPU {local_rank}) starting...")

    sam = sam_model_registry["vit_h"](checkpoint="SAM/sam_vit-h.pth").to(device=device)
    predictor = SamPredictor(sam)

    files_list = open(boxes_path)
    files = files_list.readlines()
    gpu_files = np.array_split(files, world_size)[rank]

    print("Evaluating SAM segmentation...")
    print(f"Rank {rank} processing {len(gpu_files)} samples.")

    IoU_list = []
    bckgrd_IoU_list = []
    precision_list = []
    recall_list = []
    dice_list = []
    for each_line in gpu_files:
        img, image_name, bbox, lmk = get_bounding_box(each_line, in_dir, method=method)

        if method == 'bb_no_ctr' or method == 'full_no_ctr' or method == "just_lmk":
            extra_points = lmk
            extra_points_labels = np.array([1, 1, 1, 1, 1])
        else:
            center = np.array([[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]])
            extra_points = np.vstack((lmk, center))
            extra_points_labels = np.array([1, 1, 1, 1, 1, 1])

        predictor.set_image(img)
        
        if method == "just_lmk":
            masks, _, _ = predictor.predict(
                point_coords=extra_points,
                point_labels=extra_points_labels,
                box=None,
                multimask_output=False,
            )
        else:
            masks, _, _ = predictor.predict(
                point_coords=extra_points,
                point_labels=extra_points_labels,
                box=bbox[None, :],
                multimask_output=False,
            )

        gt_mask = Image.open(gt_dir + "/" + image_name.replace(".jpg", "_mask.png")).convert("RGB") 
        gt_mask = np.array(gt_mask.resize((1024, 1024), resample=Image.NEAREST))
    
        img[masks[0]==False] = 0
        img[masks[0]==True] = 255
      
        # convert prediction and GT to binary mask
        pred = (img[..., 0] > 0).astype(np.uint8)
        gt = (gt_mask[..., 0] > 0).astype(np.uint8)

        # foreground metrics
        TP = np.sum((pred == 1) & (gt == 1))
        FP = np.sum((pred == 1) & (gt == 0))
        FN = np.sum((pred == 0) & (gt == 1))
        TN = np.sum((pred == 0) & (gt == 0))

        IoU = TP / (TP + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        dice = (2 * TP) / (2 * TP + FP + FN)

        # background metrics
        bTP = TN
        bFP = FN
        bFN = FP

        bckgrd_IoU = bTP / (bTP + bFP + bFN)
        
        IoU_list.append(IoU)
        bckgrd_IoU_list.append(bckgrd_IoU)
        precision_list.append(precision)
        recall_list.append(recall)
        dice_list.append(dice)
    
    print(f"Rank {rank} finished successfully!")
    print(f"Rank {rank}: {len(IoU_list)} IoU entries")
    dist.barrier() # waits for all the GPUS to finish their processing steps

    iou_sum = torch.tensor(sum(IoU_list), dtype=torch.float64, device=device)
    iou_count = torch.tensor(len(IoU_list), dtype=torch.float64, device=device)
    bckgrd_iou_sum = torch.tensor(sum(bckgrd_IoU_list), dtype=torch.float64, device=device)
    bckgrd_iou_count = torch.tensor(len(bckgrd_IoU_list), dtype=torch.float64, device=device)
    precision_sum = torch.tensor(sum(precision_list), dtype=torch.float64, device=device)
    precision_count = torch.tensor(len(precision_list), dtype=torch.float64, device=device)
    recall_sum = torch.tensor(sum(recall_list), dtype=torch.float64, device=device)
    recall_count = torch.tensor(len(recall_list), dtype=torch.float64, device=device)
    dice_sum = torch.tensor(sum(dice_list), dtype=torch.float64, device=device)
    dice_count = torch.tensor(len(dice_list), dtype=torch.float64, device=device)

    # sum across all GPUs
    dist.all_reduce(iou_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(iou_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(bckgrd_iou_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(bckgrd_iou_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(precision_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(precision_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(recall_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(recall_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(dice_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(dice_count, op=dist.ReduceOp.SUM)

    mIoU = (iou_sum / iou_count).item()
    mIoU_bckgrd = (bckgrd_iou_sum / bckgrd_iou_count).item()
    m_precision = (precision_sum / precision_count).item()
    m_recall = (recall_sum / recall_count).item()
    m_dice = (dice_sum / dice_count).item()

    if rank == 0:
        print("Foreground mIoU:", mIoU)
        print("Background mIoU:", mIoU_bckgrd)
        print("Average mIoU", (mIoU + mIoU_bckgrd) / 2)
        print("Precision:", m_precision)
        print("Recall:", m_recall)
        print("Dice:", m_dice)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTCNN alignment without Hydra")
    parser.add_argument("--boxes_path", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--gt", type=str)
    parser.add_argument("--method", type=str, default='full')
    parser.add_argument("--out_seg_dir", type=str)
    args = parser.parse_args()

    os.makedirs(args.out_seg_dir, exist_ok=True)

    segment_images(args.dataset, args.gt, args.boxes_path, args.method, args.out_seg_dir)