# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser

from mmcv.cnn.utils.sync_bn import revert_sync_batchnorm
from mmseg.apis import inference_segmentor, init_segmentor, segmentation_eval

from PIL import Image
import numpy as np

def main():
    parser = ArgumentParser()
    parser.add_argument('dataset_path', help='dataset path')
    parser.add_argument('gt_path', help='dataset path')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    out_dir = "/data/mcaldeir/exit_entry/CelebAMask-HQ/segmentation/" + (args.checkpoint.split("/")[-1]).split(".")[0]
    os.makedirs(out_dir, exist_ok=True)

    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    palette = [[0, 0, 0], [255, 255, 255]]

    IoU_list = []
    bckgrd_IoU_list = []
    precision_list = []
    recall_list = []
    dice_list = []
    
    for image_path in os.listdir(args.dataset_path):
        if image_path.endswith(".jpg") or image_path.endswith(".ppm"):
            result = inference_segmentor(model, os.path.join(args.dataset_path, image_path))
            gt_mask = Image.open(os.path.join(args.gt_path, image_path.replace('.jpg', '_mask.png')))
            gt_mask = gt_mask.resize((1024, 1024), resample=Image.NEAREST)
            
            IoU, bckgrd_IoU, precision, recall, dice = segmentation_eval(
                model,
                os.path.join(args.dataset_path, image_path),
                result,
                gt_mask=np.array(gt_mask).astype(np.uint8),
                out_file=os.path.join(out_dir, image_path),
                palette=palette)
            
            IoU_list.append(IoU)
            bckgrd_IoU_list.append(bckgrd_IoU)
            precision_list.append(precision)
            recall_list.append(recall)
            dice_list.append(dice)

    mIoU = sum(IoU_list) / len(IoU_list)
    mIoU_bckgrd = sum(bckgrd_IoU_list) / len(bckgrd_IoU_list)
    m_precision = sum(precision_list) / len(precision_list)
    m_recall = sum(recall_list) / len(recall_list)
    m_dice = sum(dice_list) / len(dice_list)

    print("Mean Intersection over Union (mIoU): ", mIoU)
    print("Background IoU: ", mIoU_bckgrd)
    print("Mean IoU (fg + bg): ", (mIoU + mIoU_bckgrd) / 2)
    print("Precision: ", m_precision)
    print("Recall: ", m_recall)
    print("Dice (F1): ", m_dice)

    with open("/data/mcaldeir/exit_entry/" + args.checkpoint.split('/')[-1].replace('.pth', '.txt'), 'w') as f:
        f.write('mIoU: ' + str(mIoU) + '\n')
        f.write(f"Background IoU: {mIoU_bckgrd}\n")
        f.write(f"Mean IoU (fg + bg): {(mIoU + mIoU_bckgrd) / 2}\n")
        f.write(f"Precision: {m_precision}\n")
        f.write(f"Recall: {m_recall}\n")
        f.write(f"Dice (F1): {m_dice}\n")

if __name__ == '__main__':
    main()