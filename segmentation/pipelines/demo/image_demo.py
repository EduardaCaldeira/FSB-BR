# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser

from mmcv.cnn.utils.sync_bn import revert_sync_batchnorm
from mmseg.apis import inference_segmentor, init_segmentor, generate_final_image

def main():
    parser = ArgumentParser()
    parser.add_argument('dataset_path', help='dataset path')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out_dir', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    palette = [[255, 255, 255], [0, 0, 0]]

    for image_path in os.listdir(args.dataset_path):
        if image_path.endswith(".jpg") or image_path.endswith(".ppm"):
            result = inference_segmentor(model, os.path.join(args.dataset_path, image_path))
            
            generate_final_image(
                model,
                os.path.join(args.dataset_path, image_path),
                result,
                palette=palette,
                out_file=os.path.join(args.out_dir, image_path))

if __name__ == '__main__':
    main()