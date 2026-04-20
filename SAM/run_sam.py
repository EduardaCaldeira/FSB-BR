import os
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool 
import argparse
from PIL import Image

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
    elif method == 'bb_extra':
        x1_min = 0; x2_max = img.shape[1] - 1
        y1_min = 0
        
        if x2_max == 1703:
            x1 = 0
            y1 = 0
            y2 = img.shape[0] - 1
            x2 = img.shape[1] - 1
        else:
            x1, y1, x2, y2 = np.array([float(x) for x in name_box_score[2:6]], dtype=np.float32)
            n_w = 1.5 
            n_h = 0.15
            
            w = x2 - x1
            h = y2 - y1
            x1 = max(x1 - n_w*w, x1_min)
            x2 = min(x2 + n_w*w, x2_max)
            y1 = max(y1 - n_h*h, y1_min)
            y2 = img.shape[0] - 1
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
def segment_images(in_dir, out_dir_segmentation, boxes_path, method):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    device = f"cuda:{local_rank}"
    print(f"Rank {rank} (GPU {local_rank}) starting...")

    sam = sam_model_registry["vit_h"](checkpoint="SAM/sam_vit-h.pth").to(device=device)
    predictor = SamPredictor(sam)

    files_list = open(boxes_path)
    files = files_list.readlines()
    gpu_files = np.array_split(files, world_size)[rank]

    print("Segmenting with SAM...")
    print(f"Rank {rank} processing {len(gpu_files)} samples.")
    for each_line in gpu_files:
        img, image_name, bbox, lmk = get_bounding_box(each_line, in_dir, method=method)

        if method == 'bb_no_ctr' or method == 'full_no_ctr' or method == "just_lmk" or method == 'bb_extra':
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
    
        img[masks[0]==False] = 255

        if not isinstance(img, Image.Image):
            img = Image.fromarray(img.astype(np.uint8))
        img.save(os.path.join(out_dir_segmentation, image_name))
   
    print(f"Rank {rank} finished successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTCNN alignment without Hydra")
    parser.add_argument("--boxes_path", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--folder", type=str)
    parser.add_argument("--method", type=str, default='full')
    args = parser.parse_args()

    in_dir = args.folder
    out_dir_segmentation = args.out_dir + "_sam_" + args.method
    if int(os.environ.get("RANK", 0)) == 0:
        os.makedirs(out_dir_segmentation, exist_ok=True)

    segment_images(in_dir, out_dir_segmentation, args.boxes_path, args.method)