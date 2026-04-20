import os
import numpy as np

if not hasattr(np, 'bool'):
    np.bool = bool 

import argparse
from os.path import join as ojoin
from torch.utils.data import Dataset
from retinaface import RetinaFace

from PIL import Image

thresh = 0.8
target_size = 112
gpuid = -1

retina_face = RetinaFace('./align/R50', 0, gpuid, 'net3')

def load_syn_paths(datadir, num_imgs=0):
    img_files = sorted(os.listdir(datadir))
    img_files = img_files if num_imgs == 0 else img_files[:num_imgs]
    return [ojoin(datadir, f_name) for f_name in img_files]


def load_real_paths(datadir, num_imgs=0):
    img_paths = []
    id_folders = sorted(os.listdir(datadir))
    for id in id_folders:
        img_files = sorted(os.listdir(ojoin(datadir, id)))
        img_paths += [ojoin(datadir, id, f_name) for f_name in img_files]
    img_paths = img_paths if num_imgs == 0 else img_paths[:num_imgs]
    return img_paths


def is_folder_structure(datadir):
    """checks if datadir contains folders (like CASIA) or images (synthetic datasets)"""
    img_path = sorted(os.listdir(datadir))[0]
    img_path = ojoin(datadir, img_path)
    return os.path.isdir(img_path)


class InferenceDataset(Dataset):
    def __init__(self, datadir, num_imgs=0, folder_structure=False):
        """Initializes image paths"""
        self.folder_structure = folder_structure

        if self.folder_structure:
            self.img_paths = load_real_paths(datadir, num_imgs)
        else:
            self.img_paths = load_syn_paths(datadir, num_imgs)

        self.img_paths = [path for path in self.img_paths if self.is_valid_image(path)]
        print("Amount of images:", len(self.img_paths))

    def __getitem__(self, index):
        """Reads an image from a file and corresponding label and returns."""
        img_path = self.img_paths[index]
        img_file = os.path.split(img_path)[-1]
        if self.folder_structure:
            tmp = os.path.dirname(img_path)
            img_file = ojoin(os.path.basename(tmp), img_file)

        img = Image.open(self.img_paths[index])
        img = np.array(img) 
        if img is None:
            print(f"Warning: Unable to read image {img_path}")
            return None, None 

        return img, img_file

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.img_paths)

    def is_valid_image(self, file_path):
        return os.path.splitext(file_path)[-1].lower()==".png" or os.path.splitext(file_path)[-1].lower()==".jpg"

def crop_images(folder, out_dir, extra_path_in, extra_path_out, boxes_file):
    in_dir = os.path.join(folder, extra_path_in)
    out_dir_MAD = os.path.join(out_dir, extra_path_out)

    print(out_dir_MAD)
    os.makedirs(out_dir_MAD, exist_ok=True)

    files_list = open(boxes_file)
    files = files_list.readlines()

    print("Cropping the data for MAD...")

    for idx, each_line in enumerate(files):
        name_box_score = each_line.strip().split(',')
        img_name = os.path.join(in_dir, name_box_score[0])
        img = Image.open(img_name)
        img = np.array(img) 
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

        crop_img_MAD = img[round(y1):round(y2), round(x1):round(x2)]
        if not isinstance(crop_img_MAD, Image.Image):
            crop_img_MAD = Image.fromarray(crop_img_MAD.astype(np.uint8))
        crop_img_MAD.save(os.path.join(out_dir_MAD, name_box_score[0]))
    
    print("Data cropped successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTCNN alignment without Hydra")
    parser.add_argument("--boxes_path", type=str)
    parser.add_argument("--network", type=str)
    args = parser.parse_args()

    out_dir = "/data/mcaldeir/exit_entry/IJBC"

    if args.network == "none":
        folder = "/data/mcaldeir/IJB_release/IJBC"
        extra_path_in = "loose_crop"
        extra_path_out = "MAD_crop/IJBC"
    else:
        folder = "/data/mcaldeir/exit_entry/IJBC"
        extra_path_in = "original/IJBC_"+ args.network
        extra_path_out = "MAD_crop/IJBC_"+ args.network
    
    crop_images(
        folder,
        out_dir,
        extra_path_in,
        extra_path_out,
        args.boxes_path
    )