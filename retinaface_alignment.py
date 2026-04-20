import os
import cv2
import numpy as np

if not hasattr(np, 'bool'):
    np.bool = bool 

from tqdm import tqdm
import argparse
from os.path import join as ojoin
from torch.utils.data import Dataset, DataLoader
from utils.align_trans import norm_crop
from retinaface import RetinaFace

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

        img = cv2.imread(self.img_paths[index])
        if img is None:
            print(f"Warning: Unable to read image {img_path}")
            return None, None 

        return img, img_file

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.img_paths)

    def is_valid_image(self, file_path):
        return os.path.splitext(file_path)[-1].lower()==".png" or os.path.splitext(file_path)[-1].lower()==".jpg"

def list_collate(batch):
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return [], []
    
    imgs = [item[0] for item in batch]
    names = [item[1] for item in batch]
    return imgs, names

def align_images(folder, extra_path, batchsize, num_imgs=0, evalDB=False):
    """MTCNN alignment for all images in complete_in_folder and save to complete_out_folder"""

    out_dir_MAD = os.path.join(folder, "MAD_crop", extra_path)
    out_dir = os.path.join(folder, "aligned", extra_path)
    
    in_dir = os.path.join(folder, "original", extra_path)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_MAD, exist_ok=True)

    is_folder = is_folder_structure(in_dir)
    train_dataset = InferenceDataset(
        in_dir, num_imgs=num_imgs, folder_structure=is_folder
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batchsize, shuffle=False, drop_last=False, num_workers=4, collate_fn=list_collate
    )

    skipped_imgs = []
    empty_files = []
    new_file = True
    for img_batch, img_names in tqdm(train_loader):
        for img, img_name in zip(img_batch, img_names):
            im_scale = 1.0
            scales = [im_scale]

            if img is None or not (img_name.lower().endswith(".png") or img_name.lower().endswith(".jpg")):
                empty_files.append(img_name)
                continue

            boxes, landmark = retina_face.detect(img, thresh, scales=scales, do_flip=False)

            out_path = out_dir
            out_path_MAD = out_dir_MAD
            if is_folder:
                id_dir = os.path.split(img_name)[0]
                out_path = ojoin(out_dir, id_dir)
                out_path_MAD = ojoin(out_dir_MAD, id_dir)
                img_name = os.path.split(img_name)[1]

            if landmark is None or boxes.shape[0] == 0: # no face recognized, crop to 112x112
                skipped_imgs.append(img_name)
            else: 
                # MAD cropping
                x1, y1, x2, y2, _ = boxes[0]

                n = 0.05 # add margin for cropping
                x1_min = 0; x2_max = img.shape[1] - 1
                y1_min = 0; y2_max = img.shape[0] - 1

                w = x2 - x1
                h = y2 - y1
                x1 = max(x1 - n*w, x1_min)
                y1 = max(y1 - n*h, y1_min)
                x2 = min(x2 + n*w, x2_max)
                y2 = min(y2 + n*h, y2_max)

                if extra_path.find('_') == -1:
                    if new_file:
                        mode = "w"
                    else:
                        mode = "a"
                    with open(folder + "/" + extra_path + "_bb.txt", mode, encoding="utf-8") as file:
                        lmk = landmark[0].reshape(-1)
                        str_list = [img_name, str(1), str(x1), str(y1), str(x2), str(y2), str(1)] + [str(x) for x in lmk] +[str(1), '\n']
                        file.write(",".join(str_list))
                        new_file = False
                
                crop_img_MAD = img[round(y1):round(y2), round(x1):round(x2)]
                cv2.imwrite(os.path.join(out_path_MAD, img_name), crop_img_MAD)
                
                # alignement + cropping to 112x112
                facial5points = np.array(landmark[0], dtype=np.float64)
                warped_face = norm_crop(
                    img, landmark=facial5points, image_size=112, createEvalDB=evalDB
                )
                
                cv2.imwrite(os.path.join(out_path, img_name), warped_face)
    print(skipped_imgs)
    print(f"Empty files: {len(empty_files)}")
    print(f"Images with no face or boxes: {len(skipped_imgs)}")

    if folder.find("frgc") != -1:
        with open(folder + '/' + extra_path + '.txt', 'w+') as f:
            for skipped_img in skipped_imgs:
                f.write('%s\n' %skipped_img)
        print("File written successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTCNN alignment without Hydra")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--network", type=str)
    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--evalDB", action="store_true", default=1, help="Set for eval DB alignment")
    parser.add_argument("--num_imgs", type=int, default=0, help="Max images to align (0 for all)")

    args = parser.parse_args()

    folder = "../../../../data/mcaldeir/exit_entry/" + args.dataset
    extra_path = args.dataset
    if args.network != "none":
        extra_path = extra_path + "_" + args.network

    align_images(
        folder,
        extra_path,
        args.batchsize,
        num_imgs=args.num_imgs,
        evalDB=args.evalDB
    )