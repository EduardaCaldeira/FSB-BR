import os
import random
import cv2

def copy_all_images(root_folder, target_folder, target_format="jpg"):
    """
    Copies all images from subfolders in root_folder to target_folder,
    ignoring subfolder structure, and converts them to target_format using OpenCV.
    """
    os.makedirs(target_folder, exist_ok=True)
    count = 0

    for subdir_name in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir_name)
        if not os.path.isdir(subdir_path):
            continue

        for file_name in os.listdir(subdir_path):
            src_path = os.path.join(subdir_path, file_name)
            if os.path.isfile(src_path):
                # Read the image
                img = cv2.imread(src_path)
                if img is None:
                    print(f"Failed to read {src_path}")
                    continue

                # Construct new filename with target extension
                base_name = os.path.splitext(file_name)[0]
                dst_path = os.path.join(target_folder, f"{base_name}.{target_format}")

                # Write image in target format
                success = cv2.imwrite(dst_path, img)
                if not success:
                    print(f"Failed to write {dst_path}")
                    continue

                count += 1

    print(f"Copied and converted {count} images to '{target_folder}' in {target_format.upper()} format.")
def create_pos_pairs(root_folder):
    """
    For each subfolder in root_folder, find images containing 'fa' and 'fb' in the name,
    pair each 'fa' with each 'fb', and write the pairs with label 1 to output_file
    """
    pos_pairs = []

    # iterate over all subfolders
    for subdir_name in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir_name)
        if not os.path.isdir(subdir_path):
            continue

        # find fa and fb images
        fa_images = []
        fb_images = []
        for file_name in os.listdir(subdir_path):
            if "fa" in file_name:
                fa_images.append(file_name)
            elif "fb" in file_name:
                fb_images.append(file_name)

        # pair each fa with each fb
        for fa in fa_images:
            for fb in fb_images:
                pos_pairs.append(f"{fa.replace('ppm', 'jpg')} {fb.replace('ppm', 'jpg')} 1\n")

    return pos_pairs
   
def create_neg_pairs(root_folder):
    """
    For each image in all subfolders, create negative pair with a all the images
    of the opposite type ('fa' vs 'fb') from a different folder, and label -1.
    """
    # collect fa and fb images by folder
    folder_fa_images = {}
    folder_fb_images = {}
    for subdir_name in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir_name)
        if not os.path.isdir(subdir_path):
            continue

        fa_images = [f for f in os.listdir(subdir_path) if "fa" in f and os.path.isfile(os.path.join(subdir_path, f))]
        fb_images = [f for f in os.listdir(subdir_path) if "fb" in f and os.path.isfile(os.path.join(subdir_path, f))]

        if fa_images:
            folder_fa_images[subdir_path] = fa_images
        if fb_images:
            folder_fb_images[subdir_path] = fb_images

    neg_pairs = []

    # go over all fa images
    for folder, fa_images in folder_fa_images.items():
        other_fb_folders = [f for f in folder_fb_images.keys() if f != folder]
        if not other_fb_folders:
            continue
        for fa_img in fa_images:
            for other_fb_folder in other_fb_folders:
                for other_fb_img in folder_fb_images[other_fb_folder]:
                    neg_pairs.append(f"{fa_img.replace('ppm', 'jpg')} {other_fb_img.replace('ppm', 'jpg')} -1\n")

    return neg_pairs

if __name__ == "__main__":
    folder = "/data/Biometrics/benchmarks_before_cropping/feret_db_frontal"
    output_folder = "/data/mcaldeir/exit_entry/feret/"
    pairs_folder = output_folder + "embeddings"
    os.makedirs(pairs_folder, exist_ok=True)
    output_file = pairs_folder + "/pairs.txt"

    pos_pairs = create_pos_pairs(folder)
    neg_pairs = create_neg_pairs(folder)

    all_pairs = pos_pairs + neg_pairs
    random.seed(42) 
    random.shuffle(all_pairs)
    
    with open(output_file, "w") as f:
        f.writelines(all_pairs)

    print(f"Created {len(pos_pairs)} postive pairs and {len(neg_pairs)} negative pairs in '{output_file}'.")