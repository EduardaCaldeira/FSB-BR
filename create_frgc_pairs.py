import os
import random
import cv2

def copy_all_images(root_folder, target_folder, target_format="jpg"):
    os.makedirs(target_folder, exist_ok=True)
    count = 0

    for file_name in os.listdir(root_folder):
        src_path = os.path.join(root_folder, file_name)
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
    print(f"Copied {count} images to '{target_folder}'.")

def create_pairs(root_folder):
    pos_pairs = []
    neg_pairs = []

    file_list = os.listdir(root_folder)
    for idx, file_1 in enumerate(file_list):
        for file_2 in file_list[idx+1:]:
            id_1 = file_1.split("d")[0]
            id_2 = file_2.split("d")[0]

            if id_1 == id_2:
                pos_pairs.append(f"{file_1} {file_2} 1\n")
            else:
                neg_pairs.append(f"{file_1} {file_2} -1\n") 

    random.seed(42) 
    random.shuffle(neg_pairs) # shuffle the negative pairs to randomize their order
    neg_pairs = neg_pairs[:int(len(neg_pairs)/100)] # keep only 1% of the negative pairs
    all_pairs = pos_pairs + neg_pairs

    return all_pairs, len(pos_pairs), len(neg_pairs)
    
if __name__ == "__main__":
    output_folder = "/data/mcaldeir/exit_entry/frgc/"
    pairs_folder = output_folder + "embeddings"
    os.makedirs(pairs_folder, exist_ok=True)
    output_file = pairs_folder + "/pairs.txt"

    all_pairs, len_pos_pairs, len_neg_pairs = create_pairs(output_folder + "MAD_crop/frgc")

    random.seed(42) 
    random.shuffle(all_pairs)
    
    with open(output_file, "w") as f:
        f.writelines(all_pairs)

    print(f"Created {len_pos_pairs} positive pairs and {len_neg_pairs} negative pairs in '{output_file}'.")
    print(f"Quick final len check {len(all_pairs)}")