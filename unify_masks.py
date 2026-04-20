import os
import glob
import numpy as np
from PIL import Image
from collections import defaultdict

def get_images_by_id(folders):
    """Receives a list of folders and returns a dictionary mapping IDs to their images."""
    images_by_id = defaultdict(list)
    
    for folder in folders:
        image_paths = glob.glob(os.path.join(folder, '*.png'))
        
        for image_path in image_paths:
            # extract the file ID (e.g., '00000' from '00000_placeholder.png')
            file_name = os.path.basename(image_path)
            file_id = file_name.split('_')[0]  
            
            # append image path to the corresponding ID
            images_by_id[file_id].append(image_path)
    
    return images_by_id

def create_union_mask(images):
    """Creates the union mask of the provided binary images."""
    masks = [np.array(Image.open(img)) for img in images]
    union_mask = np.any(masks, axis=0).astype(np.uint8) * 255
    return Image.fromarray(union_mask)

def save_mask(mask, save_folder, mask_name):
    """Saves the generated mask in the specified folder with the given name."""
    os.makedirs(save_folder, exist_ok=True) 
    mask.save(os.path.join(save_folder, mask_name))

def process_folders(folders, save_folder):
    """Main function to process folders, generate masks, and save them."""
    images_by_id = get_images_by_id(folders)
    
    for file_id, image_paths in images_by_id.items():
        # create the union of the binary images
        union_mask = create_union_mask(image_paths)
        
        # save the mask with the file_id (e.g., '00000_mask.png')
        mask_name = f"{str(int(file_id))}_mask.png"
        save_mask(union_mask, save_folder, mask_name)
    
    print(f"Saved masks at {save_folder}")

if __name__ == "__main__":
    input_folders = ["/data/mcaldeir/exit_entry/CelebAMask-HQ/CelebAMask-HQ-mask-anno/" + str(i) for i in range(15)]
    output_folder = "/data/mcaldeir/exit_entry/CelebAMask-HQ/CelebAMask-unified-masks"
    process_folders(input_folders, output_folder)