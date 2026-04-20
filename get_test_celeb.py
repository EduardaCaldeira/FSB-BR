import os
import pandas as pd
import shutil

def process_file(table_path, filename_list, in_dir, out_dir):
    """
    table_path: path to file containing idx, orig_idx, orig_file
    filename_list: list of filenames to match against orig_file
    in_dir: directory containing source images
    out_dir: directory to save selected images
    """

    # load table (auto-detects whitespace separated like your example)
    df = pd.read_csv(table_path, sep=r"\s+", engine="python")

    # ensure filename list is a set for fast lookup
    filename_set = set(filename_list)

    for _, row in df.iterrows():
        orig_file = row["orig_file"]

        # check if last column value is in provided list
        if orig_file in filename_set:

            # build source filename from second column
            src_filename = f"{int(row['idx'])}.jpg"  # keeps leading zeros style if needed

            src_path = os.path.join(in_dir, src_filename)
            dst_path = os.path.join(out_dir, src_filename)

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
            else:
                print(f"Missing file: {src_path}")

if __name__ == "__main__":
    table_path = "/data/mcaldeir/exit_entry/CelebAMask-HQ/CelebA-HQ-to-CelebA-mapping.txt"

    filename_list = [str(i) + ".jpg" for i in range(182638, 202600)]

    in_dir = "/data/mcaldeir/exit_entry/CelebAMask-HQ/CelebA-HQ-img"
    out_dir = "/data/mcaldeir/exit_entry/CelebAMask-HQ/CelebA-HQ-img-test"
    os.makedirs(out_dir, exist_ok=True)

    process_file(table_path, filename_list, in_dir, out_dir)