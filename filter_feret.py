import os
import shutil

def copy_filtered_files(src_root, dst_root, keywords=("fa", "fb")):
    total_files = 0
    saved_files = 0

    for root, _, files in os.walk(src_root):
        rel_path = os.path.relpath(root, src_root)
        dst_dir = os.path.join(dst_root, rel_path)

        for file in files:
            total_files += 1

            if any(k in file for k in keywords):
                os.makedirs(dst_dir, exist_ok=True)

                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_dir, file)

                shutil.copy2(src_file, dst_file)
                saved_files += 1

    print(f"Total files found: {total_files}")
    print(f"Total files saved: {saved_files}")


if __name__ == "__main__":
    source_folder = r"/data/Biometrics/benchmarks_before_cropping/feret_db/smaller"
    destination_folder = r"/data/Biometrics/benchmarks_before_cropping/feret_db_frontal"

    copy_filtered_files(source_folder, destination_folder)