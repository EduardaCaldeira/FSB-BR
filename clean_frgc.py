import os
import argparse

def remove_files(file_list_path, in_dir, network):
    emb_extensions = ["ElasticCos", "ms1mv3_arcface_r100_fp16", "TransFace-L", "SwinFace"]
    
    if network == 'none':
        complete_net = 'frgc'
    else:
        complete_net = 'frgc_' + network

    mad_dir = os.path.join(in_dir, "MAD_crop", complete_net)

    with open(file_list_path, 'r') as f:
        filenames = [line.strip() for line in f if line.strip()]

    for name in filenames:
        mad_path = os.path.join(mad_dir, name)
        if os.path.exists(mad_path):
            try:
                os.remove(mad_path)
                print(f"Removed from MAD_dir: {mad_path}")
            except Exception as e:
                print(f"Error removing {mad_path}: {e}")
        else:
            print(f"Not found in MAD_dir: {mad_path}")

        base, _ = os.path.splitext(name)
        emb_name = base + ".pt"

        for extension in emb_extensions:
            emb_path = os.path.join(in_dir, "embeddings", extension, complete_net, emb_name)

            if os.path.exists(emb_path):
                try:
                    os.remove(emb_path)
                    print(f"Removed from emb_dir: {emb_path}")
                except Exception as e:
                    print(f"Error removing {emb_path}: {e}")
            else:
                print(f"Not found in emb_dir: {emb_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove files from two directories.")
    parser.add_argument("--file_list", help="Path to txt file with filenames")
    parser.add_argument("--in_dir")
    parser.add_argument("--network")

    args = parser.parse_args()
    remove_files(args.file_list, args.in_dir, args.network)