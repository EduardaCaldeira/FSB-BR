import os
import pandas as pd
import argparse

# -----------------------------
# Parse command-line arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Create an Excel file with image paths and labels.")
parser.add_argument(
    "--image_dir",
    type=str,
    required=True,
    help="Path to the folder containing images"
)
parser.add_argument(
    "--output_excel",
    type=str,
    default="images_labels.xlsx",
    help="Path to save the output Excel file (default: images_labels.xlsx)"
)
parser.add_argument(
    "--network",
    type=str,
    default="none",
)
args = parser.parse_args()

# -----------------------------
# Gather all image paths in the directory (no subdirectories)
# -----------------------------
image_extensions = (".png", ".jpg", ".jpeg")

if args.network != "none":
    args.image_dir = args.image_dir + "_" + args.network

image_paths = [
    os.path.join(args.image_dir, f)
    for f in os.listdir(args.image_dir)
    if f.lower().endswith(image_extensions) and os.path.isfile(os.path.join(args.image_dir, f))
]

# -----------------------------
# Create DataFrame
# -----------------------------
df = pd.DataFrame({
    "image_path": image_paths,
    "label": ["bonafide"] * len(image_paths)
})

# -----------------------------
# Save to Excel
# -----------------------------
df.to_csv(args.output_excel, index=False)
print(f"Excel file saved as '{args.output_excel}' with {len(df)} entries.")