import pandas as pd
import os

def consolidate_specific_csvs(folder_path, file_list):
    all_dataframes = []

    for filename in file_list:
        file_path = os.path.join(folder_path, filename)
        
        # Check if the file exists before trying to read it
        if os.path.exists(file_path):
            print(f"Reading: {filename}")
            df = pd.read_csv(file_path)
            all_dataframes.append(df)
        else:
            print(f"Warning: {filename} not found in {folder_path}.")

    if not all_dataframes:
        return None

    # 1. Stack all dataframes together
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # 2. Remove duplicates
    # This removes rows where all column values are identical to a previous row
    clean_df = combined_df.drop_duplicates()

    return clean_df

main_path = '/data/mcaldeir/FaceMAD/Protocols'
files_to_open = ['Webmorph.csv', 'OpenCV.csv', 'MorDIFF.csv', 'MIPGAN_I.csv', 'MIPGAN_II.csv', 'FaceMorpher.csv']

df_result = consolidate_specific_csvs(main_path, files_to_open)

if df_result is not None:
    df_result.to_csv('/data/mcaldeir/FaceMAD/Protocols/SYN_MAD22.csv', index=False)