import argparse
import os
import sys
import pandas as pd
import statistics
import numpy as np
import matplotlib.pyplot as plt

from QualityModel import QualityModel

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--csv_dir', type=str, default='./data',
                        help='Root dir for csv file')
    parser.add_argument('--pairs', type=str, default='pairs.txt',
                        help='lfw pairs.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU id.')
    parser.add_argument('--model_path', type=str, default="/home/fboutros/LearnableMargin/output/ResNet50-COSQSArcFace_SmothL1",
                        help='path to pretrained evaluation.')
    parser.add_argument('--model_id', type=str, default="32572",
                        help='digit number in backbone file name')
    parser.add_argument('--backbone', type=str, default="iresnet50",
                        help=' iresnet100 or iresnet50 ')
    parser.add_argument('--score_file_name', type=str, default="quality_r50.txt",
                        help='score file name, the file will be store in the same data dir')
    parser.add_argument('--color_channel', type=str, default="BGR",
                        help='input image color channel, two option RGB or BGR')

    return parser.parse_args(argv)

def plot_scores(quality_scores, out_path):
    bins = np.linspace(min(quality_scores), max(quality_scores), 100)

    # compute histogram counts and convert to percentages
    quality_counts, _ = np.histogram(quality_scores, bins=bins)
    quality_counts = (quality_counts / quality_counts.sum()) * 100

    # plot histograms with percentage normalization
    plt.hist(bins[:-1], bins=bins, weights=quality_counts, alpha=0.6, color='blue', label="Quality Scores")

    # format y-axis to show percentages
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

    # labels and title
    plt.xlabel("!uality Score")
    plt.ylabel("Percentage of Instances")
    plt.title("Quality Scores Distribution")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()
    plt.savefig(out_path)

def read_image_list(image_list_file, image_dir=''):
    image_lists = []
    with open(image_list_file) as f:
        absolute_list=f.readlines()
        for l in absolute_list:
            image_lists.append(os.path.join(image_dir, l.rstrip()))
    return image_lists, absolute_list

def main(param):
    save_dir = os.path.join(('/').join(param.csv_dir.split('/')[0:5]), 'FIQA')
    os.makedirs(save_dir, exist_ok=True)

    face_model = QualityModel(param.model_path, param.model_id, param.gpu_id)
    
    if save_dir.find('IJBC') != -1:
        if param.csv_dir.split('/')[-1].find('_') == -1:
            img_folder = '/data/mcaldeir/IJB_release/IJBC/loose_crop'
        else:
            img_folder = '/data/mcaldeir/exit_entry/IJBC/original/' + (param.csv_dir.split('/')[-1]).split('.')[0]
        ldmk_file = '/data/mcaldeir/IJB_release/IJBC/meta/ijbc_name_5pts_score_retina.txt'

        _, quality = face_model.get_aligned_ijbc_feature(img_folder, ldmk_file, color=param.color_channel)
    else:
        dataframe = pd.read_csv(param.csv_dir)
        image_list = dataframe['image_path'].tolist()
        image_list = [img.replace('MAD_crop', 'aligned') for img in image_list]
        _, quality = face_model.get_batch_feature(image_list, batch_size=16, color=param.color_channel)

    quality = [x for xs in quality for x in xs]

    df = pd.DataFrame({'quality': quality})
    df.to_csv(os.path.join(save_dir, param.score_file_name), index=False)

    plot_scores(quality, os.path.join(save_dir, (param.csv_dir.split('/')[-1]).split('.')[0] + ".png"))

    print("Method:", (param.csv_dir.split('/')[-1]).split('.')[0])
    print("Mean quality:", statistics.mean(quality))
    print("Quality std:", statistics.stdev(quality))

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))