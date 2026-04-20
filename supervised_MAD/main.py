
import os
import copy
import cv2
import pandas as pd
import numpy as np
import csv
import logging
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_curve, auc
from utils import get_eer_threhold, get_bpcer_op, get_apcer_op
from backbones import mixnet_s

device = torch.device('cuda:0')

PRE__MEAN = [0.5, 0.5, 0.5]
PRE__STD = [0.5, 0.5, 0.5]
INPUT_SIZE = 224
EarlyStopPatience = 20

class FaceDataset(Dataset):
    def __init__(self, file_name, is_train):
        self.data = pd.read_csv(file_name)
        self.is_train = is_train
        self.train_transform = transforms.Compose(
            [
             transforms.ToPILImage(),
                transforms.Resize([INPUT_SIZE, INPUT_SIZE]),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean=PRE__MEAN,
                                 std=PRE__STD),
             ])

        self.test_transform = transforms.Compose(
            [           transforms.ToPILImage(),
                transforms.Resize([INPUT_SIZE, INPUT_SIZE]),
             transforms.ToTensor(),
             transforms.Normalize(mean=PRE__MEAN,
                                 std=PRE__STD),
             ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data.iloc[index, 0]
        label_str = self.data.iloc[index, 1]
        label = 1 if label_str == 'bonafide' else 0

        image=cv2.imread(image_path)
        try:
            if self.is_train:
                image = self.train_transform(image)
            else:
                image = self.test_transform(image)
        except ValueError:
            print(image_path)

        return image, label, image_path

def find_synmad_thresholds(prediction_scores, gt_labels, verbose=False):
    # Calculate ROC curve metrics
    fpr, tpr, threshold = roc_curve(gt_labels, prediction_scores, pos_label=1)
    bpcer = 1 - tpr
    apcer = fpr
    
    # Get EER
    eer, _, _ = get_eer_threhold(fpr, tpr, threshold)

    # Calculate APCER at fixed BPCER points
    _, apcer_bpcer20, _ = get_bpcer_op(apcer, bpcer, threshold, 0.20)
    _, apcer_bpcer10, _ = get_bpcer_op(apcer, bpcer, threshold, 0.10)
    _, apcer_bpcer1, _ = get_bpcer_op(apcer, bpcer, threshold, 0.01)

    # Calculate BPCER at fixed APCER points
    _, bpcer_apcer20, thr_bpcer_apcer20 = get_apcer_op(apcer, bpcer, threshold, 0.20)
    _, bpcer_apcer10, thr_bpcer_apcer10 = get_apcer_op(apcer, bpcer, threshold, 0.10)
    _, bpcer_apcer1, thr_bpcer_apcer1 = get_apcer_op(apcer, bpcer, threshold, 0.01)

    # Calculate AUC
    auc_score = auc(fpr, tpr)

    results = {
        "auc_score": auc_score,
        "eer": eer,
        "apcer_bpcer20": apcer_bpcer20,
        "apcer_bpcer10": apcer_bpcer10,
        "apcer_bpcer1": apcer_bpcer1,
        "bpcer_apcer20": bpcer_apcer20,
        "bpcer_apcer10": bpcer_apcer10,
        "bpcer_apcer1": bpcer_apcer1,
    }

    if verbose:
        print("\nMAD Performance Metrics:")
        print(f"AUC: {results['auc_score']:.4f}")
        print(f"EER: {results['eer']:.4f}")
        print("\nAPCER at fixed BPCER:")
        print(f"APCER@BPCER20%: {results['apcer_bpcer20']:.4f}")
        print(f"APCER@BPCER10%: {results['apcer_bpcer10']:.4f}")
        print(f"APCER@BPCER1%: {results['apcer_bpcer1']:.4f}")
        print("\nBPCER at fixed APCER:")
        print(f"BPCER@APCER20%: {results['bpcer_apcer20']:.4f}")
        print(f"BPCER@APCER10%: {results['bpcer_apcer10']:.4f}")
        print(f"BPCER@APCER1%: {results['bpcer_apcer1']:.4f}")

    print("Important thresholds:")
    print(f"thr_bpcer_apcer20:{thr_bpcer_apcer20}")
    print(f"thr_bpcer_apcer10:{thr_bpcer_apcer10}")
    print(f"thr_bpcer_apcer1:{thr_bpcer_apcer1}")

    return results

# this function assumes that all the samples are bonafide, due to the specific use case being analyzed
def evaluate_mad_performance(prediction_scores, verbose=False):
    thr_20 = 0.8891923961685699
    thr_10 = 0.8900435246161076
    thr_1 = np.inf

    prediction_scores = np.array(prediction_scores)
    bpcer_apcer20 = np.sum(prediction_scores<=thr_20)/len(prediction_scores)
    bpcer_apcer10 = np.sum(prediction_scores<=thr_10)/len(prediction_scores)
    bpcer_apcer1 = np.sum(prediction_scores<=thr_1)/len(prediction_scores)

    results = {
        "bpcer_apcer20": bpcer_apcer20,
        "bpcer_apcer10": bpcer_apcer10,
        "bpcer_apcer1": bpcer_apcer1,
    }

    if verbose:
        print("\nMAD Performance Metrics:")
        print("\nBPCER at fixed APCER:")
        print(f"BPCER@APCER20%: {results['bpcer_apcer20']:.4f}")
        print(f"BPCER@APCER10%: {results['bpcer_apcer10']:.4f}")
        print(f"BPCER@APCER1%: {results['bpcer_apcer1']:.4f}")

    return results

def run_test(test_loader, model, model_path, output_path, method):
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    prediction_scores, gt_labels, test_scores_dict = [], [], []
    with torch.no_grad():
        for inputs, labels, image_paths in tqdm(test_loader):
            inputs, labels= inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)

            for i in range(probs.shape[0]):
                prediction_scores.append(float(probs[i][1].detach().cpu().numpy()))
                gt_labels.append(int(labels[i].detach().cpu().numpy()))
                test_scores_dict.append({'image_path': image_paths[i], 'label': labels[i].item(), 'prediction_score': float(prediction_scores[i])})

        if method == 'eval':
            std_value = 0.44027915878737844
            mean_value = 0.6081200214785378
        else:
            std_value = np.std(prediction_scores)
            mean_value = np.mean(prediction_scores)

        prediction_scores = [(float(i) - mean_value) /(std_value) for i in prediction_scores]

    with open(output_path, mode='w') as csv_file:
        fieldnames = ['image_path', 'label', 'prediction_score']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for d in test_scores_dict:
            writer.writerow(d)
        print('Prediction scores writen done in', output_path)
    
    results = []
    if method == 'threshold':
        results.append(find_synmad_thresholds(prediction_scores, gt_labels))
        print('mean:', mean_value)
        print('std:', std_value)
    else:
        results.append(evaluate_mad_performance(prediction_scores))

    with open(os.path.join(output_path.replace("csv", "txt")), "w") as f:
        f.write(f"BPCER@APCER20%: {results[0]['bpcer_apcer20']:.4f}\n")
        f.write(f"BPCER@APCER10%: {results[0]['bpcer_apcer10']:.4f}\n")
        f.write(f"BPCER@APCER1%: {results[0]['bpcer_apcer1']:.4f}\n")
    return

def main(args):
    model = mixnet_s(embedding_size=128, width_scale=1.0, gdw_size=1024, shuffle=False)

    test_dataset = FaceDataset(file_name=args.test_csv_path, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    run_test(test_loader=test_loader, model=model, model_path=args.model_path, output_path=args.output_dir, method=args.method)

if __name__ == '__main__':

    cudnn.benchmark = True

    if torch.cuda.is_available():
        print('GPU is available')
        torch.cuda.manual_seed(0)
    else:
        print('GPU is not available')
        torch.manual_seed(0)

    import argparse
    parser = argparse.ArgumentParser(description='MixFaceNet model')
    parser.add_argument("--train_csv_path", default="dataset_info/train.csv", type=str, help="input path of train csv")
    parser.add_argument("--test_csv_path", default="dataset_info/test.csv", type=str, help="input path of test csv")

    parser.add_argument("--output_dir", default="output", type=str, help="path where trained model and test results will be saved")
    parser.add_argument("--model_path", default="mixfacenet_SMDD", type=str, help="path where trained model will be saved or location of pretrained weight")

    parser.add_argument("--is_train", default=True, type=lambda x: (str(x).lower() in ['true','1', 'yes']), help="train database or not")
    parser.add_argument("--is_test", default=True, type=lambda x: (str(x).lower() in ['true','1', 'yes']), help="test database or not")

    parser.add_argument("--max_epoch", default=100, type=int, help="maximum epochs")
    parser.add_argument("--batch_size", default=16, type=int, help="train batch size")
    parser.add_argument("--method", type=str)

    args = parser.parse_args()

    main(args)
