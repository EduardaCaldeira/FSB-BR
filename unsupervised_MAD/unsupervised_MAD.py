import numpy as np
import os
import csv
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from collections import defaultdict

from dataset import TestDataset
from utils import get_performance
import network

from sklearn.metrics import roc_curve, auc

def get_apcer_op(apcer, bpcer, threshold, op):
    """Returns the value of the given FMR operating point
    Definition:
    ZeroFMR: is defined as the lowest FNMR at which no false matches occur.
    Others FMR operating points are defined in a similar way.
    @param apcer: =False Match Rates
    @type apcer: ndarray
    @param bpcer: =False Non-Match Rates
    @type bpcer: ndarray
    @param op: Operating point
    @type op: float
    @returns: Index, The lowest bpcer at which the probability of apcer == op
    @rtype: float
    """
    index = np.argmin(abs(apcer - op))
    return index, bpcer[index], threshold[index]

def get_bpcer_op(apcer, bpcer, threshold, op):
    """Returns the value of the given FNMR operating point
    Definition:
    ZeroFNMR: is defined as the lowest FMR at which no non-false matches occur.
    Others FNMR operating points are defined in a similar way.
    @param apcer: =False Match Rates
    @type apcer: ndarray
    @param bpcer: =False Non-Match Rates
    @type bpcer: ndarray
    @param op: Operating point
    @type op: float
    @returns: Index, The lowest apcer at which the probability of bpcer == op
    @rtype: float
    """
    temp = abs(bpcer - op)
    min_val = np.min(temp)
    index = np.where(temp == min_val)[0][-1]

    return index, apcer[index], threshold[index]

def get_eer_threhold(fpr, tpr, threshold):
    differ_tpr_fpr_1=tpr+fpr-1.0
    index = np.nanargmin(np.abs(differ_tpr_fpr_1))
    eer = fpr[index]

    return eer, index, threshold[index]

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
        print(f"APCER@BPCER20%: {results['APCER@BPCER20%']:.4f}")
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
    thr_20 = 361.27996826171875
    thr_10 = 389.12371826171875
    thr_1 = 452.8556823730469

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

def run_test(args):
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    test_dataset = TestDataset(csv_file=args.test_csv, input_shape=args.input_shape)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=64, pin_memory=True)

    print('Number of test images:', len(test_loader.dataset))
    model = torch.nn.DataParallel(network.AEMAD(in_channels=3, features_root=args.features_root))
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    model.cuda()
    model.eval()

    mse_criterion = torch.nn.MSELoss(reduction='none').cuda()

    test_scores, gt_labels, test_scores_dict = [], [], []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            raw, labels, img_ids = data['images'].cuda(), data['labels'], data['img_path']
            _, output_raw = model(raw)

            scores = mse_criterion(output_raw, raw).cpu().data.numpy()
            scores = np.sum(np.sum(np.sum(scores, axis=3), axis=2), axis=1)
            test_scores.extend(scores)
            gt_labels.extend((1 - labels.data.numpy()))
            for j in range(labels.shape[0]):
                l = int(labels[j].detach().numpy())
                test_scores_dict.append({'image_path':img_ids[j], 'label':l, 'prediction_score': float(scores[j])})

    with open(args.output_path, mode='w') as csv_file:
        fieldnames = ['image_path', 'label', 'prediction_score']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for d in test_scores_dict:
            writer.writerow(d)
        print('Prediction scores write done in', args.output_path)
        
    if args.method == 'threshold' or args.method=='eval':
        results = []
        if args.method == 'threshold':
            results.append(find_synmad_thresholds(test_scores, gt_labels))
        else:
            results.append(evaluate_mad_performance(test_scores))

        with open(os.path.join(args.output_path.replace("csv", "txt")), "w") as f:
            f.write(f"BPCER@APCER20%: {results[0]['bpcer_apcer20']:.4f}\n")
            f.write(f"BPCER@APCER10%: {results[0]['bpcer_apcer10']:.4f}\n")
            f.write(f"BPCER@APCER1%: {results[0]['bpcer_apcer1']:.4f}\n")
    
        return

    eer, _ = get_performance(test_scores, gt_labels)
    print('Test EER:', eer*100)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    cudnn.benchmark = True

    if torch.cuda.is_available():
        print('GPU is available')
        torch.cuda.manual_seed(0)
    else:
        print('GPU is not available')
        torch.manual_seed(0)

    import argparse
    parser = argparse.ArgumentParser(description='SPL MAD')

    parser.add_argument("--test_csv", required=True, type=str, help="path of data directory including csv files")
    parser.add_argument("--model_path", required=True, type=str, help="model path")
    parser.add_argument("--output_path", default='test.csv', type=str, help="path for output prediction scores")

    parser.add_argument("--input_shape", default=(224, 224), type=tuple, help="model input shape")
    parser.add_argument("--features_root", default=64, type=int, help="feature root")
    parser.add_argument("--batch_size", default=32, type=int, help="test batch size")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--method", default="", type=str)

    args = parser.parse_args()
    run_test(args=args)