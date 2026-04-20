"""Helper for evaluation on the Labeled Faces in the Wild dataset 
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import datetime
import os
import pickle

import mxnet as mx
import numpy as np
import sklearn
import torch
from mxnet import ndarray as nd
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]


def calculate_roc(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  nrof_folds=10,
                  pca=0):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if pca == 0:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if pca > 0:
            print('doing pca on', fold_idx)
            embed1_train = embeddings1[train_set]
            embed2_train = embeddings2[train_set]
            _embed_train = np.concatenate((embed1_train, embed2_train), axis=0)
            pca_model = PCA(n_components=pca)
            pca_model.fit(_embed_train)
            embed1 = pca_model.transform(embeddings1)
            embed2 = pca_model.transform(embeddings2)
            embed1 = sklearn.preprocessing.normalize(embed1)
            embed2 = sklearn.preprocessing.normalize(embed2)
            diff = np.subtract(embed1, embed2)
            dist = np.sum(np.square(diff), 1)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set],
                actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy

def calculate_roc_scores(thresholds,
                  dist,
                  actual_issame,
                  nrof_folds=10):

    nrof_pairs = min(len(actual_issame), len(dist))
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set],
                actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                       np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val(thresholds,
                  embeddings1,
                  embeddings2,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.BSpline.construct_fast(far_train, thresholds, k=1)
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean

def calculate_val_scores(thresholds,
                  dist,
                  actual_issame,
                  far_target,
                  nrof_folds=10):
    nrof_pairs = min(len(actual_issame), len(dist))
    nrof_thresholds = len(thresholds)
    k_fold = LFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.BSpline.construct_fast(far_train, thresholds, k=1)
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean

def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))

    # print(true_accept, false_accept)
    #print(n_same, n_diff)
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)

    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]

    tpr, fpr, accuracy = calculate_roc(thresholds,
                                       embeddings1,
                                       embeddings2,
                                       np.asarray(actual_issame),
                                       nrof_folds=nrof_folds,
                                       pca=pca)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds,
                                      embeddings1,
                                      embeddings2,
                                      np.asarray(actual_issame),
                                      1e-3,
                                      nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far

def evaluate_scores(scores, actual_issame, nrof_folds=10, pca=0):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy = calculate_roc_scores(thresholds,
                                       scores,
                                       np.asarray(actual_issame),
                                       nrof_folds=nrof_folds)
    
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val_scores(thresholds,
                                      scores,
                                      np.asarray(actual_issame),
                                      1e-3,
                                      nrof_folds=nrof_folds)
    return tpr, fpr, accuracy, val, val_std, far

@torch.no_grad()
def load_bin(path, image_size):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)
    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2) # horizontal flip
            # ---- save image as jpg ----
            #from PIL import Image
            #import numpy as np
            #img_hwc = nd.transpose(img, axes=(1, 2, 0)).asnumpy()
            ## ensure uint8
            #img_hwc = np.clip(img_hwc, 0, 255).astype(np.uint8)
            #Image.fromarray(img_hwc).save(
            #    "../../../../data/mcaldeir/exit_entry/test_original.jpg",
            #    format="JPEG"
            #)
            # --------------------------------
            #exit()
            data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
        if idx % 1000 == 0:
            print('loading bin', idx)
    print(data_list[0].shape)
    return data_list, issame_list

@torch.no_grad()
def load_bin_lazy(path):
    import pickle
    with open(path, 'rb') as f:
        bins, issame_list = pickle.load(f, encoding='bytes')
    return bins, issame_list

@torch.no_grad()
def load_embeddings(path):
    root_dir = ("/").join(path.split("/")[:-2]) 
    pairs = read_pairs(os.path.join(root_dir, 'pairs.txt'))
    path_list, issame_list = get_paths(path, pairs)
    emb_list = os.listdir(path)
    emb_list = [path + "/" + f for f in emb_list if f.find('.pt') != -1] 
    return emb_list, path_list, issame_list

""" @torch.no_grad()
def test(data_set, backbone, batch_size, nfolds=10):
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = data[bb - batch_size: bb]
            time0 = datetime.datetime.now()
            img = ((_data / 255) - 0.5) / 0.5
            net_out: torch.Tensor = backbone(img)
            _embeddings = net_out.detach().cpu().numpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0].copy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    acc1 = 0.0
    std1 = 0.0
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    print('infer time', time_consumed)
    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=nfolds)
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    return acc1, std1, acc2, std2, _xnorm, embeddings_list """

""" @torch.no_grad()
def test_batched(data_set, backbone, batch_size, nfolds=10):
    print('testing verification (memory-efficient mode)..')
    bins = data_set[0] 
    issame_list = data_set[1]
    
    embeddings_list = []
    
    # process twice: once for original, once for flipped
    for flip in [False, True]:
        embeddings = None
        ba = 0
        while ba < len(bins):
            bb = min(ba + batch_size, len(bins))
            batch_bins = bins[ba:bb]
            
            # decode only this batch
            batch_tensors = []
            for b in batch_bins:
                img = mx.image.imdecode(b).asnumpy()
                if flip:
                    img = img[:, ::-1, :] # horizontal flip
                
                img = torch.from_numpy(img.copy()).permute(2, 0, 1).float()
                img = ((img / 255.0) - 0.5) / 0.5
                batch_tensors.append(img)
            
            _data = torch.stack(batch_tensors).cuda()
            net_out = backbone(_data)
            _embeddings = net_out.detach().cpu().numpy()
            
            if embeddings is None:
                embeddings = np.zeros((len(bins), _embeddings.shape[1]))
            
            embeddings[ba:bb, :] = _embeddings
            ba = bb
            
        embeddings_list.append(embeddings)

    # Calculate XNorm
    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        _norms = np.linalg.norm(embed, axis=1)
        _xnorm += np.sum(_norms)
        _xnorm_cnt += len(_norms)
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    
    _, _, accuracy, val, val_std, far = evaluate(embeddings, issame_list, nrof_folds=nfolds)
    
    acc1, std1 = 0.0, 0.0 
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    
    return acc1, std1, acc2, std2, _xnorm, embeddings_list """

@torch.no_grad()
def cossim_dist(data_set, save_png):
    print('determining cossine similarity...')
    
    pairs_path_list = data_set[1]
    issame_list = data_set[2]

    #embeddings_1 = []
    #embeddings_2 = []
    dot_list = []
    norm1_list = []
    norm2_list = []
    for item in pairs_path_list:
        path1 = item[0]
        path2 = item[1]
        embeddings_1 = torch.load(path1, weights_only=False)
        embeddings_2 = torch.load(path2, weights_only=False)

        dot = np.sum(embeddings_1 * embeddings_2)
        norm1 = np.linalg.norm(embeddings_1)
        norm2 = np.linalg.norm(embeddings_2)

        dot_list.append(dot)
        norm1_list.append(norm1)
        norm2_list.append(norm2)

    #embeddings_1 = np.stack(embeddings_1)   # shape [N, D]
    #embeddings_2 = np.stack(embeddings_2)   # shape [N, D]

    #dot = np.sum(embeddings_1 * embeddings_2, axis=1)
    #norm1 = np.linalg.norm(embeddings_1, axis=1)
    #norm2 = np.linalg.norm(embeddings_2, axis=1)

    dot_np = np.array(dot_list)
    norm1_np = np.array(norm1_list)
    norm2_np = np.array(norm2_list)

    cos_sim = dot_np / (norm1_np * norm2_np)

    issame_list=np.array(issame_list)
    genuine_scores = np.array(cos_sim[issame_list])
    imposter_scores = np.array(cos_sim[~issame_list])

    gen_max = np.max(genuine_scores)
    gen_min = np.min(genuine_scores)
    gen_avg = np.mean(genuine_scores)

    imp_max = np.max(imposter_scores)
    imp_min = np.min(imposter_scores)
    imp_avg = np.mean(imposter_scores)

    bins = np.linspace(-1, 1, 100)

    # Compute histogram counts
    gen_counts, _ = np.histogram(genuine_scores, bins=bins)
    imp_counts, _ = np.histogram(imposter_scores, bins=bins)

    # Convert counts to percentage
    gen_counts = (gen_counts / gen_counts.sum()) * 100
    imp_counts = (imp_counts / imp_counts.sum()) * 100

    # Plot histograms with percentage normalization
    plt.hist(bins[:-1], bins=bins, weights=gen_counts, alpha=0.6, color='blue', label="Genuine Pairs")
    plt.hist(bins[:-1], bins=bins, weights=imp_counts, alpha=0.6, color='red', label="Imposter Pairs")

    # Format y-axis to show percentages
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

    # Labels and title
    plt.xlabel("Similarity Score")
    plt.ylabel("Percentage of Instances")
    plt.title("Histogram of Genuine and Imposter Pairs Similarity Scores")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.show()
    plt.savefig(save_png)

    return gen_max, gen_min, gen_avg, imp_max, imp_min, imp_avg

@torch.no_grad()
def test_embeddings(data_set, nfolds=10):
    print('testing embeddings..')
    
    pairs_path_list = data_set[1]
    issame_list = data_set[2]

    embeddings = []
    for item in pairs_path_list:
        path1 = item[0]
        path2 = item[1]
        embeddings.append(torch.load(path1, weights_only=False))
        embeddings.append(torch.load(path2, weights_only=False))

    _, _, accuracy, val, val_std, far = evaluate(np.asarray(embeddings), issame_list, nrof_folds=nfolds)
    
    acc1, std1 = 0.0, 0.0 
    acc2, std2 = np.mean(accuracy), np.std(accuracy)
    
    return acc1, std1, acc2, std2

@torch.no_grad()
def test_embeddings_batched(data_set, batch_size=512, nfolds=10):
    print("testing embeddings (batched loading)...")

    pairs_path_list = data_set[1]
    issame_list = np.asarray(data_set[2])

    all_embeddings = []

    # process in chunks to avoid RAM spike
    for i in range(0, len(pairs_path_list), batch_size):
        batch = pairs_path_list[i:i+batch_size]

        emb_batch = []

        for path1, path2 in batch:
            emb1 = torch.load(path1, weights_only=False)
            emb2 = torch.load(path2, weights_only=False)

            # if the embeddings are in torch format, they are moved to the cpu and converted to NumPy
            if isinstance(emb1, torch.Tensor):
                emb1 = emb1.cpu().numpy()  
            if isinstance(emb2, torch.Tensor):
                emb2 = emb2.cpu().numpy()  

            emb_batch.append(emb1)
            emb_batch.append(emb2)

        emb_batch = np.stack(emb_batch)
        all_embeddings.append(emb_batch)

    # merge all batches (still memory-safe because each batch is small)
    embeddings = np.concatenate(all_embeddings, axis=0)

    # reuse YOUR existing evaluation (important!)
    tpr, fpr, accuracy, val, val_std, far = evaluate(
        embeddings,
        issame_list,
        nrof_folds=nfolds
    )

    acc2, std2 = np.mean(accuracy), np.std(accuracy)

    return 0.0, 0.0, acc2, std2

@torch.no_grad()
def test_embeddings_batched_efficient(data_set, batch_size=512, nfolds=10):
    print("testing embeddings (batched loading)...")

    pairs_path_list = data_set[1]
    issame_list = np.asarray(data_set[2])

    #all_embeddings = []
    scores = []

    # process in chunks to avoid RAM spike
    for i in range(0, len(pairs_path_list), batch_size):
        batch = pairs_path_list[i:i+batch_size]

        for path1, path2 in batch:
            emb1 = torch.load(path1, weights_only=False)
            emb2 = torch.load(path2, weights_only=False)

            # if the embeddings are in torch format, they are moved to the cpu and converted to NumPy
            if isinstance(emb1, torch.Tensor):
                emb1 = emb1.cpu().numpy()  
            if isinstance(emb2, torch.Tensor):
                emb2 = emb2.cpu().numpy()

            diff = np.subtract(emb1, emb2)
            dist = np.sum(np.square(diff))
            scores.append(dist)

    tpr, fpr, accuracy, val, val_std, far = evaluate_scores(
        np.asarray(scores),
        issame_list,
        nrof_folds=nfolds
    )

    acc2, std2 = np.mean(accuracy), np.std(accuracy)

    return 0.0, 0.0, acc2, std2

@torch.no_grad()
def get_embeddings(data_set, backbone, batch_size, diff_norm, nfolds=10):
    print('getting embeddings..')
    data_list = data_set[0]

    embeddings_list = []
    time_consumed = 0.0
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = data[bb - batch_size: bb]
            time0 = datetime.datetime.now()

            #if diff_norm:
            #    img_bgr = _data[:, [2, 1, 0], :, :] 
            #    img = (img_bgr - 127.5) / 128.0   
            #else:
            img = ((_data / 255) - 0.5) / 0.5

            net_out: torch.Tensor = backbone(img)
            _embeddings = net_out.detach().cpu().numpy()

            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0].copy()
    embeddings = sklearn.preprocessing.normalize(embeddings)

    return embeddings

def dumpR(data_set,
          backbone,
          batch_size,
          name='',
          data_extra=None,
          label_shape=None):
    print('dump verification embedding..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba

            _data = nd.slice_axis(data, axis=0, begin=bb - batch_size, end=bb)
            time0 = datetime.datetime.now()
            if data_extra is None:
                db = mx.io.DataBatch(data=(_data,), label=(_label,))
            else:
                db = mx.io.DataBatch(data=(_data, _data_extra),
                                     label=(_label,))
            model.forward(db, is_train=False)
            net_out = model.get_outputs()
            _embeddings = net_out[0].asnumpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    actual_issame = np.asarray(issame_list)
    outname = os.path.join('temp.bin')
    with open(outname, 'wb') as f:
        pickle.dump((embeddings, issame_list),
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL)

### NEW
import os
import torch
import mxnet as mx
from mxnet import nd

def read_pairs(pairs_filename):
    """Read pairs.txt and return a list of [img1, img2, label]"""
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue  # skip empty lines
            pair = line.split()
            if len(pair) != 3:
                print(f"Warning: skipping invalid line: {line}")
                continue
            pair[0] = pair[0].replace(".jpg", ".pt")
            pair[1] = pair[1].replace(".jpg", ".pt")
            pairs.append(pair)
    return pairs

def get_paths(lfw_dir, pairs):
    """Convert pairs to full image paths and labels"""
    path_list = []
    issame_list = []

    for pair in pairs:
        path0 = os.path.join(lfw_dir, pair[0])
        path1 = os.path.join(lfw_dir, pair[1])
        issame = int(pair[2]) == 1

        if os.path.exists(path0) and os.path.exists(path1):
            path_list.append((path0, path1))
            issame_list.append(issame)
        else:
            print(f"Warning: missing file {path0} or {path1}")

    return path_list, issame_list

def load_dataset(lfw_dir, image_size, device='cuda'):
    """Load LFW dataset as PyTorch tensors (original + flipped)"""
    base = lfw_dir.split("/")
    root_dir = '/'.join(base[:-1])
    lfw_pairs = read_pairs(os.path.join(root_dir, 'pairs.txt'))
    lfw_paths, issame_list = get_paths(lfw_dir, lfw_pairs)

    num_images = len(lfw_paths) * 2  # 2 images per pair
    lfw_data_list = []

    # Prepare storage for original and flipped images
    for flip in [0, 1]:
        # Use torch tensor on correct device
        lfw_data = torch.empty((num_images, 3, image_size[0], image_size[1]), dtype=torch.float32, device=device)
        lfw_data_list.append(lfw_data)

    for i, (path0, path1) in enumerate(lfw_paths):
        for j, path in enumerate([path0, path1]):
            # Load image with MXNet
            with open(path, 'rb') as fin:
                _bin = fin.read()
                img = mx.image.imdecode(_bin)
                img = nd.transpose(img, axes=(2, 0, 1))  # HWC -> CHW
                img_np = img.asnumpy()  # MXNet NDArray -> numpy

                for flip in [0, 1]:
                    img_proc = img_np if flip == 0 else img_np[:, :, ::-1].copy()  # horizontal flip
                    # Convert to torch tensor
                    img_tensor = torch.from_numpy(img_proc).float().to(device)
                    lfw_data_list[flip][i*2 + j][:] = img_tensor

        if (i+1) % 1000 == 0 or (i+1) == len(lfw_paths):
            print(f'loading {i+1}/{len(lfw_paths)} pairs')

    print("dataset shapes:", lfw_data_list[0].shape, lfw_data_list[1].shape)
    return lfw_data_list, issame_list

def load_dataset2(dir, image_size, device='cuda'):
    img_list = os.listdir(dir)
    img_list = [f for f in img_list if f.find('jpg') != -1] #Filtering only the files.
    data_list = []

    for flip in [0, 1]:
        data = torch.empty((len(img_list), 3, image_size[0], image_size[1]), dtype=torch.float32, device=device)
        data_list.append(data)

    for idx, img_name in enumerate(img_list):
        # Load image with MXNet
        with open(os.path.join(dir, img_name), 'rb') as fin:
            _bin = fin.read()
            img = mx.image.imdecode(_bin)
            img = nd.transpose(img, axes=(2, 0, 1))  # HWC -> CHW
            img_np = img.asnumpy()  # MXNet NDArray -> numpy

            for flip in [0, 1]:
                img_proc = img_np if flip == 0 else img_np[:, :, ::-1].copy()  # horizontal flip
                img_tensor = torch.from_numpy(img_proc).float().to(device)
                data_list[flip][idx][:] = img_tensor

    print("dataset shapes:", data_list[0].shape, data_list[1].shape)
    return data_list, img_list