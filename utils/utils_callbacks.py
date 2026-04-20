import logging
import os
import time
from typing import List

import torch

from eval import verification
from segmentation import segment
from utils.utils_logging import AverageMeter

class CallBackSegmentation(object):
    def __init__(self, frequent, rank, val_targets, rec_prefix, save_path, image_size=(112, 112)):
        self.frequent: int = frequent
        self.rank: int = rank
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.save_path: str = save_path
        if self.rank == 0:
            self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        for i in range(len(self.ver_list)):
            embeddings_list = segment.segment(
                self.ver_list[i], backbone, 10, 10)
            
    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = segment.load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, num_update, backbone: torch.nn.Module, default_step=0, finished_training=False):
        eval_step_condition = self.rank == 0 and num_update - default_step > 0 and (num_update - default_step) % self.frequent == 0
        if eval_step_condition or finished_training:
            backbone.eval()
            self.ver_test(backbone, num_update)
            backbone.train()

class CallBackVerification(object):
    def __init__(self, frequent, rank, val_targets, rec_prefix, method, image_size=(112, 112)):
        self.frequent: int = frequent
        self.rank: int = rank
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        if self.rank == 0:
            if method == 'eval' or method == 'dist':
                self.init_pairs(val_targets=val_targets, data_dir=rec_prefix)
                #self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)
            elif method == 'save_embs':
                self.init_saving(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)
            else:
                print("Invalid method, code will exit...")
                exit()

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                self.ver_list[i], backbone, 10, 10)
            logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))
            results.append(acc2)

    def ver_test_batched(self, backbone: torch.nn.Module, global_step: int):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test_batched(
                self.ver_list[i], backbone, batch_size=64) # self.ver_list[i] contains (bins, issame_list) 
            
            logging.info('[%s][%d]XNorm: %f' % (self.ver_name_list[i], global_step, xnorm))
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))
            results.append(acc2)

    def ver_embeddings(self, global_step: int):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2 = verification.test_embeddings(self.ver_list[i])
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))
            results.append(acc2)

    def ver_embeddings_batched(self, global_step: int, batch_size: int = 512):
        results = []
        # Process each verification dataset in batches
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2 = verification.test_embeddings_batched_efficient(self.ver_list[i], batch_size=batch_size)
            logging.info('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (self.ver_name_list[i], global_step, acc2, std2))
            results.append(acc2)
    
    def cosine_similarity_distribution(self, save_png):
        for i in range(len(self.ver_list)):
            gen_max, gen_min, gen_avg, imp_max, imp_min, imp_avg = verification.cossim_dist(self.ver_list[i], save_png)
        return gen_max, gen_min, gen_avg, imp_max, imp_min, imp_avg
        
    def embedding_extraction(self, backbone: torch.nn.Module, save_dir: str):
        diff_norm = False
        if save_dir.find("ms1mv2"):
            batch_size=1
            diff_norm = True
        else:
            batch_size=64

        if save_dir.find("mxnet"):
            diff_norm = True

        for idx in range(len(self.ver_list)):
            embeddings = verification.get_embeddings(self.ver_list[idx], backbone, batch_size=batch_size, diff_norm=diff_norm)
            for idx, embedding in enumerate(embeddings):
                torch.save(embedding, os.path.join(save_dir, self.img_list[idx].replace('.jpg', '.pt')))

    def init_pairs(self, val_targets, data_dir):
        for name in val_targets:
            print('init pairs')
            emb_list, pairs_path_list, issame_list = verification.load_embeddings(data_dir)
            self.ver_list.append((emb_list, pairs_path_list, issame_list)) 
            self.ver_name_list.append(name)

    def init_dataset(self, val_targets, data_dir, image_size):
        for name in val_targets:
            print('load bin')
            path = os.path.join(data_dir, name + ".bin")
            print(path)
            if os.path.exists(path):
                bins, issame_list = verification.load_bin_lazy(path)
            self.ver_list.append((bins, issame_list)) # store as a tuple
            self.ver_name_list.append(name)

    def init_saving(self, val_targets, data_dir, image_size):
        for _ in val_targets:         
            print('load dataset')
            data_set, img_list = verification.load_dataset2(data_dir, image_size)
            self.img_list = img_list
            self.ver_list.append((data_set, img_list))
            
    def __call__(self, num_update, backbone: torch.nn.Module, method, save_png=None, save_dir=None, default_step=0, finished_training=False):
        eval_step_condition = self.rank == 0 and num_update - default_step > 0 and (num_update - default_step) % self.frequent == 0
        if eval_step_condition or finished_training:
            if method == "eval":
                #self.ver_test_batched(backbone, num_update)
                #backbone.eval()
                self.ver_embeddings_batched(num_update, batch_size=1024)
                #self.ver_embeddings(num_update)
            elif method == 'dist':
                _, _, gen_avg, _, _, imp_avg = self.cosine_similarity_distribution(save_png)
                print('Genuine avg:')
                print(gen_avg)

                print('Imposter avg:')
                print(imp_avg)
            elif method == 'save_embs':
                backbone.eval()
                self.embedding_extraction(backbone, save_dir)
                backbone.train()
            else:
                print("Invalid method, code will exit...")
                exit()

class CallBackLogging(object):
    def __init__(self, frequent, rank, total_step, batch_size, world_size, writer=None, resume=0, rem_total_steps=None):
        self.frequent: int = frequent
        self.rank: int = rank
        self.time_start = time.time()
        self.total_step: int = total_step
        self.batch_size: int = batch_size
        self.world_size: int = world_size
        self.writer = writer
        self.resume = resume
        self.rem_total_steps = rem_total_steps

        self.init = False
        self.tic = 0

    def __call__(self, global_step, loss: AverageMeter, epoch: int):
        if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float('inf')

                time_now = (time.time() - self.time_start) / 3600
                # TODO: resume time_total is not working
                if self.resume:
                    time_total = time_now / ((global_step + 1) / self.rem_total_steps)
                else:
                    time_total = time_now / ((global_step + 1) / self.total_step)
                time_for_end = time_total - time_now
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end, global_step)
                    self.writer.add_scalar('loss', loss.avg, global_step)
                msg = "Speed %.2f samples/sec   Loss %.4f Epoch: %d   Global Step: %d   Required: %1.f hours" % (
                    speed_total, loss.avg, epoch, global_step, time_for_end
                )
                logging.info(msg)
                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()

class CallBackModelCheckpoint(object):
    def __init__(self, rank, output="./"):
        self.rank: int = rank
        self.output: str = output

    def __call__(self, global_step, backbone: torch.nn.Module, header: torch.nn.Module = None):
        if global_step > 100 and self.rank == 0:
            torch.save(backbone.module.state_dict(), os.path.join(self.output, str(global_step)+ "backbone.pth"))
        if global_step > 100 and header is not None:
            torch.save(header.module.state_dict(), os.path.join(self.output, str(global_step)+ "header.pth"))
