import logging
import os
from dataclasses import dataclass
from typing import List

import clip
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from src.utils.evaluation import CallBackVerification
from src.utils.logging import (AverageMeter, CallBackLogging,
                           CallBackModelCheckpoint, CallBackTensorboard)
from src.utils.utils import (evaluate_mad_performance, find_synmad_thresholds,
                         write_scores)

from .scheduler import get_scheduler
import open_clip
from torchvision.transforms import ToPILImage


@dataclass
class TestData:
    scores: torch.Tensor
    labels: torch.Tensor
    video_ids: List[str]

########  Default Trainer ########
class Trainer():
    def __init__(self, rank, world_size, model, preprocess, trainset, dataloader, train_sampler, training_type, config, header=None, test_dataloader=None, test_sampler=None):
        self.rank = rank
        self.world_size = world_size
        self.model = model
        self.preprocess = preprocess
        self.header = header
        self.trainset = trainset
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler
        self.training_type = training_type
        self.config = config

        self.start_epoch = 0
        self.global_step = self.config.global_step
        self.total_step = int(len(self.trainset) / config.batch_size / self.world_size * config.num_epoch)

        # Callback
        self.callback_logging = CallBackLogging(
            config.log_every, rank, self.total_step, config.batch_size, world_size, writer=None
        )
        self.callback_save_model = CallBackModelCheckpoint(rank, config.save_every, output=config.output_path)
        self.tensorboard_callback = CallBackTensorboard(rank, self.config)
        self.tensorboard_callback.log_hyperparameters()

        # Logging
        self.loss_log = AverageMeter()
        logging.info("Trainset lenght: %d" % len(self.trainset))
        logging.info("Total Step is: %d" % self.total_step)
        logging.info("Config is: {}".format(self.config.__dict__))


########################
########  CLIP  ########
########################
class TrainerClip(Trainer):
    def __init__(self, rank, world_size, model, preprocess, trainset, dataloader, train_sampler, training_type, config, header, test_dataloader=None, test_sampler=None):
        super().__init__(rank, world_size, model, preprocess, trainset, dataloader, train_sampler, training_type, config, header, test_dataloader, test_sampler)

    def start_training(self):
        if self.training_type == "MAD_training":
            self.MAD_training()
        elif self.training_type == "MAD_training_only_header":
            self.MAD_training_only_header()
        elif self.training_type == "test_clip":
            self.test_clip()
        else:
            raise ValueError()

    def gather_test_data(self, scores, labels, video_ids):
        local_data = TestData(
            scores=scores.cpu(), labels=labels.cpu(), video_ids=video_ids
        )
        gathered_data = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_data, local_data)

        if self.rank == 0:
            all_scores = []
            all_labels = []
            all_video_ids = []

            for data in gathered_data:
                all_scores.append(data.scores)
                all_labels.append(data.labels)
                all_video_ids.extend(data.video_ids)

            return (torch.cat(all_scores), torch.cat(all_labels), all_video_ids)
        return None, None, None

    def test_model(self, epoch, test_data):
        self.model.module.eval()
        self.header.eval()
        results = []
        for i, testdata in enumerate(self.test_dataloader):
            raw_test_scores, gt_labels = [], []
            raw_test_img_pths = []
            with torch.no_grad():
                for _, (raw, labels, img_paths) in enumerate(testdata):
                    raw = raw.cuda(self.rank, non_blocking=True)
                    labels = labels.cuda(self.rank, non_blocking=True)

                    features_image = self.model.module.encode_image(raw)

                    output, _ = self.header(F.normalize(features_image), labels)
                    # print(output)
                    logits = output.softmax(dim=1)[:, 1]
                    # print(logits)
                    raw_test_scores.append(logits)
                    gt_labels.append(labels)
                    for j in range(raw.shape[0]):
                        raw_test_img_pths.append(img_paths[j])

                raw_test_scores = torch.cat(raw_test_scores)
                gt_labels = torch.cat(gt_labels)
                scores, labels, img_pathes = self.gather_test_data(
                    raw_test_scores, gt_labels, raw_test_img_pths
                )
            if self.rank == 0:
                raw_test_scores = scores.cpu().numpy()
                gt_labels = labels.cpu().numpy()
                if epoch == self.config.num_epoch - 1:
                    out_path = os.path.join(self.config.output_path, test_data[i] + ".csv")
                    write_scores(img_pathes, raw_test_scores, gt_labels, out_path)
                raw_test_stats = [np.mean(raw_test_scores), np.std(raw_test_scores)]
                raw_test_scores = (raw_test_scores - raw_test_stats[0]) / raw_test_stats[1]

                results.append(evaluate_mad_performance(raw_test_scores, gt_labels))

        self.model.module.train()
        self.header.train()
        if self.rank == 0:
            return results
        else:
            return None

    def MAD_training(self):
        # Optimizer
        optimizer_model = torch.optim.AdamW(
            params=[{'params': self.model.parameters()}], betas=(0.9, 0.999),
            lr=self.config.lr_model, weight_decay=self.config.weight_decay
        )
        optimizer_header = torch.optim.AdamW(
            params=[{'params': self.header.parameters()}], betas=(0.9, 0.999),
            lr=self.config.lr_header, weight_decay=self.config.weight_decay
        )

        # Scheduler
        scheduler_model = get_scheduler(
                scheduler_type=self.config.scheduler_type,
                optimizer_model=optimizer_model,
                epoch=self.config.num_epoch,
                warmup=self.config.warmup,
                num_warmup_epochs=self.config.num_warmup_epochs,
                T_0=self.config.T_0,
                T_mult=self.config.T_mult,
                eta_min=self.config.lr_model, #self.config.eta_min,
                lr_func_drop=self.config.lr_func_drop,
        )
        scheduler_header = get_scheduler(
                scheduler_type=self.config.scheduler_type,
                optimizer_model=optimizer_header,
                epoch=self.config.num_epoch,
                warmup=self.config.warmup,
                num_warmup_epochs=self.config.num_warmup_epochs,
                T_0=self.config.T_0,
                T_mult=self.config.T_mult,
                eta_min=self.config.lr_header, #self.config.eta_min,
                lr_func_drop=self.config.lr_func_drop,
        )

        for epoch in range(self.start_epoch, self.config.num_epoch):
            self.train_sampler.set_epoch(epoch)
            for _, (images, target) in enumerate(self.dataloader):
                self.global_step += 1

                images = images.cuda(self.rank, non_blocking=True)
                target = target.cuda(self.rank, non_blocking=True)

                features_image = self.model.module.encode_image(images)
                _, loss_image = self.header(F.normalize(features_image), target)

                loss_image.backward()

                clip_grad_norm_(self.model.module.parameters(), max_norm=self.config.max_norm, norm_type=2)
                clip_grad_norm_(self.header.parameters(), max_norm=self.config.max_norm, norm_type=2)

                optimizer_model.step()
                optimizer_header.step()

                self.loss_log.update(loss_image.item(), 1)
                self.tensorboard_callback.log_info(
                    global_step=self.global_step,
                    loss=loss_image.item(),
                    learning_rate=scheduler_model.get_last_lr()[0],
                    model=self.model.module.visual
                )
                self.callback_logging(self.global_step, self.loss_log, epoch)

                optimizer_model.zero_grad()
                optimizer_header.zero_grad()

            scheduler_model.step()
            scheduler_header.step()
            if epoch == 39:
                results = self.test_model(epoch, self.config.test_data)
                if self.rank == 0:
                    for i, result in enumerate(results):
                        logging.info(
                            f'Dataset: {self.config.test_data[i]}, Epoch: {epoch}, auc: {result["auc_score"]}, eer: {result["eer"]}, apcer_bpcer20: {result["apcer_bpcer20"]}, apcer_bpcer10: {result["apcer_bpcer10"]}, apcer_bpcer1: {result["apcer_bpcer1"]}, bpcer_apcer20: {result["bpcer_apcer20"]}, bpcer_apcer10: {result["bpcer_apcer10"]}, bpcer_apcer1: {result["bpcer_apcer1"]}'
                        )

            self.callback_save_model(epoch, self.model.module, self.header)
        self.tensorboard_callback.close()

    def MAD_training_only_header(self):
        # Freeze the clip model
        for param in self.model.module.parameters():
            param.requires_grad = False
        self.model.module.eval()

        optimizer_header = torch.optim.AdamW(
            params=[{'params': self.header.parameters()}], betas=(0.9, 0.999),
            lr=self.config.lr_header, weight_decay=self.config.weight_decay
        )

        scheduler_header = get_scheduler(
                scheduler_type=self.config.scheduler_type,
                optimizer_model=optimizer_header,
                epoch=self.config.num_epoch,
                warmup=self.config.warmup,
                num_warmup_epochs=self.config.num_warmup_epochs,
                T_0=self.config.T_0,
                T_mult=self.config.T_mult,
                eta_min=self.config.lr_header,
                lr_func_drop=self.config.lr_func_drop,
        )

        for epoch in range(self.start_epoch, self.config.num_epoch):
            self.train_sampler.set_epoch(epoch)
            for _, (images, target) in enumerate(self.dataloader):
                self.global_step += 1

                images = images.cuda(self.rank, non_blocking=True)
                target = target.cuda(self.rank, non_blocking=True)

                # image
                with torch.no_grad():
                    features_image = self.model.module.encode_image(images)

                _, loss_image = self.header(F.normalize(features_image), target)

                loss_image.backward()

                clip_grad_norm_(self.header.parameters(), max_norm=self.config.max_norm, norm_type=2)

                optimizer_header.step()

                self.loss_log.update(loss_image.item(), 1)
                self.tensorboard_callback.log_info(
                    global_step=self.global_step,
                    loss=loss_image.item(),
                    learning_rate=scheduler_header.get_last_lr()[0],
                    model=self.model.module.visual
                )

                self.callback_logging(self.global_step, self.loss_log, epoch)
                optimizer_header.zero_grad()

            scheduler_header.step()
            if epoch == 39:
                results = self.test_model(epoch, self.config.test_data)
                if self.rank == 0:
                    for i, result in enumerate(results):
                        logging.info(
                            f'Dataset: {self.config.test_data[i]}, Epoch: {epoch}, auc: {result["auc_score"]}, eer: {result["eer"]}, apcer_bpcer20: {result["apcer_bpcer20"]}, apcer_bpcer10: {result["apcer_bpcer10"]}, apcer_bpcer1: {result["apcer_bpcer1"]}, bpcer_apcer20: {result["bpcer_apcer20"]}, bpcer_apcer10: {result["bpcer_apcer10"]}, bpcer_apcer1: {result["bpcer_apcer1"]}'
                        )

            self.callback_save_model(epoch, self.model.module, self.header)
        self.tensorboard_callback.close()

    def determine_scores(self, text_features, current_output_path, model_name, method):
        results = []

        for i, testdata in enumerate(self.test_dataloader):
            if self.test_sampler:
                self.test_sampler[i].set_epoch(0)

            raw_test_scores, gt_labels = [], []
            raw_test_img_pths = []
            with torch.no_grad():
                for _, (raw, labels, img_paths) in enumerate(testdata):
                    raw = raw.cuda(self.rank, non_blocking=True)
                    labels = labels.cuda(self.rank, non_blocking=True)

                    image_features = self.model.module.encode_image(raw)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                    if self.config.method == 'avg':
                        logits_per_image = (100.0 * image_features @ text_features.T).softmax(
                           dim=-1
                        )                        
                    else: 
                        logits_per_image = (100.0 * image_features @ text_features.T)
                        break_len = int(len(logits_per_image.T)/2)
                        part_1 = logits_per_image[:, 0:break_len].max(dim=1, keepdim=True).values
                        part_2 = logits_per_image[:, break_len:].max(dim=1, keepdim=True).values
                        logits_per_image = torch.cat([part_1, part_2], dim=1).softmax(
                           dim=-1
                        )                        

                    raw_scores = logits_per_image[:, 1]
                    raw_test_scores.append(raw_scores)
                    gt_labels.append(labels)

                    for j in range(raw.shape[0]):
                        raw_test_img_pths.append(img_paths[j])

                raw_test_scores = torch.cat(raw_test_scores)
                gt_labels = torch.cat(gt_labels)
                scores, labels, img_pathes = self.gather_test_data(
                    raw_test_scores, gt_labels, raw_test_img_pths
                )

            if self.rank == 0:
                raw_test_scores = scores.cpu().numpy()
                gt_labels = labels.cpu().numpy()

                out_path = os.path.join(current_output_path, self.config.test_data[i] + ".csv")
                write_scores(img_pathes, raw_test_scores, gt_labels, out_path)

                if method == 'threshold':
                    raw_test_stats = [np.mean(raw_test_scores), np.std(raw_test_scores)]
                    print("mean and std:", raw_test_stats)
                    results.append(find_synmad_thresholds(raw_test_scores, gt_labels))
                else: 
                    if model_name == "MADPromptS":
                        raw_test_stats = np.array([0.00047382814, 0.004171985])
                    elif model_name== "MADation":
                        raw_test_stats = np.array([0.00051609194, 0.002589177])
                         
                    raw_test_scores = (raw_test_scores - raw_test_stats[0]) / raw_test_stats[1]

                    results.append(evaluate_mad_performance(raw_test_scores, model_name))               
                    with open(os.path.join(current_output_path, self.config.test_data[i] + ".txt"), "w") as f:
                        f.write(f"BPCER@APCER20%: {results[i]['bpcer_apcer20']:.4f}\n")
                        f.write(f"BPCER@APCER10%: {results[i]['bpcer_apcer10']:.4f}\n")
                        f.write(f"BPCER@APCER1%: {results[i]['bpcer_apcer1']:.4f}\n")
    
    def get_text_features(self, morphing_prompts, bonafide_prompts):
        with torch.no_grad():
            prompts = morphing_prompts + bonafide_prompts 

            text_inputs = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
            text_features = self.model.module.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            if len(morphing_prompts) > 1:
                if self.config.method == 'avg':
                    text_features = torch.cat([text_features[0:len(morphing_prompts)].mean(dim=0, keepdim=True), text_features[len(morphing_prompts):].mean(dim=0, keepdim=True)], dim=0)

                text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features
    
    def encode_siglip_image(self, processor, model, image, device="cuda"):
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = F.normalize(image_features, dim=-1)
        return image_features

    def encode_siglip_text(self, processor, model, texts, device="cuda"):
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def test_clip(self):
        self.model.module.eval()

        templates_presentation = [
            'frontal {}.',
            'profile {}.',
            'tilted {}.',
            'rotated {}.',
            'upward {}.',
            'downward {}.',
            'sideways {}.',
            'leftward {}.',
            'rightward {}.',
            'angled {}.',
            'inclined {}.',
            'declined {}.',
            'oblique {}.',
            'twisted {}.',
            'turned {}.',
            'slanted {}.',
            'offcenter {}.',
            'misaligned {}.',
            'skewed {}.',
            'asymmetric {}.',
        ]

        templates_appearance = [
            'bearded {}.',
            'moustached {}.',
            'smiling {}.',
            'frowning {}.',
            'eyeglasses {}.',
            'sunglasses {}.',
            'wrinkled {}.',
            'balding {}.',
            'occluded {}.',
            'scarred {}.',
            'pierced {}.',
            'tanned {}.',
            'pale {}.',
            'makeup {}.',
            'freckled {}.',
            'chubby-cheeked {}.',
            'sweaty {}.',
            'dirty {}.',
            'blinking {}.',
            'tearful {}.',
        ]

        self.config.output_path = self.config.output_path + '/' + self.config.model_name 

        if self.config.model_name == "MADation":
            morphing_prompts = ["face image morphing attack"]
            bonafide_prompts = ["bona-fide presentation"]

            current_output_path = self.config.output_path + "/" + self.config.method
            os.makedirs(current_output_path, exist_ok=True)

            text_features = self.get_text_features(morphing_prompts, bonafide_prompts)
            self.determine_scores(text_features, current_output_path, self.config.model_name, self.config.eval_method)
        
        else:
            # face presentation + face appearance
            templates = templates_presentation + templates_appearance
            morphing_prompts = [template.format("face image morphing attack") for template in templates]
            bonafide_prompts = [template.format("bona-fide presentation") for template in templates]
        
            current_output_path = self.config.output_path + "/" + self.config.method + "/face_presentation_face_appearance"
            os.makedirs(current_output_path, exist_ok=True)

            text_features = self.get_text_features(morphing_prompts, bonafide_prompts)
            self.determine_scores(text_features, current_output_path, self.config.model_name, self.config.eval_method)