import sys
import torch
import logging
import clip

from PIL import Image
import open_clip

from transformers import Dinov2Model, SiglipModel, AutoProcessor, AutoModel

from transformers import BlipProcessor, BlipModel

class BLIPModel():
    def __init__(self, rank, backbone_size):
        if backbone_size == 'base':
            model_name = "Salesforce/blip-image-captioning-base"
        elif backbone_size == 'large':
            model_name = "Salesforce/blip-image-captioning-large"

        # Load the BLIP model and processor
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.backbone = BlipModel.from_pretrained(model_name)
        self.backbone.to(rank)


class SigLIPModel():
    def __init__(self, rank, backbone_size):
        model_map = {
            "ViT-B/16": "google/siglip-base-patch16-224",
            "ViT-G/14": "google/siglip-so400m-patch14-384",
        }

        if backbone_size not in model_map:
            raise ValueError(f"Unsupported SigLIP backbone size: {backbone_size}")

        model_checkpoint = model_map[backbone_size]
        self.backbone = SiglipModel.from_pretrained(model_checkpoint)
        self.processor = AutoProcessor.from_pretrained(model_checkpoint)

        self.backbone.to(rank)

class ClipModel():
    def __init__(self, rank, model_name):
        self.backbone, _ = clip.load(model_name, device="cuda", jit=False)
        self.backbone.to(rank)

        for param in self.backbone.parameters():
            if param.dtype == torch.float16:
                param.data = param.data.to(torch.float32)

class DINOv2Model():
    def __init__(self, rank, backbone_size):
        self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-" + backbone_size)
        self.backbone.to(rank)

class OpenClipModel():
    def __init__(self, rank, model_name):
        model_name = model_name.replace("/", "-")
        self.backbone, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained='laion400m_e31')
        self.backbone.to(rank)

        for param in self.backbone.parameters():
            if param.dtype == torch.float16:
                param.data = param.data.to(torch.float32)

class DeCLIPModel():
    def __init__(self, rank):
        # Model checkpoint or config name for DeCLIP
        model_name = "facebook/declip-base"  # only ViT-B/32 is publicly available
        self.backbone = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.backbone.to(rank)