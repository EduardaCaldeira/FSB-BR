import logging

from .lora import apply_lora_clip
from .loralib.utils import mark_only_lora_as_trainable
from src.utils.utils import print_trainable_parameters
import torch

def apply_lora_model(rank, model, **kwargs):
    name = kwargs["model_name"]
    print(kwargs["training_type"])
    
    logging.info("Before LoRA : " + print_trainable_parameters(model.backbone))

    if kwargs["training_type"] == "test_clip":
        apply_lora_clip(
            model=model, 
            training_type=kwargs["training_type"], 
            model_name=kwargs["backbone_size"], 
            target_modules=kwargs["lora_target_modules"],
            lora_rank=kwargs["lora_r"], 
            lora_alpha=kwargs["lora_a"], 
            lora_dropout=kwargs["lora_dropout"], 
            device=rank, 
            position="all"
        )
        madation_params = torch.load('src/backbone/madation_backbone.pth')
        model.backbone.load_state_dict(madation_params) 
    elif name == "clip":
        logging.info("Add LoRA layers ...")
        apply_lora_clip(
            model=model, 
            training_type=kwargs["training_type"], 
            model_name=kwargs["backbone_size"], 
            target_modules=kwargs["lora_target_modules"],
            lora_rank=kwargs["lora_r"], 
            lora_alpha=kwargs["lora_a"], 
            lora_dropout=kwargs["lora_dropout"], 
            device=rank, 
            position="all"
        )
        mark_only_lora_as_trainable(model.backbone)
    else:
        raise ValueError()
    
    logging.info("After LoRA : " + print_trainable_parameters(model.backbone))