import logging
from .model import ClipModel

def get_model(rank, **kwargs):
    name = kwargs["model_name"]

    logging.info("Loading model: " + name + " " + kwargs["backbone_size"])

    if name == "MADPromptS" or name == "MADation":
        clip_model = ClipModel(
            rank=rank,
            model_name=kwargs["backbone_size"]
        )
        return clip_model, None
    else:
        raise ValueError()
    """ elif name == "MADation":
        madation_params = torch.load('src/backbone/madation_backbone.pth')
        print(clip_model)
        print('check2')
        #print(madation_params)
        clip_model.backbone.load_state_dict(madation_params)
        return clip_model, None """
    

def get_output_dim(**kwargs):
    name = kwargs["model_name"]

    if name == "MADPromptS" or name == "MADation":
        backbone_embeddings = {
            "ViT-B/32": 512,
            "ViT-B/16": 512,
            "ViT-L/14": 768,
            "ViT-L/14@336px": 768,
        }

        logging.info("Transformer dimension: " + str(backbone_embeddings[kwargs["backbone_size"]]))

        return backbone_embeddings[kwargs["backbone_size"]]
    else:
        raise ValueError()
