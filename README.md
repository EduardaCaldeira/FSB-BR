# This is the official repository of the paper "On the Impact of Face Segmentation-Based Background Removal on Recognition and Morphing Attack Detection" (accepted at FG 2026)

## Pre-Trained Portrait Segmentation Models Download
You can download the weights of the pre-trained models used for portrait segmentation in this work [here](https://github.com/hukenovs/easyportrait) (FPN + ResNet50, SegFormer-B0, BiSeNetv2, DANet, Fast SCNN and FCN + MobileNetv2) and [here](https://github.com/facebookresearch/segment-anything) (SAM).

## How to Run?

# Pre-Processing

**FERET:** Run `filter_feret.py` to copy FERET's frontal images to a new folder. Run `create_feret_pairs.py` to create genuine and impostor pairs with FERET's frontal images, following the pairing protocol discribed in the paper.

**FRGCv2:** Since FRGCv2 contains several folders, the first step is to save all the images considered for the experiment in the same folder. 

# Segmentation
**FPN + ResNet50, SegFormer-B0, BiSeNetv2, DANet, Fast SCNN and FCN + MobileNetv2:** Run `segmentation/pipelines/demo/image_demo.py`. The segmentation network and the target dataset can be altered directly in `segmentation/pipelines/demo.sh`.

**SAM:** Run `SAM/run_sam.py`. The target dataset and method can be selected directly in `run_sam.sh`. To follow the protocol described in the paper, the method should be set to `full_no_ctr` for FERET and IJB-C, and to `bb_extra` for FRGCv2. 

# Alignment and MAD crop
**FERET and FRGCv2:** Run `retinaface_alignment.py` to align the FERET, FRGCv2 and their segmented versions. This script also uses the bounding boxes extracted for each sample to perform the MAD crop. The segmentation network and target dataset can be altered directly in `create_dataset.sh` (for the unsegmented dataset, the network should be defined as `"none"`).

**IJB-C:** Contrary to the other datasets, IJB-C does not need to be passed through RetinaFace, as the landmarks and bounding boxes coordinates are already provided in the dataset. While the alignment is directly performed during evaluation, it is still necessary to save the MAD-cropped version of the samples, by running  `MAD_crop_IJBC.py`. The segmentation network can be altered directly in `MAD_crop_IJBC.sh`.

# FR Evaluation

**FERET and FRGCv2:** 
1. Run `save_FR_embeddings.py` to save the FR embeddings extracted by each FR network for FERET, FRGCv2, and their segmented versions. The segmentation network, target dataset and FR model used for extracting the embeddings can be altered in `save_FR_embs.sh`.
   
   1.1. **FRGCv2:** As described in the paper, some FRGCv2 samples are not properly aligned and should thus be excluded from the evaluation protocol. Run `clean_frgc.py` to remove the samples for which the alignment failed for **at least** one segmentation network from all relevant folders (**note:** this should only be ran after generating the embeddings for all the FRGCv2 (unsegmented and/or segmented) versions considered in the global evaluation, as the same list of samples should be used for evaluation for all methods).
   
   1.2. Run `create_frgc_pairs.py` to create genuine and impostor pairs with FRGCv2 images, following the pairing protocol discribed in the paper.
   
2. Set `--method eval` in `eval_FR.sh` to run `efficient_eval_FR.py` and get the FR evaluation results. Set `--method dist` in `eval_FR.sh` to run `efficient_eval_FR.py` and get the genuine and impostor scores distributions and metrics.

# Face Image Quality Assessement
# MAD Evaluation

# Plots

# Segmentation Methods Evaluation

## Citation

If you use any of the code, datasets or models provided in this repository, please cite the correspondent paper:

TO BE RELEASED

## License

<pre>This project is licensed under the terms of the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. 
Copyright (c) 2025 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt </pre>
