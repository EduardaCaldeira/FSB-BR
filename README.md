# This is the official repository of the paper "On the Impact of Face Segmentation-Based Background Removal on Recognition and Morphing Attack Detection" (accepted at FG 2026)

## Pre-Trained Portrait Segmentation Models Download
You can download the weights of the pre-trained models used for portrait segmentation in this work [here](https://github.com/hukenovs/easyportrait) (FPN + ResNet50, SegFormer-B0, BiSeNetv2, DANet, Fast SCNN and FCN + MobileNetv2) and [here](https://github.com/facebookresearch/segment-anything) (SAM).

## How to Run?

# Pre-Processing

**FERET:** Run `filter_feret.py` to copy FERET's frontal images to a new folder. Run `create_feret_pairs.py` to create genuine and impostor pairs with FERET's frontal images, following the pairing protocol discribed in the paper.

**FRGCv2:** Since FRGCv2 contains several folders, the first step is to save all the images considered for the experiment in the same folder. Run `create_frgc_pairs.py` to create genuine and impostor pairs with FRGCv2 images, following the pairing protocol discribed in the paper.

# Segmentation
+ clean FRGCv2

# Face Image Quality Assessement

# FR Evaluation

# MAD Evaluation

# Segmentation Methods Evaluation

## Citation

If you use any of the code, datasets or models provided in this repository, please cite the correspondent paper:

TO BE RELEASED

## License

<pre>This project is licensed under the terms of the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. 
Copyright (c) 2025 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt </pre>
