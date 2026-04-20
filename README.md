# This is the official repository of the paper "On the Impact of Face Segmentation-Based Background Removal on Recognition and Morphing Attack Detection" (accepted at FG 2026)

## Downloads

### Pre-Trained Portrait Segmentation Models
You can download the weights of the pre-trained models used for portrait segmentation in this work [here](https://github.com/hukenovs/easyportrait) (FPN + ResNet50, SegFormer-B0, BiSeNetv2, DANet, Fast SCNN and FCN + MobileNetv2) and [here](https://github.com/facebookresearch/segment-anything) (SAM).

### FR Evaluation Models
The weigths of the pre-trained FR models used in this work can be downloaded through the following links: [ElasticFace](https://github.com/fdbtrs/ElasticFace), [ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch), [SwinFace](https://github.com/lxq1000/SwinFace), [TransFace](https://github.com/DanJun6737/TransFace). These models should be saved inside a folder named `FR_models`

### MAD Evaluation Models
The weigths of the pre-trained MAD models used in this work can be downloaded through the following links: [SPL](https://github.com/meilfang/SPL-MAD), [MixFaceNet-MAD](https://github.com/naserdamer/SMDD-Synthetic-Face-Morphing-Attack-Detection-Development-dataset). These models should be saved inside the folders named `unsupervised_MAD` and `supervised_MAD`, respectively.

### FIQA Evaluation Model
The weights of CR-FIQA model can be downloaded [here](https://github.com/fdbtrs/cr-fiqa). This model should be saved inside the folders named `CR-FIQA`.

## How to Run?

### Pre-Processing

**FERET:** Run `filter_feret.py` to copy FERET's frontal images to a new folder. Run `create_feret_pairs.py` to create genuine and impostor pairs with FERET's frontal images, following the pairing protocol discribed in the paper.

**FRGCv2:** Since FRGCv2 contains several folders, the first step is to save all the images considered for the experiment in the same folder. 

### Segmentation
**FPN + ResNet50, SegFormer-B0, BiSeNetv2, DANet, Fast SCNN and FCN + MobileNetv2:** Run `segmentation/pipelines/demo/image_demo.py`. The segmentation network and the target dataset can be altered directly in `segmentation/pipelines/demo.sh`

**SAM:** Run `SAM/run_sam.py`. The target dataset and method can be selected directly in `run_sam.sh`. To follow the protocol described in the paper, the method should be set to `full_no_ctr` for FERET and IJB-C, and to `bb_extra` for FRGCv2. 

### Alignment and MAD crop
**FERET and FRGCv2:** Run `retinaface_alignment.py` to align the FERET, FRGCv2 and their segmented versions. This script also uses the bounding boxes extracted for each sample to perform the MAD crop. The segmentation network and target dataset can be altered directly in `create_dataset.sh` (for the unsegmented dataset, the network should be defined as `"none"`).

**IJB-C:** Contrary to the other datasets, IJB-C does not need to be passed through RetinaFace, as the landmarks and bounding boxes coordinates are already provided in the dataset. While the alignment is directly performed during evaluation, it is still necessary to save the MAD-cropped version of the samples, by running  `MAD_crop_IJBC.py`. The segmentation network can be altered directly in `MAD_crop_IJBC.sh`

### FR Evaluation

**FERET and FRGCv2:** 
1. Run `save_FR_embeddings.py` to save the FR embeddings extracted by each FR network for FERET, FRGCv2, and their segmented versions. The segmentation network, target dataset and FR model used for extracting the embeddings can be altered in `save_FR_embs.sh`
   
   1.1. **FRGCv2:** As described in the paper, some FRGCv2 samples are not properly aligned and should thus be excluded from the evaluation protocol. Run `clean_frgc.py` to remove the samples for which the alignment failed for **at least** one segmentation network from all relevant folders (**note:** this should only be ran after generating the embeddings for all the FRGCv2 (unsegmented and/or segmented) versions considered in the global evaluation, as the same list of samples should be used for evaluation for all methods).
   
   1.2. Run `create_frgc_pairs.py` to create genuine and impostor pairs with FRGCv2 images, following the pairing protocol discribed in the paper.
   
2. Set `--method eval` in `eval_FR.sh` to run `efficient_eval_FR.py` and get the FR evaluation results. Set `--method dist` in `eval_FR.sh` to run `efficient_eval_FR.py` and get the genuine and impostor scores distributions and metrics.

**IJB-C:** Run `eval_ijbc.py`. The segmentation network and FR model used for extracting the embeddings can be altered in `runIJBEval.sh`

### Face Image Quality Assessement

Run `CR-FIQA/getQualityScore.py` to extract the face image quality assessment metrics. The segmentation network and target dataset can be altered in `FIQA.sh`

### MAD Evaluation

1. Run `MAD_format.py` to obtain the `.csv` file used as a reference during MAD. The segmentation network and target dataset can be altered in `MAD_format.sh`
2. Get the MAD thresholds for the MAD22 dataset:

   2.1. Run `join_synmad.py` to join the original subsets of MAD22 (FaceMorpher, MIPGAN_I, MIPGAN_II, OpenCV, Webmorph) and its extension MorDIFF.
   
   2.2. **SPL:** Set `--method="threshold"` in `unsupervised_MAD.sh` and run to obtain the thresholds for SPL.

   2.3. **MixFaceNet-MAD:**  Set `--method "threshold"` in `supervised_MAD.sh` and run to obtain the thresholds and normalization statistics for MixFaceNet-MAD.

   2.4. **MADPromptS:** Set `--eval_method="threshold"` in `MAD_triggering.sh` and run to obtain the thresholds and normalization statistics for MADPromptS.
   
3. Run the MAD evaluations:
   
  3.1. **SPL:** Alter the threshold values in `unsupervised_MAD/unsupervised_NAD.py` to the thresholds obtained for SPL in step 2.2 and run `unsupervised_MAD.sh` with `--method="eval"` to perform the MAD evaluation on SPL.
   
  3.2. **MixFaceNet-MAD:** Alter the threshold and normalization statistics values in `supervised_MAD/main.py` to the values obtained for MixFaceNet-MAD in step 2.3 and run `supervised_MAD.sh` with `--method "eval"` to perform the MAD evaluation on MixFaceNet-MAD.
  
  3.3. **MADPromptS:** Alter the threshold and normalization statistics values in `src/utils/utils.py` and `src/training/trainer.py`, respectively, to the values obtained for MADPromptS in step 2.4 and run `MAD_triggering.sh` with `--method "eval"` to perform the MAD evaluation on MADPromptS.

### Plots
1. Alter the metric values in `plot.py` to the values obtained for FR evaluation and Δ, to obtain the first part of the joint visualization of IJB-C results.
2. Alter the metric values in `plot_MAD.py` to the values obtained for MAD evaluation, to obtain the second part of the joint visualization of IJB-C results.

## Segmentation Methods Evaluation
1. Run `unify_masks.py` to get the union of the GT masks representing different facial elements for each sample of the CelebAMask-HQ dataset.
2. Run `get_test_celeb.py` to copy the test set of the CelebAMask-HQ dataset into a new folder.
3. Run `segmentation/pipelines/demo/segmentation_eval.py` and `SAM/eval_sam.py` to obtain the evaluation results.

## Citation

If you use any of the code, datasets or models provided in this repository, please cite the correspondent paper:

TO BE RELEASED

## License

<pre>This project is licensed under the terms of the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. 
Copyright (c) 2025 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt </pre>
