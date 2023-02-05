# MIPT (MSAI) - Computer Vision for Postal cards Library Thesis project

This repository contains materials and description of the experiments undertaken during our thesis work.

### Image Retrieval Experiment
Target - show that AI-based solutions can confidently detect post cards depicting same landmarks. <br> 
Approach - understand theory, find existing technological solutions appropriate for the task, select best one, based on it come up with workable POC, demonstrate convincing results.  

Tangible Results achieved:
Four nn-models were trained and tested for the IR task, all proved capability of detecting same landmark postcards. 
KCP (1K manually Categorized Postcards) Data Set with domain-specific images (postcards).

Open question:
issues with out-of-domain pictures (non-landmarks).
issues with pictures containing huge blank borders on the edges.
issues with pictures depicting river banks, lakes, bushes.

Steps to reproduce the experiment:

1. Make sure you have python, pytorch and other libraries listed in ```requirements.txt``` are installed.
2. For model training and embeddings generation you'll need a CUDA compatible device (RTX 30X0, 40X0 graphic card) for 2 weeks x 24hr period along with its drivers.
3. Download Google Landmarks v2 dataset (500Mb): https://github.com/cvdfoundation/google-landmark <br>
GLMv2 has several versions, you need ```train clean``` version of train index (train_clean.csv ) along with complete test part of the DS. 
4. Train one or more neuro-network model(s) with code provided in this repository.
```
python train.py --config <model number>
ex: python train.py --config config6
```
Pre-trained weights for one of the models are available here: https://drive.google.com/file/d/1cg5CB8Fn7leSFcRZdgQ46JEm4QSGHK7W 

5. Download KCP (1K manually Categorized Postcards) Data Set:
https://drive.google.com/file/d/1eAiH5o32u8Ctt0UxcGX_PY3-S238wbT6

6. Calculate KCP image embeddings with the following command: 
```
python embeddings.py <directory where KCP is extracted>
ex.: python embeddings.py f:/KCP
```
Note: KCP zip archive contains pre-generated embeddings from of the models, you may skip this part.

7. Run benchmark on KCP:
```
python img_retrieval_benchmark.py <directory where KCP is extracted>
ex.: python img_retrieval_benchmark f:/KCP
```
You should receive something like that as the script output: <br>
*Threshold: 0.297, Acc: 0.924, F1:0.925, Precision:0.917, Recall: 0.933*

Threshold - optimal threshold of similarity used to judge if 2 images are similar or not (calculated by the tool).

Accuracy, F1, Precision, Recall are standard metrics describing model performance under optimal threshold.
