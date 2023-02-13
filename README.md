# MIPT (MSAI) - Computer Vision for Postal cards Library Thesis project

This repository contains materials and description of the experiments undertaken during our thesis work.

### Image Retrieval Experiment
Target - show that AI-based solution can confidently detect post cards depicting same landmarks and identify it limitations. <br> 
Approach - understand theory, find existing technological solutions appropriate for the task, select best one, based on it come up with workable POC, demonstrate convincing results.  

Tangible Results achieved:
- KISA (1K Image Similarity Assembly) Data Set with domain-specific images (postcards) was created.
- Four nn-models were trained and tested for the IR task, all demonstrated acceptable results.

Open question:
- issues with out-of-domain pictures (non-landmarks).
- issues with pictures containing huge blank borders on the edges.
- issues with pictures depicting river banks, lakes, bushes.

Steps to reproduce the experiment:

1. Make sure you have python, pytorch and other libraries listed in ```requirements.txt``` are installed.
2. Clone this repository into your local IDE
3. For model training and embeddings generation you'll need a CUDA compatible device (graphic card) for ~2 weeks x 24hr period along with its drivers.
4. Download Google Landmarks v2 dataset (500Mb): https://github.com/cvdfoundation/google-landmark <br>
GLMv2 has several versions, you need ```train clean``` version of train index (train_clean.csv ) along with complete test part of the DS. 
5. Train one or more neuro-network model(s) with code provided in this repository.
```
python train.py --config <model number>
ex: python train.py --config config6
```
Pre-trained weights for one of the models are available here: https://drive.google.com/file/d/1cg5CB8Fn7leSFcRZdgQ46JEm4QSGHK7W 

6. Download KISA Data Set:
https://drive.google.com/file/d/1eAiH5o32u8Ctt0UxcGX_PY3-S238wbT6

7. Calculate KISA image embeddings with the following command: 
```
python embeddings.py <directory where KCP is extracted>
ex.: python embeddings.py f:/KCP
```
Note: KISA zip archive contains pre-generated embeddings from of the models, so you may skip this step.

8. Run benchmark on KISA:
```
python img_retrieval_benchmark.py <directory where KCP is extracted>
ex.: python img_retrieval_benchmark f:/KCP
```
You should receive the following as the script output: <br>
```
Landmark postcards: Threshold:0.27900, Acc:0.964, F1:0.963, Precision:0.972, Recall:0.955
Blend of landmark and non-landmark postcards: Threshold:0.32100, Acc:0.911, F1:0.912, Precision:0.900, Recall:0.924
```

Threshold - optimal threshold of similarity used to judge if 2 images are similar or not (calculated by the tool).
Accuracy, F1, Precision, Recall are standard metrics describing model performance under optimal threshold.

### KISA (1K Image Similarity Assembly) Data Set
Dataset is intended to measure image similarity performance on scanned postal cards (pre-computer age, 1900 - 1930 A.D.). 

Image pairs generation:
DS Python wrapper generates 2308 image pairs: 
1154 "positive" pairs assembled from images of same landmark category, labeled as "positive" since both images in the pair depict similar objects.
1154 "negative" pairs, constructed from images of different landmark categories (or of non-landmark categories). Since two images in such pair have non-related content the pair by design should have a low mutual similarity. 

DS content:
Total 1001 images: 489 landmark and 512 non-landmark images.
Total 101 landmark categories.
873 scanned postcards and portraits, 482 of them are landmarks images.
128 internet era images, 7 of them depicting landmarks.

Image-Landmark relation:
Each landmark images is labeled with one or more landmark category, 
there are at least 3 and 5 images on average associated with each landmark category.
