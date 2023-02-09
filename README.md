# MIPT (MSAI) - Computer Vision for Postal cards Library Thesis project

This repository contains materials and description of the experiments undertaken during our thesis work.

### Image Retrieval Experiment
Target - show that AI-based solutions can confidently detect post cards depicting same landmarks. <br> 
Approach - understand theory, find existing technological solutions appropriate for the task, select best one, based on it come up with workable POC, demonstrate convincing results.  

Tangible Results achieved:
- Four nn-models were trained and tested for the IR task, all proved capability of detecting same landmark postcards. 
- KISA (1K Image Similarity Assembly) Data Set with domain-specific images (postcards).

Open question:
- issues with out-of-domain pictures (non-landmarks).
- issues with pictures containing huge blank borders on the edges.
- issues with pictures depicting river banks, lakes, bushes.

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

5. Download KISA Data Set:
https://drive.google.com/file/d/1eAiH5o32u8Ctt0UxcGX_PY3-S238wbT6

6. Calculate KISA image embeddings with the following command: 
```
python embeddings.py <directory where KCP is extracted>
ex.: python embeddings.py f:/KCP
```
Note: KISA zip archive contains pre-generated embeddings from of the models, you may skip this part.

7. Run benchmark on KISA:
```
python img_retrieval_benchmark.py <directory where KCP is extracted>
ex.: python img_retrieval_benchmark f:/KCP
```
You should receive something like that as the script output: <br>
*Threshold: 0.297, Acc: 0.924, F1:0.925, Precision:0.917, Recall: 0.933*

Threshold - optimal threshold of similarity used to judge if 2 images are similar or not (calculated by the tool).

Accuracy, F1, Precision, Recall are standard metrics describing model performance under optimal threshold.

### KISA (1K Image Similarity Assembly) Data Set
Dataset is intended to measure image similarity performance on scanned postal cards (pre-computer age, 1900 - 1930 A.D.). 

Image pairs generation:
DS Python wrapper generates 2302 image pairs: 
1151 "positive" pairs assembled from images of same landmark category, labeled as "positive" since both images in the pair depict similar objects.
1151 "negative" pairs, constructed from images of different landmark categories and images of non-landmark type. Since two images in such pair have non-related content the pair by design should have a low mutual similarity. 

DS content:
Total 1000 images: 489 landmark and 511 non-landmark images.
Total 101 landmark categories.
872 scanned postcards and portraits, 482 of them are landmarks images.
128 internet era images, 7 of them depicting landmarks.

Image-Landmark relation:
Each landmark images is labeled with one or more landmark category, 
there are at least 3 and on average 5 images associated with each landmark category.
