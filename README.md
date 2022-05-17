# Boosting Anomaly Detection using Unsupervised Diverse Test-Time Augmentation
The official code of the paper "Boosting Anomaly Detection using Unsupervised Diverse Test-Time Augmentation".

![proposed framework](https://github.com/nivgold/ADTTA/blob/main/fig1.jpg)

## Abstract

> Test-time augmentation (TTA) involves aggregating the predictions of several synthetic versions of a given test sample. TTA produces different points of view for a specific test instance and might decrease its prediction bias.
Anomaly Detection is a well-known task that involves identifying abnormal events that occur relatively infrequently and can have dangerous consequences. There exist no studies that utilize TTA for tabular anomaly detection. We propose the Test-Time Augmentation for anomaly Detection (TTAD) technique - a TTA-based method to improve anomaly detection performance. TTAD augments a test instance based on its nearest neighbors while using multiple methods to produce the augmentations, including a trained k-Means centroids' and SMOTE. Our approach utilizes a Siamese Network to learn an advanced distance metric when retrieving a test instance's neighbors. We show that the anomaly detector that uses our TTA approach achieves significantly higher AUC results for all evaluated datasets.


## Repository Files

- ├──`data/`
  - └──`[dataset]/`
    - ├──`[dataset]_features.npy`: The preprocessed features of `[dataset]`
    - ├──`[dataset]_labels.npy`: The labels of `[dataset]`
    - ├──`[dataset]_pairs_X.npy`: The features of the pairs used for training the Siamese network of `[dataset]`
    - └──`[dataset]_pairs_y.npy`: The labels of the pairs used for training the Siamese network of `[dataset]`
- ├──`src/`: The source code implementation of our approach 


## Datasets

These are the datasets we used in our experiments and described in the paper.

As noted, the datasets were taken from [ODDS](http://odds.cs.stonybrook.edu/).
|Dataset|#Samples|#Dim|Outliers|
|:---:|:---:|:---:|:---:|
|[Annthyroid](http://odds.cs.stonybrook.edu/annthyroid-dataset/)|7200|6|7.42 (%)|
|[Cardio](http://odds.cs.stonybrook.edu/cardiotocogrpahy-dataset/)|1831|21|9.6 (%)|
|[Mammo](http://odds.cs.stonybrook.edu/mammography-dataset/)|11183|6|2.32 (%)|
|[Satellite](http://odds.cs.stonybrook.edu/satellite-dataset/)|6435|36|32 (%)|
|[Seismic](http://odds.cs.stonybrook.edu/seismic-dataset/)|2584|11|6.5 (%)|
|[Thyroid](http://odds.cs.stonybrook.edu/thyroid-disease-dataset/)|3772|6|2.5 (%)|
|[Vowels](http://odds.cs.stonybrook.edu/japanese-vowels-data/)|1456|12|3.4 (%)|
|[Yeast](https://archive.ics.uci.edu/ml/datasets/Yeast)|1364|8|4.7 (%)|


## Dependencies

The required dependencies are specified in `environment.yml`.

For setting up the environment, use [Anaconda](https://www.anaconda.com/):
```bash
$ conda env create -f environment.yml
$ conda activate adtta
```


## Running the Code

**NOTES:**

- A valid dataset name (instead of mammo) can be only one from the described earlier.
- It is important to run the scripts from being inside `src/` (i.e. ```$ cd src/```)

---

* ### **Preprocessing**
        
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A preprocessing phase on a desired dataset.

```bash
$ cd src/
$ python preprocess.py --dataset mammo
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;From this script, 4 files are going to be generated in the desired dataset `data/` folder:

&emsp;&emsp;&emsp;&emsp;- `mammo_features.npy`

&emsp;&emsp;&emsp;&emsp;- `mammo_labels.npy`

&emsp;&emsp;&emsp;&emsp;- `mammo_pairs_X.npy`

&emsp;&emsp;&emsp;&emsp;- `mammo_pairs_y.npy`


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;With these files, you can run the train and test.

---

* ### **Train & Test**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;To train the estimator, as well as the Nearest Neighbors models (both Euclidean and Siamese) run:

```bash
$ cd src/
$ python main.py --dataset mammo --neighbors 10 --augmentations 7
```
