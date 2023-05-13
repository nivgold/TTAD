# Boosting Anomaly Detection using Unsupervised Diverse Test-Time Augmentation
The official code of the paper "Boosting Anomaly Detection using Unsupervised Diverse Test-Time Augmentation".

![proposed framework](https://github.com/nivgold/TTAD/blob/master/framework.jpg)

## Abstract

> Anomaly detection is a well-known task that involves the identification of abnormal events that occur relatively infrequently. Methods for improving anomaly detection performance have been widely studied. However, no studies utilizing test-time augmentation (TTA) for anomaly detection in tabular data have been performed. TTA involves aggregating the predictions of several synthetic versions of a given test sample; TTA produces different points of view for a specific test instance and might decrease its prediction bias. We propose the Test-Time Augmentation for anomaly Detection (TTAD) technique, a TTA-based method aimed at improving anomaly detection performance. TTAD augments a test instance based on its nearest neighbors; various methods, including the k-Means centroid and SMOTE methods, are used to produce the augmentations. Our technique utilizes a Siamese network to learn an advanced distance metric when retrieving a test instance's neighbors. Our experiments show that using our TTA technique significantly improves the performance of anomaly detection algorithms, as evidenced by the higher AUC results achieved on all datasets evaluated. Specifically, we observed average improvements of 0.037 AUC (3.7\%) using Autoencoder, 0.016 AUC (1.6\%) using OC-SVM, and 0.023 AUC (2.3\%) using LOF.


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
|[Cardiotocography](http://odds.cs.stonybrook.edu/cardiotocogrpahy-dataset/)|1831|21|9.6 (%)|
|[Mammography](http://odds.cs.stonybrook.edu/mammography-dataset/)|11183|6|2.32 (%)|
|[Satellite](http://odds.cs.stonybrook.edu/satellite-dataset/)|6435|36|32 (%)|
|[Seismic](http://odds.cs.stonybrook.edu/seismic-dataset/)|2584|11|6.5 (%)|
|[Thyroid](http://odds.cs.stonybrook.edu/thyroid-disease-dataset/)|3772|6|2.5 (%)|
|[Vowels](http://odds.cs.stonybrook.edu/japanese-vowels-data/)|1456|12|3.4 (%)|
|[Yeast](https://archive.ics.uci.edu/ml/datasets/Yeast)|1364|8|4.7 (%)|
|[Wine](https://archive.ics.uci.edu/ml/datasets/Wine)|178|13|7.7 (%)|
|[Satimage](https://archive.ics.uci.edu/ml/datasets/Statlog+%28Landsat+Satellite%29)|5803|36|1.2 (%)|


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

## TTAD Usage
```python
from sklear.ensemble import RandomForest
from sklearn,model_selection import train_test_split
from sklear.data import make_data

X, y = make_data(num_samples=1000, num_class=2)
X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.3)

```

## Normal Flow
```python
rfc = RandomForest()
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
```

## TTAD
```python
from ttad import TTADClassifier

rfc = RandomForest()
ttad_clf = TTADClassifier(model=rfc)

ttad_clf.fit(X_train)
y_pred = ttad_clf.predict(X_test)
```

## Changing Distance Metric
```python
from ttad.data_selector.distance_metric import Adaptive

adaptive_dm = Adaptive()
ttad_clf = TTADClassifier(model=rfc,
                          distance_metric=adaptive_dm)
```

## Augmentation Producer Method
```python
from ttad.augmentation_producer import SMOTE

smote_ap = SMOTE()
ttad_clf = TTADClassifier(model=rfc,
                          augmentation_producer=smote_ap)
```

# Citing TTAD
If you use TTAD, please cite our work:
```
@article{cohen2023boosting,
  title={Boosting Anomaly Detection Using Unsupervised Diverse Test-Time Augmentation},
  author={Cohen, Seffi and Goldshlager, Niv and Rokach, Lior and Shapira, Bracha},
  journal={Information Sciences},
  year={2023},
  publisher={Elsevier}
}
```