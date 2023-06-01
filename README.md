# Boosting Anomaly Detection using Unsupervised Diverse Test-Time Augmentation
The official code of the paper "Boosting Anomaly Detection using Unsupervised Diverse Test-Time Augmentation".

![proposed framework](https://github.com/nivgold/TTAD/blob/master/framework.jpg)

## Abstract

> Anomaly detection is a well-known task that involves the identification of abnormal events that occur relatively infrequently. Methods for improving anomaly detection performance have been widely studied. However, no studies utilizing test-time augmentation (TTA) for anomaly detection in tabular data have been performed. TTA involves aggregating the predictions of several synthetic versions of a given test sample; TTA produces different points of view for a specific test instance and might decrease its prediction bias. We propose the Test-Time Augmentation for anomaly Detection (TTAD) technique, a TTA-based method aimed at improving anomaly detection performance. TTAD augments a test instance based on its nearest neighbors; various methods, including the k-Means centroid and SMOTE methods, are used to produce the augmentations. Our technique utilizes a Siamese network to learn an advanced distance metric when retrieving a test instance's neighbors. Our experiments show that using our TTA technique significantly improves the performance of anomaly detection algorithms, as evidenced by the higher AUC results achieved on all datasets evaluated. Specifically, we observed average improvements of 0.037 AUC (3.7\%) using Autoencoder, 0.016 AUC (1.6\%) using OC-SVM, and 0.023 AUC (2.3\%) using LOF.

# Research Details
To see the paper's evaluation implementation, including exact datasets that were used, hyperparameters, architectures, etc., go to `ttad/research_code` and refer to its README

# Install
Using `pip` you can install the TTAD package:
```bash
pip install ttad
```


# TTAD Usage Example
```python
from sklear.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklear.datasets import make_classification

X, y = make_classification(n_samples=1000,
                           n_features=10,
                           n_informative=3,
                           n_classes=2
                          )
X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.3)

```

## Normal Flow
```python
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
```

## Using TTAD
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