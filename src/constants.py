evaluated_algorithms = [
    'WO_TTA_Baseline',
    'Gaussian_TTA_Baseline',
    'Euclidean_SMOTE_TTA',
    'Siamese_SMOTE_TTA',
    'Euclidean_Kmeans_TTA',
    'Siamese_Kmeans_TTA'
]

# define anomaly detection estimators from the Experiments section
evaluated_estimators = [
    'Autoencoder',
    # 'Isolation Forest',
    'One-Class SVM',
    'Local Outlier Factor'
]