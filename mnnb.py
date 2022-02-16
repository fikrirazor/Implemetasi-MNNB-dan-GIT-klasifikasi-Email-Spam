from __future__ import division
import numpy as np
np.set_printoptions(precision=6)
import preprocessing

class mnnb(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        
        y = preprocessing.label_ke_numerik(y)

        # Mengambil index Ham dan Spam dari masing-masing data
        y_index = {}
        for c in np.unique(y):
            y_index[c] = np.where(y == c)[0]
        # Mencocokan Index ke data X
        X_index = []
        for i in range(len(y_index)):
            X_index.append(X[y_index[i]])
        
        count_sample = X.shape[0]
        
        self.class_log_prior_ = [np.log(i.shape[0] / count_sample) for i in X_index]
        sumrow = []
        
        for i in range(len(X_index)):
            sumrow.append(np.array(X_index[i].sum(axis=0)) )
        # Menggabungkan menjadi matrix
        count = np.vstack(sumrow) + self.alpha
        self.feature_log_prob_ = np.log(count / count.sum(axis=1)[np.newaxis].T)
        return self

    def predict_log_proba(self, X):
            return [(self.feature_log_prob_ * x.A).sum(axis=1) + self.class_log_prior_
                    for x in X]

    def predict(self, X):
            return np.argmax(self.predict_log_proba(X), axis=1)
    
