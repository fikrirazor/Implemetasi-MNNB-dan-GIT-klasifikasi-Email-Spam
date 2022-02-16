'''
Created on Thu Feb 14 18:34:19 2019

@Author : Fikri Rozan Imadudin
'''
import numpy as np
import preprocessing

class gitb(object):
    '''
    Menghitung Complete Gini Index Text A antara dua variabel kontinu

    Parameters
    ----------
    alpha : smoothing agar tidak terjadi 0/0 infinite
    X : berupa sparsing matrix term document frequency
    y : kelas harus berupa angka
    Returns
    -------
    gita : matrix kelas X banyaknya fitur
    
    References
    ----------
    [1] Park, H., Kwon, S., & Kwon, H. C. (2010, June). Complete gini-index text (git) feature-selection algorithm for text classification. 
    In The 2nd International Conference on Software Engineering and Data Mining (pp. 366-371). IEEE.

    '''
    def __init__(self,alpha=1.0):
        self.alpha=alpha
    
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
        # Menjumlahkan Tiap Fitur/Row
        sumrow = []
        for i in range(len(X_index)):
            sumrow.append(np.array(X_index[i].sum(axis=0)) )
        # Menggabungkan menjadi matrix
        count = np.vstack(sumrow) + self.alpha
        # Menjumlahkan semua Fitur Spam dan Ham
        jumlahw = np.sum(count, axis=0)
        #np.seterr(divide='ignore', invalid='ignore')
        pcw=count/(jumlahw+count)
        # Gini Index Text A
        gita=(pcw)**2
        # Gini Index Text B
        self.gitb=np.abs(gita/np.log2(jumlahw))
        return self.gitb
