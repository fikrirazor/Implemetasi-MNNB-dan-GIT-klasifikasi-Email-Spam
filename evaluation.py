import preprocessing

import numpy as np
np.set_printoptions(precision=6)

def find_TP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))
    
def find_TN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))
   
def find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))
    
def find_FN(y_true, y_pred):
    return sum((y_true== 1 ) & (y_pred == 0))
    
def fit(y_true, y_pred):
    y_true = preprocessing.label_ke_numerik(y_true)
    y_pred = preprocessing.label_ke_numerik(y_pred)
    TP = find_TP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    return TP,FN,FP,TN

def my_confusion_matrix(y_true, y_pred):
    TP,FN,FP,TN = fit(y_true,y_pred)
    return np.array([[TP,TN],[FP,FN]])

def accuracy_score(y_true, y_pred):
    # calculates the fraction of samples predicted correctly
    TP,FN,FP,TN = fit(y_true,y_pred)  
    return (TP + TN ) / (TP + FP + FN + TN)

def recall(y_true, y_pred):
    TP,FN,FP,TN = fit(y_true,y_pred)
    return  TP / (TP + FN)

def precision(y_true, y_pred):
    TP,FN,FP,TN = fit(y_true,y_pred)
    return TP/(TP + FP)
    
def f1(y_true, y_pred ):
    TP,FN,FP,TN = fit(y_true,y_pred)
    return (0 if precision==0 and recall==0 else ((2 * TP)/(2*TP+FP+FN)))