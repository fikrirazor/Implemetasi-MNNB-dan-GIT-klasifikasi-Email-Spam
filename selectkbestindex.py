
import numpy as np

class selectkbestindex(object):
    def __init__(self,k):
        self.k = k
    
    def get(self,git):
        # Mendapatkan hasil terbaik pada GIT berdasarkan kelasnya
        subsetmax = git.max(axis=0)
        # Mendapatkan index terbaik
        self.xterbaik = np.argsort(subsetmax)[-self.k:]
        return self.xterbaik

