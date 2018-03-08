# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 21:44:42 2018

@author: Levent
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DimensionReduction():
    def Pca(X):
        
        x_std = StandardScaler().fit_transform(X)
        
        # x_std özellikleri ve kolonları
        features = x_std.T 
        #Kovaryans matrisi
        covariance_matrix = np.cov(features)
        #Dizinin özdeğerleri ve doğru özvektörlerini hesapla
        eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
        
        projected_X = x_std.dot(eig_vecs.T[0])
        
        result = pd.DataFrame(projected_X, columns=['PC1'])
        return result