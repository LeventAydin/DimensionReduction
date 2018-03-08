# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 21:37:42 2018

@author: Levent
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from DimensionReduction import DimensionReduction as dr

#Dataseti oku
dataFrame = pd.read_csv( "Data\\data.tsv", header=0, 
 delimiter=",", quoting=2 )

#datanın x ve y verileri
X = dataFrame[['Left-Weight','Left-Distance','Right-Weight','Right-Distance']]
Y = dataFrame[['Class-Name']]

#PCA Fonksiyonuna gönder
result = dr.Pca(X)

result['y-axis'] = 0.0
result['label'] = Y

#Grafik olarak göster
sns.lmplot('PC1', 'y-axis', data=result, fit_reg=False,  # x-axis, y-axis, data, no line
           scatter_kws={"s": 50}, # marker size
           hue="label") # color

# title
plt.title('PCA result')
