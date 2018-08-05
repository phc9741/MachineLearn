import pandas as pd
import io
import requests
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()  # for plot styling
from sklearn.cluster import KMeans

from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


from sklearn.datasets.samples_generator import make_blobs

"""
AUthor: Pierre Hau Cruikshank
        Columbia University

Comments: This is QMSS Machine Learning for Social Science homework 03
        
"""

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
red = requests.get(url).content
red_ds = pd.read_csv(io.StringIO(red.decode('utf-8')), sep=';')
print("----------- This is data set for RED wine\n")
print(red_ds.describe())

print("----------- These are the column names for RED wine\n")
print(red_ds.columns.values)

X  = np.array(red_ds).astype(float)

fix_acid_red = red_ds['fixed acidity'].astype('float64')
volatile_acidity_red = red_ds['volatile acidity'].astype('float64')
citric_acid_red = red_ds['citric acid'].astype('float64')
residual_sugarred = red_ds['residual sugar'].astype('float64')
chlorides_red = red_ds['chlorides'].astype('float64')
free_sulfur_dioxide_red = red_ds['free sulfur dioxide'].astype('float64')
total_sulfur_dioxide_red = red_ds['total sulfur dioxide'].astype('float64')
density_red = red_ds['density'].astype('float64')
pH_red = red_ds['pH'].astype('float64')
sulphates_red = red_ds['sulphates'].astype('float64')
alcohol_red = red_ds['alcohol'].astype('float64')
quality_red = red_ds['quality'].astype('float64')

header = red_ds.columns
#print(red.decode())

fig = plt.figure(figsize=(14,12))

feat_comb_1 = [1,2,3,4,5]
feat_comb_2 = [1,2,3,4,5]
#This next line was alluded to above, it essentially gives the transpose of X
feature_array = [X[:,j] for j in range(len(header))]

nfeat = len(feat_comb_1)

for j in range(nfeat):
    for k in range(nfeat):
        #subplot takes 3 arguments.
        # If the final plot is going to be 4 subplots x 4 subplots for example,
        # both of these arguments must be equal to 4.
        # The third argument should be incremented sequentially and matplotlib will then decide, for example
        # that in the case of a 5x5 matrix of plots, the 9th plot should be in the 4th plot on the second row
        plt.subplot(nfeat, nfeat, j + 1 + k * nfeat)
        plt.scatter(feature_array[feat_comb_1[j]], feature_array[feat_comb_2[k]])
        plt.xlabel(header[feat_comb_1[j]])
        plt.ylabel(header[feat_comb_2[k]])
        fig.tight_layout()

plt.show()


fig, axes = plt.subplots(figsize=(16,6))
bp = plt.boxplot(X)
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='o')
plt.xlabel('Features')
plt.ylabel('Value')
axes.set_xticklabels(header, rotation=270)
plt.grid()
plt.show()