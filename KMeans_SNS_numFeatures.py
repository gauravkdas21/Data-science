# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 12:06:03 2019

@author: Gaurav.Das
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 20:56:40 2019

@author: Gaurav.Das
"""

# Only numercial features for modelling

import pandas as pd
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
import time

warnings.filterwarnings("ignore")

os.chdir(r"D:\PERSONAL DATA\IMARTICUS\PYTHON Materials\ML")

data = pd.read_csv('data\\SNS\\snsdata.csv')

data.describe()

data.describe(include = 'O')

data.dtypes

data.isnull().any()

# gender and age has missing values

# gender is categorical
data.gender.value_counts()
data.gender.value_counts(dropna = False)
data.gender.value_counts(dropna = False)/data.shape[0]
# Here, we see that 2,724 records (9 percent) have missing gender data. Interestingly, there are over
# four times as many females as males in the SNS data, suggesting that males are not as inclined to 
# use SNS websites as females.

# age is continuous
data.age.describe()

data.age.isnull().sum()
data.age.isnull().sum()/data.shape[0]

sns.distplot(data.age.fillna(data.age.median()))

# A total of 5,086 records (17 percent) have missing ages. Also concerning is the fact that the
# minimum and maximum values seem to be unreasonable; it is unlikely that a 3 year old or a 106 year 
# old is attending high school. To ensure that these extreme values don’t cause problems for the 
# analysis, we’ll need to clean them up before moving on.

# A more reasonable range of ages for the high school students includes those who are at least 13 
# years old and not yet 20 years old. Any age value falling outside this range should be treated 
# the same as missing data-we cannot trust the age provided. To recode the age variable, we can use 
# the ifelse() function, assigning teenagethevalueofteenage if the age is at least 13 and less than 
# 20 years; otherwise, it will receive the value NA:

data.loc[(data.age < 13), 'age'] = np.nan
data.loc[(data.age >= 20), 'age'] = np.nan

# By rechecking the summary() output, we see that the age range now follows a distribution that 
# looks much more like an actual high school:

data.age.isnull().sum()
data.age.describe()

# Unfortunately, now we’ve created an even larger missing data problem. We’ll need to find a way to deal with these values before continuing with our analysis.

# Data preparation - dummy coding missing values

data['gender'] = data.gender.fillna('Unknown')
data.gender.value_counts()

df_gender = pd.get_dummies(data.gender, drop_first = 'True')

# Data preparation - imputing the missing values

np.mean(data.age)

data[['age', 'gradyear']].groupby(['gradyear'], as_index=False).mean().sort_values(by='gradyear', ascending=True)

data.loc[(data.age.isnull()) & (data.gradyear == 2006), 'age'] = 18.656
data.loc[(data.age.isnull()) & (data.gradyear == 2007), 'age'] = 17.706
data.loc[(data.age.isnull()) & (data.gradyear == 2008), 'age'] = 16.768
data.loc[(data.age.isnull()) & (data.gradyear == 2009), 'age'] = 15.819

data.age.isnull().any()

data1 = data.drop(['gender'], axis = 1)
df = pd.concat([data1, df_gender], axis = 1)

data = df.copy()

#remove erroneous cluster data
index_remove = df[(df['friends'] == 44) & (df['age'] == 18.119)].index.values.tolist()[0]
df =  df.drop(df.index[[index_remove]])


# training a model on the data

from sklearn import preprocessing

## scaling
scaler = preprocessing.StandardScaler().fit(df[['age', 'friends']])
dfs = scaler.transform(df[['age', 'friends']])

df[['age', 'friends']] = dfs

df = df.drop(['gradyear', 'age', 'friends', 'M', 'Unknown'], axis = 1)

data = df.copy()

## USE dfs (cotaines continuous features Age and Friends) for fitting 
from sklearn.cluster import KMeans

#kmeans = KMeans(n_clusters=5, random_state=0).fit(df)

sil = []
# Use silhouette coefficient to determine the best number of clusters
from sklearn.metrics import silhouette_score

for k in  list(range(2,11)): #[4,5,6,7,8]:
    kmeans = KMeans(n_clusters=k).fit(dfs)
    
    silhouette_avg = silhouette_score(dfs, kmeans.labels_)
    
    print('Silhouette Score for %i Clusters: %0.4f' % (k, silhouette_avg))
    
    sil.append(silhouette_avg)


krange = list(range(2,11))
plt.plot(krange, sil)
plt.xlabel("$K$")
plt.ylabel("Silhoutte score")
plt.show()
 
    
    
from sklearn import cluster
import numpy as np

sse = []
krange = list(range(2,11))
X = dfs  #df.values
for n in krange:
    model = cluster.KMeans(n_clusters=n, random_state=3)
    model.fit_predict(X)
    cluster_assignments = model.labels_
    centers = model.cluster_centers_
    sse.append(np.sum((X - centers[cluster_assignments]) ** 2))

plt.plot(krange, sse)
plt.xlabel("$K$")
plt.ylabel("Sum of Squares")
plt.show()

# 3 optimal clusters

# check groups

kmeans = KMeans(n_clusters=3).fit(dfs)

data['cluster'] = kmeans.labels_ #model.predict(df)

print('Cluster summary:')
summary = data.groupby(['cluster']).mean()
summary['count'] = data['cluster'].value_counts()
summary = summary.sort_values(by='count', ascending=False)
print(summary)
print(data.cluster.value_counts())

silhouette_avg = silhouette_score(dfs, kmeans.labels_)
print(silhouette_avg)

kmeans.inertia_


'''
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

y = labels
sse = {}
accuracy = []
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    labels_pred = kmeans.labels_
#     print(labels_pred.shape)

    # check how many of the samples were correctly labeled
    correct_labels = sum(labels == labels_pred)
    accuracy.append(correct_labels/float(y.size))
#     print("Result: %d out of %d samples were correctly labeled. when k = %d " % (correct_labels, y.size,k))
    print("correct %.02f percent classification at k = %d" % (correct_labels/float(y.size) * 100 ,k))
    
    get_cluster_metric(y, kmeans.labels_)

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()
'''