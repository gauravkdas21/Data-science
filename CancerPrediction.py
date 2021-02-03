# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:51:53 2019

@author: Gaurav.Das
"""

# ----------------------------- Breast Cancer Prediction ----------------------------------------------------------

# Problem Statement 
# Breast cancer is one of the most common cancers among women in the world. Early detection of 
# breast cancer is essential in reducing their life losses. Build a predictive model using machine
# learning algorithms to predict whether the tumor is benign or malignant. 

# Data Description 
# Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. 
# The mean, standard error and "worst" or largest (mean of the three largest values) of these 
# features were computed for each image, resulting in 30 features. 


import pandas as pd
import os
import warnings
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
import time

warnings.filterwarnings("ignore")

data = pd.read_csv('data\\cancerData\\cancerdata.csv')

# list all columns (for reference)
data.columns

data = data.drop(['id'], axis = 1)

# target variable (diagnosis)

data['diagnosis'].value_counts() # balanced data set

# convert the response to numeric values and store as a new column
data['diagnosis'] = data.diagnosis.map({'B':0, 'M':1})

#data.isnull().sum()
data.describe()
data.info()
#data.dtypes
data.shape

major = np.round(data.diagnosis.value_counts()[0]/data.shape[0],3)*100
minor = 100 - major

# so, balanced data

# Plot histograms for each variable
data.hist(figsize = (10, 10))
plt.show()

# Create scatter plot matrix
#scatter_matrix(data, figsize = (18,18))
#plt.show()

################################### Wihtout scaling #####################################
t0 = time.time()

print('*'*80)
print('WITHOUT SCALING:')

#loading libraries

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X = np.array(data.iloc[:,1:])
y = np.array(data['diagnosis'])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)

knn = KNeighborsClassifier(n_neighbors = 12)
knn.fit(X_train,y_train)

knn.score(X_test,y_test)


#Performing cross validation
neighbors = []
cv_scores = []
from sklearn.model_selection import cross_val_score
#perform 10 fold cross validation
for k in range(1,51,2):
    neighbors.append(k)
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn,X_train,y_train,cv=10, scoring = 'accuracy')
    cv_scores.append(scores.mean())
    
    
#Misclassification error versus k
MSE = [1-i for i in cv_scores]


#determining the best k
optimal_k = neighbors[MSE.index(min(MSE))]
print('The optimal number of neighbors is: %d ' %optimal_k)

#plot misclassification error versus k

#plt.figure(figsize = (10,6))
#plt.plot(neighbors, MSE)
#plt.xlabel('Number of neighbors (K)')
#plt.ylabel('Misclassification Error')
#plt.show()

#Without Hyper Parameters Tuning
#importing the metrics module
from sklearn import metrics
#making the instance
model = KNeighborsClassifier(n_jobs=-1)
#learning
model.fit(X_train,y_train)
#Prediction
prediction=model.predict(X_test)
#evaluation(Accuracy)
print("Accuracy:",metrics.accuracy_score(prediction,y_test))
#evaluation(Confusion Metrix)
print("Confusion Matrix:\n",metrics.confusion_matrix(prediction,y_test))


#With Hyper Parameters Tuning
#importing modules
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
#making the instance
model = KNeighborsClassifier(n_jobs=-1)
#Hyper Parameters Set
params = {'n_neighbors':[9,11,13,15],
          'leaf_size':[25,30,35,40],
          'weights':['uniform', 'distance'],
          'algorithm':['auto', 'ball_tree','kd_tree','brute'],
          'n_jobs':[-1]}
#Making models with hyper parameters sets
model1 = GridSearchCV(model, param_grid=params, n_jobs=1)
#Learning
model1.fit(X_train,y_train)
#The best hyper parameters set
print("Best Hyper Parameters:\n",model1.best_params_)
#Prediction
prediction=model1.predict(X_test)
#evaluation(Accuracy)
print("Accuracy after tuning:",metrics.accuracy_score(prediction,y_test))
#evaluation(Confusion Metrix)
print("Confusion Matrix after tuning:\n",metrics.confusion_matrix(prediction,y_test))

#The default hyper parameters of KNN are not bad in this case

############################### AFter scaling ##################################################
print('*'*80)
print('AFTER SCALING:')
#loading libraries

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#X = np.array(data.iloc[:,1:])
#y = np.array(data['diagnosis'])

scaler = StandardScaler()
Xs = scaler.fit_transform(X) # X has been scaled using z transformation (normal assumption from plotting)

Xs_train, Xs_test, y_train, y_test = train_test_split(Xs,y, test_size = 0.33, random_state = 42)

knn = KNeighborsClassifier(n_neighbors = 12)
knn.fit(Xs_train,y_train)

knn.score(Xs_test,y_test)


#Performing cross validation
neighbors = []
cv_scores_s = []
#perform 10 fold cross validation
for k in range(1,51,2):
    neighbors.append(k)
    knn = KNeighborsClassifier(n_neighbors = k)
    scores_s = cross_val_score(knn,Xs_train,y_train,cv=10, scoring = 'accuracy')
    cv_scores_s.append(scores_s.mean())
    
    
#Misclassification error versus k
MSE_s = [1-x for x in cv_scores_s]

#determining the best k
optimal_k_s = neighbors[MSE_s.index(min(MSE_s))]
print('The optimal number of neighbors is: %d ' %optimal_k_s)

#plot misclassification error versus k

plt.figure(figsize = (10,6))
plt.plot(neighbors, MSE_s)
plt.xlabel('Number of neighbors (K)')
plt.ylabel('Misclassification Error')
plt.show()


#Without Hyper Parameters Tuning
#importing the metrics module
from sklearn import metrics
#making the instance
model = KNeighborsClassifier(n_jobs=-1)
#learning
model.fit(Xs_train,y_train)
#Prediction
prediction_s = model.predict(Xs_test)
#evaluation(Accuracy)
print("Accuracy:",metrics.accuracy_score(prediction_s,y_test))
#evaluation(Confusion Metrix)
print("Confusion Metrix:\n",metrics.confusion_matrix(prediction_s,y_test))


#With Hyper Parameters Tuning
#importing modules
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
#making the instance
model = KNeighborsClassifier(n_jobs=-1)
#Hyper Parameters Set
params = {'n_neighbors':[5,6,7,8,9,10],
          'leaf_size':[1,2,3,5],
          'weights':['uniform', 'distance'],
          'algorithm':['auto', 'ball_tree','kd_tree','brute'],
          'n_jobs':[-1]}
#Making models with hyper parameters sets
model1 = GridSearchCV(model, param_grid=params, n_jobs=1)
#Learning
model1.fit(Xs_train,y_train)
#The best hyper parameters set
print("Best Hyper Parameters:\n",model1.best_params_)
#Prediction
prediction_s = model1.predict(Xs_test)
#evaluation(Accuracy)
print("Accuracy after tuning:",metrics.accuracy_score(prediction_s,y_test))
#evaluation(Confusion Metrix)
print("Confusion Metrix after tuning:\n",metrics.confusion_matrix(prediction_s,y_test))

#The default hyper parameters of KNN are not bad


# Model Comparison between scaled and not scaled -----------------------------------

print('*'*80)
print('Comparison')

from sklearn.metrics import roc_auc_score, roc_curve

# for un-scaled
knn.fit(X_train, y_train)
pred_prob = knn.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  pred_prob)
auc = roc_auc_score(y_test, pred_prob)
plt.plot(fpr,tpr,label=", auc="+str(np.round(auc,3)))
plt.legend(loc=4)
plt.tight_layout()

# for scaled
knn.fit(Xs_train, y_train)
pred_prob = knn.predict_proba(Xs_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  pred_prob)
auc_s = roc_auc_score(y_test, pred_prob)
plt.plot(fpr,tpr,label=", auc_s="+str(np.round(auc_s,3)))
plt.legend(loc=4)
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.title('ROC')
plt.tight_layout()



#plot misclassification error versus k

dic = {'K':neighbors, 'MSE':[i*100 for i in MSE], 'MSE_s':[j*100 for j in MSE_s]}
df = pd.DataFrame.from_dict(dic)#.transpose()

plt.figure(figsize = (10,6))
plt.plot('K', 'MSE', data=df, marker='o', color='blue', linewidth=2)
plt.plot('K', 'MSE_s', data=df, marker='^', color='red', linewidth=2)
plt.legend()
plt.xlabel('Number of neighbors (K)')
plt.ylabel('Error %')
plt.show()

print('*'*80)
print('Optimal no. of neighbors without scaling: ',optimal_k)
print('Optimal no. of neighbors after scaling: ',optimal_k_s)

print('*'*80)
print('F1 score without scaling:\n', metrics.classification_report(y_test, prediction)) 
print('*'*80)
print('F1 score after scaling:\n', metrics.classification_report(y_test, prediction_s)) 

t1 = time.time()
t = t1-t0

print('Time taken for model completion: '+ str(np.round(t,2)) + ' secs')
