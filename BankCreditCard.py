# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 20:39:25 2019

@author: Gaurav.Das
"""

import pandas as pd
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

data_original = pd.read_csv('data\\BankCreditCard\\BankCreditCard.csv')

data = data_original

# list all columns (for reference)
data.columns

#  Default_Payment (response)
data = data.drop(['Customer ID'], axis = 1)

data.isnull().sum()

data.describe()

catCols = ["Gender", "Academic_Qualification", "Marital", "Repayment_Status_Jan", "Repayment_Status_Feb",
            "Repayment_Status_March", "Repayment_Status_April", "Repayment_Status_May", "Repayment_Status_June",
            "Default_Payment"]

## ------------------------------------------EDA ------------------------------------------------------------------------------------------

## Target variable -------------------------------------------------------------------

sns.countplot(x=data['Default_Payment'])

# percentage of 0's and 1's
np.round(data.Default_Payment.value_counts()/data.shape[0],2)

## imbalanced data

## EDA categorical independent features ------------------------------------------------

# Set up the matplotlib figure
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.despine(left=True)
sns.countplot(x=data['Gender'], hue = 'Default_Payment', data = data, ax = axes[0][0])
sns.countplot(x=data['Academic_Qualification'], hue = 'Default_Payment', data = data, ax = axes[1][0])
sns.countplot(x=data['Marital'],  hue = 'Default_Payment', data = data, ax = axes[0][1])
plt.setp(axes, yticks=[])
plt.tight_layout()


def createProportions(df,colName, dependentColName):
    
    tab = pd.crosstab(df[colName],  df[dependentColName],margins = False)
    prop = []
    for i in range(tab.shape[0]):
        value = tab.iloc[i,1]/tab.iloc[i,0]
        prop.append(value)
    tab['prop'] = prop

    return tab

createProportions(data,'Marital', 'Default_Payment')
createProportions(data,'Academic_Qualification', 'Default_Payment')
createProportions(data,'Gender', 'Default_Payment')


# Set up the matplotlib figure
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(3, 2, figsize=(7, 7), sharex=True)
sns.despine(left=True)
sns.countplot(x=data['Repayment_Status_Jan'], hue = 'Default_Payment', data = data, ax = axes[0][0])
sns.countplot(x=data['Repayment_Status_Feb'], hue = 'Default_Payment', data = data, ax = axes[0][1])
sns.countplot(x=data['Repayment_Status_March'],  hue = 'Default_Payment', data = data, ax = axes[1][0])
sns.countplot(x=data['Repayment_Status_April'], hue = 'Default_Payment', data = data, ax = axes[1][1])
sns.countplot(x=data['Repayment_Status_May'], hue = 'Default_Payment', data = data, ax = axes[2][0])
sns.countplot(x=data['Repayment_Status_June'],  hue = 'Default_Payment', data = data, ax = axes[2][1])
plt.setp(axes, yticks=[])
plt.tight_layout()

createProportions(data,'Repayment_Status_June', 'Default_Payment')
createProportions(data,'Repayment_Status_Jan', 'Default_Payment')


# group the academic levels as 1,2 and >= 3  OR 0(1) and 1(2,3,4) but 2nd technique will not be proper 
# due to distribution of target values in 2 and 3, let's see what happens afetr technique 1

def club(df, feature, a, b, newValue):
    
    for i in range(a, b):
        df[feature][df[feature] == i] = newValue
    
    x = df[feature].value_counts()
    
    return x

def labelCh(df, feature, a, b):
    
    for i in range(a, b):
        df[feature][df[feature] == i] = i-1
    
    x = df[feature].value_counts()
    
    return x

club(data, 'Repayment_Status_Jan', 2, 7, 1)
club(data, 'Repayment_Status_Feb', 2, 7, 1)
club(data, 'Repayment_Status_March', 2, 7, 1)
club(data, 'Repayment_Status_April', 2, 7, 1)
club(data, 'Repayment_Status_May', 2, 7, 1)
club(data, 'Repayment_Status_June', 2, 7, 1)

print(data['Academic_Qualification'].value_counts())
club(data, 'Academic_Qualification', 4, 7, 3)

sns.countplot(x=data['Academic_Qualification'], hue = 'Default_Payment', data = data)
plt.tight_layout()

data[['Academic_Qualification', 'Default_Payment']].groupby(['Academic_Qualification'], as_index=False).mean().sort_values(by='Academic_Qualification', ascending=True)
# Here percentage conversion is the same for 2 and 3 levels
# group the academic levels as 0(1) and 1(2,3): makes sense post grad approx.== professionals and elite classes
data['Academic_Qualification'][data['Academic_Qualification']==1] = 0
club(data, 'Academic_Qualification', 2, 4, 1)

# group the Gender levels as 0 and 1 instead of 1 and 2
labelCh(data, 'Gender', 1, 3)

# group the marital levels as (0,1) as 0 (due to proprotion of blue)
# and (2,3) as 1; (due to proprotion of red)
print(data['Marital'].value_counts())
club(data, 'Marital', 1, 2, 0)
club(data, 'Marital', 2, 4, 1)


## Rechecking the plots
# Set up the matplotlib figure
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.despine(left=True)
sns.countplot(x=data['Gender'], hue = 'Default_Payment', data = data, ax = axes[0][0])
sns.countplot(x=data['Academic_Qualification'], hue = 'Default_Payment', data = data, ax = axes[1][0])
sns.countplot(x=data['Marital'],  hue = 'Default_Payment', data = data, ax = axes[0][1])
plt.setp(axes, yticks=[])
plt.tight_layout()

# Set up the matplotlib figure
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(3, 2, figsize=(7, 7), sharex=True)
sns.despine(left=True)
sns.countplot(x=data['Repayment_Status_Jan'], hue = 'Default_Payment', data = data, ax = axes[0][0])
sns.countplot(x=data['Repayment_Status_Feb'], hue = 'Default_Payment', data = data, ax = axes[0][1])
sns.countplot(x=data['Repayment_Status_March'],  hue = 'Default_Payment', data = data, ax = axes[1][0])
sns.countplot(x=data['Repayment_Status_April'], hue = 'Default_Payment', data = data, ax = axes[1][1])
sns.countplot(x=data['Repayment_Status_May'], hue = 'Default_Payment', data = data, ax = axes[2][0])
sns.countplot(x=data['Repayment_Status_June'],  hue = 'Default_Payment', data = data, ax = axes[2][1])
plt.setp(axes, yticks=[])
plt.tight_layout()

dataCAT = data[catCols].drop(['Default_Payment'], axis = 1)

### EDA - numerical features ----------------------------------------------------------------------
cols = data.columns.tolist()
numCols = [cols[i] for i in range(len(cols)) if i == 0 or i == 4] #names(data3)[c(1,5,12:23)]
numCols2 = cols[11:-1]
numCols.extend(numCols2)

dataNUM = data[numCols]

# Univariate analysis
# Set up the matplotlib figure
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(3, 3, figsize=(7, 7), sharex=True)
sns.despine(left=True)
sns.distplot(dataNUM.iloc[:,0], ax = axes[0][0])
sns.distplot(dataNUM.iloc[:,1], ax = axes[0][1])
sns.distplot(dataNUM.iloc[:,2], ax = axes[0][2])
sns.distplot(dataNUM.iloc[:,3], ax = axes[1][0])
sns.distplot(dataNUM.iloc[:,4], ax = axes[1][1])
sns.distplot(dataNUM.iloc[:,5], ax = axes[1][2])
sns.distplot(dataNUM.iloc[:,6], ax = axes[2][0])
sns.distplot(dataNUM.iloc[:,7], ax = axes[2][1])
plt.setp(axes, yticks=[])
plt.tight_layout()


sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2, 3, figsize=(7, 7), sharex=True)
sns.despine(left=True)
sns.distplot(dataNUM.iloc[:,8], ax = axes[0][0])
sns.distplot(dataNUM.iloc[:,9], ax = axes[0][1])
sns.distplot(dataNUM.iloc[:,10], ax = axes[0][2])
sns.distplot(dataNUM.iloc[:,11], ax = axes[1][0])
sns.distplot(dataNUM.iloc[:,12], ax = axes[1][1])
sns.distplot(dataNUM.iloc[:,13], ax = axes[1][2])
plt.setp(axes, yticks=[])
plt.tight_layout()

# Credit, Jan-Bill, March-bill, April-bill, all Previous_payments

#sns.heatmap(dataNUM, cmap="YlGnBu")
corr = dataNUM.corr()
ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)
#ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right')

# correlation is high among Bill_Amount, not Previous_Payments
dataNUM[['Jan_Bill_Amount','Feb_Bill_Amount']].corr().iloc[1,0]



# drop all but June_Bills and June Previous_Payments
dataNUM.columns
# dropped columns are more
dataNUM = dataNUM[['Credit_Amount', 'Age_Years','June_Bill_Amount', 'Previous_Payment_June']]

#outlier check

sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.despine(left=True)
sns.boxplot(y = dataNUM.iloc[:,0], x = data['Default_Payment'], data = data, ax = axes[0][0])
sns.boxplot(y = dataNUM.iloc[:,1], x = data['Default_Payment'], data = data, ax = axes[0][1])
sns.boxplot(y = dataNUM.iloc[:,2], x = data['Default_Payment'], data = data, ax = axes[1][0])
sns.boxplot(y = dataNUM.iloc[:,3], x = data['Default_Payment'], data = data, ax = axes[1][1])
plt.setp(axes, yticks=[])
plt.tight_layout()

#outliers

#print(np.percentile(dataNUM['Credit_Amount'].values, 1), np.percentile(dataNUM['Credit_Amount'].values, 99))
#print(np.percentile(dataNUM['Age_Years'].values, 1), np.percentile(dataNUM['Age_Years'].values, 99))
#print(np.percentile(dataNUM['June_Bill_Amount'].values, 1), np.percentile(dataNUM['June_Bill_Amount'].values, 99))
#print(np.percentile(dataNUM['Previous_Payment_June'].values, 1), np.percentile(dataNUM['Previous_Payment_June'].values, 99))

dataNUM_list = [dataNUM]

for dataset in dataNUM_list:
    dataset.loc[dataset.Credit_Amount < np.percentile(dataNUM['Credit_Amount'].values, 1), 'Credit_Amount' ] = np.percentile(dataNUM['Credit_Amount'].values, 1)
    dataset.loc[dataset.Credit_Amount > np.percentile(dataNUM['Credit_Amount'].values, 99), 'Credit_Amount' ] = np.percentile(dataNUM['Credit_Amount'].values, 99)
    
    dataset.loc[dataset.Age_Years < np.percentile(dataNUM['Age_Years'].values, 1), 'Age_Years' ] = np.percentile(dataNUM['Age_Years'].values, 1)
    dataset.loc[dataset.Age_Years > np.percentile(dataNUM['Age_Years'].values, 99), 'Age_Years' ] = np.percentile(dataNUM['Age_Years'].values, 99)
    
    dataset.loc[dataset.June_Bill_Amount < np.percentile(dataNUM['June_Bill_Amount'].values, 1), 'June_Bill_Amount' ] = np.percentile(dataNUM['June_Bill_Amount'].values, 1)
    dataset.loc[dataset.June_Bill_Amount > np.percentile(dataNUM['June_Bill_Amount'].values, 99), 'June_Bill_Amount' ] = np.percentile(dataNUM['June_Bill_Amount'].values, 99)
    
    dataset.loc[dataset.Previous_Payment_June < np.percentile(dataNUM['Previous_Payment_June'].values, 1), 'Previous_Payment_June' ] = np.percentile(dataNUM['Previous_Payment_June'].values, 1)
    dataset.loc[dataset.Previous_Payment_June > np.percentile(dataNUM['Previous_Payment_June'].values, 99), 'Previous_Payment_June' ] = np.percentile(dataNUM['Previous_Payment_June'].values, 99)


# Method1 with outlier, Mehotd2 without outlier

# Method2

# Method: transforming numercical to categroical nature by quantile grouping  

def clubLabelEncoder(df, feature, k):
    
    df[feature +'_band'] = pd.qcut(df[feature], k)
    x = df[feature + '_band'].value_counts().index.tolist()
    
    intervals = []
    for i in range(len(x)):
        leftInt = x[i].left
        rtInt = x[i].right
        intervals.append(leftInt)
        intervals.append(rtInt)
    
    intervals_ = sorted(list(set(intervals)))
    
    for i in range(len(intervals_)-1):
        
        df.loc[(df[feature] > intervals_[i]) & (df[feature] <= intervals_[i+1]), feature] = i
    
    df = df.iloc[:,:-1]
    
    return df[feature].value_counts()


clubLabelEncoder(dataNUM, 'Age_Years', 4)
dataNUM['Age_Years'].value_counts()

clubLabelEncoder(dataNUM, 'Credit_Amount', 4)
clubLabelEncoder(dataNUM, 'June_Bill_Amount', 4)
clubLabelEncoder(dataNUM, 'Previous_Payment_June', 4)

dataNUM = dataNUM.iloc[:,:-4] #removing four band columns from the last

## Scaling
#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler(feature_range  =(-1,1)).fit(dataNUM)
#Xs = scaler.transform(dataNUM)

#dataNUM1 = pd.DataFrame(Xs)
#dataNUM1.columns = dataNUM.columns

dataCAT.iloc[:,0].value_counts() # already brinary,  no need for dummy encoding
#catCols1 = catCols[:-1]
#dataCAT1 = pd.get_dummies(data=dataCAT, columns=catCols1, drop_first=True)

df = pd.concat([dataCAT, dataNUM], axis = 1) # only independent deatures
df['Default_Payment'] = data['Default_Payment'] # add the target column



#------------------------- rectified data for SVM, scaled numercial features-------------
dataNUM_ = data[['Credit_Amount', 'Age_Years', 'June_Bill_Amount', 'Previous_Payment_June']]
df_ = pd.concat([dataCAT, dataNUM_], axis = 1)
df_['Default_Payment'] = data['Default_Payment'] 

## ---------------------------------------Modelling with SMOTE--------------------------------------------------------------------------------

#df_ has the numerical variables as they were along with the other discrete features #(clubbed them)
#df has the categorized numerical features along with the other discrete features #(clubbed them)
#df2 has the numerical variables as they were along with the other discrete features #(unclubbed)


from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE 
from sklearn.svm import SVC 
from sklearn.metrics import roc_auc_score, roc_curve
import time


X = df_.iloc[:,0:-1]
y = df_['Default_Payment']

sm = SMOTE(random_state = 2) 
X_sm, y_sm = sm.fit_sample(X, y.ravel()) 

print('With imbalance treatment:'.upper())
print('Before SMOTE:',X.shape, y.shape)
print('After SMOTE, Xs: {}'.format(X_sm.shape)) 
print('After SMOTE, y: {}'.format(y_sm.shape)) 
print("After SMOTE, counts of '1': {}".format(sum(y_sm == 1))) 
print("After SMOTE, counts of '0': {}".format(sum(y_sm == 0))) 
print("Before SMOTE, counts of '1': {}".format(sum(y == 1))) 
print("Before SMOTE, counts of '0': {}".format(sum(y == 0))) 
print('\n')
print('*'*80)


X_smU = [X_sm[i][:9] for i in range(len(X_sm))] #cat
X_smS = [X_sm[i][9:] for i in range(len(X_sm))] #num

# Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range  =(-1,1)).fit(X_smS)
Xs_smS = scaler.transform(X_smS)

#X_sm = Xs_smS.tolist()

df1 = pd.DataFrame(columns = dataCAT.columns.tolist() , data = X_smU)
df2 = pd.DataFrame(columns = dataNUM.columns.tolist() , data = Xs_smS)

DF = pd.concat([df1, df2], axis = 1)
#X_sm = DF.values

X_scaled = DF.iloc[:,0:-1] # scaled version
y_sm

#split into 70:30 ratio 
X_train_sm, X_val_sm, y_train_sm, y_val_sm = train_test_split(X_sm, y_sm, test_size = 0.3, random_state = 0)
# for scaled data use X_scaled instead of X_sm 


t1 = time.time()
# Naked Modelling

svm = SVC() #SVC(kernel='linear') 

clf = svm.fit(X_train_sm, y_train_sm.ravel()) 
score1 = svm.score(X_train_sm, y_train_sm)
score2 = svm.score(X_val_sm, y_val_sm)
pred = svm.predict(X_val_sm) 
  
# print classification report 
print('With SMOTE:'.upper())
print('train accuracy: ', score1)
print('test accuracy: ', score2)
print('F1 score:\n', classification_report(y_val_sm, pred)) 
print('*'*80)
#print('\n')

#pred_tr = svm.predict(X_train_sm) 
#print(classification_report(y_train_sm, pred_tr))


# ROC Curve
svm_p = SVC(probability = True) # for probability
clf_p = svm_p.fit(X_train_sm, y_train_sm.ravel()) # for probability

pred_proba = svm_p.predict_proba(X_val_sm)[::,1]
fpr, tpr, _ = roc_curve(y_val_sm,  pred_proba)
auc = roc_auc_score(y_val_sm, pred_proba)
plt.plot(fpr,tpr,label="Smote, auc="+str(np.round(auc,3)))
plt.legend(loc=4)
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.title('ROC')
plt.tight_layout()

t2 = time.time()
t = np.round((t2-t1)/60)
print('Time for naked model: ', t + ' mins') 
print('*'*80)

# --------------------------------homework------------------------------------------------------

# repeat the modelling for df2 with scaling

# columns should be 14

#split into 70:30 ratio 
X_train_sm, X_val_sm, y_train_sm, y_val_sm = train_test_split(X_scaled, y_sm, test_size = 0.3, random_state = 0)

t1 = time.time()
# Naked Modelling

svm = SVC() #SVC(kernel='linear') 

clf = svm.fit(X_train_sm, y_train_sm.ravel()) 
score1 = svm.score(X_train_sm, y_train_sm)
score2 = svm.score(X_val_sm, y_val_sm)
pred = svm.predict(X_val_sm) 
  
# print classification report 
print('With SMOTE:'.upper())
print('train accuracy: ', score1)
print('test accuracy: ', score2)
print('F1 score:\n', classification_report(y_val_sm, pred)) 
print('*'*80)
#print('\n')

#-------------------------------------------------------------------------------------------
# repeat the modelling for df2 with scaling for all numerical --- create df2_ without clubbing
dataCAT_original = data_original[catCols]
data_allNumeric = pd.concat([dataCAT_original, dataNUM_], axis = 1)
data_allNumeric['Default_Payment'] = data['Default_Payment']

X = data_allNumeric.iloc[:,0:-1]
y = data_allNumeric['Default_Payment']

X = df_.iloc[:,0:-1]
y = df_['Default_Payment']

sm = SMOTE(random_state = 2) 
X_sm, y_sm = sm.fit_sample(X, y.ravel()) 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range  =(-1,1)).fit(X_sm)
Xscaled = scaler.transform(X_sm)


#split into 70:30 ratio 
X_train_sm, X_val_sm, y_train_sm, y_val_sm = train_test_split(Xscaled, y_sm, test_size = 0.3, random_state = 0)

t1 = time.time()
# Naked Modelling

svm = SVC() #SVC(kernel='linear') 

clf = svm.fit(X_train_sm, y_train_sm.ravel()) 
score1 = svm.score(X_train_sm, y_train_sm)
score2 = svm.score(X_val_sm, y_val_sm)
pred = svm.predict(X_val_sm) 
  
# print classification report 
print('With SMOTE:'.upper())
print('train accuracy: ', score1)
print('test accuracy: ', score2)
print('F1 score:\n', classification_report(y_val_sm, pred)) 
print('*'*80)
#print('\n')




#### GRID SEARCH WITH SMOTE for SVM---------------------------------------------------
t3 = time.time()

from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import metrics

parameters = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma' : [0.001, 0.01, 0.1, 1], 'kernel': ['rbf', 'linear']}
svm_gs = GridSearchCV(svm,parameters)
svm_gs.fit(X_train_sm,y_train_sm)
preds = svm_gs.predict(X_val_sm)
score3 = svm_gs.score(X_train_sm, y_train_sm)
score4 = svm_gs.score(X_val_sm, y_val_sm)
predG = svm_gs.predict(X_val_sm)

print('GRID SEARCH WITH SMOTE:')
print('Using best parameters:',svm_gs.best_params_)
print('Train accuracy:', np.round(score3,3))
print('Test accuracy:', np.round(score4,3))
print('F1 score:\n', classification_report(y_val_sm, predG)) 

# ROC Curve
pred_proba_gs = svm_gs.predict_proba(X_val_sm)[::,1]
fpr_gs, tpr_gs, _ = roc_curve(y_val_sm,  pred_proba)
auc_gs = roc_auc_score(y_val_sm, pred_proba)
plt.plot(fpr_gs,tpr_gs,label="Gs-Smote, auc="+str(np.round(auc_gs,3)))
plt.legend(loc=4)
plt.tight_layout()
print('*'*80)

t4 = time.time()
t = np.round((t4-t3)/60)
print('Time for GS: ', t + ' mins') 

#### GRID SEARCH WITH SMOTE with CROSS VALIDATION---------------------------------
t5 = time.time()

def SVM_gridSearch(X,y,nfolds):
    #create a dictionary of all values we want to test
    parameters = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma' : [0.001, 0.01, 0.1, 1], 'kernel': ['rbf', 'linear']}
    # decision tree model
    svm = SVC()
    #use gridsearch to val all values
    svm_gscv = GridSearchCV(svm, parameters, cv=nfolds)
    #fit model to data
    svm_gscv.fit(X, y)
    #find score
    accuracy = svm_gscv.score(X, y)
    
    return svm_gscv.best_params_, accuracy, svm_gscv

print('GRID SEARCH WITH SMOTE & CROSS VALIDATION -- DT:')
best_param, score5, svm_gscv = SVM_gridSearch(X_train_sm,y_train_sm, 4)
score6 = svm_gscv.score(X_val_sm, y_val_sm)
predGC = svm_gscv.predict(X_val_sm)
print('Using best parameters:',best_param)
print('Train accuracy:', np.round(score5,3))
print('Test accuracy:', np.round(score6,3))
print('F1 score:\n', classification_report(y_val_sm, predGC))

## ROC curve
pred_proba_gscv = svm_gscv.predict_proba(X_val_sm)[::,1]
fpr_gscv, tpr_gscv, _ = roc_curve(y_val_sm,  pred_proba_gscv)
auc_gscv = metrics.roc_auc_score(y_val_sm, pred_proba_gscv)
plt.plot(fpr_gscv,tpr_gscv,label="Gs-Smote-CV, auc="+str(np.round(auc_gscv,3)))
plt.legend(loc=4)
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.title('ROC')
plt.tight_layout()

t6 = time.time()
t = np.round((t6-t5)/60)
print('Time for GS-CV: ', t + ' mins') 
