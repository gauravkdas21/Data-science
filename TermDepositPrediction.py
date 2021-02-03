#!/usr/bin/env python
# coding: utf-8

# Data Set information 
# 
# 
#     Source:
# 
#     [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014
# 
# 
#     Data Set Information:
# 
#     The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 
# 
#     There are four datasets: 
#     1) bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014]
#     2) bank-additional.csv with 10% of the examples (4119), randomly selected from 1), and 20 inputs.
#     3) bank-full.csv with all examples and 17 inputs, ordered by date (older version of this dataset with less inputs). 
#     4) bank.csv with 10% of the examples and 17 inputs, randomly selected from 3 (older version of this dataset with less inputs). 
#     The smallest datasets are provided to val more computationally demanding machine learning algorithms (e.g., SVM). 
# 
#     The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).
# 
# 
#     Attribute Information:
# 
#     Input variables:
#     # bank client data:
#     1 - age (numeric)
#     2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
#     3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
#     4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
#     5 - default: has credit in default? (categorical: 'no','yes','unknown')
#     6 - housing: has housing loan? (categorical: 'no','yes','unknown')
#     7 - loan: has personal loan? (categorical: 'no','yes','unknown')
#     # related with the last contact of the current campaign:
#     8 - contact: contact communication type (categorical: 'cellular','telephone') 
#     9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
#     10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
#     11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
#     # other attributes:
#     12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
#     13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
#     14 - previous: number of contacts performed before this campaign and for this client (numeric)
#     15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
#     # social and economic context attributes
#     16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
#     17 - cons.price.idx: consumer price index - monthly indicator (numeric) 
#     18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric) 
#     19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
#     20 - nr.employed: number of employees - quarterly indicator (numeric)
# 
#     Output variable (desired target):
#     21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

# ## Problem Statement :
#     
# The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. 
# The classification goal is to predict if the client will subscribe a term deposit.



import pandas as pd
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


warnings.filterwarnings("ignore")

bank = pd.read_csv('data\\bank\\bank-additional-full.csv', sep=';')

# list all columns (for reference)
bank.columns


#  y (response)

# convert the response to numeric values and store as a new column
bank['outcome'] = bank.y.map({'no':0, 'yes':1})

#bank.head()

bank = bank.drop(['y'], axis = 1)
#bank.head()

bank.isna().sum()

bank.describe()

bank.describe(include = ['O'])

## EDA

# outcome

sns.countplot(x=bank['outcome'])
plt.xlabel('Subscribed for Term deposit')

np.round(len(bank['outcome'][bank['outcome'] == 0])/bank.shape[0],2)

# imbalanced data

#bank.head()

bank.dtypes


catCols = bank.dtypes[bank.dtypes == 'object'].index.tolist()
catCols

# oR
data_num = bank.select_dtypes(include = ['float64', 'int64'])

data_cat = bank.select_dtypes(include = ['object'])

#data_cat.head()

# for categorical

# EDUCATION
data_cat['education'].value_counts()

data_cat['education'] = data_cat.education.map({'basic.4y':'Basic', 'basic.6y':'Basic', 'basic.9y': 'Basic',
        'high.school':'HighSchool', 'illiterate':'Illiterate', 'professional.course':'ProfessionalCourse',
        'university.degree':'UniversityDegree','unknown':'Unknown' })

#data_cat.education[data_cat.education=='basic.4y']='Basic'
#data_cat.education[data_cat.education=='basic.6y']='Basic'
#data_cat.education[data_cat.education=='basic.9y']='Basic'
#data_cat.education[data_cat.education=='high.school']='HighSchool'
#data_cat.education[data_cat.education=='illiterate']='Illiterate'
#data_cat.education[data_cat.education=='professional.course']='ProfessionalCourse'
#data_cat.education[data_cat.education=='university.degree']='UniversityDegree'
#data_cat.education[data_cat.education=='unknown']='Unknown'

data_cat['education'].value_counts()

data_cat.education[data_cat.education=='Illiterate']='Basic'

def categoryRename(df, colName, oldName, newName):
    
    df[colName][df[colName]==oldName]=newName
    #x = df[colName].value_counts()
    return df


data_cat['education'].value_counts()
data_cat['outcome'] = bank['outcome']


plt.figure(figsize=(5,3))
sns.countplot(y='education',hue='outcome',data=data_cat)
plt.tight_layout()

data_cat['job'].value_counts()
plt.figure(figsize=(6,5))
sns.countplot(y='job',hue='outcome',data=bank)
plt.tight_layout()

data_cat['marital'].value_counts()
data_cat.marital[data_cat.marital=='unknown']='divorced' # because very less/nil subsribers
data_cat['marital'].value_counts()
plt.figure(figsize=(5,3))
sns.countplot(x='marital',hue='outcome',data=data_cat)
plt.tight_layout()

data_cat['default'].value_counts()
data_cat['outcome'][data_cat['default']=='yes']

tab = pd.crosstab(data_cat['default'],  data_cat['outcome'],margins = False)
prop = []
for i in range(tab.shape[0]):
    value = tab.iloc[i,1]/tab.iloc[i,0]
    prop.append(value)
tab['prop'] = prop

def createProportions(df,colName, dependentColName):
    
    tab = pd.crosstab(df[colName],  df[dependentColName],margins = False)
    prop = []
    for i in range(tab.shape[0]):
        value = tab.iloc[i,1]/tab.iloc[i,0]
        prop.append(value)
    tab['prop'] = prop

    return tab


# club yes with unknown based on 1 proportion
data_cat = categoryRename(data_cat,'default', 'yes', 'unknown')

plt.figure(figsize=(5,3))
sns.countplot(x='default',hue='outcome',data=data_cat)
plt.tight_layout()

data_cat['housing'].value_counts()
plt.figure(figsize=(5,3))
sns.countplot(x='housing',hue='outcome',data=bank)
plt.tight_layout()

data_cat['loan'].value_counts()
plt.figure(figsize=(5,3))
sns.countplot(x='loan',hue='outcome',data=bank)
plt.tight_layout()

data_cat['contact'].value_counts()
plt.figure(figsize=(5,3))
sns.countplot(x='contact',hue='outcome',data=bank)
plt.tight_layout()

data_cat['month'].value_counts()
plt.figure(figsize=(5,3))
sns.countplot(x='month',hue='outcome',data=bank)
plt.tight_layout()

data_cat['day_of_week'].value_counts()
plt.figure(figsize=(5,3))
sns.countplot(x='day_of_week',hue='outcome',data=bank)
plt.tight_layout()
# can remove day column ---- HW
# Or club them based on proportion
tab1 = createProportions(data_cat,'day_of_week', 'outcome')

data_cat['poutcome'].value_counts()
plt.figure(figsize=(5,3))
sns.countplot(x='poutcome',hue='outcome',data=bank)
plt.tight_layout()
plt.show()
plt.close()
tab2 = createProportions(data_cat,'poutcome', 'outcome')


# EDA for numerical features -----------------------------------------------

'''age, duration are truly contiuous
 campaign, previous are categorical but are already label encoded
 rest are categorical actually, have to change'''

plt.figure(figsize = (6,5))
sns.distplot(data_num['age'])
plt.show()

data_num['pdays'].value_counts()
tab3 = createProportions(data_num,'pdays', 'outcome')
##### ideallly remove this column, because most customers are new (999 has max occurences)

data_num.pdays[data_num.pdays==999]=35 ## 
sns.distplot(data_num['pdays'])
plt.show()

data_num['pdays_band'] = pd.cut(data_num['pdays'], 5)
data_num['pdays_band'].value_counts()
data_num[['pdays_band', 'outcome']].groupby(['pdays_band'], as_index=False).mean().sort_values(by='pdays_band', ascending=True)

data_num.loc[ data_num['pdays'] <= 7.0, 'pdays'] = 0
data_num.loc[(data_num['pdays'] > 7.0) & (data_num['pdays'] <= 14.0), 'pdays'] = 1
data_num.loc[(data_num['pdays'] > 14.0) & (data_num['pdays'] <= 21), 'pdays'] = 2
data_num.loc[(data_num['pdays'] > 21.0) & (data_num['pdays'] <= 28), 'pdays'] = 3
data_num.loc[ data_num['pdays'] > 28, 'pdays'] = 4

data_num.pdays.value_counts()

def clubLabelEncoder(df, feature, k):
    
    #df[feature + '_band'] = pd.cut(df[feature], k)
    #data = df[[feature + '_band', target]].groupby([feature + '_band'], as_index = False).mean().sort_values(by = feature + '_band', ascending = True)
    #x = data[feature + '_band'].tolist()
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

# emp.var.rate
data_num['emp.var.rate'].value_counts()
data_num.loc[ data_num['emp.var.rate'] <= 0, 'emp.var.rate'] = 0
data_num.loc[ data_num['emp.var.rate'] > 0, 'emp.var.rate'] = 1
data_num['emp.var.rate'].value_counts()

data_num['cons.price.idx'].value_counts()
data_num['cons.price.idx_band'] = pd.cut(data_num['cons.price.idx'], 4)

data_num.loc[ data_num['cons.price.idx'] <= 92.842, 'cons.price.idx'] = 0
data_num.loc[(data_num['cons.price.idx'] > 92.842) & (data_num['cons.price.idx'] <= 93.484), 'cons.price.idx'] = 1
data_num.loc[(data_num['cons.price.idx'] > 93.484) & (data_num['cons.price.idx'] <= 94.126), 'cons.price.idx'] = 2
data_num.loc[data_num['cons.price.idx'] > 94.126, 'cons.price.idx'] = 3

data_num['cons.conf.idx'].value_counts()
data_num['cons.conf.idx_band'] = pd.cut(data_num['cons.conf.idx'], 4)
data_num['cons.conf.idx_band'].value_counts()
#remove column cons.conf.idx

data_num['nr.employed'].value_counts()
data_num['nr.employed_band'] = pd.cut(data_num['nr.employed'], 4)
data_num['nr.employed_band'].value_counts()

data_num.loc[ data_num['nr.employed'] <= 5029.735, 'nr.employed'] = 0
data_num.loc[(data_num['nr.employed'] > 5029.735) & (data_num['nr.employed'] <= 5095.85), 'nr.employed'] = 1
data_num.loc[(data_num['nr.employed'] > 5095.85) & (data_num['nr.employed'] <= 5161.975), 'nr.employed'] = 2
data_num.loc[data_num['nr.employed'] > 5161.975, 'nr.employed'] = 3

# have to categrize the numericals age, duration, euribor3m 
sns.distplot(data_num['age'])
data_num['age_band'] = pd.qcut(data_num['age'], 3)
data_num['age_band'].value_counts()

data_num.loc[ data_num['age'] <= 34.0, 'age'] = 0
data_num.loc[(data_num['age'] > 34.0) & (data_num['age'] <= 44.0), 'age'] = 1
data_num.loc[data_num['age'] > 44, 'age'] = 2

sns.distplot(data_num['duration'])
plt.show()
plt.close()
data_num['duration_band'] = pd.qcut(data_num['duration'], 3)
data_num['duration_band'].value_counts()

data_num.loc[ data_num['duration'] <= 126.0, 'duration'] = 0
data_num.loc[(data_num['duration'] > 126.0) & (data_num['duration'] <= 258.0), 'duration'] = 1
data_num.loc[data_num['duration'] > 258.0, 'duration'] = 2


sns.distplot(data_num['euribor3m'])
plt.show()
plt.close()
data_num['euribor3m_band'] = pd.qcut(data_num['euribor3m'], 3)
data_num['euribor3m_band'].value_counts()

data_num.loc[ data_num['euribor3m'] <= 4.021, 'euribor3m'] = 0
data_num.loc[(data_num['euribor3m'] > 4.021) & (data_num['euribor3m'] <= 4.958), 'euribor3m'] = 1
data_num.loc[data_num['euribor3m'] > 4.958, 'euribor3m'] = 2
# can remove this column euribor3m

listRemove = data_num.columns.tolist()[-7:]
data_num1 = data_num.drop(listRemove, axis = 1)
data_num1 = data_num1.drop(['cons.conf.idx'], axis = 1)

#for i in catCols[:-1]:
#    dummies = pd.get_dummies(data_cat1[i])
#    data_cat1[dummies.columns] = dummies

data_cat1 = data_cat.drop(['day_of_week', 'outcome'], axis = 1)

catCols1 = catCols.remove('day_of_week')

data_cat1 = pd.get_dummies(data=data_cat1, columns=catCols1, drop_first=True)

data = pd.concat([data_cat1, data_num1], axis = 1)
print('\n')
print('*'*80)



import time

t0 = time.time()
# ----------------------------------- MODELLING-----------------------------------------------------------------------------

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.tree import DecisionTreeClassifier
from confusionMatrix import plotConfusionMatrix

X = data.iloc[:,0:-1]
y = data['outcome']

## Method 0: without SMOTE---------------------------------------

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 0)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train.ravel()) 
clf.score(X_train, y_train)
clf.score(X_val, y_val)

predictions_ = clf.predict(X_val) 
  
# print classification report 
print('Without imbalance treatment:'.upper())
print(classification_report(y_val, predictions_)) 
print(confusion_matrix(y_val, predictions_))
f1= f1_score(y_val,predictions_, average='micro')
plt.figure(figsize=(5,3))
cnf_mat = confusion_matrix(y_val,predictions_)
plotConfusionMatrix(cnf_mat, 2,title='F1_micro {:.3f}'.format(f1))
print('*'*80)
#print('\n')

sum(y_val == 0)
sum(y_val == 1)
sum(predictions_ == 0)
sum(predictions_ == 1)


## Method1: SMOTE on both train and validation sets------------------------------------------------------------------

sm = SMOTE(random_state = 2) 
X_res, y_res = sm.fit_sample(X, y.ravel()) 

print('With imbalance treatment:'.upper())
print('After OverSampling, X: {}'.format(X_res.shape)) 
print('After OverSampling, y: {}'.format(y_res.shape)) 
print("After OverSampling, counts of '1': {}".format(sum(y_res == 1))) 
print("After OverSampling, counts of '0': {}".format(sum(y_res == 0))) 
print('\n')

#split into 70:30 ratio 
X_train_res, X_val_res, y_train_res, y_val_res = train_test_split(X_res, y_res, test_size = 0.3, random_state = 0)

clf.fit(X_train_res, y_train_res.ravel()) 
clf.score(X_train_res, y_train_res)
clf.score(X_val_res, y_val_res)

predictions = clf.predict(X_val_res) 
  
# print classification report 
print('After imbalance treatment:'.upper())
print(classification_report(y_val_res, predictions)) 
print(confusion_matrix(y_val_res, predictions))
print('*'*80)
#print('\n')

sum(y_val_res == 0)
sum(y_val_res == 1)
sum(predictions == 0)
sum(predictions == 1)

confusion_matrix(y_val_res, predictions)


#### GRID SEARCH WITH SMOTE---------------------------------------------------

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import metrics

#parameters={'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}
parameters = {'max_depth': np.arange(3, 10)} # pruning
tree = GridSearchCV(clf,parameters)
tree.fit(X_train_res,y_train_res)
preds = tree.predict(X_val_res)
accu = tree.score(X_val_res, y_val_res)

print('GRID SEARCH WITH SMOTE -- DT:')
print('Using best parameters:',tree.best_params_)
print('Accuracy:', np.round(accu,3))

y_pred_proba_ = tree.predict_proba(X_val_res)[::,1]
fpr, tpr, _ = roc_curve(y_val_res,  y_pred_proba_)
auc = roc_auc_score(y_val_res, y_pred_proba_)
plt.plot(fpr,tpr,label="Gs-Smote-DT, auc="+str(np.round(auc,3)))
plt.legend(loc=4)
plt.tight_layout()

print('*'*80)

#### GRID SEARCH WITH SMOTE with CROSS VALIDATION---------------------------------

def dtree_grid_search(X,y,nfolds):
    #create a dictionary of all values we want to test
    param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(15, 30)}
    # decision tree model
    dtree_model = DecisionTreeClassifier()
    #use gridsearch to val all values
    dtree_gscv = GridSearchCV(dtree_model, param_grid, cv=nfolds)
    #fit model to data
    dtree_gscv.fit(X, y)
    #find score
    score = dtree_gscv.score(X, y)
    
    return dtree_gscv.best_params_, score, dtree_gscv

print('GRID SEARCH WITH SMOTE & CROSS VALIDATION -- DT:')
best_param, acc, model = dtree_grid_search(X_train_res,y_train_res, 4)
acc = model.score(X_val_res, y_val_res)
print('Using best parameters:',best_param)
print('accuracy:', np.round(acc,3))

## ROC curve
y_pred_proba = model.predict_proba(X_val_res)[::,1]
fpr, tpr, _ = roc_curve(y_val_res,  y_pred_proba)
auc = metrics.roc_auc_score(y_val_res, y_pred_proba)
plt.plot(fpr,tpr,label="Gs-Smote-cv-DT, auc="+str(np.round(auc,3)))
plt.legend(loc=4)
plt.tight_layout()

#from sklearn.tree import export_graphviz
#import pydotplus
#from io import StringIO
#from PIL import Image


#dot_data = StringIO()
#export_graphviz(clf, out_file=dot_data,  
#                filled=True, rounded=True,
#                special_characters=True)
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#Image(graph.create_png())

# The ‘value’ row in each node tells us how many of the observations that were sorted into 
# that node fall into each of our two categories. 

#-------------------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier

def RF_grid_search(X,y,nfolds):
    
    #create a dictionary of all values we want to test
    param_grid = {'criterion':['gini','entropy'],'max_depth': np.arange(11, 19),
                  'n_estimators': [100,300]}
    #randomForest model without gridSrearch
    rf = RandomForestClassifier()
    #use gridsearch to val all values
    rf_gscv = GridSearchCV(rf, param_grid, cv=nfolds)
    #fit model to data
    rf_gscv.fit(X, y) # with grid search
    #find score
    score_gscv = rf_gscv.score(X, y) # with grid search
    
    return rf, rf_gscv.best_params_, rf_gscv, score_gscv  

print('*'*80)
print('GRID SEARCH WITH SMOTE & CROSS VALIDATION -- RF:')
rf, best_param_rf, model_rf, acc_rf = RF_grid_search(X_train_res,y_train_res, 4)
acc_rf = model_rf.score(X_val_res, y_val_res)
print('Using best parameters:',best_param_rf)
print('accuracy with Gs:', np.round(acc_rf,3))

## ROC curve
y_pred_proba_rf = model_rf.predict_proba(X_val_res)[::,1]
fpr_rf, tpr_rf, _rf = roc_curve(y_val_res,  y_pred_proba_rf)
auc_rf = metrics.roc_auc_score(y_val_res, y_pred_proba_rf)
plt.plot(fpr_rf,tpr_rf,label="Gs-Smote-cv-RF, aucRF="+str(np.round(auc_rf,3)))
plt.legend(loc=4)
plt.tight_layout()


rf.fit(X_train_res, y_train_res)
# Get numerical feature importances
importances = list(rf.feature_importances_)
feature_list = data.columns.tolist()[:-1]
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

print(rf.estimators_)

feature_importances_df = pd.DataFrame(importances, index = feature_list,
                             columns=['importance']).sort_values('importance', ascending=False)


#-------------------------------------------------------------------------------------------
# import XGBoost classifier and accuracy
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#instantiate model and train
model_G = XGBClassifier(learning_rate = 0.05, n_estimators=300, max_depth=5)
model_G.fit(X_train_res, y_train_res)

# make predictions for test set
y_pred = model_G.predict(X_val_res)
preds = [round(value) for value in y_pred]

accG = model_G.score(X_val_res,y_val_res)

print('*'*80)
print('SMOTE with XGB:')
print("Accuracy without Gs: %.2f%%" % (accG * 100.0))
print('*'*80)
y_pred_proba_G = model_G.predict_proba(X_val_res)[::,1]
fpr_G, tpr_G, _G = roc_curve(y_val_res,  y_pred_proba_G)
auc_G = metrics.roc_auc_score(y_val_res, y_pred_proba_G)
plt.plot(fpr_G,tpr_G,label="Smote-GB, aucG="+str(np.round(auc_G,3)))
plt.legend(loc=4)
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.title('ROC')
plt.tight_layout()

from xgboost import plot_importance
# plot feature importance
plt.rcParams["figure.figsize"] = (14, 7)
plot_importance(model_G)

from numpy import sort
from sklearn.feature_selection import SelectFromModel

thresholds = sort(model_G.feature_importances_)

feature_num = []
acc = []

for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model_G, threshold=thresh, prefit=True)
    select_X_train_res = selection.transform(X_train_res)
    # train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train_res, y_train_res)
    # eval model
    select_X_val_res = selection.transform(X_val_res)
    y_pred = selection_model.predict(select_X_val_res)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_val_res, predictions)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train_res.shape[1], accuracy*100.0))
    feature_num.append(select_X_train_res.shape[1])
    acc.append(accuracy*100.0)

plt.plot(feature_num, acc)
plt.xlabel('No. of features')
plt.ylabel('Accuracy')
plt.xlim(0,45)
plt.ylim(75,100)
plt.show()


# Essentially this bit of code trains and tests the model by iteratively removing features by
# their importance, recording the model’s accuracy along the way. This allows you to easily 
# remove features without simply using trial and error. Although not shown here, this approach 
# can also be applied to other parameters (learning_rate, max_depth, etc) of the model to 
# automatically try different tuning variables. I wont go into the details of tuning the model,
# however the great number of tuning parameters is one of the reasons XGBoost so popular.

t1 = time.time()
t = t1-t0

print('Time taken for model completion: '+ str(np.round(t/60,2)) + ' mins')

## 7 Techniques to Handle Imbalanced Data---------------------------------------------------
#Use the right evaluation metrics. ...
#Resample the training set. ...
#Use K-fold Cross-Validation in the right way. ...
#Ensemble different resampled datasets. ...
#Resample with different ratios. ...
#Cluster the abundant class. ...
#Design your own models.


## 2nd Method: NearMiss on both train and validation sets-------------------------------------------------
#from imblearn.under_sampling import NearMiss 
#nr = NearMiss()
#
#X_miss, y_miss = nr.fit_sample(X, y.ravel()) 
#
#print('After OverSampling, the shape of X: {}'.format(X_miss.shape)) 
#print('After OverSampling, the shape of y: {} \n'.format(y_miss.shape)) 
#  
#print("After OverSampling, counts of label '1': {}".format(sum(y_miss == 1))) 
#print("After OverSampling, counts of label '0': {}".format(sum(y_miss == 0))) 
#
##split into 70:30 ratio 
#X_train_miss, X_val_miss, y_train_miss, y_val_miss = train_test_split(X_miss, y_miss, test_size = 0.3, random_state = 0)
#
#clf.fit(X_train_miss, y_train_miss.ravel()) 
#clf.score(X_train_miss, y_train_miss)
#
#predictions2 = clf.predict(X_val_miss) 
#  
## print classification report 
#print(classification_report(y_val_miss, predictions2)) 


## 3rd method: SMOTE only on train-------------------------------------------------

## split into 70:30 ratio
#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 0)
#
#X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel()) 
#
#print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape)) 
#print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape)) 
#  
#print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1))) 
#print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0))) 
#
#clf.fit(X_train_res, y_train_res.ravel()) 
#clf.score(X_train_res, y_train_res)
#clf.score(X_val, y_val)
#
#predictions3 = clf.predict(X_val) 
#  
## print classification report 
#print(classification_report(y_val, predictions3)) 
#
#sum(y_val == 0)
#sum(y_val == 1)
#sum(predictions3 == 0)
#sum(predictions3 == 1)
#
#confusion_matrix(y_val, predictions3)
