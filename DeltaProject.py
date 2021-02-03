# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:18:48 2019

@author: Gaurav.Das
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:18:48 2019

@author: Gaurav.Das
"""


import os
import pandas as pd
import numpy as np

#pd.options.mode.chained_assignment = None

t1 = pd.read_csv(r"data\Delta\borrower_table.csv")

t2 = pd.read_csv(r"data\Delta\loan_table.csv")

df = pd.merge(t1, t2, how = 'outer', on = 'loan_id')

train = df[df['loan_granted'] == 1]

test = df[df['loan_granted'] == 0]

train1 = train.drop(['loan_granted'], axis = 1)

test1 = test.drop(['loan_granted'], axis = 1)

## EDA of y
# check for data imbalance
y0 = len(train1[train1['loan_repaid'] == 0])/len(train1)

y1 = 1 - y0

# Here 0 represents minority class or defaulters, 1 represents majority class

# for common preprocessing, we combine train and test
combined = pd.concat([train1, test1], axis = 0)

combined = combined.drop(['date', 'loan_id'], axis = 1)


# treat nan as zero, as people not paying previous loans OR as a separate category 2
combined['fully_repaid_previous_loans'] = combined['fully_repaid_previous_loans'].fillna(0)

combined['currently_repaying_other_loans'] = combined['currently_repaying_other_loans'].fillna(0)

combined['avg_percentage_credit_card_limit_used_last_year'] = combined['avg_percentage_credit_card_limit_used_last_year'].fillna(0)

combined.shape

combined.describe()

combined.columns

## EDA discrete features

catCols = ['is_first_loan', 'fully_repaid_previous_loans',
       'currently_repaying_other_loans',
       'is_employed',
       'dependent_number', 'loan_purpose', 'loan_repaid'] #discrete data

numCols = ['total_credit_card_limit', 'avg_percentage_credit_card_limit_used_last_year',
           'saving_amount','checking_amount', 'yearly_salary', 'age'] #continuous data

# y = combined['loan_repaid']

catCombined = pd.DataFrame(data = combined,columns = catCols)

import matplotlib.pyplot as plt
import seaborn as sns

# some columns are already binary coded, so look out for other columns

# col: dependent_number
sns.set(style="white", palette="muted", color_codes=True)
plt.figure(figsize=(8,4))
sns.countplot(x=train1['dependent_number'], hue = 'loan_repaid', data = train1)

# % proportion of people repaying the loan for different levels of dependents
len(train1.loc[(train1['dependent_number'] == 0) & (train1['loan_repaid'] == 1), 'dependent_number'])/len(train1[train1['dependent_number'] == 0])
len(train1.loc[(train1['dependent_number'] == 1) & (train1['loan_repaid'] == 1), 'dependent_number'])/len(train1[train1['dependent_number'] == 1])
len(train1.loc[(train1['dependent_number'] == 2) & (train1['loan_repaid'] == 1), 'dependent_number'])/len(train1[train1['dependent_number'] == 2])
len(train1.loc[(train1['dependent_number'] == 3) & (train1['loan_repaid'] == 1), 'dependent_number'])/len(train1[train1['dependent_number'] == 3])
len(train1.loc[(train1['dependent_number'] == 4) & (train1['loan_repaid'] == 1), 'dependent_number'])/len(train1[train1['dependent_number'] == 4])
len(train1.loc[(train1['dependent_number'] == 5) & (train1['loan_repaid'] == 1), 'dependent_number'])/len(train1[train1['dependent_number'] == 5])
len(train1.loc[(train1['dependent_number'] == 6) & (train1['loan_repaid'] == 1), 'dependent_number'])/len(train1[train1['dependent_number'] == 6])
len(train1.loc[(train1['dependent_number'] == 7) & (train1['loan_repaid'] == 1), 'dependent_number'])/len(train1[train1['dependent_number'] == 7])
len(train1.loc[(train1['dependent_number'] == 8) & (train1['loan_repaid'] == 1), 'dependent_number'])/len(train1[train1['dependent_number'] == 8])


def club(df, feature, a, b, newValue):
    
    for i in range(a, b):
        df.loc[df[feature] == i, feature] = newValue
    
    x = df[feature].value_counts()
    
    return x

catCombined.loc[catCombined['dependent_number'] == 1, 'dependent_number'] = 0
catCombined['dependent_number'].value_counts()

#clubbing 2 to 6 as 1
club(catCombined, 'dependent_number', 2, 7, 1)

#clubbing 7 and 8 as 2
club(catCombined, 'dependent_number', 7, 9, 2)

#checking proprotion again
len(train1.loc[(train1['dependent_number'] == 0) & (train1['loan_repaid'] == 1), 'dependent_number'])/len(train1[train1['dependent_number'] == 0])
len(train1.loc[(train1['dependent_number'] == 1) & (train1['loan_repaid'] == 1), 'dependent_number'])/len(train1[train1['dependent_number'] == 1])
len(train1.loc[(train1['dependent_number'] == 2) & (train1['loan_repaid'] == 1), 'dependent_number'])/len(train1[train1['dependent_number'] == 2])

# we can club 0 and 1 since proportion is in same range 70s
# or keep them as 3 levels and dummify

catCombined.loc[catCombined['dependent_number'] == 1, 'dependent_number'] = 0
catCombined['dependent_number'].value_counts()

catCombined.loc[catCombined['dependent_number'] == 2, 'dependent_number'] = 1
catCombined['dependent_number'].value_counts()


# col: loan_purpose
sns.set(style="white", palette="muted", color_codes=True)
plt.figure(figsize=(8,5))
sns.countplot(x=train1['loan_purpose'], hue = 'loan_repaid', data = train1)

#chcking count/proportion of the levels
train1['loan_purpose'].value_counts()#/train1.shape[0]

# for string to int, use map function (map takes in a dictionary)
# from the barplot, it seems that business, investment, home are very similar
# other and emergency can be clubbed as well (total 2 levels) OR emergency_funds can be made 2 (total 3 levels)
loanPurposeMap = {"other": 0, "business": 1, "emergency_funds": 0, "investment": 1, "home": 1}
# make emergency_funds as 2 and then do dummy encoding or keep them as strings before dummifying

catCombined['loan_purpose'] = catCombined['loan_purpose'].map(loanPurposeMap)


## EDA continuous features

numCombined = pd.DataFrame(data = combined,columns = numCols)

numCombined.columns


# col: total_credit_card_limit
sns.distplot(numCombined['total_credit_card_limit'])

# Method: transforming numercical to categroical nature by quantile grouping  

def clubLabelEncoder(df, feature, k): # k = no. of groups
    
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
    
    y = df[feature].value_counts()
    
    return y

clubLabelEncoder(numCombined, 'total_credit_card_limit', 4)

clubLabelEncoder(numCombined, 'avg_percentage_credit_card_limit_used_last_year', 4)

clubLabelEncoder(numCombined, 'saving_amount', 4)

clubLabelEncoder(numCombined, 'checking_amount', 4)

clubLabelEncoder(numCombined, 'yearly_salary', 2)

clubLabelEncoder(numCombined, 'age', 4)

numCombined = numCombined.iloc[:, :-6] # dropping the 6 unwanted intervals columns

data = pd.concat([numCombined, catCombined], axis = 1)

# while splitting, rest the index and drop it too
dataTrain = data.iloc[:47654, :].reset_index(drop = True) # or select by data['loan_repaid'] != 'NaN'

dataTest = data.iloc[47654:, :].reset_index(drop = True) # or select by data['loan_repaid'] == 'NaN'
dataTest = dataTest.iloc[:,:-1] #removing the unknown (nan) dependent column from test



# Modelling----------------------------------------------------------------------------------------


import time

t0 = time.time()

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE 
#from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import roc_auc_score, roc_curve


X = dataTrain.iloc[:,:-1]
y = dataTrain['loan_repaid']


## Method A: without SMOTE--------------------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, random_state = 0)

clf = DecisionTreeClassifier(random_state=0)

clf.fit(X_train, y_train.ravel()) 
clf.score(X_train, y_train)
clf.score(X_val, y_val)

predictions_ = clf.predict(X_val) 
  
# print classification report 
print('Without imbalance treatment:'.upper())
print(classification_report(y_val, predictions_)) 
print('*'*80)
#print('\n')

sum(y_val == 0)
sum(y_val == 1)
sum(predictions_ == 0)
sum(predictions_ == 1)

confusion_matrix(y_val, predictions_)


#### GRID SEARCH----------------------------------------------------------------

from sklearn.model_selection import GridSearchCV
from sklearn import metrics

#parameters={'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}
parameters = {'max_depth': np.arange(3, 10)}
tree = GridSearchCV(clf,parameters)
tree.fit(X_train,y_train)
tree.score(X_train,y_train)
preds = tree.predict(X_val)
accu = tree.score(X_val, y_val)

print('GRID SEARCH -- DT:')
print('Using best parameters:',tree.best_params_)
print('Accuracy:', np.round(accu,3))

y_pred_proba_ = tree.predict_proba(X_val)[::,1]
fpr, tpr, _ = roc_curve(y_val,  y_pred_proba_)
auc = roc_auc_score(y_val, y_pred_proba_)
#plt.plot(fpr,tpr,label="Gs-DT, auc="+str(np.round(auc,3)))
#plt.legend(loc=4)
#plt.tight_layout()

print('*'*80)

#### GRID SEARCH with CROSS VALIDATION---------------------------------------

def dtree_grid_search(X,y,nfolds):
    #create a dictionary of all values we want to test
    param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(3, 15)}
    # decision tree model
    dtree_model = DecisionTreeClassifier()
    #use gridsearch to val all values
    dtree_gscv = GridSearchCV(dtree_model, param_grid, cv=nfolds)
    #fit model to data
    dtree_gscv.fit(X, y)
    #find score
    score = dtree_gscv.score(X, y)
    
    return dtree_gscv.best_params_, score, dtree_gscv

print('GRID SEARCH & CROSS VALIDATION -- DT:')
best_param, acc, model = dtree_grid_search(X_train,y_train, 4)
model.score(X_train, y_train)
acc = model.score(X_val, y_val)
print('Using best parameters:',best_param)
print('accuracy:', np.round(acc,3))

## ROC curve
y_pred_proba = model.predict_proba(X_val)[::,1]
fpr_, tpr_, __ = roc_curve(y_val,  y_pred_proba)
auc_ = metrics.roc_auc_score(y_val, y_pred_proba)
#plt.plot(fpr_,tpr_,label="Gs-cv-DT, auc="+str(np.round(auc_,3)))
#plt.legend(loc=4)
#plt.tight_layout()

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
# that node fall into each of our three categories. 

#-------------------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier

def RF_grid_search(X,y,nfolds):
    
    #create a dictionary of all values we want to test
    param_grid = {'criterion':['gini','entropy'],'max_depth': np.arange(8,13), #because 9 was in DT
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
print('GRID SEARCH & CROSS VALIDATION -- RF:')
rf, best_param_rf, model_rf, acc_rf = RF_grid_search(X_train,y_train, 4)
model_rf.score(X_train, y_train)
acc_rf = model_rf.score(X_val, y_val)
print('Using best parameters:',best_param_rf)
print('accuracy with Gs:', np.round(acc_rf,3))
## ROC curve
y_pred_proba_rf = model_rf.predict_proba(X_val)[::,1]
fpr_rf, tpr_rf, _rf = roc_curve(y_val,  y_pred_proba_rf)
auc_rf = metrics.roc_auc_score(y_val, y_pred_proba_rf)
#plt.plot(fpr_rf,tpr_rf,label="Gs-Smote-cv-RF, aucRF="+str(np.round(auc_rf,3)))
#plt.legend(loc=4)
#plt.tight_layout()


# feature importance from RF
rf.fit(X_train, y_train)
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
model_G = XGBClassifier(learning_rate = 0.05, n_estimators=300, max_depth=5) # max_depht = 9
model_G.fit(X_train, y_train)

# make predictions for test set
y_pred = model_G.predict(X_val)
preds = [round(value) for value in y_pred]

model_G.score(X_train, y_train)
accG = model_G.score(X_val,y_val)

print('*'*80)
print('XGB:')
print("Accuracy: %.2f%%" % (accG * 100.0))
print('*'*80)
y_pred_proba_G = model_G.predict_proba(X_val)[::,1]
fpr_G, tpr_G, _G = roc_curve(y_val,  y_pred_proba_G)
auc_G = metrics.roc_auc_score(y_val, y_pred_proba_G)
#plt.plot(fpr_G,tpr_G,label="GB, aucG="+str(np.round(auc_G,3)))
#plt.legend(loc=4)
#plt.ylabel('Sensitivity')
#plt.xlabel('1 - Specificity')
#plt.title('ROC')
#plt.tight_layout()

from xgboost import plot_importance
# plot feature importance
plot_importance(model_G)
plt.show()

from numpy import sort
from sklearn.feature_selection import SelectFromModel

thresholds = sort(model_G.feature_importances_)

feature_num = []
acc = []

# feature selection from XB
for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model_G, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBClassifier()
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_val = selection.transform(X_val)
    y_pred = selection_model.predict(select_X_val)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_val, predictions)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))
    feature_num.append(select_X_train.shape[1])
    acc.append(accuracy*100.0)

plt.plot(feature_num, acc)
plt.xlabel('No. of features')
plt.ylabel('Accuracy')
plt.xlim(0,13)
plt.ylim(70,100)
plt.show()


# Essentially this bit of code trains and tests the model by iteratively removing features by
# their importance, recording the model’s accuracy along the way. This allows you to easily 
# remove features without simply using trial and error. Although not shown here, this approach 
# can also be applied to other parameters (learning_rate, max_depth, etc) of the model to 
# automatically try different tuning variables. I wont go into the details of tuning the model,
# however the great number of tuning parameters is one of the reasons XGBoost so popular.

#t1 = time.time()
#t = t1-t0
#
#print('Time taken for model completion: '+ str(np.round(t/60,2)) + ' mins')



## MethodB: with SMOTE------------------------------------------------------------------

sm = SMOTE(random_state = 2) 
X_res, y_res = sm.fit_sample(X, y.ravel()) 

print('With imbalance treatment:'.upper())
print('Before SMOTE X, Y:',X.shape, y.shape)
print('After SMOTE, X: {}'.format(X_res.shape)) 
print('After SMOTE, y: {}'.format(y_res.shape)) 

print("Before SMOTE, counts of '1': {}".format(sum(y == 1))) 
print("Before SMOTE, counts of '0': {}".format(sum(y == 0))) 
print("After SMOTE, counts of '1': {}".format(sum(y_res == 1))) 
print("After SMOTE, counts of '0': {}".format(sum(y_res == 0))) 

print('\n')
#print('*'*80)

#split into 70:30 ratio 
X_train_res, X_val_res, y_train_res, y_val_res = train_test_split(X_res, y_res, test_size = 0.3, random_state = 0)


clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train_res, y_train_res.ravel()) 

clf.score(X_train_res, y_train_res)
clf.score(X_val_res, y_val_res)

predictions = clf.predict(X_val_res) 
  
# print classification report 
print(classification_report(y_val_res, predictions)) 
print('*'*80)
#print('\n')

sum(y_val_res == 0)
sum(y_val_res == 1)
sum(predictions == 0)
sum(predictions == 1)

confusion_matrix(y_val_res, predictions)

#### GRID SEARCH WITH SMOTE----------------------------------------------------------------

from sklearn.model_selection import GridSearchCV
from sklearn import metrics

#parameters={'min_samples_split' : range(10,500,20),'max_depth': range(1,20,2)}
parameters = {'max_depth': np.arange(3, 10)}
tree = GridSearchCV(clf,parameters)
tree.fit(X_train_res,y_train_res)
preds = tree.predict(X_val_res)
tree.score(X_train_res, y_train_res)
accu = tree.score(X_val_res, y_val_res)

print('GRID SEARCH WITH SMOTE -- DT:')
print('Using best parameters:',tree.best_params_)
print('Accuracy:', np.round(accu,3))

y_pred_proba__sm = tree.predict_proba(X_val_res)[::,1]
fpr_sm, tpr_sm, _sm = roc_curve(y_val_res,  y_pred_proba__sm)
auc_sm = roc_auc_score(y_val_res, y_pred_proba__sm)
#plt.plot(fpr_sm,tpr_sm,label="Gs-Smote-DT, auc="+str(np.round(auc_sm,3)))
#plt.legend(loc=4)
#plt.tight_layout()

print('*'*80)

#### GRID SEARCH WITH SMOTE with CROSS VALIDATION---------------------------------------

def dtree_grid_search(X,y,nfolds):
    #create a dictionary of all values we want to test
    param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(3, 15)}
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
model.score(X_train_res, y_train_res)
acc = model.score(X_val_res, y_val_res)
print('Using best parameters:',best_param)
print('accuracy:', np.round(acc,3))

## ROC curve
y_pred_proba_sm = model.predict_proba(X_val_res)[::,1]
fpr__sm, tpr__sm, __sm = roc_curve(y_val_res,  y_pred_proba_sm)
auc__sm = metrics.roc_auc_score(y_val_res, y_pred_proba_sm)
#plt.plot(fpr__sm,tpr__sm,label="Gs-Smote-cv-DT, auc_sm="+str(np.round(auc__sm,3)))
#plt.legend(loc=4)
#plt.tight_layout()

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
# that node fall into each of our three categories. 

#-------------------------------------------------------------------------------------------

def RF_grid_search(X,y,nfolds):
    
    #create a dictionary of all values we want to test
    param_grid = {'criterion':['gini','entropy'],'max_depth': np.arange(8,13), #because 9 was in DT
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
model_rf.score(X_train_res, y_train_res)
acc_rf = model_rf.score(X_val_res, y_val_res)
print('Using best parameters:',best_param_rf)
print('accuracy with Gs:', np.round(acc_rf,3))
## ROC curve
y_pred_proba_rf_sm = model_rf.predict_proba(X_val_res)[::,1]
fpr_rf_sm, tpr_rf_sm, _rf_sm = roc_curve(y_val_res,  y_pred_proba_rf_sm)
auc_rf_sm = metrics.roc_auc_score(y_val_res, y_pred_proba_rf_sm)
#plt.plot(fpr_rf_sm,tpr_rf_sm,label="Gs-Smote-cv-RF, aucRF_sm="+str(np.round(auc_rf_sm,3)))
#plt.legend(loc=4)
#plt.tight_layout()


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
model_G = XGBClassifier(learning_rate = 0.05, n_estimators=300, max_depth=5) # max_depth = 9
model_G.fit(X_train_res, y_train_res)

# make predictions for test set
y_pred = model_G.predict(X_val_res)
preds = [round(value) for value in y_pred]

model_G.score(X_train_res,y_train_res)
accG = model_G.score(X_val_res,y_val_res)

print('*'*80)
print('SMOTE with XGB:')
print("Accuracy: %.2f%%" % (accG * 100.0))
print('*'*80)
y_pred_proba_G_sm = model_G.predict_proba(X_val_res)[::,1]
fpr_G_sm, tpr_G_sm, _G_sm = roc_curve(y_val_res,  y_pred_proba_G_sm)
auc_G_sm = metrics.roc_auc_score(y_val_res, y_pred_proba_G_sm)
#plt.plot(fpr_G_sm,tpr_G_sm,label="Smote-GB, aucG_sm="+str(np.round(auc_G_sm,3)))
#plt.legend(loc=4)
#plt.ylabel('Sensitivity')
#plt.xlabel('1 - Specificity')
#plt.title('ROC')
#plt.tight_layout()

from xgboost import plot_importance
# plot feature importance
plot_importance(model_G)
plt.show()

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
plt.xlim(0,13)
plt.ylim(70,100)
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

dt_train = 0.936
dt_test = 0.868

dtG_train = 0.89087
dtG_test = 0.8919

dtGcv_train = 0.8964
dtGcv_test = 0.8943

rf_train = 0.9046
rf_test = 0.8995

xb_train = 0.9002
xb_test = 0.9004

dt_train_sm =  0.94791
dt_test_sm = 0.89182

dtG_train_sm = 0.9041
dtG_test_sm = 0.9002

dtGcv_train_sm = 0.91095
dtGcv_test_sm = 0.90116

rf_train_sm = 0.9190
rf_test_sm = 0.9084

xb_train_sm = 0.91644
xb_test_sm = 0.91549


TrainScore = [dt_train, dtG_train, dtGcv_train, rf_train, xb_train, 
               dt_train_sm, dtG_train_sm, dtGcv_train_sm, rf_train_sm, xb_train_sm]

TestScore = [dt_test, dtG_test, dtGcv_test, rf_test, xb_test, 
               dt_test_sm, dtG_test_sm, dtGcv_test_sm, rf_test_sm, xb_test_sm]


Scores = pd.DataFrame([TrainScore, TestScore], columns = ['DT', 'DTg', 'DTgcv', 'RF', 'XB', 
               'DT_sm', 'DTg_sm', 'DTgcv_sm', 'RF_sm', 'XB_sm'], index = ['Train', 'Test'])

Scores_ = Scores.transpose()

plt.figure(figsize = (9,4))
plt.plot(Scores_.iloc[:,0], 'b-',  marker='o', linewidth=2, markersize=7)
plt.plot(Scores_.iloc[:,1], 'r--',  marker='^', linewidth=2, markersize=8)
plt.xlabel('Model', fontweight='bold', fontsize = 13)
plt.ylabel('Accuracy', fontweight='bold', fontsize = 13)
plt.ylim(0.86, 0.96)
plt.legend(Scores_.columns)
plt.title('Model comparison', fontweight='bold', fontsize = 14)
plt.show()



# PLOTTING the ROC Curves for all the models

plt.plot(fpr,tpr,label="Gs-DT, auc="+str(np.round(auc,3)))
plt.legend(loc=4)
plt.tight_layout()

plt.plot(fpr_,tpr_,label="Gs-cv-DT, auc="+str(np.round(auc_,3)))
plt.legend(loc=4)
plt.tight_layout()

plt.plot(fpr_rf,tpr_rf,label="Gs-cv-RF, auc="+str(np.round(auc_rf,3)))
plt.legend(loc=4)
plt.tight_layout()

plt.plot(fpr_G,tpr_G,label="XB, auc="+str(np.round(auc_G,3)))
plt.legend(loc=4)
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.title('ROC')
plt.tight_layout()


plt.plot(fpr_sm,tpr_sm,label="Gs-Smote-DT, auc="+str(np.round(auc_sm,3)))
plt.legend(loc=4)
plt.tight_layout()

plt.plot(fpr__sm,tpr__sm,label="Gs-Smote-cv-DT, auc="+str(np.round(auc__sm,3)))
plt.legend(loc=4)
plt.tight_layout()

plt.plot(fpr_rf_sm,tpr_rf_sm,label="Gs-Smote-cv-RF, auc="+str(np.round(auc_rf_sm,3)))
plt.legend(loc=4)
plt.tight_layout()

plt.plot(fpr_G_sm,tpr_G_sm,label="Smote-XB, auc="+str(np.round(auc_G_sm,3)))
plt.legend(loc=4)
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.title('ROC')
plt.tight_layout()


#----------------------- FINAL PREDICTION on the Test data (dataTest) -----------------------------

# selecting the best model: XGBoost with SMOTE

# Since XB-SMOTE model (model_G, lines 638) uses X_train_res (lines 639) which is an array, 
# so we have to convert dataTest (dataframe) to an array, say XTest for predicting on it
 
XTest = dataTest.to_numpy() #converting dataframe to array using .to_numpy()

Pred_final = model_G.predict(XTest) 

#checking count of the classes
sum(Pred_final == 0)
sum(Pred_final == 1)

# saving the results to a csv file: predictions against loan_id

# Since, loan_id is series and Pred_final is an array, convert both to lists first
data = np.array([test1['loan_id'].tolist(), Pred_final.tolist()]).transpose()
# converting to an array of lists

Preds = pd.DataFrame(data, columns = ['loan_id','loan_repaid_pred'])

Preds.to_csv(r'data\\Delta\\Delta_prediction.csv', index = False)



