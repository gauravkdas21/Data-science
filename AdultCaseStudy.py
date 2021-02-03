# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 19:23:32 2019

@author: Gaurav.Das
"""

# Data Manipulation 
import numpy as np
import pandas as pd

# Visualization 
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from mpl_toolkits.mplot3d import Axes3D

# Feature Selection and Encoding
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize


# Machine learning 
import sklearn.ensemble as ske
from sklearn import datasets, model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf

# Grid and Random Search
import scipy.stats as st
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Metrics
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc

# Managing Warnings 
import warnings
warnings.filterwarnings('ignore')


# Download
DATASET = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
)

# Load Training and Test Data Sets
headers = ['age', 'workclass', 'fnlwgt', 
           'education', 'education-num', 
           'marital-status', 'occupation', 
           'relationship', 'race', 'sex', 
           'capital-gain', 'capital-loss', 
           'hours-per-week', 'native-country', 
           'predclass']
training_raw = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', 
                       header=None, 
                       names=headers,na_values=["?"])
test_raw = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', 
                      header=None, 
                      names=headers,na_values=["?"])


# Join Datasets
dataset_raw = training_raw.append(test_raw)
dataset_raw.reset_index(inplace=True)
dataset_raw.drop('index',inplace=True,axis=1)

dataset_raw1 = dataset_raw.dropna()
dataset_raw = dataset_raw1

from math import*

import math
import seaborn as sns
# Let’s plot the distribution of each feature
def plot_distribution(dataset, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(dataset.shape[1]) / cols)
    for i, column in enumerate(dataset.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if dataset.dtypes[column] == np.object:
            g = sns.countplot(y=column, data=dataset)
            substrings = [s.get_text()[:18] for s in g.get_yticklabels()]
            g.set(yticklabels=substrings)
            plt.xticks(rotation=25)
        else:
            g = sns.distplot(dataset[column])
            plt.xticks(rotation=25)
    
plot_distribution(dataset_raw, cols=2, width=20, height=40, hspace=0.45, wspace=0.5)


# How many missing values are there in our dataset?
missingno.matrix(dataset_raw, figsize = (30,5))

missingno.bar(dataset_raw, sort='ascending', figsize = (30,5))


dataset_bin = pd.DataFrame() # To contain dataframe with discretised continuous variables #Categorical
dataset_con = pd.DataFrame() # To contain dataframe with continuous variables #continuous


# Let's recode the Class Feature
dataset_raw.loc[dataset_raw['predclass'] == ' >50K', 'predclass'] = 1
dataset_raw.loc[dataset_raw['predclass'] == ' >50K.', 'predclass'] = 1
dataset_raw.loc[dataset_raw['predclass'] == ' <=50K', 'predclass'] = 0
dataset_raw.loc[dataset_raw['predclass'] == ' <=50K.', 'predclass'] = 0


(dataset_raw['predclass'] == ' <=50K').value_counts()
(pd.DataFrame(dataset_raw['predclass'].value_counts()).index)

dataset_bin['predclass'] = dataset_raw['predclass'] #for the target variable
dataset_con['predclass'] = dataset_raw['predclass'] #for the target variable

plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=(20,5)) 
sns.countplot(y="predclass", data=dataset_bin);


dataset_raw.age.value_counts()

dataset_raw.age = dataset_raw.age.astype(float)

dataset_bin['age'] = pd.cut(dataset_raw['age'], 10) # discretised #categorical
dataset_con['age'] = dataset_raw['age'] # non-discretised #continuous

plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=(20,5)) 
sns.countplot(y="age", data=dataset_bin)

sns.distplot(dataset_con.loc[dataset_con['predclass'] == 1]['age'], kde_kws={"label": ">$50K (1)"})
sns.distplot(dataset_con.loc[dataset_con['predclass'] == 0]['age'], kde_kws={"label": "<$50K (0)"})

plt.style.use('fivethirtyeight')
plt.figure(figsize=(20,6)) 
sns.countplot(y="workclass", data=dataset_raw)


# Create buckets for Workclass
dataset_raw.loc[dataset_raw['predclass'] == ' >50K', 'predclass'] = 1

dataset_raw.loc[dataset_raw['workclass'] == ' Without-pay', 'workclass'] = 'Not Working' #spaces
dataset_raw.loc[dataset_raw['workclass'] == ' Never-worked', 'workclass'] = 'Not Working'
dataset_raw.loc[dataset_raw['workclass'] == ' Federal-gov', 'workclass'] = 'Fed-gov'
dataset_raw.loc[dataset_raw['workclass'] == ' State-gov', 'workclass'] = 'Non-fed-gov'
dataset_raw.loc[dataset_raw['workclass'] == ' Local-gov', 'workclass'] = 'Non-fed-gov'
dataset_raw.loc[dataset_raw['workclass'] == ' Self-emp-not-inc', 'workclass'] = 'Self-emp'
dataset_raw.loc[dataset_raw['workclass'] == ' Self-emp-inc', 'workclass'] = 'Self-emp'
dataset_raw.loc[dataset_raw['workclass'] == ' Private', 'workclass'] = 'Private'

dataset_bin['workclass'] = dataset_raw['workclass']
dataset_con['workclass'] = dataset_raw['workclass']

fig = plt.figure(figsize=(20,6)) 
sns.countplot(y="workclass", data=dataset_bin)

plt.figure(figsize=(20,8)) 
sns.countplot(y="occupation", data=dataset_raw)

# Create buckets for Occupation
dataset_raw.loc[dataset_raw['occupation'] == ' Adm-clerical', 'occupation'] = 'Admin'
dataset_raw.loc[dataset_raw['occupation'] == ' Armed-Forces', 'occupation'] = 'Military'
dataset_raw.loc[dataset_raw['occupation'] == ' Craft-repair', 'occupation'] = 'Manual Labour'
dataset_raw.loc[dataset_raw['occupation'] == ' Exec-managerial', 'occupation'] = 'Office Labour'
dataset_raw.loc[dataset_raw['occupation'] == ' Farming-fishing', 'occupation'] = 'Manual Labour'
dataset_raw.loc[dataset_raw['occupation'] == ' Handlers-cleaners', 'occupation'] = 'Manual Labour'
dataset_raw.loc[dataset_raw['occupation'] == ' Machine-op-inspct', 'occupation'] = 'Manual Labour'
dataset_raw.loc[dataset_raw['occupation'] == ' Other-service', 'occupation'] = 'Service'
dataset_raw.loc[dataset_raw['occupation'] == ' Priv-house-serv', 'occupation'] = 'Service'
dataset_raw.loc[dataset_raw['occupation'] == ' Prof-specialty', 'occupation'] = 'Professional'
dataset_raw.loc[dataset_raw['occupation'] == ' Protective-serv', 'occupation'] = 'Military'
dataset_raw.loc[dataset_raw['occupation'] == ' Sales', 'occupation'] = 'Office Labour'
dataset_raw.loc[dataset_raw['occupation'] == ' Tech-support', 'occupation'] = 'Office Labour'
dataset_raw.loc[dataset_raw['occupation'] == ' Transport-moving', 'occupation'] = 'Manual Labour'

dataset_bin['occupation'] = dataset_raw['occupation']
dataset_con['occupation'] = dataset_raw['occupation']


fig = plt.figure(figsize=(20,8))
sns.countplot(y="occupation", data=dataset_bin)


plt.figure(figsize=(20,10)) 
sns.countplot(y="native-country", data=dataset_raw)

dataset_raw['native-country'] = dataset_raw['native-country'].str.replace(' ', '') #remove spaces before country strings

dataset_raw.loc[dataset_raw['native-country'] == 'Cambodia'                    , 'native-country'] = 'SE-Asia'
dataset_raw.loc[dataset_raw['native-country'] == 'Canada'                      , 'native-country'] = 'British-Commonwealth'    
dataset_raw.loc[dataset_raw['native-country'] == 'China'                       , 'native-country'] = 'China'       
dataset_raw.loc[dataset_raw['native-country'] == 'Columbia'                    , 'native-country'] = 'South-America'    
dataset_raw.loc[dataset_raw['native-country'] == 'Cuba'                        , 'native-country'] = 'South-America'        
dataset_raw.loc[dataset_raw['native-country'] == 'Dominican-Republic'          , 'native-country'] = 'South-America'
dataset_raw.loc[dataset_raw['native-country'] == 'Ecuador'                     , 'native-country'] = 'South-America'     
dataset_raw.loc[dataset_raw['native-country'] == 'El-Salvador'                 , 'native-country'] = 'South-America' 
dataset_raw.loc[dataset_raw['native-country'] == 'England'                     , 'native-country'] = 'British-Commonwealth'
dataset_raw.loc[dataset_raw['native-country'] == 'France'                      , 'native-country'] = 'Euro_Group_1'
dataset_raw.loc[dataset_raw['native-country'] == 'Germany'                     , 'native-country'] = 'Euro_Group_1'
dataset_raw.loc[dataset_raw['native-country'] == 'Greece'                      , 'native-country'] = 'Euro_Group_2'
dataset_raw.loc[dataset_raw['native-country'] == 'Guatemala'                   , 'native-country'] = 'South-America'
dataset_raw.loc[dataset_raw['native-country'] == 'Haiti'                       , 'native-country'] = 'South-America'
dataset_raw.loc[dataset_raw['native-country'] == 'Holand-Netherlands'          , 'native-country'] = 'Euro_Group_1'
dataset_raw.loc[dataset_raw['native-country'] == 'Honduras'                    , 'native-country'] = 'South-America'
dataset_raw.loc[dataset_raw['native-country'] == 'Hong'                        , 'native-country'] = 'China'
dataset_raw.loc[dataset_raw['native-country'] == 'Hungary'                     , 'native-country'] = 'Euro_Group_2'
dataset_raw.loc[dataset_raw['native-country'] == 'India'                       , 'native-country'] = 'British-Commonwealth'
dataset_raw.loc[dataset_raw['native-country'] == 'Iran'                        , 'native-country'] = 'Euro_Group_2'
dataset_raw.loc[dataset_raw['native-country'] == 'Ireland'                     , 'native-country'] = 'British-Commonwealth'
dataset_raw.loc[dataset_raw['native-country'] == 'Italy'                       , 'native-country'] = 'Euro_Group_1'
dataset_raw.loc[dataset_raw['native-country'] == 'Jamaica'                     , 'native-country'] = 'South-America'
dataset_raw.loc[dataset_raw['native-country'] == 'Japan'                       , 'native-country'] = 'APAC'
dataset_raw.loc[dataset_raw['native-country'] == 'Laos'                        , 'native-country'] = 'SE-Asia'
dataset_raw.loc[dataset_raw['native-country'] == 'Mexico'                      , 'native-country'] = 'South-America'
dataset_raw.loc[dataset_raw['native-country'] == 'Nicaragua'                   , 'native-country'] = 'South-America'
dataset_raw.loc[dataset_raw['native-country'] == 'Outlying-US(Guam-USVI-etc)'  , 'native-country'] = 'South-America'
dataset_raw.loc[dataset_raw['native-country'] == 'Peru'                        , 'native-country'] = 'South-America'
dataset_raw.loc[dataset_raw['native-country'] == 'Philippines'                 , 'native-country'] = 'SE-Asia'
dataset_raw.loc[dataset_raw['native-country'] == 'Poland'                      , 'native-country'] = 'Euro_Group_2'
dataset_raw.loc[dataset_raw['native-country'] == 'Portugal'                    , 'native-country'] = 'Euro_Group_2'
dataset_raw.loc[dataset_raw['native-country'] == 'Puerto-Rico'                 , 'native-country'] = 'South-America'
dataset_raw.loc[dataset_raw['native-country'] == 'Scotland'                    , 'native-country'] = 'British-Commonwealth'
dataset_raw.loc[dataset_raw['native-country'] == 'South'                       , 'native-country'] = 'Euro_Group_2'
dataset_raw.loc[dataset_raw['native-country'] == 'Taiwan'                      , 'native-country'] = 'China'
dataset_raw.loc[dataset_raw['native-country'] == 'Thailand'                    , 'native-country'] = 'SE-Asia'
dataset_raw.loc[dataset_raw['native-country'] == 'Trinadad&Tobago'             , 'native-country'] = 'South-America'
dataset_raw.loc[dataset_raw['native-country'] == 'United-States'               , 'native-country'] = 'United-States'
dataset_raw.loc[dataset_raw['native-country'] == 'Vietnam'                     , 'native-country'] = 'SE-Asia'
dataset_raw.loc[dataset_raw['native-country'] == 'Yugoslavia'                  , 'native-country'] = 'Euro_Group_2'

dataset_bin['native-country'] = dataset_raw['native-country']
dataset_con['native-country'] = dataset_raw['native-country']

fig = plt.figure(figsize=(20,8)) 
sns.countplot(y="native-country", data=dataset_bin)

#Education
plt.figure(figsize=(20,8)) 
sns.countplot(y="education", data=dataset_raw)

dataset_raw['education'] = dataset_raw['education'].str.replace(' ', '') #remove spaces before country strings

dataset_raw.loc[dataset_raw['education'] == '10th'          , 'education'] = 'Dropout'
dataset_raw.loc[dataset_raw['education'] == '11th'          , 'education'] = 'Dropout'
dataset_raw.loc[dataset_raw['education'] == '12th'          , 'education'] = 'Dropout'
dataset_raw.loc[dataset_raw['education'] == '1st-4th'       , 'education'] = 'Dropout'
dataset_raw.loc[dataset_raw['education'] == '5th-6th'       , 'education'] = 'Dropout'
dataset_raw.loc[dataset_raw['education'] == '7th-8th'       , 'education'] = 'Dropout'
dataset_raw.loc[dataset_raw['education'] == '9th'           , 'education'] = 'Dropout'
dataset_raw.loc[dataset_raw['education'] == 'Assoc-acdm'    , 'education'] = 'Associate'
dataset_raw.loc[dataset_raw['education'] == 'Assoc-voc'     , 'education'] = 'Associate'
dataset_raw.loc[dataset_raw['education'] == 'Bachelors'     , 'education'] = 'Bachelors'
dataset_raw.loc[dataset_raw['education'] == 'Doctorate'     , 'education'] = 'Doctorate'
dataset_raw.loc[dataset_raw['education'] == 'HS-Grad'       , 'education'] = 'HS-Graduate'
dataset_raw.loc[dataset_raw['education'] == 'Masters'       , 'education'] = 'Masters'
dataset_raw.loc[dataset_raw['education'] == 'Preschool'     , 'education'] = 'Dropout'
dataset_raw.loc[dataset_raw['education'] == 'Prof-school'   , 'education'] = 'Professor'
dataset_raw.loc[dataset_raw['education'] == 'Some-college'  , 'education'] = 'HS-Graduate'

dataset_bin['education'] = dataset_raw['education']
dataset_con['education'] = dataset_raw['education']

fig = plt.figure(figsize=(20,8)) 
sns.countplot(y="education", data=dataset_bin)

plt.figure(figsize=(20,8)) 
sns.countplot(y="marital-status", data=dataset_raw)

dataset_raw['marital-status'] = dataset_raw['marital-status'].str.replace(' ', '') #remove spaces before country strings

dataset_raw.loc[dataset_raw['marital-status'] == 'Never-married'        , 'marital-status'] = 'Never-Married'
dataset_raw.loc[dataset_raw['marital-status'] == 'Married-AF-spouse'    , 'marital-status'] = 'Married'
dataset_raw.loc[dataset_raw['marital-status'] == 'Married-civ-spouse'   , 'marital-status'] = 'Married'
dataset_raw.loc[dataset_raw['marital-status'] == 'Married-spouse-absent', 'marital-status'] = 'Not-Married'
dataset_raw.loc[dataset_raw['marital-status'] == 'Separated'            , 'marital-status'] = 'Separated'
dataset_raw.loc[dataset_raw['marital-status'] == 'Divorced'             , 'marital-status'] = 'Separated'
dataset_raw.loc[dataset_raw['marital-status'] == 'Widowed'              , 'marital-status'] = 'Widowed'

dataset_bin['marital-status'] = dataset_raw['marital-status']
dataset_con['marital-status'] = dataset_raw['marital-status']

fig = plt.figure(figsize=(20,8)) 
sns.countplot(y="marital-status", data=dataset_bin)

sns.distplot(dataset_raw.loc[dataset_raw['predclass'] == 1]['fnlwgt'], kde_kws={"label": ">$50K (1)"})
sns.distplot(dataset_raw.loc[dataset_raw['predclass'] == 0]['fnlwgt'], kde_kws={"label": "<$50K (0)"})

# use Cut function to bin the data in equally sized buckets
dataset_bin['fnlwgt'] = pd.cut(dataset_raw['fnlwgt'], 10)
dataset_con['fnlwgt'] = dataset_raw['fnlwgt']

fig = plt.figure(figsize=(20,8)) 
sns.countplot(y="fnlwgt", data=dataset_bin);

# use the Cut function to bin the data in equally sized buckets
sns.distplot(dataset_raw.loc[dataset_raw['predclass'] == 1]['education-num'], kde_kws={"label": ">$50K (1)"})
sns.distplot(dataset_raw.loc[dataset_raw['predclass'] == 0]['education-num'], kde_kws={"label": "<$50K (0)"})

dataset_bin['education-num'] = pd.cut(dataset_raw['education-num'], 10)
dataset_con['education-num'] = dataset_raw['education-num']

fig = plt.figure(figsize=(20,8)) 
sns.countplot(y="education-num", data=dataset_bin)

# use Cut function to bin the data in equally sized buckets
sns.distplot(dataset_raw.loc[dataset_raw['predclass'] == 1]['hours-per-week'], kde_kws={"label": ">$50K (1)"})
sns.distplot(dataset_raw.loc[dataset_raw['predclass'] == 0]['hours-per-week'], kde_kws={"label": "<$50K (0)"})

dataset_bin['hours-per-week'] = pd.cut(dataset_raw['hours-per-week'], 10)
dataset_con['hours-per-week'] = dataset_raw['hours-per-week']

fig = plt.figure(figsize=(20,8)) 
plt.subplot(1, 2, 1)
sns.countplot(y="hours-per-week", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con['hours-per-week'])

# use the Cut function to bin the data in equally sized buckets
dataset_bin['capital-gain'] = pd.cut(dataset_raw['capital-gain'], 5)
dataset_con['capital-gain'] = dataset_raw['capital-gain']

fig = plt.figure(figsize=(20,8)) 
plt.subplot(1, 2, 1)
sns.countplot(y="capital-gain", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con['capital-gain'])

# use the Cut function to bin the data in equally sized buckets
dataset_bin['capital-loss'] = pd.cut(dataset_raw['capital-loss'], 5)
dataset_con['capital-loss'] = dataset_raw['capital-loss']

fig = plt.figure(figsize=(20,8)) 
plt.subplot(1, 2, 1)
sns.countplot(y="capital-loss", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con['capital-loss'])


dataset_con['sex'] = dataset_bin['sex'] = dataset_raw['sex']
dataset_con['race'] = dataset_bin['race'] = dataset_raw['race']
dataset_con['relationship'] = dataset_bin['relationship'] = dataset_raw['relationship']

fig = plt.figure(figsize=(20,8)) 
sns.countplot(y="race", data=dataset_bin)

## Bi-variate Analysis
# Plot a count of the categories from each categorical feature split by prediction class: salary - predclass.
def plot_bivariate_bar(dataset, hue, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    dataset = dataset.select_dtypes(include=[np.object])
    plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(dataset.shape[1]) / cols)
    for i, column in enumerate(dataset.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if dataset.dtypes[column] == np.object:
            g = sns.countplot(y=column, hue=hue, data=dataset)
            substrings = [s.get_text()[:10] for s in g.get_yticklabels()]
            g.set(yticklabels=substrings)
            
plot_bivariate_bar(dataset_con, hue=dataset_raw['predclass'], cols=2, width=20, height=40, hspace=0.4, wspace=0.5)

# Effect of Marital Status and Education on Income, across Marital Status.
plt.style.use('fivethirtyeight')
g = sns.FacetGrid(dataset_con, col='marital-status', size=4, aspect=.9)
g = g.map(sns.boxplot, 'predclass', 'education-num')

# Trends on the Sex, Education, HPW and Age impact on Income.
plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=(20,8)) 
plt.subplot(1, 3, 1)
sns.violinplot(x='sex', y='education-num', hue='predclass', data=dataset_con, split=True, scale='count')

plt.subplot(1, 3, 2)
sns.violinplot(x='sex', y='hours-per-week', hue='predclass', data=dataset_con, split=True, scale='count')

plt.subplot(1, 3, 3)
sns.violinplot(x='sex', y='age', hue='predclass', data=dataset_con, split=True, scale='count')


# Interaction between pairs of features.
sns.pairplot(dataset_con[['age','education-num','hours-per-week','predclass','capital-gain','capital-loss']], 
             hue="predclass", 
             diag_kind="kde",
             size=4) # Rectify!!

# creating interaction features
dataset_con['age-hours'] = dataset_con['age'] * dataset_con['hours-per-week']

dataset_bin['age-hours'] = pd.cut(dataset_con['age-hours'], 10)
dataset_con['age-hours'] = dataset_con['age-hours']


plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=(20,8)) 
plt.subplot(1, 2, 1)
sns.countplot(y="age-hours", data=dataset_bin)
plt.subplot(1, 2, 2)
sns.distplot(dataset_con.loc[dataset_con['predclass'] == 1]['age-hours'], kde_kws={"label": ">$50K (1)"})
sns.distplot(dataset_con.loc[dataset_con['predclass'] == 0]['age-hours'], kde_kws={"label": "<$50K (0)"})


#creating interaction features between categorical variables
dataset_bin['sex-marital'] = dataset_con['sex-marital'] = dataset_con['sex'] + dataset_con['marital-status']

fig = plt.figure(figsize=(20,8)) 
sns.countplot(y="sex-marital", data=dataset_bin)


# One Hot Encoded (dummy variable creation) for all labels before performing Machine Learning
one_hot_cols = dataset_bin.columns.tolist()
one_hot_cols.remove('predclass')
dataset_bin_enc = pd.get_dummies(dataset_bin, columns=one_hot_cols)

dataset_bin_enc.head()


# Label Encode all labels
from sklearn.preprocessing import LabelEncoder
dataset_con_enc = dataset_con.apply(LabelEncoder().fit_transform)

dataset_con_enc.head()


#multivariate
# Create a correlation plot of both datasets.
fig = plt.figure(figsize=(25,10)) 
plt.subplot(1, 2, 1)
# Generate a mask for the upper triangle
mask = np.zeros_like(dataset_bin_enc.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(dataset_bin_enc.corr(), 
            vmin=-1, vmax=1, 
            square=True, 
            cmap=sns.color_palette("RdBu_r", 100), 
            mask=mask, 
            linewidths=.5)
plt.subplot(1, 2, 2)

mask = np.zeros_like(dataset_con_enc.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(dataset_con_enc.corr(), 
            vmin=-1, vmax=1, 
            square=True, 
            cmap=sns.color_palette("RdBu_r", 100), 
            mask=mask, 
            linewidths=.5)

mask = np.zeros_like(dataset_con_enc.corr(), dtype=np.bool)  #No point doing for dataset_bin_enc (discretised)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(dataset_con_enc.corr(), 
            vmin=-1, vmax=1, 
            square=True, 
            cmap=sns.color_palette("RdBu_r", 100), 
            mask=mask, 
            linewidths=.9);
            
'''Step 5: Dimensinality Reduction using Random Forest and PCA'''

# Use Random Forest to get an insight on Feature Importance
clf = RandomForestClassifier()
plt.style.use('fivethirtyeight')
clf.fit(dataset_con_enc.drop('predclass', axis=1), dataset_con_enc['predclass'])
importance = clf.feature_importances_
importance = pd.DataFrame(importance, index=dataset_con_enc.drop('predclass', axis=1).columns, columns=["Importance"])
importance.sort_values(by='Importance', ascending=True).plot(kind='barh', figsize=(20,len(importance)/2))


#extract X and y for both con and bin

features_con_enc = list(dataset_con_enc.columns)[1:]
# Separating out the features
x_con = dataset_con_enc.loc[:, features_con_enc].values
# Separating out the target
y_con = dataset_con_enc.loc[:,['predclass']].values

features_bin_enc = list(dataset_bin_enc.columns)[1:]
# Separating out the features
x_bin = dataset_bin_enc.loc[:, features_bin_enc].values
# Separating out the target
y_bin = dataset_bin_enc.loc[:,['predclass']].values


#apply PCA on con
#we can apply standard scaler on con
from sklearn import preprocessing
std_scale = preprocessing.StandardScaler().fit(dataset_con_enc.drop(['predclass', 'sex'], axis=1))
std_scale
x_con_std = std_scale.transform(dataset_con_enc.drop(['predclass', 'sex'], axis=1))

#apply PCA on con
from sklearn.decomposition import PCA
pca_con = PCA(n_components=10)
pca_fit_con = pca_con.fit_transform(x_con_std)
dataset_con_enc_pc = pd.DataFrame(data = pca_fit_con, columns = ['PC1',
'PC2',
'PC3',
'PC4',
'PC5',
'PC6',
'PC7',
'PC8',
'PC9',
'PC10'])
 
    
#apply PCA on bin
pca_bin = PCA(n_components=75)
pca_fit_bin = pca_bin.fit_transform(x_bin)
dataset_bin_enc_pc = pd.DataFrame(data = pca_fit_bin, columns = ['PC1',
'PC2',
'PC3',
'PC4',
'PC5',
'PC6',
'PC7',
'PC8',
'PC9',
'PC10',
'PC11',
'PC12',
'PC13',
'PC14',
'PC15',
'PC16',
'PC17',
'PC18',
'PC19',
'PC20',
'PC21',
'PC22',
'PC23',
'PC24',
'PC25',
'PC26',
'PC27',
'PC28',
'PC29',
'PC30',
'PC31',
'PC32',
'PC33',
'PC34',
'PC35',
'PC36',
'PC37',
'PC38',
'PC39',
'PC40',
'PC41',
'PC42',
'PC43',
'PC44',
'PC45',
'PC46',
'PC47',
'PC48',
'PC49',
'PC50',
'PC51',
'PC52',
'PC53',
'PC54',
'PC55',
'PC56',
'PC57',
'PC58',
'PC59',
'PC60',
'PC61',
'PC62',
'PC63',
'PC64',
'PC65',
'PC66',
'PC67',
'PC68',
'PC69',
'PC70',
'PC71',
'PC72',
'PC73',
'PC74',
'PC75'])
    
# Graphing the variance per feature
plt.style.use('fivethirtyeight')
plt.figure(figsize=(18,10)) 

plt.subplot(1, 2, 1)
plt.xlabel('PCA Feature')
plt.ylabel('Variance')
plt.title('PCA for Continous dataset')
plt.bar(range(0, pca_con.explained_variance_ratio_.size), pca_con.explained_variance_ratio_)
#youcan increase the PCs from 10 to 12

plt.subplot(1, 2, 2)
plt.xlabel('PCA Feature')
plt.ylabel('Variance')
plt.title('PCA for Discretised dataset')
plt.bar(range(0, pca_bin.explained_variance_ratio_.size), pca_bin.explained_variance_ratio_)

# ----------------- variance doesn't explain properly, so use cumsum

# Graphing the cumsum variance per feature
plt.style.use('fivethirtyeight')
plt.figure(figsize=(18,10)) 

plt.subplot(1, 2, 1)
plt.xlabel('PCA Feature')
plt.ylabel('Variance')
plt.title('PCA for Continous dataset')
plt.bar(range(0, pca_con.explained_variance_ratio_.cumsum().size), pca_con.explained_variance_ratio_.cumsum())
#youcan increase the PCs from 10 to 12

plt.subplot(1, 2, 2)
plt.xlabel('PCA Feature')
plt.ylabel('Variance')
plt.title('PCA for Discretised dataset')
plt.bar(range(0, pca_bin.explained_variance_ratio_.cumsum().size), pca_bin.explained_variance_ratio_.cumsum())

# ----- increase 10 to 12 for con, reduce 75 to 70 for bin
# re-apply PCA on con
from sklearn.decomposition import PCA
pca_con1 = PCA(n_components=12)
pca_fit_con1 = pca_con1.fit_transform(x_con_std)
dataset_con_enc_pc1 = pd.DataFrame(data = pca_fit_con1, columns = ['PC1',
'PC2',
'PC3',
'PC4',
'PC5',
'PC6',
'PC7',
'PC8',
'PC9',
'PC10',
'PC11',
'PC12'])
 
    
#apply PCA on bin
pca_bin1 = PCA(n_components=70)
pca_fit_bin1 = pca_bin1.fit_transform(x_bin)
dataset_bin_enc_pc1 = pd.DataFrame(data = pca_fit_bin1, columns = ['PC1',
'PC2',
'PC3',
'PC4',
'PC5',
'PC6',
'PC7',
'PC8',
'PC9',
'PC10',
'PC11',
'PC12',
'PC13',
'PC14',
'PC15',
'PC16',
'PC17',
'PC18',
'PC19',
'PC20',
'PC21',
'PC22',
'PC23',
'PC24',
'PC25',
'PC26',
'PC27',
'PC28',
'PC29',
'PC30',
'PC31',
'PC32',
'PC33',
'PC34',
'PC35',
'PC36',
'PC37',
'PC38',
'PC39',
'PC40',
'PC41',
'PC42',
'PC43',
'PC44',
'PC45',
'PC46',
'PC47',
'PC48',
'PC49',
'PC50',
'PC51',
'PC52',
'PC53',
'PC54',
'PC55',
'PC56',
'PC57',
'PC58',
'PC59',
'PC60',
'PC61',
'PC62',
'PC63',
'PC64',
'PC65',
'PC66',
'PC67',
'PC68',
'PC69',
'PC70'])
    
# Re-Graphing the cumsum variance per feature
plt.style.use('fivethirtyeight')
plt.figure(figsize=(18,10)) 

plt.subplot(1, 2, 1)
plt.xlabel('PCA Feature')
plt.ylabel('Variance')
plt.title('PCA for Continous dataset')
plt.bar(range(0, pca_con1.explained_variance_ratio_.cumsum().size), pca_con1.explained_variance_ratio_.cumsum())
#youcan increase the PCs from 10 to 12

plt.subplot(1, 2, 2)
plt.xlabel('PCA Feature')
plt.ylabel('Variance')
plt.title('PCA for Discretised dataset')
plt.bar(range(0, pca_bin1.explained_variance_ratio_.cumsum().size), pca_bin1.explained_variance_ratio_.cumsum())

# Create datasets for model fitting
# CON
X_con = dataset_con_enc_pc1.values # X_con means with pca, x_con measn without pca
y_con #already created
# Splitting the Training and Test data sets
from sklearn.model_selection import train_test_split
Xtrain_con, Xtest_con, ytrain_con, ytest_con  = train_test_split(X_con, y_con,test_size=0.30,random_state=1234)

# BIN
X_bin = dataset_bin_enc_pc1.values # X_bin means with pca, x_bin measn without pca
y_bin #already created
# Splitting the Training and Test data sets
Xtrain_bin, Xtest_bin, ytrain_bin, ytest_bin  = train_test_split(X_bin, y_bin,test_size=0.30,random_state=1234)


''' Model fitting on both CON and BIN using Grid Search: 
2. Logistic Regression
3. Random Forest
4. XGBoost
1. KNN
5. SVM 

ROC, cofusion matrix

plot model scores in a dataframe -barplot

Model_name, Score, 0_f1_score, 1_f1_score'''