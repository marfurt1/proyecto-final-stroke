#import libraries

import pandas as pd
import numpy as np
import seaborn as sns 
import math
import warnings 
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, 
                             auc, 
                             precision_score,
                             recall_score,
                             f1_score, 
                             roc_auc_score, RocCurveDisplay, roc_curve,
                             confusion_matrix, classification_report)

from imblearn.under_sampling import NearMiss
from collections import Counter

from imblearn.over_sampling import RandomOverSampler

from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

#Load dataset
df_raw = pd.read_csv('../data/raw/healthcare-dataset-stroke-data.csv')

#usamos este en el drive compartido
df_raw = pd.read_csv('/content/drive/MyDrive/4geeks/Proyecto/colab/Copia de healthcare-dataset-stroke-data.csv')

#BMI Missing Value and outliers
#Create Age bins 
# The labels of bins
labels = ['0 - 4','5 - 9','10 - 19','20 - 29', '30 - 39', '40 - 49', '50 - 59','60 - 69', '70 - +']
# Define the ages between bins
bins = [0,5,10,20,30,40, 50, 60, 70, np.inf]

# pd.cut each column, with each bin closed on left and open on right
df_raw['age_bins'] = pd.cut(df_raw['age'], bins=bins, labels=labels, right=False)

#Calculate the bmi value depend of age bins and gender. using mean value.
df_raw['bmi_new'] = df_raw.groupby(["age_bins","gender"])['bmi'].transform(lambda x: x.fillna(x.mean()))
#Set the value of missing value
df_raw['bmi'].fillna(df_raw['bmi_new'], inplace = True)

#Remove 2 outliers
df_raw.drop(df_raw[(df_raw['bmi'] > 80)].index, inplace=True)

#Transformation of category feature, and remove feature
#set age as int
df_raw['stroke']=df_raw['stroke'].astype(int)
df_raw['age']=df_raw['age'].astype(int)
df_raw['heart_disease']=df_raw['heart_disease'].astype(int)
df_raw['hypertension']=df_raw['hypertension'].astype(int)

# Encoding the 'Sex' column
df_raw['gender'] = df_raw['gender'].map({'Male': 0, 'Female' : 1, 'Other': 2})
df_raw['gender'] = df_raw['gender'].astype(int)

# Encoding the 'Residence_type' column
df_raw['Residence_type'] = df_raw['Residence_type'].map({'Urban': 0, 'Rural' : 1})
df_raw['Residence_type']=df_raw['Residence_type'].astype(int)

# Encoding the 'smoking status' column
df_raw['smoking_status'] = df_raw['smoking_status'].map({'Unknown': 0, 'never smoked' : 1, 'smokes': 2 , 'formerly smoked':3})
df_raw['smoking_status']=df_raw['smoking_status'].astype(int)

# Encoding the 'ever_married' column
df_raw['ever_married'] = df_raw['ever_married'].apply(lambda x: 1 if x == 'Yes' else 0)
df_raw['ever_married'] =df_raw['ever_married'].astype(int)

# Encoding the 'work_type' column
df_raw['work_type'] = df_raw['work_type'].map({'Private' : 0, 'Self-employed': 1, 'children': 2 , 'Govt_job':3, 'Never_worked':4})
df_raw['work_type'] =df_raw['work_type'].astype(int)

df_raw.drop(["age_bins","bmi_new","id"],axis=1,inplace=True)

# falta traer el modelo