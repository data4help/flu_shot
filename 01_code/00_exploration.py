
### Packages
# Basic
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import math

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Pre-processing
from category_encoders import TargetEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

# Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

# Forecasting
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

# Clustering
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Upsampling
import imblearn

# Self written
import _func_plotting as self_plot
import _func_pre as self_func

import importlib

importlib.reload(self_plot)
importlib.reload(self_func)

# Paths
MAIN_PATH = r'C:/Users/DEPMORA1/Documents/Projects/flu_shot'
RAW_PATH = rf'{MAIN_PATH}/data/raw'
DATA_PATH = rf'{MAIN_PATH}/data/preprocessed'
GRAPH_PATH = rf'{MAIN_PATH}/output/graphs'

# %% Data import

# Data
train_features = pd.read_csv(rf'{RAW_PATH}/training_features.csv')
train_labels = pd.read_csv(rf'{RAW_PATH}/training_labels.csv')
test_features = pd.read_csv(rf'{RAW_PATH}/test_features.csv')
total_train = pd.merge(train_features, train_labels, on='respondent_id')
total_train.drop(columns=['respondent_id'], inplace=True)

total_train = total_train.loc[:1_000, :]

# %% Exploration of Dependent Variable

""" At first it could be interesting to see whether the two dependent variables are balanced. For that we create a
count-plot. Here we see that the h1n1 vaccine variable is relatively balanced, whereas the seasonal variable
is nicely balanced. We therefore will consider playing a bit with sampling methods for the h1n1 vaccine variable """

target_columns = ['h1n1_vaccine', 'seasonal_vaccine']
num_target_columns = len(target_columns)
fig, axs = plt.subplots(ncols=2, figsize=(num_target_columns * 10, 10))
axs = axs.ravel()
for i, column in enumerate(target_columns):
    sns.countplot(total_train.loc[:, column], ax=axs[i])
path = rf'{GRAPH_PATH}/dependent_count_plot.png'
fig.savefig(path, bbox_inches='tight')

# %% Exploration - Most correlated features

"""In order to see what the important variables are we take a first look at the most and least correlated variables
with the target. From there we already have some sort of an idea in which direction to look"""

NUM_ROWS_SHOWN = 10
for y_column in target_columns:
    fig, axs = plt.subplots(figsize=([num_target_columns * 10, 10]), ncols=num_target_columns)
    axs = axs.ravel()
    series_corr = total_train.corr().loc[:, y_column].drop(target_columns)
    for i, (bool_order, title) in enumerate(zip([True, False], ['Lowest Correlations', 'Highest Correlations'])):
        target_corr_sorted = series_corr.sort_values(ascending=bool_order)[:NUM_ROWS_SHOWN]
        sns.heatmap(pd.DataFrame(target_corr_sorted).T, yticklabels=False, ax=axs[i])
        axs[i].set_title(title)
    axs[0].set_ylabel(y_column)
    path = rf'{GRAPH_PATH}\{y_column}.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close()

# %% Feature Engineering - Creation of scoring variables

"""In order for the model to better pick up information from multiple columns, we can give the model the information
in a better to understand format. For that we create a score for being careful, and for being in the risk group.
This score is nothing other than a sum of binary variables which indicate carefulness/ or being part of a risk
group."""

# Careful People
careful_behavior = [
    'behavioral_antiviral_meds',
    'behavioral_avoidance',
    'behavioral_face_mask',
    'behavioral_wash_hands',
    'behavioral_large_gatherings',
    'behavioral_outside_home',
    'behavioral_touch_face',
]

# Risk Group
risk_group_columns = [
    'chronic_med_condition',
    'health_worker',
]

df_careful_corr = total_train.loc[:, careful_behavior].corr()
fig, axs = plt.subplots(figsize=(10, 10))
sns.heatmap(df_careful_corr, ax=axs)
path = rf'{GRAPH_PATH}\careful_correlation_matrix.png'
fig.savefig(path, bbox_inches='tight')

# %% Missing Observations

""" To get a better impression how many observations are actually missing, we plot bar-charts for every column.
We can see from the chart below that for some variables a significant amount of observations are missing. Further,
we basically have missing values for every column, which is why we will spend more time on a sophisticated
imputation method"""

missing_pct = total_train.agg(lambda x: x.isnull().sum() / len(x) * 100)
missing_pct_index = missing_pct.reset_index()
missing_pct_index.rename(columns={0: "Percentage Missing", "index": "Columns"}, inplace=True)
sorted_long_missing_data = missing_pct_index.sort_values(by="Percentage Missing", ascending=False)
fig, axs = plt.subplots(figsize=(20, 10))
plot = sns.barplot(data=sorted_long_missing_data, x="Columns", y="Percentage Missing", ax=axs)
plt.xticks(rotation=90)
path = rf"{GRAPH_PATH}/missing_observation_threshold.png"
fig.savefig(path, bbox_inches='tight')
