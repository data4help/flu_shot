
# %% Preliminaries

### Packages
# Basic
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

# Plotting
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Pre-processing
from category_encoders import TargetEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

# Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

# Forecasting
from sklearn.linear_model import LogisticRegression

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

# %% Exploration of Dependent Variable

""" At first it could be interesting to see whether the two dependent variables are balanced. For that we create a
count-plot. Here we see that the h1n1 vaccine variable is relatively balanced, whereas the seasonal variable
is nicely balanced. We therefore will consider playing a bit with sampling methods for the h1n1 vaccine variable """

target_columns = ['h1n1_vaccine', 'seasonal_vaccine']
fig, axs = plt.subplots(ncols=2, figsize=(len(target_columns) * 10, 10))
axs = axs.ravel()
for i, column in enumerate(target_columns):
    sns.countplot(total_train.loc[:, column], ax=axs[i])
path = rf'{GRAPH_PATH}/dependent_count_plot.png'
fig.savefig(path, bbox_inches='tight')

# %% Exploration of Independent Variables

""" We start looking at the distribution of variables. We will plot pie-charts for categorical variables and histograms
for continuous/ numeric variables. For that we have to adjust for the case where we have a numeric variable,
which is actually a categorical variable, disguised as a numeric one. This is done by checking how often each value
occurs

The biggest insight of the graphing is that all variables are of categorical nature, which mean that we will
have to think about valid encoding methods. Secondly we find several columns with a significant amount of missing
observations."""

CATEGORICAL_THRESHOLD = 5/100 * len(total_train)
feature_columns = list(set(total_train.columns) - set(target_columns))

for column in tqdm(feature_columns):
    series = total_train.loc[:, column]
    no_nan_series = series.dropna()
    highest_occurring_category = max(no_nan_series.value_counts())
    if highest_occurring_category > CATEGORICAL_THRESHOLD:
        object_series = series.fillna(-1)
        self_plot.categorical_plot(object_series)
    else:
        self_plot.numerical_plot(no_nan_series)

# %% Feature Engineering - Beginning

"""In order to see what the important variables are we take a first look at the most and least correlated variables
with the target. From there we already have some sort of an idea in which direction to look"""

NUM_ROWS_SHOWN = 15
num_target_columns = len(target_columns)
for bool_order, title in zip([True, False], ['Lowest Correlations', 'Highest Correlations']):
    fig, axs = plt.subplots(figsize=([ num_target_columns * 10, 10]), ncols=num_target_columns)
    axs = axs.ravel()
    for i, y_column in enumerate(target_columns):
        target_corr_sorted = pd.get_dummies(total_train).corr().sort_values(y_column, ascending=bool_order)
        target_corrs_top_ten = target_corr_sorted.loc[:, [y_column]][:NUM_ROWS_SHOWN].T
        sns.heatmap(target_corrs_top_ten, yticklabels=False, ax=axs[i])
        axs[i].set_title(y_column)
    axs[0].set_ylabel(title)
    path = rf'{GRAPH_PATH}\{title}.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close()

# %% Feature Engineering - Subjective Answers

"""The dataset contains several variables which has subjective answering - for example how would you rate the risk
of getting the seasonal flu. Given that the differentation between 'Very Low' and 'Somewhat Low' is not comparable
between participants, we reduce the possible answers from five to three. This potentially reduces confusion.
It is not possible to treat them as a numeric variable either given that the answer choice 'Don't Know' does not lay
in the middle between low and high"""

new_opinion_cat = {1: 'Low', 2: 'Low', 3: 'No Idea', 4: 'High', 5: 'High'}
opinion_variables = [x for x in total_train.columns if x.startswith('opinion')]
for var in opinion_variables:
    total_train.loc[:, var] = total_train.loc[:, var].map(new_opinion_cat)

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

df_careful_corr = total_train.loc[:, careful_behavior].corr()
fig, axs = plt.subplots(figsize=(10, 10))
sns.heatmap(df_careful_corr, ax=axs)
path = rf'{GRAPH_PATH}\careful_correlation_matrix.png'
fig.savefig(path, bbox_inches='tight')

total_train.loc[:, 'careful_score'] = total_train.loc[:, careful_behavior].sum(axis=1)

# Risk Group
risk_group_columns = [
    'chronic_med_condition',
    'health_worker',
]
old_series = total_train.loc[:, 'age_group'] == '65+ Years'
total_train.loc[:, 'risk_score'] = total_train.loc[:, risk_group_columns].sum(axis=1) + old_series

# %% Feature Engineering - What kind of city do you live in

"""What we would like to indicate now are more characteristics about every city/occupation/industry individually.
Particularly we would like to find out what the average income/ age / etc. is for all aforementioned grouping variables.
For that we create plots as well as ranking variables."""

groupby_variables = [
    'employment_industry',
    'employment_occupation',
    'hhs_geo_region'
]

sorting_variables = [
    'h1n1_concern',
    'careful_score',
    'income_poverty',
    'careful_score',
    'income_poverty',
]
"""
for grouping in tqdm(groupby_variables):
    for secondary in sorting_variables:
        new_var_name = f'{grouping}_{secondary}_ranking'
        total_train.loc[:, new_var_name] = self_func.create_ranking(grouping, secondary, total_train)
        self_plot.grouping_insights(grouping, secondary, total_train)
"""
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

# %% Pre-Processing Pipeline

"""For the imputation we try out different imputation methods, ranging from non at all, over simple ones such as
median and mean. Finally we try a range of classifiers. For doing that we have to pre-process our data a bit. For doing
that we separate all features into two buckets. One bucket for all features with fewer or equal to five categories,
the other bucket for the rest. The latter bucket will be target encoded, the former bucket transformed to dummies.
"""

# Column Separator
NUM_CAT_THRESHOLD = 5
features_fewer_than_six = [x for x in feature_columns if self_func.fewer_than_cutoff(x, total_train, NUM_CAT_THRESHOLD)]
features_more_than_five = list(set(feature_columns) - set(features_fewer_than_six))

# Pre-processing pipeline
target_encoding_transformer = make_pipeline(
    TargetEncoder(handle_missing='return_nan'),
    MinMaxScaler()
)
dummy_transformer = make_pipeline(
    OneHotEncoder(handle_missing='return_nan')
)
transforming_pipeline = ColumnTransformer(
    transformers=[
        ('TargetEncoding', target_encoding_transformer, features_more_than_five),
        ('Dummy', dummy_transformer, features_fewer_than_six)
    ]
)

# %% Imputation

"""For the imputation we will use KNN Imputer from Sklearn. In order to decide how many neigbours we try out different
ones and run a GridSearch to predict the two target variables

From the results we can see that the best method seems to be """

path_preprocessed_data_file = rf'{DATA_PATH}/preprocessed_data.pickle'
if not os.path.isfile(path_preprocessed_data_file):
    # Conducting imputations
    N_SPLITS = 5
    n_neighbors = [10, 25, 50, 75, 100, 150, 200]
    example_clf = LogisticRegression(max_iter=1_000)
    df_imputed = pd.DataFrame()
    dict_processed_data = {}
    for y_column in tqdm(target_columns):
        y_series = total_train.loc[:, y_column]
        X_raw = total_train.loc[:, feature_columns]
        X_missing = transforming_pipeline.fit_transform(X_raw, y_series)

        df_temp_results = self_func.knn_imputing(X_missing, y_series, example_clf, n_neighbors, N_SPLITS)
        dict_processed_data[y_column] = self_func.imputing_data(X_raw, y_series, df_temp_results, transforming_pipeline)
        df_imputed = pd.concat([df_imputed, df_temp_results], axis=0)

    # Plot imputation performance
    df_imputed = df_imputed.astype({'score': float})
    grouped_imputed = df_imputed.groupby(['neighbors', 'target']).mean()
    reshaped_table = pd.pivot_table(grouped_imputed, values='score', index='neighbors',columns='target')
    fig, axs = plt.subplots(figsize=(10, 10))
    sns.heatmap(data=reshaped_table, annot=True, fmt='.4g')
    path = rf'{GRAPH_PATH}/imputation.png'
    fig.savefig(path, bbox_inches='tight')

    # Save preprocessed data
    with open(path_preprocessed_data_file, 'wb') as f:
        pickle.dump(dict_processed_data, f)
else:
    dict_processed_data = pickle.load(open(path_preprocessed_data_file, 'rb'))

# %% Initial Modelling

"""After getting the data ready, it is now time to try out all kind of different models, we start by throwing
all kind of models at the data with their standard parameter. We then take the model which performed the best
initially to test out some hyper-parameter. Given the scoring of the data science problem"""

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# prepare models
models = []
models.append(('LR', LogisticRegression(max_iter=1_000)))
models.append(('MLP', MLPClassifier(max_iter=1_000)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RFT', RandomForestClassifier()))

results = pd.DataFrame(index=[x[0] for x in models], columns=target_columns)
scoring='roc_auc'
for y_column in target_columns:
    y_data = total_train.loc[:, y_column]
    X_data = dict_processed_data[y_column]
    for name, model in tqdm(models):
        kfold = model_selection.KFold(n_splits=10, random_state=28, shuffle=True)
        cv_results = model_selection.cross_val_score(model, X_data, y_data, cv=kfold, scoring=scoring)
        results.loc[name, y_column] = np.mean(cv_results)

# Plot imputation performance
results.columns.name = 'Target'
results.index.name = 'CV Score'

fig, axs = plt.subplots(figsize=(10, 10))
sns.heatmap(data=results.astype(float), annot=True, ax=axs, fmt='.4g')
path = rf'{GRAPH_PATH}/model_performance.png'
fig.savefig(path, bbox_inches='tight')

# %% Hpyer-Parameter Tuning for Final Models

"""Now it is time to train hyper-parameterized models for """





