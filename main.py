
# %% Preliminaries

### Packages
# Basic
import pandas as pd
import numpy as np
from tqdm import tqdm

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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso, LinearRegression

# Self written
import _func_plotting as self_plot
import _functions as self_func

import importlib
importlib.reload(self_plot)
importlib.reload(self_func)

# Paths
MAIN_PATH = r'C:/Users/DEPMORA1/Documents/Projects/flu_shot'
DATA_PATH = rf'{MAIN_PATH}/data'
OUTPUT_PATH = rf'{MAIN_PATH}/output'

# %% Data import

# Data
train_features = pd.read_csv(rf'{DATA_PATH}/training_features.csv')
train_labels = pd.read_csv(rf'{DATA_PATH}/training_labels.csv')
test_features = pd.read_csv(rf'{DATA_PATH}/test_features.csv')
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
path = rf'{OUTPUT_PATH}/dependent_count_plot.png'
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
    path = rf'{OUTPUT_PATH}\{title}.png'
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

# %% Feature Engineering - What kind of city do you live in

""""""

grouping_variables = [
    'employment_industry',
    'employment_occupation',
    'hhs_geo_region'
]

secondary_variables = [
    'doctor_recc_h1n1',
    'opinion_seas_risk',
    'income_poverty',
    'doctor_recc_seasonal',
    'race',
    'health_insurance'
]

for grouping in grouping_variables:
    for secondary in secondary_variables:
        self_plot.grouping_insights(grouping, secondary, total_train)


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
path = rf'{OUTPUT_PATH}\careful_correlation_matrix.png'
fig.savefig(path, bbox_inches='tight')

total_train.loc[:, 'careful_score'] = total_train.loc[:, careful_behavior].sum(axis=1)

# Risk Group
risk_group_columns = [
    'chronic_med_condition',
    'health_worker',
]
old_series = total_train.loc[:, 'age_group'] == '65+ Years'
total_train.loc[:, 'risk_score'] = total_train.loc[:, risk_group_columns].sum(axis=1) + old_series

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
path = rf"{OUTPUT_PATH}/missing_observation_threshold.png"
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

preprocessing_pipeline = ColumnTransformer(
    transformers=[
        ('TargetEncoding', target_encoding_transformer, features_more_than_five),
        ('Dummy', dummy_transformer, features_fewer_than_six)
    ]
)

# %% Imputation

"""For the imputation we try out several imputation methods. We start by using simpler methods, such as
mean and median imputation, before then moving on to model based imputation. Here we use BayesRidge and
LinearRegression. The error metric chosen is roc_auc, given that this also represents the error metrics used
by the overall challenge.

From the results we can see that the best method seems to be """

# Preliminaries
target_clf = GradientBoostingClassifier()
N_SPLITS = 5
simple_methods = ['mean', 'median', 'most_frequent']
impute_classifiers = [
    LinearRegression(),
]

#
df_imputed = pd.DataFrame()
for y_column in tqdm(target_columns):
    y_series = total_train.loc[:, y_column]

    X_raw = total_train.loc[:, feature_columns]
    X_missing = preprocessing_pipeline.fit_transform(X_raw, y_series)

    df_simply_imputed = self_func.simple_imputing(X_missing, y_series, target_clf, simple_methods, N_SPLITS)
    df_iteratively_imputed = self_func.iterative_imputing(X_missing, y_series, target_clf, impute_classifiers, N_SPLITS)
    df_imputed = pd.concat([df_imputed, df_iteratively_imputed, df_simply_imputed], axis=0)

fig, axs = plt.subplots(figsize=(10, 10))
sns.boxplot(data=df_imputed, x='score', y='method' ,hue='target')
path = rf'{OUTPUT_PATH}/imputation.png'
fig.savefig(path, bbox_inches='tight')




















