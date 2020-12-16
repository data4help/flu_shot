
# %% Preliminaries

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

total_train = total_train.loc[:1000, :]

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

NUM_ROWS_SHOWN = 10
for y_column in target_columns:
    fig, axs = plt.subplots(figsize=([ num_target_columns * 10, 10]), ncols=num_target_columns)
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

# %% Feature Engineering - What kind of city/ profession/ industry do you live/work in

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

for grouping in tqdm(groupby_variables):
    for secondary in sorting_variables:
        new_var_name = f'{grouping}_{secondary}_ranking'
        total_train.loc[:, new_var_name] = self_func.create_ranking(grouping, secondary, total_train)
        self_plot.grouping_insights(grouping, secondary, total_train)

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
    sns.heatmap(data=reshaped_table, annot=True, fmt='.5g')
    path = rf'{GRAPH_PATH}/imputation.png'
    fig.savefig(path, bbox_inches='tight')

    # Save preprocessed data
    with open(path_preprocessed_data_file, 'wb') as f:
        pickle.dump(dict_processed_data, f)
else:
    dict_processed_data = pickle.load(open(path_preprocessed_data_file, 'rb'))

# %% Feature Engineering - Creation of Clustering Variables

"""We have to do that now in order to have a non-missing dataset"""

# Finding most correlated variables
pca_model = PCA(n_components=2)
NUM_OF_VARS = 20
for y_column in tqdm(target_columns):

    # Find the most correlated variables
    y_series = total_train.loc[:, y_column]
    processed_data = dict_processed_data[y_column].copy()
    processed_data.loc[:, y_column] = y_series
    df_corr = processed_data.corr().abs().drop(y_column)
    most_correlated_variables = df_corr.loc[:, y_column].sort_values(ascending=False)[:NUM_OF_VARS].index

    # Create PCA and DBSCAN clusters
    df_relevant = processed_data.loc[:, list(most_correlated_variables)]
    pca_factors = pca_model.fit_transform(df_relevant)
    cluster = DBSCAN(eps=0.1, min_samples=10).fit_predict(pca_factors)

    # Plotting
    fig, axs = plt.subplots(figsize=(10, 10))
    sns.scatterplot(x=pca_factors[:, 0], y=pca_factors[:, 1], hue=cluster, palette='tab10', ax=axs)
    path = rf'{GRAPH_PATH}/pca_{y_column}.png'
    fig.savefig(path, bbox_inches='tight')

    # Mean Encode the Clusters
    df_clusters_target = pd.DataFrame({'target': y_series, 'clusters': cluster})
    cluster_encoding_map = df_clusters_target.groupby(['clusters'])['target'].mean()
    mean_encoded_cluster = df_clusters_target.loc[:, 'clusters'].map(cluster_encoding_map)
    dict_processed_data[y_column].loc[:, 'cluster_rank'] = mean_encoded_cluster

# %% Initial Modelling

"""After getting the data ready, it is now time to try out all kind of different models, we start by throwing
all kind of models at the data with their standard parameter. We then take the model which performed the best
initially to test out some hyper-parameter.

From the information we get from the final plot we see that the Gradient Boosting Classifier is the strongest classifier
and therefore also the one we will continue with"""

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
        kfold = KFold(n_splits=10, random_state=28, shuffle=True)
        cv_results = cross_val_score(model, X_data, y_data, cv=kfold, scoring=scoring)
        results.loc[name, y_column] = np.mean(cv_results)

# Plot imputation performance
results.columns.name = 'Target'
results.index.name = 'CV Score'

fig, axs = plt.subplots(figsize=(10, 10))
sns.heatmap(data=results.astype(float), annot=True, ax=axs, fmt='.4g')
path = rf'{GRAPH_PATH}/model_performance.png'
fig.savefig(path, bbox_inches='tight')

# %% Feature Selection

"""From the """

# Feature Importance
gbm_clf = GradientBoostingClassifier()
dict_relevant_data = {}
THRESHOLD_IMPORTANCE = 1 / 100

fig, axs = plt.subplots(figsize=(20, 10), ncols=num_target_columns)
axs = axs.ravel()
for i, y_column in enumerate(target_columns):

    y_data = total_train.loc[:, y_column]
    X_data = dict_processed_data[y_column]

    clf = gbm_clf.fit(X_data, y_data)
    feature_importance = clf.feature_importances_
    df_feature_importance = pd.DataFrame(np.reshape(feature_importance, (-1, 1)),
                                         index=X_data.columns, columns=['columns'])
    df_above_one_percent = df_feature_importance.query('columns >= @THRESHOLD_IMPORTANCE')
    sorted_features = df_above_one_percent.sort_values(by='columns', ascending=False)
    dict_relevant_data[y_column] = dict_processed_data[y_column].loc[:, list(sorted_features.index)]

    sorted_features.plot.bar(ax=axs[i])
path = rf'{GRAPH_PATH}/feature_importance.png'
fig.savefig(path, bbox_inches='tight')

# %% Hyper-Parameter Tuning

"""Now we tune the hyper-parameter of the Gradient Boosting. Furthermore, we apply a SMOTE algorithm to up-sample
the minority class. As we found out in the beginning, the h1n1 vaccine is quite imbalanced (~20% of majority class).
We try out three different levels for the SMOTE algorithm, either the status quo, midway sampling or exactly the same
amount.

"""

gb_grid_params = {
    'gradientboostingclassifier__learning_rate': [1/pow(10, x) for x in range(1, 3)],
    'gradientboostingclassifier__max_depth': np.linspace(3, 30, num=3, dtype=int),
    'gradientboostingclassifier__min_samples_leaf': np.linspace(1, 30, num=3, dtype=int),
    'gradientboostingclassifier__n_estimators': [5],
}

over_sampler = imblearn.over_sampling.SMOTE(random_state=28)
gb_model = GradientBoostingClassifier(random_state=28)
gb_pipeline = imblearn.pipeline.make_pipeline(over_sampler, gb_model)

dict_gridsearch_results = {}
for y_column in tqdm(target_columns):

    # Get the SMOTE params
    num_target_values = total_train.loc[:, y_column].value_counts()
    current_balance = math.ceil(min(num_target_values) / max(num_target_values) * 100) / 100
    smote_grid_params = {'smote__sampling_strategy': np.linspace(current_balance, 1, num=3)}

    # Initialize the GridSearch
    grid_params = {**gb_grid_params, **smote_grid_params}
    gb_gs_clf = GridSearchCV(gb_pipeline, grid_params, n_jobs=-1, cv=5, scoring='roc_auc')

    # Get the data
    y_data = total_train.loc[:, y_column]
    X_data = dict_relevant_data[y_column]

    # Train the model
    gb_gs_clf.fit(X_data, y_data)
    dict_gridsearch_results[y_column] = pd.DataFrame.from_dict(gb_gs_clf.cv_results_)

# %% Actual Prediction

# Find the best parameters
for y_column in target_columns:
    df_params = dict_gridsearch_results[y_column]
    best_params = df_params.sort_values(by='rank_test_score').loc[0, 'params']

gb_pipeline = imblearn.pipeline.make_pipeline(over_sampler, gb_model)
gb_pipeline.set_params(**best_params)

final_pipeline = make_pipeline(
    transforming_pipeline,
    KNNImputer(n_neighbors=100),
    gb_pipeline.set_params(**best_params)
)

final_pipeline.fit(X_data)











