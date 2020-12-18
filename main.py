
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
GRAPH_PATH = rf'{MAIN_PATH}/graphs'

# %% Data import

# Data
train_features = pd.read_csv(rf'{RAW_PATH}/training_features.csv')
train_labels = pd.read_csv(rf'{RAW_PATH}/training_labels.csv')
test_features = pd.read_csv(rf'{RAW_PATH}/test_features.csv')

# Variable Access
target_columns = ['h1n1_vaccine', 'seasonal_vaccine']
feature_columns = list(set(train_features.columns) - set(target_columns))

TEST_SAMPLE_LENGTH = 1_000
train_features = train_features.loc[:TEST_SAMPLE_LENGTH, :]
train_labels = train_labels.loc[:TEST_SAMPLE_LENGTH, target_columns[0]]

# %% Feature Engineering - Subjective Answers

"""The dataset contains several variables which has subjective answering - for example how would you rate the risk
of getting the seasonal flu. Given that the differentation between 'Very Low' and 'Somewhat Low' is not comparable
between participants, we reduce the possible answers from five to three. This potentially reduces confusion.
It is not possible to treat them as a numeric variable either given that the answer choice 'Don't Know' does not lay
in the middle between low and high"""


class OpinionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X_data, y_data=None):
        return self

    def transform(self, X_data, y_data=None):
        new_opinion_cat = {1: 'Low', 2: 'Low', 3: 'No Idea', 4: 'High', 5: 'High'}
        opinion_variables = [x for x in X_data.columns if x.startswith('opinion')]
        for var in opinion_variables:
            X_data.loc[:, var] = X_data.loc[:, var].map(new_opinion_cat)
        return X_data


# %% Feature Engineering - Creation of scoring variables

"""In order for the model to better pick up information from multiple columns, we can give the model the information
in a better to understand format. For that we create a score for being careful, and for being in the risk group.
This score is nothing other than a sum of binary variables which indicate carefulness/ or being part of a risk
group."""

# Careful People
careful_behavior_columns = [
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


class ScoringVariableCreation(BaseEstimator, TransformerMixin):

    def __init__(self, careful_columns, risk_group_columns):
        self.careful_columns = careful_columns
        self.risk_group_columns = risk_group_columns

    def fit(self, X_data, y_data=None):
        return self

    def transform(self, X_data, y_data=None):
        # Careful score
        X_data.loc[:, 'careful_score'] = X_data.loc[:, self.careful_columns].sum(axis=1)

        # Risk group score
        bool_old_people = X_data.loc[:, 'age_group'] == '65+ Years'
        X_data.loc[:, 'risk_group_score'] = X_data.loc[:, self.risk_group_columns].sum(axis=1) + bool_old_people
        return X_data


# %% Pre-Processing Pipeline

"""For the imputation we try out different imputation methods, ranging from non at all, over simple ones such as
median and mean. Finally we try a range of classifiers. For doing that we have to pre-process our data a bit. For doing
that we separate all features into two buckets. One bucket for all features with fewer or equal to five categories,
the other bucket for the rest. The latter bucket will be target encoded, the former bucket transformed to dummies.
"""


class TransformerPipeline(BaseEstimator, TransformerMixin):

    def __init__(self, cat_threshold, feature_columns):
        self.cat_threshold = cat_threshold
        self.feature_columns = feature_columns

    def fewer_than_cutoff(self, column_name, data, n):
        series = data.loc[:, column_name]
        num_categories = len(series.value_counts())
        return num_categories <= n

    def fit(self, X_data, y_data=None):
        columns_less_than_six = [x for x in self.feature_columns if
                                 self.fewer_than_cutoff(x, X_data, self.cat_threshold)]
        columns_more_than_five = list(set(X_data.columns) - set(columns_less_than_six))

        # Pre-processing pipeline
        target_encoding_transformer = make_pipeline(
            TargetEncoder(handle_missing='return_nan'),
            MinMaxScaler()
        )
        dummy_transformer = make_pipeline(
            OneHotEncoder(handle_missing='return_nan')
        )
        self.transforming_pipeline = ColumnTransformer(
            transformers=[
                ('TargetEncoding', target_encoding_transformer, columns_more_than_five),
                ('Dummy', dummy_transformer, columns_less_than_six)
            ]
        )

        self.fitted_transformer = self.transforming_pipeline.fit(X_data, y_data)

        return self

    def transform(self, X_data, y_data=None):
        processed_data = self.fitted_transformer.transform(X_data)
        column_names = self_func.get_feature_names(self.transforming_pipeline)
        df_processed_data = pd.DataFrame(processed_data, columns=column_names)
        return df_processed_data

# %% Imputation

from sklearn.impute import KNNImputer

class KNNImputation():

    def __init__(self, N_SPLITS, n_neighbors, classification_model):
        self.N_SPLITS = N_SPLITS
        self.n_neighbors_list = n_neighbors
        self.classification_model = classification_model

    def plot_imputing_score(self, data, target_name):
        fig, axs = plt.subplots(figsize=(10, 10))
        sns.boxplot(data=data, x='neighbors', y='score', ax=axs)
        axs.set_title(target_name)
        path = rf'{GRAPH_PATH}/imputation_{target_name}.png'
        fig.savefig(path, bbox_inches='tight')

    def fit(self, X_data, y_data=None):

        # Set Up
        index = list(range(self.N_SPLITS * len(self.n_neighbors_list)))
        columns = ['score', 'neighbors']
        df_imputed = pd.DataFrame(index=index, columns=columns)
        target_name = y_data.name

        # Trying out different neighbors
        for i, n_neighbors in enumerate(self.n_neighbors_list):
            imputer = KNNImputer(n_neighbors=n_neighbors)
            estimator = make_pipeline(imputer, self.classification_model)

            range_begin, range_end = (i * self.N_SPLITS), ((i+1) * self.N_SPLITS - 1)
            df_imputed.loc[range_begin:range_end, 'score'] = cross_val_score(
                estimator, X_data, y_data, scoring='roc_auc', cv=self.N_SPLITS
            )

            df_imputed.loc[range_begin:range_end, 'neighbors'] = n_neighbors

        # Assessing Results and Fitting best model
        self.plot_imputing_score(df_imputed, target_name)
        df_imputed.loc[:, 'score'] = df_imputed.loc[:, 'score'].astype(float)
        best_n_neighbors = df_imputed.groupby('neighbors')['score'].mean().idxmax()
        self.best_imputer = KNNImputer(n_neighbors=best_n_neighbors).fit(X_data)
        return self

    def transform(self, X_data, y_data=None):

        columns = X_data.columns
        imputed_X_data = self.best_imputer.transform(X_data)

        df_imputed = pd.DataFrame(imputed_X_data, columns=columns)
        assert not df_imputed.isna().any().any(), 'Still Missing Data!'
        return df_imputed



# Conducting imputations
N_SPLITS = 5
n_neighbors = [10, 25, 50, 75, 100, 150, 200]
# n_neighbors = [2, 5]
example_clf = LogisticRegression(max_iter=1_000)


pipe = make_pipeline(
    # Feature Engineering
    OpinionTransformer(),
    ScoringVariableCreation(careful_behavior_columns, risk_group_columns),
    TransformerPipeline(5, feature_columns),

    # Imputation
    KNNImputation(N_SPLITS, n_neighbors, example_clf)

    # Clustering
)


pipe.fit(train_features, train_labels)
transformed_data = pipe.transform(test_features)
print(transformed_data)



# # %% Feature Engineering - Creation of Clustering Variables
#
# """We have to do that now in order to have a non-missing dataset"""
#
# # Finding most correlated variables
# pca_model = PCA(n_components=2)
# NUM_OF_VARS = 20
# for y_column in tqdm(target_columns):
#     # Find the most correlated variables
#     y_series = total_train.loc[:, y_column]
#     processed_data = dict_processed_data[y_column].copy()
#     processed_data.loc[:, y_column] = y_series
#     df_corr = processed_data.corr().abs().drop(y_column)
#     most_correlated_variables = df_corr.loc[:, y_column].sort_values(ascending=False)[:NUM_OF_VARS].index
#
#     # Create PCA and DBSCAN clusters
#     df_relevant = processed_data.loc[:, list(most_correlated_variables)]
#     pca_factors = pca_model.fit_transform(df_relevant)
#     cluster = DBSCAN(eps=0.1, min_samples=10).fit_predict(pca_factors)
#
#     # Plotting
#     fig, axs = plt.subplots(figsize=(10, 10))
#     sns.scatterplot(x=pca_factors[:, 0], y=pca_factors[:, 1], hue=cluster, palette='tab10', ax=axs)
#     path = rf'{GRAPH_PATH}/pca_{y_column}.png'
#     fig.savefig(path, bbox_inches='tight')
#
#     # Mean Encode the Clusters
#     df_clusters_target = pd.DataFrame({'target': y_series, 'clusters': cluster})
#     cluster_encoding_map = df_clusters_target.groupby(['clusters'])['target'].mean()
#     mean_encoded_cluster = df_clusters_target.loc[:, 'clusters'].map(cluster_encoding_map)
#     dict_processed_data[y_column].loc[:, 'cluster_rank'] = mean_encoded_cluster
#
# # %% Initial Modelling
#
# """After getting the data ready, it is now time to try out all kind of different models, we start by throwing
# all kind of models at the data with their standard parameter. We then take the model which performed the best
# initially to test out some hyper-parameter.
# From the information we get from the final plot we see that the Gradient Boosting Classifier is the strongest classifier
# and therefore also the one we will continue with"""
#
# # prepare models
# models = []
# models.append(('LR', LogisticRegression(max_iter=1_000)))
# models.append(('MLP', MLPClassifier(max_iter=1_000)))
# models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('CART', DecisionTreeClassifier()))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC()))
# models.append(('GBM', GradientBoostingClassifier()))
# models.append(('RFT', RandomForestClassifier()))
#
# results = pd.DataFrame(index=[x[0] for x in models], columns=target_columns)
# scoring = 'roc_auc'
# for y_column in target_columns:
#     y_data = total_train.loc[:, y_column]
#     X_data = dict_processed_data[y_column]
#     for name, model in tqdm(models):
#         kfold = KFold(n_splits=10, random_state=28, shuffle=True)
#         cv_results = cross_val_score(model, X_data, y_data, cv=kfold, scoring=scoring)
#         results.loc[name, y_column] = np.mean(cv_results)
#
# # Plot imputation performance
# results.columns.name = 'Target'
# results.index.name = 'CV Score'
#
# fig, axs = plt.subplots(figsize=(10, 10))
# sns.heatmap(data=results.astype(float), annot=True, ax=axs, fmt='.4g')
# path = rf'{GRAPH_PATH}/model_performance.png'
# fig.savefig(path, bbox_inches='tight')
#
# # %% Feature Selection
#
# """From the """
#
# # Feature Importance
# gbm_clf = GradientBoostingClassifier()
# dict_relevant_data = {}
# THRESHOLD_IMPORTANCE = 1 / 100
#
# fig, axs = plt.subplots(figsize=(20, 10), ncols=num_target_columns)
# axs = axs.ravel()
# for i, y_column in enumerate(target_columns):
#     y_data = total_train.loc[:, y_column]
#     X_data = dict_processed_data[y_column]
#
#     clf = gbm_clf.fit(X_data, y_data)
#     feature_importance = clf.feature_importances_
#     df_feature_importance = pd.DataFrame(np.reshape(feature_importance, (-1, 1)),
#                                          index=X_data.columns, columns=['columns'])
#     df_above_one_percent = df_feature_importance.query('columns >= @THRESHOLD_IMPORTANCE')
#     sorted_features = df_above_one_percent.sort_values(by='columns', ascending=False)
#     dict_relevant_data[y_column] = dict_processed_data[y_column].loc[:, list(sorted_features.index)]
#
#     sorted_features.plot.bar(ax=axs[i])
# path = rf'{GRAPH_PATH}/feature_importance.png'
# fig.savefig(path, bbox_inches='tight')
#
# # %% Hyper-Parameter Tuning
#
# """Now we tune the hyper-parameter of the Gradient Boosting. Furthermore, we apply a SMOTE algorithm to up-sample
# the minority class. As we found out in the beginning, the h1n1 vaccine is quite imbalanced (~20% of majority class).
# We try out three different levels for the SMOTE algorithm, either the status quo, midway sampling or exactly the same
# amount.
# """
#
# gb_grid_params = {
#     'gradientboostingclassifier__learning_rate': [1 / pow(10, x) for x in range(1, 3)],
#     'gradientboostingclassifier__max_depth': np.linspace(3, 30, num=3, dtype=int),
#     'gradientboostingclassifier__min_samples_leaf': np.linspace(1, 30, num=3, dtype=int),
#     'gradientboostingclassifier__n_estimators': [5],
# }
#
# over_sampler = imblearn.over_sampling.SMOTE(random_state=28)
# gb_model = GradientBoostingClassifier(random_state=28)
# gb_pipeline = imblearn.pipeline.make_pipeline(over_sampler, gb_model)
#
# dict_gridsearch_results = {}
# for y_column in tqdm(target_columns):
#     # Get the SMOTE params
#     num_target_values = total_train.loc[:, y_column].value_counts()
#     current_balance = math.ceil(min(num_target_values) / max(num_target_values) * 100) / 100
#     smote_grid_params = {'smote__sampling_strategy': np.linspace(current_balance, 1, num=3)}
#
#     # Initialize the GridSearch
#     grid_params = {**gb_grid_params, **smote_grid_params}
#     gb_gs_clf = GridSearchCV(gb_pipeline, grid_params, n_jobs=-1, cv=5, scoring='roc_auc')
#
#     # Get the data
#     y_data = total_train.loc[:, y_column]
#     X_data = dict_relevant_data[y_column]
#
#     # Train the model
#     gb_gs_clf.fit(X_data, y_data)
#     dict_gridsearch_results[y_column] = pd.DataFrame.from_dict(gb_gs_clf.cv_results_)
#
# # %% Actual Prediction
#
# # Find the best parameters
# for y_column in target_columns:
#     # Get the data
#     y_data = total_train.loc[:, y_column]
#     X_data = dict_relevant_data[y_column]
#
#     df_params = dict_gridsearch_results[y_column]
#     best_params = df_params.sort_values(by='rank_test_score').loc[0, 'params']
#
#     gb_smote_pipeline = imblearn.pipeline.make_pipeline(over_sampler, gb_model)
#
#     final_pipeline = make_pipeline(
#         transforming_pipeline,
#         KNNImputer(n_neighbors=100),
#         gb_smote_pipeline
#     )
#
#     final_pipeline.fit(X_data, y_data)
#
#
#





