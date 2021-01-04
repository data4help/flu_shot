
# %% Preliminaries

# Packages ----

# Basics
import pandas as pd
import os
import numpy as np
import sys
import copy
from tqdm import tqdm
import pickle
import math
from datetime import date

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Pipelines
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Feature Importance
from sklearn.ensemble import GradientBoostingClassifier

# Predictions
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix

# Upsampling
import imblearn

# Helper Functions
sys.path.insert(1, r'C:/Users/DEPMORA1/Documents/Projects/flu_shot/01_code')
import _config

# Paths
MAIN_PATH = r'C:/Users/DEPMORA1/Documents/Projects/flu_shot'
RAW_PATH = rf'{MAIN_PATH}/00_raw'
DATA_PATH = rf'{MAIN_PATH}/02_data'
OUTPUT_PATH = rf'{MAIN_PATH}/03_output'

# %% Data

data_dict = {}
target_columns = ['h1n1_vaccine', 'seasonal_vaccine']
saved_files = os.listdir(DATA_PATH)
for target, file in zip(saved_files, target_columns):
    infile = open(f'{DATA_PATH}/{target}', 'rb')
    data_dict[file] = pickle.load(infile)
    infile.close()

test_features = pd.read_csv(rf'{RAW_PATH}/test_features.csv')
train_labels = pd.read_csv(rf'{RAW_PATH}/training_labels.csv')
train_labels = train_labels.loc[:1_000, :]

# %% Feature Selection


class FeatureSelection(TransformerMixin):

    def __init__(self, cutoff_threshold):
        self.cutoff_threshold = cutoff_threshold

    def plotting_feature_importance(self, df_sorted_features, target_name):
        """This method plots the importance of all features"""

        fig, axs = plt.subplots(figsize=(10, 10))
        df_sorted_features.plot.bar(ax=axs)
        path = rf'{OUTPUT_PATH}/00_graphs/feature_importance_{target_name}.png'
        fig.savefig(path, bbox_inches='tight')

    def fit(self, X_data, y_data=None):
        """We fit a basic Gradient Boosting and extract the feature importance"""

        # Feature Importance
        target_name = y_data.name
        gbm_clf = GradientBoostingClassifier()

        clf = gbm_clf.fit(X_data, y_data)
        feature_importance = clf.feature_importances_

        reshaped_feature_importance = np.reshape(feature_importance, (-1, 1))
        data_columns = X_data.columns
        df_feature_importance = pd.DataFrame(reshaped_feature_importance, index=data_columns, columns=['columns'])
        df_above_one_percent = df_feature_importance.query('columns >= @self.cutoff_threshold')
        sorted_features = df_above_one_percent.sort_values(by='columns', ascending=False)
        self.relevant_columns = list(sorted_features.index)
        self.plotting_feature_importance(sorted_features, target_name)

        return self

    def transform(self, X_data, y_data=None):
        """Here we simply use the columns which are important"""

        df_relevant_columns = X_data.loc[:, self.relevant_columns]
        return df_relevant_columns


# %% Hyper-Parameter Tuning


class FinalPrediction(BaseEstimator, TransformerMixin):

    def __init__(self, model_parameters):
        self.model_parameters = model_parameters
        self.BEST_PARAMETERS_FOR_SHOW = 100

    def plot_grid_params_importance(self, df_grid_results, target_name):
        """This method plots out of the top 100 best models how often did which hyper-parameter the best."""

        df_sorted_grid_results = df_grid_results.sort_values(by='rank_test_score')
        df_top_parameters = df_sorted_grid_results.query('rank_test_score <= @self.BEST_PARAMETERS_FOR_SHOW')
        list_of_best_params = list(df_top_parameters.loc[:, 'params'])
        series_top_parameters = pd.DataFrame.from_dict(list_of_best_params)

        ncols = 2
        nrows = math.ceil(series_top_parameters.shape[1] / ncols)
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(nrows*10, ncols*10))
        axs = axs.ravel()
        for i, col_name in enumerate(series_top_parameters):
            series = series_top_parameters.loc[:, col_name]
            sns.countplot(x=series, ax=axs[i])
        path = rf'{OUTPUT_PATH}/00_graphs/best_hyper_{target_name}.png'
        fig.savefig(path, bbox_inches='tight')

    def obtain_best_model(self, X_data, y_data, df_grid_results, gb_pipeline):
        """This method extracts the best model for all hyper-parameters"""

        best_pipe = copy.deepcopy(gb_pipeline)

        # Finding best params
        df_best_params = df_grid_results.query('rank_test_score==1')
        if len(df_best_params) > 1:
            df_best_params = df_best_params.sort_values(by='mean_fit_time').iloc[[0], :]
        dict_best_params = df_best_params.loc[:, 'params'].values[0]
        best_pipe.set_params(**dict_best_params)
        best_pipe.fit(X_data, y_data)
        return best_pipe

    def best_model_performance(self, best_model, X_data, y_data, target_name):
        """This method plots the confusion matrix of the best classifier"""

        # TODO: evaluation of best classifier, confusion matrix and roc curve would be appropriate here
        fig, axs = plt.subplots(figsize=(10, 10))
        plot_confusion_matrix(best_model, X_data, y_data, ax=axs)
        path = rf'{OUTPUT_PATH}/00_graphs/model_performance_{target_name}.png'
        fig.savefig(path, bbox_inches='tight')

    def fit(self, X_data, y_data=None):
        """Here we find the best model by implementing a pipeline and gridsearching it. It is important to note that
        we use imblearn pipelines given that we also specify a SMOTE parameter"""

        # Adding SMOTE Parameters
        num_target_values = y_data.value_counts()
        current_balance = math.ceil(min(num_target_values) / max(num_target_values) * 100) / 100
        smote_grid_params = {'smote__sampling_strategy': np.linspace(current_balance, 1, num=3)}
        grid_params = {**self.model_parameters, **smote_grid_params}

        # Creating the final pipeline
        over_sampler = imblearn.over_sampling.SMOTE(random_state=28)
        gb_model = GradientBoostingClassifier(random_state=28)
        gb_pipeline = imblearn.pipeline.make_pipeline(over_sampler, gb_model)
        gb_gs_clf = GridSearchCV(gb_pipeline, grid_params, n_jobs=-1, cv=5, scoring='roc_auc')

        # Plot the results of the tuning
        gb_gs_clf.fit(X_data, y_data)
        df_grid_results = pd.DataFrame.from_dict(gb_gs_clf.cv_results_)
        target_name = y_data.name
        self.plot_grid_params_importance(df_grid_results, target_name)

        # Obtain the best pipeline
        self.best_model = self.obtain_best_model(X_data, y_data, df_grid_results, gb_pipeline)
        self.best_model_performance(self.best_model, X_data, y_data, target_name)
        return self

    def predict_proba(self, X_data, y_data=None):
        """This last method is for predicting the probabilities that an observation is taking the vaccine"""

        predictions = self.best_model.predict_proba(X_data)
        prediction_to_get_vaccine = predictions[:, 1]
        return prediction_to_get_vaccine

# %% Predicting

"""After specifying all classes above, we now create the final pipeline and set all parameters needed for the
classes"""

CUTOFF_THRESHOLD = 1 / 100

gb_grid_params = {
    'gradientboostingclassifier__learning_rate': [1 / pow(10, x) for x in range(1, 3)],
    'gradientboostingclassifier__max_depth': np.linspace(3, 30, num=3, dtype=int),
    'gradientboostingclassifier__min_samples_leaf': np.linspace(1, 30, num=3, dtype=int),
    'gradientboostingclassifier__n_estimators': [5],
}

df_results = pd.DataFrame(columns=target_columns, index=test_features.index)
predicting_pipe = make_pipeline(
    FeatureSelection(CUTOFF_THRESHOLD),
    FinalPrediction(gb_grid_params)
)

for target in tqdm(target_columns):
    y_series = train_labels.loc[:, target]
    train_features_copy = data_dict[target]['train'].copy()
    test_features_copy = data_dict[target]['test'].copy()

    predicting_pipe.fit(train_features_copy, y_series)
    df_results.loc[:, target] = predicting_pipe.predict_proba(test_features_copy)

date_string = date.today().strftime("%m_%d_%y")
file_name = f'test_{date_string}'
df_results.to_csv(rf'{OUTPUT_PATH}/01_predictions/{file_name}.csv')
