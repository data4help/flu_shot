
# %% Preliminaries

# Packages

# Classical
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Forecasting
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# %% Plotting


def fewer_than_cutoff(column_name, data, n):
    """This function indicates whether a certain series within a dataframe has fewer or equal than n categories"""
    series = data.loc[:, column_name]
    num_categories = len(series.value_counts())
    return num_categories <= n


# %% Imputing Techniques


def simple_imputing(X_data, y_data, classifier, simple_methods, num_splits):
    """This function imputes the missing values by using simple methods such as mean and median.
    Then it reports the score"""
    df_simple_imputer = pd.DataFrame(index=list(range(len(simple_methods) * num_splits)),
                                     columns=['score', 'method', 'target'])

    for i, strategy in enumerate(simple_methods):
        estimator = make_pipeline(
            SimpleImputer(missing_values=np.nan, strategy=strategy),
            classifier
        )
        row_begin, row_end = (i*num_splits), ((i+1)*num_splits)-1
        df_simple_imputer.loc[row_begin:row_end, 'score'] = cross_val_score(
            estimator, X_data, y_data, scoring='roc_auc', cv=num_splits
        )
        df_simple_imputer.loc[row_begin:row_end, 'method'] = strategy
        df_simple_imputer.loc[row_begin:row_end, 'target'] = y_data.name
    return df_simple_imputer


def iterative_imputing(X_data, y_data, classifier, impute_classifiers, num_splits):
    """This function imputes the missing values with model-based methods."""
    df_iterative_imputer = pd.DataFrame(index=list(range(len(impute_classifiers) * num_splits)),
                                        columns=['score', 'method', 'target'])

    for i, impute_classifier in enumerate(impute_classifiers):
        estimator = make_pipeline(
            IterativeImputer(random_state=0, estimator=impute_classifier),
            classifier
        )
        row_begin, row_end = (i*num_splits), ((i+1)*num_splits)-1
        df_iterative_imputer.loc[row_begin:row_end, 'score'] = cross_val_score(
            estimator, X_data, y_data, scoring='roc_auc', cv=num_splits
        )
        df_iterative_imputer.loc[row_begin:row_end, 'method'] = impute_classifier.__class__.__name__
        df_iterative_imputer.loc[row_begin:row_end, 'target'] = y_data.name
    return df_iterative_imputer
