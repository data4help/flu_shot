
# %% Preliminaries

# Packages

# Classical
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Forecasting
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline

# %% Column Divider


def fewer_than_cutoff(column_name, data, n):
    """This function indicates whether a certain series within a dataframe has fewer or equal than n categories"""
    series = data.loc[:, column_name]
    num_categories = len(series.value_counts())
    return num_categories <= n


# %% Ranking variables


def create_ranking(grouping_variable, secondary_variable, data):
    """This function creates a ranking for the grouping variable according the secondary variable"""
    method = 'count'
    ranking_map = data.groupby(grouping_variable)[secondary_variable].agg([method]).rank()
    ranked_grouping_variable = data.loc[:, grouping_variable].map(ranking_map.to_dict()[method])
    return ranked_grouping_variable


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
            IterativeImputer(max_iter=100, random_state=0, estimator=impute_classifier, imputation_order='ascending'),
            classifier
        )
        row_begin, row_end = (i*num_splits), ((i+1)*num_splits)-1
        df_iterative_imputer.loc[row_begin:row_end, 'score'] = cross_val_score(
            estimator, X_data, y_data, scoring='roc_auc', cv=num_splits
        )
        df_iterative_imputer.loc[row_begin:row_end, 'method'] = impute_classifier.__class__.__name__
        df_iterative_imputer.loc[row_begin:row_end, 'target'] = y_data.name
    return df_iterative_imputer


# %% Extracting columns names from imputation method


def onehotencoder_namechange(original_list, list_binary_variables):
    """
    This function is a add-on to the feature name extraction below. It helps to also transform the
    binary variables
    """
    for i, bin_var in enumerate(list_binary_variables):
        original_list = [x.replace(f"__x{i}_", f"__{bin_var}_") for x in original_list]
    return original_list


def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """

    # Remove the internal helper function
    # check_is_fitted(column_transformer)

    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
            # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                          "provide get_feature_names. "
                          "Will return input column names if available"
                          % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]

    ### Start of processing
    feature_names = []

    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))

    for name, trans, column, _ in l_transformers:
        if type(trans) == Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names) == 0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))

    return feature_names
