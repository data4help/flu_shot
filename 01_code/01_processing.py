
# %% Preliminaries

# Packages

# Basic
import pandas as pd
from tqdm import tqdm
import pickle

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Pre-processing
from category_encoders import TargetEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

# Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

# Imputation
from sklearn.impute import KNNImputer

# Clustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Forecasting
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Helper Functions
import sys
sys.path.insert(1, r'C:/Users/DEPMORA1/Documents/Projects/flu_shot/01_code')
import _functions_processing as help_proc
import _config

# Paths
MAIN_PATH = r'C:/Users/DEPMORA1/Documents/Projects/flu_shot'
RAW_PATH = rf'{MAIN_PATH}/00_raw'
DATA_PATH = rf'{MAIN_PATH}/02_data'
OUTPUT_PATH = rf'{MAIN_PATH}/03_output'

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
train_labels = train_labels.loc[:TEST_SAMPLE_LENGTH, :]

# %% Feature Engineering - Subjective Answers


class OpinionTransformer(TransformerMixin):
    """The dataset contains several variables which has subjective answering - for example how would you rate the risk
    of getting the seasonal flu. Given that the differentation between 'Very Low' and 'Somewhat Low' is not comparable
    between participants, we reduce the possible answers from five to three. This potentially reduces confusion.
    It is not possible to treat them as a numeric variable either given that the answer choice 'Don't Know' does not lay
    in the middle between low and high"""
    def __init__(self):
        pass

    def fit(self, X_data=None, y_data=None):
        return self

    def transform(self, X_data, y_data=None):
        new_opinion_cat = {1: 'Low', 2: 'Low', 3: 'No Idea', 4: 'High', 5: 'High'}
        opinion_variables = [x for x in X_data.columns if x.startswith('opinion')]
        for var in opinion_variables:
            X_data.loc[:, var] = X_data.loc[:, var].map(new_opinion_cat)
        return X_data


# %% Feature Engineering - Creation of scoring variables


class ScoringVariableCreation(TransformerMixin):
    """In order for the model to better pick up information from multiple columns, we can give the model the information
    in a better to understand format. For that we create a score for being careful, and for being in the risk group.
    This score is nothing other than a sum of binary variables which indicate carefulness/ or being part of a risk
    group."""

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


class TransformerPipeline(TransformerMixin):
    """"""
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
        column_names = help_proc.get_feature_names(self.transforming_pipeline)
        df_processed_data = pd.DataFrame(processed_data, columns=column_names)
        return df_processed_data


# %% Imputation


class KNNImputation(TransformerMixin):
    """"""

    def __init__(self, n_neighbors, classification_model):
        self.n_neighbors_list = n_neighbors
        self.classification_model = classification_model
        self.N_SPLITS = 10

    def plot_imputing_score(self, data, target_name):
        fig, axs = plt.subplots(figsize=(10, 10))
        sns.boxplot(data=data, x='neighbors', y='score', ax=axs)
        axs.set_title(target_name)
        path = rf'{OUTPUT_PATH}/00_graphs/imputation_{target_name}.png'
        fig.savefig(path, bbox_inches='tight')
        plt.close()

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


# %% Clustering


class CreateClusters(TransformerMixin):
    """"""

    def __init__(self):
        pass

    def plotting_clusters(self, pca_factors, clusters, target_name):
        fig, axs = plt.subplots(figsize=(10, 10))
        sns.scatterplot(x=pca_factors[:, 0], y=pca_factors[:, 1], hue=clusters, palette='tab10', ax=axs)
        path = rf'{OUTPUT_PATH}/00_graphs/pca_{target_name}.png'
        fig.savefig(path, bbox_inches='tight')
        plt.close()

    def fit(self, X_data, y_data=None):
        # Creating of PCA Factors
        pca_model = PCA(n_components=2)
        self.pca_fitted_model = pca_model.fit(X_data)
        pca_factors = self.pca_fitted_model.transform(X_data)

        # Creating of clusters
        n_clusters = len(set(y_data))
        target_name = y_data.name
        kmeans_model = KMeans(n_clusters=n_clusters)
        self.kmeans_model_fitted = kmeans_model.fit(pca_factors)
        clusters = self.kmeans_model_fitted.predict(pca_factors)
        self.plotting_clusters(pca_factors, clusters, target_name)

        return self

    def transform(self, X_data, y_data=None):
        test_pca_factors = self.pca_fitted_model.transform(X_data)
        clusters = self.kmeans_model_fitted.predict(test_pca_factors)
        X_data.loc[:, 'pca_clusters'] = clusters
        return X_data


# %% Pre-Processing the data

# Scoring Variables
careful_behavior_columns = [x for x in train_features.columns if x.startswith('behavioral')]
risk_group_columns = ['chronic_med_condition', 'health_worker']

# Imputation
#list_n_neighbors = [10, 25, 50, 75, 100, 150, 200]
list_n_neighbors = [2, 5]
example_clf = LogisticRegression(max_iter=1_000)

preprocessing_pipe = make_pipeline(
    OpinionTransformer(),
    ScoringVariableCreation(careful_behavior_columns, risk_group_columns),
    TransformerPipeline(5, feature_columns),
    KNNImputation(list_n_neighbors, example_clf),
    CreateClusters()
)

for target in tqdm(target_columns):
    y_series = train_labels.loc[:, target]
    preprocessing_pipe.fit(train_features, y_series)

    dict_processed_data = {
        'test': preprocessing_pipe.transform(test_features),
        'train': preprocessing_pipe.transform(train_features)
    }

    file_name = rf'{DATA_PATH}/{target}_data_dict.pickle'
    file_object = open(file_name, 'wb')
    pickle.dump(dict_processed_data, file_object)
    file_object.close()
