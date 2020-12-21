# %% Preliminaries

# Packages ----

# Basic
import os
import pickle
import pandas as pd
from tqdm import tqdm
import sys

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

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

train_labels = pd.read_csv(rf'{RAW_PATH}/training_labels.csv')
train_labels = train_labels.loc[:1_000, :]

# %% Finding optimal Model

# Initializing models to be tried out
model_dict = {
    # 'LR': LogisticRegression(max_iter=1_000),
    'MLP': MLPClassifier(max_iter=1_000),
    # 'LDA': LinearDiscriminantAnalysis(),
    'KNN': KNeighborsClassifier(),
    # 'CART': DecisionTreeClassifier(),
    # 'NB': GaussianNB(),
    # 'SVM': SVC(),
    # 'GBM': GradientBoostingClassifier(),
    # 'RFT': RandomForestClassifier()
}

n_splits = 2
scoring = 'roc_auc'
model_names = model_dict.keys()

number_targets = len(target_columns)
fig, axs = plt.subplots(ncols=number_targets, nrows=number_targets, figsize=(10*number_targets, 10), sharey=True)
axs = axs.ravel()
for i, (target_name, data) in enumerate(data_dict.items()):

    X_data_train = data['train'].copy()
    y_data_train = train_labels.loc[:, target_name].copy()

    df_results = pd.DataFrame(index=list(range(len(model_dict) * n_splits)), columns=['Scores', 'Models'])
    for j, (name, model) in tqdm(enumerate(model_dict.items())):
        kfold = KFold(n_splits=n_splits, random_state=28, shuffle=True)
        cv_results = cross_val_score(model, X_data_train, y_data_train, cv=kfold, scoring=scoring)
        start_index, end_index = (j * n_splits), (((j+1) * n_splits) -1)
        df_results.loc[start_index:end_index, 'Scores'] = cv_results
        df_results.loc[start_index:end_index, 'Models'] = name

    df_results.loc[:, 'Scores'] = df_results.loc[:, 'Scores'].astype(float)
    mean_score = df_results.groupby('Models')['Scores'].agg(['mean']).T
    sns.boxplot(x='Models', y='Scores', data=df_results, showmeans=True, ax=axs[i])
    sns.heatmap(mean_score, annot=True, fmt='.4', ax=axs[i+number_targets])
    axs[i].set_title(f'Target Variable: {target_name}')
path = rf'{OUTPUT_PATH}/00_graphs/model_test.png'
fig.savefig(path, bbox_inches='tight')
plt.close()


print(best_model.values)





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





