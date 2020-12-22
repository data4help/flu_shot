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
fig_boxplot, axs_boxplot = plt.subplots(ncols=number_targets, figsize=(10*number_targets, 10), sharey=True)
axs_boxplot = axs_boxplot.ravel()

fig_heatmap, axs_heatmap = plt.subplots(ncols=number_targets, figsize=(10*number_targets, 10), sharey=True)
axs_heatmap = axs_heatmap.ravel()

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
    mean_score = df_results.groupby('Models')['Scores'].agg(['mean'])
    sns.boxplot(x='Models', y='Scores', data=df_results, showmeans=True, ax=axs_boxplot[i])
    sns.heatmap(mean_score, annot=True, fmt='.4', ax=axs_heatmap[i])
    axs_boxplot[i].set_title(f'Target Variable: {target_name}')
    axs_heatmap[i].set_title(f'Target Variable: {target_name}')

fig_boxplot.savefig(rf'{OUTPUT_PATH}/00_graphs/model_test_boxplot.png', bbox_inches='tight')
fig_heatmap.savefig(rf'{OUTPUT_PATH}/00_graphs/model_test_heatmap.png', bbox_inches='tight')
plt.close()
