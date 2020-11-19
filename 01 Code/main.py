
# %% Preliminaries

# Packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
MAIN_PATH = r"/Users/paulmora/Documents/projects/flu_shot"
RAW_PATH = rf"{MAIN_PATH}/00 Raw"
CODE_PATH = rf"{MAIN_PATH}/01 Code"
DATA_PATH = rf"{MAIN_PATH}/02 Data"
OUTPUT_PATH = rf"{MAIN_PATH}/03 Output"

# Data
train_features = pd.read_csv(rf"{RAW_PATH}/training_features.csv")
train_labels = pd.read_csv(rf"{RAW_PATH}/training_labels.csv")
test_features = pd.read_csv(rf"{RAW_PATH}/test_features.csv")

# Putting data together
total_train = pd.merge(train_features, train_labels, on="respondent_id")
total_train.loc[:, "type"] = "train"
test_features.loc[:, "type"] = "test"
total_data = pd.concat([total_train, test_features], axis=0)

# %% Settings

# Quick Access
target_columns = ["h1n1_vaccine", "seasonal_vaccine"]

# Matplotlib sizes
SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# %% Missing Observations

"""
First we check how many observations are missing for train and test. We are only interested in using a model-based
approach for the imputation if the amount of missing observation is seriously high (e.g. 5%)
"""

missing_pct = total_data.groupby(by="type").agg(lambda x: x.isnull().sum() / len(x) * 100)
missing_pct_index = missing_pct.reset_index()
long_missing_data = pd.melt(missing_pct_index, id_vars=["type"], value_vars=train_features.columns)
sorted_long_missing_data = long_missing_data.sort_values(by="value", ascending=False)
fig, axs = plt.subplots(figsize=(20, 10))
plot = sns.barplot(data=sorted_long_missing_data, x="variable", y="value", hue="type", ax=axs)
axs.set_ylabel("Percentage Missing")
plt.xticks(rotation=90)
path = rf"{OUTPUT_PATH}/missing_observation_threshold.png"
fig.savefig(path, bbox_inches='tight')

"""
We see that we have some variables which have around 50% missing data. For that reason we already could think
of dropping those columns. Before doing that we will try to gauge their importance by looking whether we find
any correlation between these columns and any of the target.
"""

CRITICAL_THRESHOLD = 40
bool_above_threshold = missing_pct.loc["train", :] > CRITICAL_THRESHOLD
critical_columns = list(bool_above_threshold.index[bool_above_threshold])


def correlation_check(column, target, data):

    joined_df = data.loc[:, [column, target]]
    non_nan_joined_df = joined_df.dropna()
    dummy_data = pd.get_dummies(non_nan_joined_df)
    corr_df = dummy_data.corr()
    return corr_df.loc[:, target]


fig, axs = plt.subplots(ncols=len(critical_columns),
                        figsize=(5*10, 10), sharey=True)
axs = axs.ravel()
i = 0
for column in critical_columns:
    column_df = pd.DataFrame()
    for target in target_columns:
        column_df.loc[:, target] = correlation_check(column, target, total_data)
    sns.heatmap(column_df.transpose())
    plt.yticks(rotation=0)
    i += 1

# %% Exploration of Dependent Variable

"""
At first it could be interesting to see whether the two dependent variables are balanced. For that we create a
count-plot.
"""

dependent_columns = ["h1n1_vaccine", "seasonal_vaccine"]
fig, axs = plt.subplots(ncols=2, figsize=(len(dependent_columns) * 10, 10))
axs = axs.ravel()
for i, column in enumerate(dependent_columns):
    sns.countplot(total_train.loc[:, column], ax=axs[i])
path = rf"{OUTPUT_PATH}/dependent_count_plot.png"
fig.savefig(path, bbox_inches='tight')

"""
The h1n1 vaccine is relatively imbalanced <- Potential up-sampling. The seasonal one is nicely balanced
"""

# %% Exploration of Independent Variables



