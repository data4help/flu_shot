
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
total_train = pd.concat([train_features, train_labels], axis=1)
total_train.drop(columns=["respondent_id"], inplace=True)
total_train.loc[:, "type"] = "train"

test_features.loc[:, "type"] = "test"
total_data = pd.concat([total_train, test_features], axis=0)

# %% Matplotlib Settings

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





"""
We now divide the variables into two categories, namely into 'easy' and 'difficult' to impute. That classification
decides which imputation method we can use.
"""

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



