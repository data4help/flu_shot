
# %% Preliminaries

# Packages
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
MAIN_PATH = r'C:/Users/DEPMORA1/Documents/Projects/flu_shot'
DATA_PATH = rf'{MAIN_PATH}/data'
OUTPUT_PATH = rf'{MAIN_PATH}/output'

# %%


# %% Matplotlib Settings

SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30
plt.rc("font", size=BIGGER_SIZE)
plt.rc("axes", titlesize=SMALL_SIZE)
plt.rc("axes", labelsize=MEDIUM_SIZE)
plt.rc("xtick", labelsize=SMALL_SIZE)
plt.rc("ytick", labelsize=SMALL_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE)
plt.rc("figure", titlesize=BIGGER_SIZE)
matplotlib.use("Agg")

# %% Distributional Analysis


def categorical_plot(series):
    """This function plots pie charts out of categorical variables"""
    value_count_series = series.value_counts()
    labels = value_count_series.keys()
    fig, axs = plt.subplots(figsize=(10, 10))
    axs.pie(x=value_count_series, labels=labels, pctdistance=0.5)
    axs.set_title(series.name)
    path = rf'{OUTPUT_PATH}/distribution/{series.name}'
    fig.savefig(path, bbox_inches='tight')
    plt.close()


def numerical_plot(series):
    fig, axs = plt.subplots(figsize=(10, 10))
    sns.histplot(series, ax=axs)
    axs.set_title(series.name)
    path = rf'{OUTPUT_PATH}/distribution/{series.name}'
    fig.savefig(path, bbox_inches='tight')
    plt.close()
