
# %% Preliminaries

# Packages
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
    """This function plots histogram out of all continuous variable"""
    fig, axs = plt.subplots(figsize=(10, 10))
    sns.histplot(series, ax=axs)
    axs.set_title(series.name)
    path = rf'{OUTPUT_PATH}/distribution/{series.name}'
    fig.savefig(path, bbox_inches='tight')
    plt.close()

# %% County Plots


def grouping_insights(grouping_variable, secondary_variable, data):
    """This function plots the relative distribution of the secondary variable within each category of the
    grouping variable"""

    # Getting data in shape
    columns = [grouping_variable, secondary_variable]
    df_count_by_category = data.loc[:, columns].groupby(columns).size()
    counts = df_count_by_category.groupby(grouping_variable).sum()
    df_relative_count = df_count_by_category.div(counts, level=0).reset_index()
    df_relative_count_reshaped = pd.pivot_table(df_relative_count, values=0,
                                                columns=secondary_variable, index=grouping_variable)
    # Plotting and saving figure
    fig, axs = plt.subplots(figsize=([len(columns) * 10, 10]))
    df_relative_count_reshaped.plot.barh(stacked=True, ax=axs)
    path = rf'{OUTPUT_PATH}\grouping\{grouping_variable}_{secondary_variable}.png'
    fig.savefig(path, bbox_inches='tight')

