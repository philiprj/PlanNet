import sys
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from relnet.evaluation.file_paths import FilePaths
sys.path.append('/relnet')


def get_file_paths():
    # Gets the file saves paths
    parent_dir = '/experiment_data'
    experiment_id = 'development'
    file_paths = FilePaths(parent_dir, experiment_id)
    return file_paths


def save_figure(data, filename):

    if (data.iloc[-1, 0] == "Test set performance"):
        df = data.iloc[0:-1, :]
    elif (data.iloc[-1, 0] == "Baseline test set performance"):
        df = data.iloc[0:-2, :]
    else:
        df = data

    # Set the style
    sns.set_style("darkgrid")
    # Create a plot
    fig = sns.lineplot(x="steps", y="performance", data=df)
    # Set Titles and labels
    plt.title("Validation Set Performance")
    plt.xlabel('Step')
    plt.ylabel('Validation Performance')
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':

    run_name = 'bspgg-ws-30-2-0.9'

    # file_paths = get_file_paths()
    figure_root = Path('/experiment_data/development/figures')
    data_root = Path('/experiment_data/development/models/eval_histories')
    data_path = data_root / f"{run_name}_history.csv"
    figure_path = figure_root / f"{run_name}.png"

    if data_path.exists():
        df = pd.read_csv(data_path, names=['steps', 'performance'])
        save_figure(df, figure_path)
    else:
        raise BaseException("No such file in directory")