import sys
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from relnet.evaluation.file_paths import FilePaths
sys.path.append('/relnet')

data_root = Path('/experiment_data/development/models/eval_histories')
figure_root = Path('/experiment_data/development/figures')


def plot_tests(game="bspgg", graph="ba", m="4"):

    figure_path = figure_root / f"test_{game}_{graph}_{m}.png"

    # for i, j in zip([15, 25, 50, 100], [21, 60, 123, 248]):
    for i in [15, 25, 50, 100]:
        run_name = f"{game}_{graph}_{i}_{m}"
        test_path = data_root / run_name / "test.csv"
        if test_path.exists():
            if i == 15:
                df = pd.read_csv(test_path, names=['br_sw', 'test_sw', 'random_baseline_sw'], skiprows=1)
                df["n_nodes"] = [i] * df.shape[0]
            else:
                df1 = pd.read_csv(test_path, names=['br_sw', 'test_sw', 'random_baseline_sw'], skiprows=1)
                df1["n_nodes"] = [i] * df1.shape[0]
                frames = [df, df1]
                df = pd.concat(frames)
        else:
            raise BaseException("No such file in directory")

    # Set the style
    sns.set_style("darkgrid")
    fig = sns.lineplot(x="n_nodes", y="br_sw", data=df, label='Best Response', ci=90)
    fig = sns.lineplot(x="n_nodes", y="test_sw", data=df, label='PlanNet', ci=90)
    fig = sns.lineplot(x="n_nodes", y="random_baseline_sw", data=df, label='Random Agent', ci=90)
    fig.set(xlim=(min(df['n_nodes']), max(df['n_nodes'])))
    plt.title("Test Set Performance")
    plt.xlabel('Number of nodes')
    plt.ylabel('Test Set Social Welfare')
    plt.savefig(figure_path)
    plt.close()


def save_raw(game, graph, n, m):

    run_name = f"{game}_{graph}_{n}_{m}"
    raw_path = data_root / run_name / "raw.csv"
    figure_path = figure_root / f"validation_{run_name}.png"

    if raw_path.exists():
        df = pd.read_csv(raw_path, names=['steps', 'initial', 'final'], skiprows=1)
    else:
        raise BaseException("No such file in directory")

    # Set the style
    sns.set_style("darkgrid")
    # Create a plot
    fig = sns.lineplot(x="steps", y="initial", data=df, label='Best Response', ci=90)
    fig = sns.lineplot(x="steps", y="final", data=df, label='PlanNet', ci=90)
    fig.set(ylim=(df["initial"].mean() * 0.95, min(1, df["final"].mean() * 1.05)))
    fig.set(xlim=(0, max(df['steps'])))
    # Set Titles and labels
    plt.title("Validation Set Performance")
    plt.xlabel('Step')
    plt.ylabel('Validation Performance')
    plt.savefig(figure_path)
    plt.close()


if __name__ == '__main__':

    # plot_tests(game="bspgg", graph="er", m="2")
    save_raw(game="majority", graph="er", n="25", m="39")
