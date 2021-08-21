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

    for i, j in zip([15, 25, 50, 100], [21, 60, 123, 248]):
    # for i in [15, 25, 50, 100]:
        run_name = f"{game}_{graph}_{i}_{j}"
        test_path = data_root / run_name / "test.csv"
        if test_path.exists():
            if i == 15:
                df = pd.read_csv(test_path, names=['br_sw', 'test_sw', 'random_baseline_sw', 'baseline'], skiprows=1)
                df["n_nodes"] = [i] * df.shape[0]
            else:
                df1 = pd.read_csv(test_path, names=['br_sw', 'test_sw', 'random_baseline_sw', 'baseline'], skiprows=1)
                df1["n_nodes"] = [i] * df1.shape[0]
                frames = [df, df1]
                df = pd.concat(frames)
        else:
            raise BaseException("No such file in directory")

    if game == 'majority':
        df = pd.melt(df,
                     id_vars=['n_nodes'],
                     value_vars=['br_sw', 'test_sw', 'random_baseline_sw', 'baseline'],
                     var_name='test_type',
                     value_name='Objective')
    else:
        del df['baseline']
        df = pd.melt(df,
                     id_vars=['n_nodes'],
                     value_vars=['br_sw', 'test_sw', 'random_baseline_sw'],
                     var_name='test_type',
                     value_name='Objective')


    # Set the style
    sns.set_style("darkgrid")
    g = sns.lineplot(data=df, x="n_nodes", y="Objective", hue='test_type', ci=90, legend=False)
    g.set(xlim=(min(df['n_nodes']), max(df['n_nodes'])))
    plt.title("Test Set Performance")
    plt.xlabel('Number of nodes')
    plt.ylabel('Test Set Social Welfare')
    if game == 'bspgg':
        g.set(ylim=(0.68, 0.825))
        plt.legend(title='Test Type', loc='lower right', labels=["Best Response", "PlanNet", "Random Agent"])
    else:
        g.set(ylim=(0.93, 1.005))
        plt.legend(title='Test Type', loc='lower right', labels=["Best Response", "PlanNet", "Random Agent",
                                                                 "Policy Baseline"])
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

    sns.set_style("darkgrid")
    fig = sns.lineplot(x="steps", y="initial", data=df, label='Best Response', ci=90)
    fig = sns.lineplot(x="steps", y="final", data=df, label='PlanNet', ci=90)
    fig.set(ylim=(df["initial"].mean() * 0.95, min(1, df["final"].mean() * 1.05)))
    fig.set(xlim=(0, max(df['steps'])))
    plt.title("Validation Set Performance")
    plt.xlabel('Step')
    plt.ylabel('Validation Performance')
    plt.savefig(figure_path)
    plt.close()


def plot_oos(game, graph, n, m, curriculum=False):
    sns.set_style("darkgrid")
    if curriculum:
        run_name = f"curriculum_{game}_{graph}"
    else:
        run_name = f"{game}_{graph}_{n}_{m}"
    oos_path = data_root / run_name / "out_of_sample.csv"
    figure_path = figure_root / f"oos_{run_name}.png"

    if oos_path.exists():
        df = pd.read_csv(oos_path, names=['Best Response', 'PlanNet', 'nodes', 'Graph type'], skiprows=1)
    else:
        raise BaseException("No such file in directory")

    # Approach plots initial and final values, cluttered graph
    # df = pd.melt(df, id_vars=['nodes', 'graph_type'], value_vars=['Best Response', 'PlanNet'],
    #              var_name='Baseline', value_name='Objective')
    # g = sns.lineplot(data=df, x="nodes", y="Objective",
    #                    hue='graph_type', style='Baseline', style_order=['PlanNet', 'Best Response'],
    #                    ci=90, legend="brief")

    df['improvement'] = df['PlanNet'] - df['Best Response']
    g = sns.lineplot(data=df, x="nodes", y="improvement", hue='Graph type', ci=90,
                     legend=False)

    g.set(xlim=(min(df['nodes']), max(df['nodes'])))
    if game == "bspgg":
        g.set(ylim=(-0.02, 0.08))
    else:
        g.set(ylim=(-0.02, 0.05))
    plt.title(f"Test set performance for out of sample graphs")
    plt.xlabel('Number of nodes')
    plt.ylabel('Change in Social Welfare')
    plt.legend(title='Graph type', loc='lower right', labels=["Barabási–Albert", "Watts–Strogatz", "Erdős–Rényi"])
    plt.savefig(figure_path)
    plt.close()


if __name__ == '__main__':

    # plot_tests(game="bspgg", graph="er", m="2")
    # save_raw(game="bspgg", graph="ba", n="100", m="4")
    # for n in [15, 25, 50, 100]:
    plot_oos(game="bspgg", graph="ws", n=25, m=2, curriculum=True)