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
p_er_bspgg = [0.3, 0.2, 0.1, 0.05]
# p_er_maj = [0.1, 0.05, 0.02, 0.005]
p_er_maj = [19, 39, 62, 149]


def plot_tests():

    figure_path = figure_root / f"test_graphs.png"
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(12, 8))
    # fig.suptitle('Test set performance', fontsize=16)


    for i, game in enumerate(['bspgg', 'majority']):
        for j, graph in enumerate(['ba', 'ws', 'er']):
            node_vals = []
            for k, nodes in enumerate([15, 25, 50, 100]):
                if game == 'bspgg':
                    if (graph == 'ba') or (graph == 'ws'):
                        m = 4
                    else:
                        m = p_er_bspgg[k]
                else:
                    if graph == 'ba':
                        m = 1
                    elif graph == 'ws':
                        m = 2
                    else:
                        m = p_er_maj[k]
                run_name = f"{game}_{graph}_{nodes}_{m}"
                test_path = data_root / run_name / "test.csv"
                if test_path.exists():
                    df = pd.read_csv(test_path,
                                     names=['br_sw', 'test_sw', 'random_baseline_sw', 'baseline'],
                                     skiprows=1)
                else:
                    raise BaseException("No such file in directory")
                df["n_nodes"] = [nodes] * df.shape[0]
                node_vals.append(df)

            df = pd.concat(node_vals, axis=0, ignore_index=True)

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

            p = sns.lineplot(ax=ax[i, j], data=df, x="n_nodes", y="Objective", hue='test_type',
                             ci=95, legend=False, palette=sns.color_palette('deep', n_colors=df.test_type.nunique()))
            p.set_xlim(left=15, right=100)
            p.set_xlabel('Number of nodes', fontsize=14)
            if game == 'bspgg':
                p.set_ylim(bottom=0.68, top=0.825)
                if j == 0:
                    p.legend(title='Test Type', loc='lower left', labels=["Best Response", "PlanNet", "Random Agent"],
                             fontsize=14)
            else:
                p.set_ylim(bottom=0.93, top=1.001)
                if j == 0:
                    p.legend(title='Test Type', loc='lower left',
                             labels=["Best Response", "PlanNet", "Random Agent", "Policy Baseline"], fontsize=14)
            if i == 0:
                p.set_ylabel(r'Best-shot Public Goods Game'
                             '\n'
                             r'$G^{test}$ social welfare',
                             fontsize=14)
            else:
                p.set_ylabel(r'Majority game'
                             '\n'
                             r'$G^{test}$ social welfare',
                             fontsize=14)
            if (j == 0) and (i == 0):
                p.set_title("Barabási–Albert", fontsize=14)
            elif (j == 1) and (i == 0):
                p.set_title("Watts–Strogatz", fontsize=14)
            elif (j == 2) and (i == 0):
                p.set_title("Erdős–Rényi", fontsize=14)

    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()


def save_raw():
    figure_path = figure_root / f"validation_graphs.png"
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(12, 8))

    for i, game in enumerate(['bspgg', 'majority']):
        for j, graph in enumerate(['ba', 'ws', 'er']):
            node_vals = []
            for k, nodes in enumerate([15, 25, 50, 100]):
                if game == 'bspgg':
                    if (graph == 'ba') or (graph == 'ws'):
                        m = 4
                    else:
                        m = p_er_bspgg[k]
                else:
                    if graph == 'ba':
                        m = 1
                    elif graph == 'ws':
                        m = 2
                    else:
                        m = p_er_maj[k]
                run_name = f"{game}_{graph}_{nodes}_{m}"
                raw_path = data_root / run_name / "raw.csv"
                if raw_path.exists():
                    df = pd.read_csv(raw_path, names=['steps', 'initial', 'final'], skiprows=1)
                else:
                    raise BaseException("No such file in directory")
                df['change'] = df['final'] - df['initial']
                del df['final']
                del df['initial']
                df["n_nodes"] = [nodes] * df.shape[0]
                node_vals.append(df)

            df = pd.concat(node_vals, axis=0, ignore_index=True)
            p = sns.lineplot(ax=ax[i, j], data=df, x="steps", y="change", hue='n_nodes',
                             ci=95, legend=False, palette=sns.color_palette('deep', n_colors=df.n_nodes.nunique()))
            p.set_xlim(left=0, right=1000)
            p.set_ylim(bottom=-0.02, top=0.08)
            p.set_xlabel('Training Steps', fontsize=14)
            if i == 0:
                p.set_ylabel(r'Best-shot Public Goods Game'
                             '\n'
                             r'$G^{valid}$ improvement',
                             fontsize=14)
            else:
                p.set_ylabel(r'Majority game'
                             '\n'
                             r'$G^{valid}$ improvement',
                             fontsize=14)
            if (j == 0) and (i == 0):
                p.set_title("Barabási–Albert", fontsize=14)
            elif (j == 1) and (i == 0):
                p.set_title("Watts–Strogatz", fontsize=14)
            elif (j == 2) and (i == 0):
                p.set_title("Erdős–Rényi", fontsize=14)
            if (i == 0) and (j == 0):
                p.legend(title='n nodes', loc='upper left', labels=["15", "25", "50", "100"])
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()


def plot_oos(game='bspgg'):
    figure_path = figure_root / f"oos_{game}_graphs.png"
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(12, 8))
    p_er_bspgg = [0.3, 0.1, 0.05]
    # p_er_maj = [0.1, 0.05, 0.02, 0.005]
    p_er_maj = [19, 62, 149]

    for j, graph in enumerate(['ba', 'ws', 'er']):
        for k, nodes in enumerate([15, 50, 100]):
            if game == 'bspgg':
                if (graph == 'ba') or (graph == 'ws'):
                    m = 4
                else:
                    m = p_er_bspgg[k]
            else:
                if graph == 'ba':
                    m = 1
                elif graph == 'ws':
                    m = 2
                else:
                    m = p_er_maj[k]

            run_name = f"{game}_{graph}_{nodes}_{m}"
            raw_path = data_root / run_name / "out_of_sample.csv"
            if raw_path.exists():
                df = pd.read_csv(raw_path, names=['br', 'PlanNet', 'nodes', 'g_type'], skiprows=1)
            else:
                raise BaseException("No such file in directory")
            df['improvement'] = df['PlanNet'] - df['br']

            p = sns.lineplot(ax=ax[j, k], data=df, x="nodes", y="improvement", hue='g_type',
                             ci=95, legend=False, palette=sns.color_palette('deep', n_colors=df.g_type.nunique()))

            p.set_xlim(left=15, right=100)
            p.set_ylim(bottom=-0.02, top=0.065)
            p.set_xlabel('Test graph number of nodes', fontsize=14)
            if j == 0:
                p.set_ylabel(r'Barabási–Albert'
                             '\n'
                             r'$G^{test}$ change')
            elif j == 1:
                p.set_ylabel(r'Watts–Strogatz'
                             '\n'
                             r'$G^{test}$ change', fontsize=14)
            else:
                p.set_ylabel(r'Erdős–Rényi'
                             '\n'
                             r'$G^{test}$ change', fontsize=14)
            if (j == 0) and (k == 0):
                p.set_title("Trained on n=15", fontsize=14)
            elif (j == 0) and (k == 1):
                p.set_title("Trained on n=50", fontsize=14)
            elif (j == 0) and (k == 2):
                p.set_title("Trained on n=100", fontsize=14)
            # if (j == 1) and (k == 2):
            #     p.legend(title='Test graph type', loc='center right', bbox_to_anchor=(1.2, 1.0),
            #              labels=["Barabási–Albert", "Watts–Strogatz", "Erdős–Rényi"])

    labels = ["Barabási–Albert", "Watts–Strogatz", "Erdős–Rényi"]
    fig.legend(labels=labels, loc='upper center', bbox_to_anchor=(0.5, 1), borderaxespad=0, fontsize=11, ncol=3)
    plt.tight_layout(pad=3.2)
    plt.savefig(figure_path)
    plt.close()


def plot_oos_curriculum():
    figure_path = figure_root / f"curriculum_oos_graphs.png"
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(12, 8))

    for i, game in enumerate(['bspgg', 'majority']):
        for j, graph in enumerate(['ba', 'ws']):
            run_name = f"curriculum_{game}_{graph}"
            raw_path = data_root / run_name / "out_of_sample.csv"
            if raw_path.exists():
                df = pd.read_csv(raw_path, names=['br', 'PlanNet', 'nodes', 'g_type'], skiprows=1)
            else:
                raise BaseException("No such file in directory")
            df['improvement'] = df['PlanNet'] - df['br']

            p = sns.lineplot(ax=ax[i, j], data=df, x="nodes", y="improvement", hue='g_type',
                             ci=95, legend=False, palette=sns.color_palette('deep', n_colors=df.g_type.nunique()))
            p.set_xlim(left=15, right=100)
            p.set_ylim(bottom=-0.02, top=0.065)
            p.set_xlabel('Test graph number of nodes', fontsize=14)
            if i == 0:
                p.set_ylabel(r'Best-shot public goods game'
                             '\n'
                             r'$G^{test}$ change', fontsize=14)
            elif i == 1:
                p.set_ylabel(r'Majority game'
                             '\n'
                             r'$G^{test}$ change', fontsize=14)
            if (i == 0) and (j == 0):
                p.set_title("Trained on Barabási–Albert graphs", fontsize=14)
            elif (i == 0) and (j == 1):
                p.set_title("Trained on Watts–Strogatz graphs", fontsize=14)
            # if (i == 0) and (j == 0):
            #     p.legend(title='Test graph type', loc='upper left',
            #              labels=["Barabási–Albert", "Watts–Strogatz", "Erdős–Rényi"])

    labels = ["Barabási–Albert", "Watts–Strogatz", "Erdős–Rényi"]
    fig.legend(labels=labels, loc='upper center', bbox_to_anchor=(0.5, 1), borderaxespad=0, fontsize=11, ncol=3)
    plt.tight_layout(pad=3.2)
    plt.savefig(figure_path)
    plt.close()


def institution_plot():
    sns.set_style("darkgrid")
    figure_path = figure_root / f"Institution_graphs.png"
    fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(12, 8))
    headers = ['Only Tax', 'PlanNet + Tax', 'rand_end', 'policy_end']
    for i, m in enumerate([1, 3]):
        for j, graph in enumerate(['ba', 'ws']):
            tax_vals = []
            if graph == 'ws':
                m += 1
            for t in ['0.0', '0.001', '0.1']:
                node_vals = []
                for n in [15, 25, 50]:
                    run_name_inst = f"MaxContribution_institution_{graph}_{n}_{m}_{t}"
                    inst_path = data_root / run_name_inst / "test.csv"
                    if inst_path.exists():
                        df = pd.read_csv(inst_path, index_col=None, names=headers, skiprows=1)
                        df['n_nodes'] = [n] * df.shape[0]
                        node_vals.append(df)
                    else:
                        raise BaseException("No such file in directory")
                df = pd.concat(node_vals, axis=0, ignore_index=True)
                df['Tax'] = [t] * df.shape[0]
                tax_vals.append(df)

            df = pd.concat(tax_vals, axis=0, ignore_index=True)
            del df['policy_end']
            del df['rand_end']
            df = pd.melt(df,
                         id_vars=['n_nodes', 'Tax'],
                         value_vars=['Only Tax', 'PlanNet + Tax'],
                         var_name='PlanNet / Tax',
                         value_name='Objective')

            p = sns.lineplot(ax=ax[j, i],
                             data=df,
                             x="n_nodes",
                             y="Objective",
                             hue='Tax',
                             style='PlanNet / Tax',
                             style_order=['PlanNet + Tax', 'Only Tax'],
                             ci=95,
                             legend=None,
                             palette=sns.color_palette(palette='deep', n_colors=df.Tax.nunique()))

            p.set_xlim(left=15, right=50)
            p.set_ylim(bottom=0.2, top=1.001)
            p.set_xlabel('Number of nodes', fontsize=14)
            if j == 0:
                p.set_ylabel(r'Barabási–Albert'
                             '\n'
                             r'$\mathcal{F}_{MC}$ on $G^{test}$',
                             fontsize=14)
            elif j == 1:
                p.set_ylabel(r'Watts–Strogatz'
                             '\n'
                             r'$\mathcal{F}_{MC}$ on $G^{test}$',
                             fontsize=14)
            if (i == 0) and (j == 0):
                p.title.set_text("m = 1")
            elif (i == 1) and (j == 0):
                p.title.set_text("m = 3")
            elif (i == 0) and (j == 1):
                p.title.set_text("k = 2")
            if (i == 1) and (j == 1):
                p.title.set_text("k = 4")
    labels = ['No Tax + PlanNet', 'No Tax + No PlanNet',
              '0.001 Tax + PlanNet', '0.001 Tax',
              '0.1 Tax + PlanNet', '0.1 Tax']
    fig.legend(labels=labels, loc='upper center', bbox_to_anchor=(0.5, 1), borderaxespad=0, fontsize=11, ncol=6)
    plt.tight_layout(pad=3.2)
    plt.savefig(figure_path)
    plt.close()


def real_world_plot():
    sns.set_style("darkgrid")
    figure_path = figure_root / f"Institution_real_world.png"
    fig, ax = plt.subplots(2, 3, sharey='row', figsize=(12, 8))
    headers = ['Only Tax', 'PlanNet + Tax', 'rand_end', 'policy_end']

    for i, graph in enumerate(['karate', 'saw_mill']):
        for j, type in enumerate(['ba', 'ws', 'real']):
            tax_vals = []
            for k, t in enumerate(['0.0', '0.001', '0.1']):
                if j == 0:
                    if i == 0:
                        run_name_inst = f"karate_15_1_ba_{t}"
                    else:
                        run_name_inst = f"saw_mill_15_1_ba_{t}"
                elif j == 1:
                    if i == 0:
                        run_name_inst = f"karate_15_2_ws_{t}"
                    else:
                        run_name_inst = f"saw_mill_15_2_ws_{t}"
                else:
                    if i == 0:
                        run_name_inst = f"MaxContribution_institution_karate_{t}"
                    else:
                        run_name_inst = f"MaxContribution_institution_saw_mill_{t}"
                inst_path = data_root / run_name_inst / "test.csv"
                if inst_path.exists():
                    df = pd.read_csv(inst_path, index_col=None, names=headers, skiprows=1)
                else:
                    raise BaseException("No such file in directory")

                df['Tax'] = [t] * df.shape[0]
                tax_vals.append(df)
            df = pd.concat(tax_vals, axis=0, ignore_index=True)
            del df['policy_end']
            del df['rand_end']
            df = pd.melt(df,
                         id_vars=['Tax'],
                         value_vars=['Only Tax', 'PlanNet + Tax'],
                         var_name='PlanNet_Tax',
                         value_name='Objective')

            g = sns.barplot(ax=ax[i, j],
                            data=df,
                            x="Tax",
                            y="Objective",
                            hue='PlanNet_Tax',
                            ci=95,
                            palette=sns.color_palette(palette='deep', n_colors=df.PlanNet_Tax.nunique()),
                            capsize=0.05)

            g.set_ylim(bottom=0.0, top=1.001)
            g.set_xlabel('Tax', fontsize=14)
            if (i == 0) and (j == 0):
                g.set_ylabel(r'Zachary karate club'
                             '\n'
                             r'$\mathcal{F}_{MC}$ on $G^{test}$',
                             fontsize=14)
            elif (i == 1) and (j == 0):
                g.set_ylabel(r'Saw mill'
                             '\n'
                             r'$\mathcal{F}_{MC}$ on $G^{test}$',
                             fontsize=14)
            if (i == 0) and (j == 0):
                g.title.set_text("Trained on Barabási–Albert graphs")
            elif (i == 0) and (j == 1):
                g.title.set_text("Trained on Watts–Strogatz graphs")
            elif (i == 0) and (j == 2):
                g.title.set_text("Trained on real world graphs")
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()


if __name__ == '__main__':

    # plot_tests()
    # save_raw()
    # plot_oos(game='bspgg')
    # plot_oos_curriculum()
    # institution_plot()
    real_world_plot()
