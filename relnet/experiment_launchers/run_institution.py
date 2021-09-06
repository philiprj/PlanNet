import sys
from pathlib import Path

sys.path.append('/relnet')
from relnet.agent.pytorch_agent import PyTorchAgent
from relnet.agent.rnet_dqn.rnet_dqn_agent import RNetDQNAgent
from relnet.agent.baseline.baseline_agent import RandomAgent
from relnet.environment.graph_edge_env import GraphEdgeEnv
from relnet.evaluation.file_paths import FilePaths
from relnet.state.network_generators import NetworkGenerator, BANetworkGenerator, WSNetworkGenerator, \
    GNMNetworkGenerator
from relnet.objective_functions.social_welfare import MaxContribution
from relnet.experiment_launchers.run_rnet_dqn import get_file_paths, get_rand_options


def get_gen_params():
    # Defines the network generation parameters
    gp = {}
    # Define the type of graph
    gp['type'] = 'ws'   # ['ba', 'ws', 'er']
    # Nodes
    gp['n'] = 25
    # Parameters for Barabasi-Albert Graph
    gp['m_ba'] = 3
    # Parameters for Watts-Strogatz Graph
    gp['k_ws'], gp['p_ws'] = 4, 0.5
    # Number of edges compared to nodes
    gp['m_percentage_er'] = 3
    # gp['m'] = NetworkGenerator.compute_number_edges(gp['n'], gp['m_percentage_er'])
    gp['m'] = 149
    gp['t'] = 0.0
    return gp


def institution_get_identifier_prefix(gen_params, institution):

    if gen_params['type'] == 'ba':
        m = gen_params['m_ba']
    elif gen_params['type'] == 'ws':
        m = gen_params['k_ws']
    else:
        m = gen_params['m']

    t = gen_params['t']
    if institution:
        return f"MaxContribution_institution_{gen_params['type']}_{gen_params['n']}_{m}_{t}"
    else:
        return f"MaxContribution_institution_{gen_params['type']}_{gen_params['n']}_{m}_0.0"


def get_options(file_paths, gen_params):
    game_type = 'majority'
    institution = True

    options = {"log_progress": True,
               "log_filename": str(file_paths.construct_log_filepath()),
               "log_tf_summaries": True,
               "random_seed": 42,
               "models_path": file_paths.models_dir,
               "game_type": game_type,
               "institution": institution,
               'model_identifier_prefix': institution_get_identifier_prefix(gen_params, institution),
               "restore_model": False,
               "validation_check_interval": 50,
               'reward_shaping': True,
               'pytorch_full_print': False,
               'max_validation_consecutive_steps': 700,
               'double': True,
               'soft': True,
               'tau': 0.1,
               'discount': 0.99}
    return options


if __name__ == '__main__':

    for t in [0.1]:
        for n in [15, 25, 50]:
            num_training_steps = 200
            num_train_graphs = 1000
            num_validation_graphs = 100
            num_test_graphs = 100
            gen_params = get_gen_params()
            gen_params['n'] = n
            gen_params['t'] = t

            file_paths = get_file_paths()
            options = get_options(file_paths, gen_params)
            storage_root = Path('/experiment_data/stored_graphs')
            original_dataset_dir = Path('/experiment_data/real_world_graphs/processed_data')
            kwargs = {'store_graphs': True,
                      'graph_storage_root': storage_root,
                      'game_type': options["game_type"],
                      'enforce_connected': True,
                      'institution': options['institution'],
                      'tax': t}
            if gen_params['type'] == 'ba':
                gen = BANetworkGenerator(**kwargs)
            elif gen_params['type'] == 'ws':
                gen = WSNetworkGenerator(**kwargs)
            elif gen_params['type'] == 'er':
                gen = GNMNetworkGenerator(**kwargs)
            train_graph_seeds, validation_graph_seeds, test_graph_seeds = \
                NetworkGenerator.construct_network_seeds(num_train_graphs, num_validation_graphs, num_test_graphs)

            train_graphs = gen.generate_many(gen_params, train_graph_seeds)
            validation_graphs = gen.generate_many(gen_params, validation_graph_seeds)

            test_graphs = gen.generate_many(gen_params, test_graph_seeds)
            edge_percentage = 10.0
            # Here we change the object to maximise contributing agents, rather than social welfare.
            target_env = GraphEdgeEnv(MaxContribution, edge_percentage)
            agent = RNetDQNAgent(target_env)
            agent.setup(options, agent.get_default_hyperparameters())

            # If not training comment out and just set up test history file
            agent.train(train_graphs, validation_graphs, num_training_steps)
            # agent.update_test_history()

            test_perf = agent.eval(test_graphs, test_set=True)
            # Compare to the random baseline
            rand_agent = RandomAgent(target_env)
            rand_agent.setup(get_rand_options(file_paths), None)
            baseline_perf = rand_agent.eval(test_graphs, test_set=True, baseline=True)
            # Log raw results
            agent.log_test_performance(test_perf,
                                       baseline_perf,
                                       agent.initial_obj_values,
                                       agent.final_obj_values,
                                       rand_agent.final_obj_values)
