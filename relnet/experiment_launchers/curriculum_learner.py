import sys
from pathlib import Path

sys.path.append('/relnet')
from relnet.agent.rnet_dqn.rnet_dqn_agent import RNetDQNAgent
from relnet.environment.graph_edge_env import GraphEdgeEnv
from relnet.state.network_generators import NetworkGenerator, BANetworkGenerator, WSNetworkGenerator, \
    GNMNetworkGenerator
from relnet.objective_functions.social_welfare import SocialWelfare
from relnet.experiment_launchers.run_rnet_dqn import get_file_paths


def get_curriculum_gen_params():
    # Defines the network generation parameters
    gp = {}
    # Define the type of graph
    gp['type'] = 'ba'   # ['ba', 'ws', 'er']
    # Nodes
    gp['n'] = 15
    # Parameters for Barabasi-Albert Graph
    gp['m_ba'] = 4
    # Parameters for Watts-Strogatz Graph
    gp['k_ws'], gp['p_ws'] = 4, 0.5
    # Number of edges compared to nodes
    gp['m_percentage_er'] = 3
    # gp['m'] = NetworkGenerator.compute_number_edges(gp['n'], gp['m_percentage_er'])
    gp['m'] = 149
    return gp


def curriculum_identifier_prefix(gen_params, gtype):

    return f"curriculum_{gtype}_{gen_params['type']}"


def get_options(file_paths, gen_params):

    game_type = 'bspgg'  # ['majority', 'bspgg']
    options = {"log_progress": True,
               "log_filename": str(file_paths.construct_log_filepath()),
               "log_tf_summaries": True,
               "random_seed": 42,
               "models_path": file_paths.models_dir,
               "game_type": game_type,
               'model_identifier_prefix': curriculum_identifier_prefix(gen_params, game_type),
               "restore_model": False,
               "validation_check_interval": 50,
               'reward_shaping': True,
               'pytorch_full_print': False,
               'double': True,
               'soft': True,
               'tau': 0.1,
               'discount': 0.99,
               'restore_end': True}
    return options


if __name__ == '__main__':
    """
    This file only runs training of agent using curriculum learning - 
    Testing is performed in out of sample file 
    """
    num_training_steps = 1000
    num_train_graphs = 1000
    num_validation_graphs = 100
    num_test_graphs = 100
    gen_params = get_curriculum_gen_params()
    file_paths = get_file_paths()
    options = get_options(file_paths, gen_params)
    storage_root = Path('/experiment_data/stored_graphs')
    original_dataset_dir = Path('/experiment_data/real_world_graphs/processed_data')
    kwargs = {'store_graphs': True,
              'graph_storage_root': storage_root,
              'game_type': options["game_type"],
              'enforce_connected': True}
    if gen_params['type'] == 'ba':
        gen = BANetworkGenerator(**kwargs)
    elif gen_params['type'] == 'ws':
        gen = WSNetworkGenerator(**kwargs)
    elif gen_params['type'] == 'er':
        gen = GNMNetworkGenerator(**kwargs)

    edge_percentage = 10.0
    target_env = GraphEdgeEnv(SocialWelfare, edge_percentage)
    agent = RNetDQNAgent(target_env)
    agent.setup(options, agent.get_default_hyperparameters())
    n_nodes = [15, 25, 50, 100]
    for i, n in enumerate(n_nodes):
        gen_params['n'] = n
        train_graph_seeds, validation_graph_seeds, _ = \
            NetworkGenerator.construct_network_seeds(num_train_graphs, num_validation_graphs, num_test_graphs)
        train_graphs = gen.generate_many(gen_params, train_graph_seeds)
        validation_graphs = gen.generate_many(gen_params, validation_graph_seeds)
        if n == 15:
            agent.train(train_graphs, validation_graphs, num_training_steps)
        else:
            options['restore_model'] = True
            agent = RNetDQNAgent(target_env)
            agent.setup(options, agent.get_default_hyperparameters())
            agent.train(train_graphs, validation_graphs, num_training_steps)
