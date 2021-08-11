import sys
from copy import deepcopy
from pathlib import Path

sys.path.append('/relnet')
from relnet.agent.pytorch_agent import PyTorchAgent
from relnet.agent.rnet_dqn.rnet_dqn_agent import RNetDQNAgent
from relnet.agent.baseline.baseline_agent import RandomAgent
from relnet.environment.graph_edge_env import GraphEdgeEnv
from relnet.evaluation.file_paths import FilePaths
from relnet.state.network_generators import NetworkGenerator, BANetworkGenerator, WSNetworkGenerator, \
    GNMNetworkGenerator
from relnet.objective_functions.social_welfare import SocialWelfare


def get_gen_params():
    # Defines the network generation parameters
    gp = {}
    # Define the type of graph
    gp['type'] = 'er' # ['ba', 'ws', 'er']
    # Nodes
    gp['n'] = 25
    # Parameters for Barabasi-Albert Graph
    gp['m_ba'] = 1
    # Parameters for Watts-Strogatz Graph
    gp['k_ws'], gp['p_ws'] = 2, 0.5
    # Number of edges compared to nodes
    gp['m_percentage_er'] = 13
    gp['m'] = NetworkGenerator.compute_number_edges(gp['n'], gp['m_percentage_er'])
    print(gp['m'] / gp['n'])
    return gp


def get_identifier_prefix(gen_params, gtype):

    if gen_params['type'] == 'ba':
        m = gen_params['m_ba']
    elif gen_params['type'] == 'ws':
        m = gen_params['k_ws']
    else:
        m = gen_params['m']

    return f"{gtype}_{gen_params['type']}_{gen_params['n']}_{m}"

def get_options(file_paths, gen_params):
    # Defines the general options for logging/saving results
    game_type = 'majority' # ['majority', 'bspgg', 'pgg']

    options = {"log_progress": True,
               "log_filename": str(file_paths.construct_log_filepath()),
               "log_tf_summaries": True,
               "random_seed": 42,
               "models_path": file_paths.models_dir,
               "game_type": game_type,
               'model_identifier_prefix': get_identifier_prefix(gen_params, game_type),
               "restore_model": False,
               "validation_check_interval": 50,
               'reward_shaping': True,
               'pytorch_full_print': False,
               'max_validation_consecutive_steps': 2000,
               'double': True,
               'soft': True,
               'tau': 0.1,
               'discount': 0.99}
    return options


def get_rand_options(file_paths):
    options = {"log_progress": True,
               "log_filename": str(file_paths.construct_log_filepath()),
               "log_tf_summaries": True,
               "random_seed": 42,
               "models_path": file_paths.models_dir,
               'model_identifier_prefix': 'majority-random'}
    return options


def get_file_paths():
    # Gets the file saves paths
    parent_dir = '/experiment_data'
    experiment_id = 'development'
    file_paths = FilePaths(parent_dir, experiment_id)
    return file_paths


if __name__ == '__main__':
    # Defines the number of training steps, and graphs for train/validation/test
    num_training_steps = 1000
    num_train_graphs = 1000
    num_validation_graphs = 100
    num_test_graphs = 100

    # Gets the training parameters and save paths
    gen_params = get_gen_params()
    file_paths = get_file_paths()
    # Defines the training logging details
    options = get_options(file_paths, gen_params)
    storage_root = Path('/experiment_data/stored_graphs')
    original_dataset_dir = Path('/experiment_data/real_world_graphs/processed_data')

    # Setting the game type to play
    kwargs = {'store_graphs': True,
              'graph_storage_root': storage_root,
              'game_type': options["game_type"],
              'enforce_connected': True}
    # Generate graphs class using Barabasi–Albert

    if gen_params['type'] == 'ba':
        gen = BANetworkGenerator(**kwargs)
    elif gen_params['type'] == 'ws':
        gen = WSNetworkGenerator(**kwargs)
    elif gen_params['type'] == 'er':
        gen = GNMNetworkGenerator(**kwargs)

    # Generate random seeds for create graphs
    train_graph_seeds, validation_graph_seeds, test_graph_seeds = \
        NetworkGenerator.construct_network_seeds(num_train_graphs, num_validation_graphs, num_test_graphs)

    # Creates our graphs - just lists so we could append different graph sizes to the list
    train_graphs = gen.generate_many(gen_params, train_graph_seeds)
    validation_graphs = gen.generate_many(gen_params, validation_graph_seeds)
    test_graphs = gen.generate_many(gen_params, test_graph_seeds)

    # Edge percentage - number of moves as percentage of initial edges
    edge_percentage = 10.0
    # Creates the graph environment which the agent will work with
    target_env = GraphEdgeEnv(SocialWelfare, edge_percentage)
    # Init the DQN agent and set it up with hyperparameters
    agent = RNetDQNAgent(target_env)
    agent.setup(options, agent.get_default_hyperparameters())
    # Training for given number of steps - validating on validation graphs
    agent.train(train_graphs, validation_graphs, num_training_steps)
    # Evaluate the best model agent on the test graphs
    test_perf = agent.eval(test_graphs, test_set=True)

    # Get baseline performance
    rand_agent = RandomAgent(target_env)
    rand_agent.setup(get_rand_options(file_paths), None)
    baseline_perf = rand_agent.eval(test_graphs, test_set=True, baseline=True)
    agent.log_test_performance(test_perf,
                               baseline_perf,
                               agent.initial_obj_values,
                               agent.final_obj_values,
                               rand_agent.final_obj_values)
