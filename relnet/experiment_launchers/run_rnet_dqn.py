import sys
from pathlib import Path

sys.path.append('/relnet')
from relnet.agent.pytorch_agent import PyTorchAgent
from relnet.agent.rnet_dqn.rnet_dqn_agent import RNetDQNAgent
from relnet.environment.graph_edge_env import GraphEdgeEnv
from relnet.evaluation.file_paths import FilePaths
from relnet.objective_functions.objective_functions import CriticalFractionTargeted, CriticalFractionRandom
from relnet.state.network_generators import NetworkGenerator, BANetworkGenerator


def get_gen_params():
    # Defines the network generation parameters
    gp = {}
    # Nodes
    gp['n'] = 25
    gp['m_ba'] = 2
    # Number of edges compared to nodes
    gp['m_percentage_er'] = 20
    gp['m'] = NetworkGenerator.compute_number_edges(gp['n'], gp['m_percentage_er'])
    return gp


def get_options(file_paths):
    # Defines the general options for logging/saving results
    options = {"log_progress": True,
               "log_filename": str(file_paths.construct_log_filepath()),
               "log_tf_summaries": True,
               "random_seed": 42,
               "models_path": file_paths.models_dir,
               "restore_model": False}
    return options


def get_file_paths():
    # Gets the file saves paths
    parent_dir = '/experiment_data'
    experiment_id = 'development'
    file_paths = FilePaths(parent_dir, experiment_id)
    return file_paths


if __name__ == '__main__':
    # Defines the number of training steps, and graphs for train/validation/test
    num_training_steps = 100
    num_train_graphs = 2
    num_validation_graphs = 1
    num_test_graphs = 1
    # Gets the training parameters and save paths
    gen_params = get_gen_params()
    file_paths = get_file_paths()
    # Defines the training logging details
    options = get_options(file_paths)
    storage_root = Path('/experiment_data/stored_graphs')
    original_dataset_dir = Path('/experiment_data/real_world_graphs/processed_data')
    kwargs = {'store_graphs': False, 'graph_storage_root': storage_root}
    # Generate graphs class using Barabasi–Albert
    gen = BANetworkGenerator(**kwargs)
    # Generate random seeds for create graphs
    train_graph_seeds, validation_graph_seeds, test_graph_seeds = NetworkGenerator.construct_network_seeds(num_train_graphs, num_validation_graphs, num_test_graphs)
    # Creates our graphs
    train_graphs = gen.generate_many(gen_params, train_graph_seeds)
    validation_graphs = gen.generate_many(gen_params, validation_graph_seeds)
    test_graphs = gen.generate_many(gen_params, test_graph_seeds)
    # Edge percentage tells us how many changes we can make as a fraction of initial edges
    edge_percentage = 2.5
    # Objective function parameters
    obj_fun_kwargs = {"random_seed": 42, "num_mc_sims": gen_params['n'] * 2}
    # Creates the graph environment which the agent will work with
    targ_env = GraphEdgeEnv(CriticalFractionRandom(), obj_fun_kwargs, edge_percentage)
    # Init the DQN agent and set it up with hyperparameters
    agent = RNetDQNAgent(targ_env)
    agent.setup(options, agent.get_default_hyperparameters())
    # Training for given number of steps - validating on validation graphs
    agent.train(train_graphs, validation_graphs, num_training_steps)
    # Evaluate the best model agent on the test graphs
    avg_perf = agent.eval(test_graphs)
