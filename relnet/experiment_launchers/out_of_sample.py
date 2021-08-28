import sys
from pathlib import Path
from copy import deepcopy

sys.path.append('/relnet')
from relnet.agent.rnet_dqn.rnet_dqn_agent import RNetDQNAgent
from relnet.environment.graph_edge_env import GraphEdgeEnv
from relnet.state.network_generators import NetworkGenerator, BANetworkGenerator, WSNetworkGenerator, \
    GNMNetworkGenerator
from relnet.objective_functions.social_welfare import SocialWelfare
from relnet.experiment_launchers.run_rnet_dqn import get_identifier_prefix, get_file_paths
from relnet.experiment_launchers.curriculum_learner import curriculum_identifier_prefix


def get_gen_params():
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
    gp['m'] = NetworkGenerator.compute_number_edges(gp['n'], gp['m_percentage_er'])
    return gp


def get_options_oos(file_paths, gen_params):
    game_type = 'bspgg'  # ['majority', 'bspgg']

    options = {"random_seed": 42,
               "models_path": file_paths.models_dir,
               "game_type": game_type,
               'model_identifier_prefix': curriculum_identifier_prefix(gen_params, game_type),
               "restore_model": True,
               'double': True}
    return options


if __name__ == '__main__':
    ntg, nvg, num_test_graphs = 1000, 100, 100
    gen_params = get_gen_params()
    file_paths = get_file_paths()
    storage_root = Path('/experiment_data/stored_graphs')
    original_dataset_dir = Path('/experiment_data/real_world_graphs/processed_data')

    for N in [15, 25, 50, 100]:
        gen_params['n'] = N
        options = get_options_oos(file_paths, gen_params)
        kwargs = {'store_graphs': True,
                  'graph_storage_root': storage_root,
                  'game_type': options["game_type"],
                  'enforce_connected': True}
        _, _, test_graph_seeds = NetworkGenerator.construct_network_seeds(ntg, nvg, num_test_graphs)
        target_env = GraphEdgeEnv(SocialWelfare, edge_budget_percentage=10.0)
        agent = RNetDQNAgent(target_env)
        # Will load the agent defined in options - same for all tests
        agent.setup(options, agent.get_default_hyperparameters())
        # Set up logging system for test sets
        agent.setup_out_of_sample()
        er_m = [[19, 39, 62, 149], [21, 60, 123, 248]]
        for g in ['ba', 'ws', 'er']:
            for i, n in enumerate([15, 25, 50, 100]):
                gen_params_copy = deepcopy(gen_params)
                gen_params_copy['type'] = g
                gen_params_copy['n'] = n
                if g == 'ba':
                    if options["game_type"] == 'majority':
                        gen_params_copy['m_ba'] = 1
                    else:
                        gen_params['m_ba'] = 4
                    gen = BANetworkGenerator(**kwargs)
                elif g == 'ws':
                    if options["game_type"] == 'majority':
                        gen_params_copy['m_ba'] = 2
                    else:
                        gen_params_copy['m_ba'] = 4
                    gen = WSNetworkGenerator(**kwargs)
                else:
                    if options["game_type"] == 'majority':
                        gen_params_copy['m'] = er_m[0][i]
                    else:
                        gen_params_copy['m'] = er_m[1][i]
                    gen = GNMNetworkGenerator(**kwargs)
                # Set up new graphs and test
                test_graphs = gen.generate_many(gen_params_copy, test_graph_seeds)
                test_perf = agent.eval(test_graphs, test_set=True)
                agent.log_oos_performance(init_obj=agent.initial_obj_values,
                                          final_obj=agent.final_obj_values,
                                          n_nodes=n,
                                          graph_type=g)
