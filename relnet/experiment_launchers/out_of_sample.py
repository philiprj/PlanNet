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
    gp = {}
    gp['type'] = 'ws'   # ['ba', 'ws', 'er']
    gp['n'] = 15
    gp['m_ba'] = 1
    gp['k_ws'], gp['p_ws'] = 2, 0.5
    gp['er_p'] = 0.3
    return gp


def get_options_oos(file_paths, gen_params):
    game_type = 'majority'  # ['majority', 'bspgg']

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
    p_er_bspgg = [0.3, 0.2, 0.1, 0.05]
    p_er_maj = [0.1, 0.07, 0.05, 0.03]
    p_er_bspgg1 = [0.3, 0.1, 0.05]
    p_er_maj1 = [0.1, 0.05, 0.03]

    for g_type in ['ba', 'ws']:
        gen_params['type'] = g_type

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
                        gen_params_copy['er_p'] = p_er_maj[i]
                    else:
                        gen_params_copy['er_p'] = p_er_bspgg[i]
                    gen = GNMNetworkGenerator(**kwargs)
                # Set up new graphs and test
                test_graphs = gen.generate_many(gen_params_copy, test_graph_seeds)
                test_perf = agent.eval(test_graphs, test_set=True)
                agent.log_oos_performance(init_obj=agent.initial_obj_values,
                                          final_obj=agent.final_obj_values,
                                          n_nodes=n,
                                          graph_type=g)
