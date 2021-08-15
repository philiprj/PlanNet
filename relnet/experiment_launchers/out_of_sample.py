import sys
from pathlib import Path

sys.path.append('/relnet')
from relnet.agent.rnet_dqn.rnet_dqn_agent import RNetDQNAgent
from relnet.environment.graph_edge_env import GraphEdgeEnv
from relnet.evaluation.file_paths import FilePaths
from relnet.state.network_generators import NetworkGenerator, BANetworkGenerator, WSNetworkGenerator, \
    GNMNetworkGenerator
from relnet.objective_functions.social_welfare import SocialWelfare
from relnet.experiment_launchers.run_rnet_dqn import get_identifier_prefix, get_file_paths, get_gen_params


def get_options_oos(file_paths, gen_params):
    game_type = 'majority'  # ['majority', 'bspgg', 'pgg']

    options = {"random_seed": 42,
               "models_path": file_paths.models_dir,
               "game_type": game_type,
               'model_identifier_prefix': get_identifier_prefix(gen_params, game_type),
               "restore_model": True,
               'double': True}
    return options


if __name__ == '__main__':
    ntg, nvg, num_test_graphs = 1000, 100, 100
    gen_params = get_gen_params()
    file_paths = get_file_paths()
    options = get_options_oos(file_paths, gen_params)
    storage_root = Path('/experiment_data/stored_graphs')
    original_dataset_dir = Path('/experiment_data/real_world_graphs/processed_data')
    kwargs = {'store_graphs': True,
              'graph_storage_root': storage_root,
              'game_type': options["game_type"],
              'enforce_connected': True}
    _, _, test_graph_seeds = NetworkGenerator.construct_network_seeds(ntg, nvg, num_test_graphs)
    for g in ['ba', 'ws']:
        for n in [15, 25, 50, 100]:
            gen_params['type'] = g
            gen_params['n'] = n
            if g == 'ba':
                if options["game_type"] == 'majority':
                    gen_params['m_ba'] = 1
                else:
                    gen_params['m_ba'] = 4
                gen = BANetworkGenerator(**kwargs)
            else:
                if options["game_type"] == 'majority':
                    gen_params['m_ba'] = 2
                else:
                    gen_params['m_ba'] = 4
                gen = WSNetworkGenerator(**kwargs)

            test_graphs = gen.generate_many(gen_params, test_graph_seeds)
            edge_percentage = 10.0
            target_env = GraphEdgeEnv(SocialWelfare, edge_percentage)
            agent = RNetDQNAgent(target_env)
            agent.setup(options, agent.get_default_hyperparameters())
            test_perf = agent.eval(test_graphs, test_set=True)

            # TODO
            # Set up logging system for test sets
