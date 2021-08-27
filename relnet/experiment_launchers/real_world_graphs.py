import sys
from pathlib import Path

sys.path.append('/relnet')
from relnet.agent.rnet_dqn.rnet_dqn_agent import RNetDQNAgent
from relnet.agent.baseline.baseline_agent import RandomAgent
from relnet.environment.graph_edge_env import GraphEdgeEnv
from relnet.state.network_generators import NetworkGenerator, SawMill, KarateClub
from relnet.objective_functions.social_welfare import SocialWelfare, MaxContribution
from relnet.experiment_launchers.run_rnet_dqn import get_file_paths, get_rand_options
from relnet.experiment_launchers.run_institution import institution_get_identifier_prefix


def get_gen_params():
    gp = {}
    gp['type'] = 'ws'   # ['ba', 'ws', 'er']
    gp['n'] = 15
    gp['m_ba'] = 1
    gp['k_ws'], gp['p_ws'] = 2, 0.5
    gp['t'] = 0.1
    if gp['type'] == 'ws':
        gp['m'] = gp['k_ws']
    else:
        gp['m'] = gp['m_ba']
    return gp


def get_options_real_world(file_paths, gen_params):
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
               "restore_model": True,
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
    storage_root = Path('/experiment_data/real_world_graphs')
    original_dataset_dir = Path('/experiment_data/real_world_graphs/processed_data')
    gen_params = get_gen_params()
    file_paths = get_file_paths()
    options = get_options_real_world(file_paths, gen_params)
    edge_percentage = 10.0
    target_env = GraphEdgeEnv(MaxContribution, edge_percentage)
    agent = RNetDQNAgent(target_env)
    agent.setup(options, agent.get_default_hyperparameters())

    for t in [0.0, 0.001, 0.1, 0.2]:
        print(f"Tax: {t}")
        for n in [15, 25, 50]:
            gen_params['n'] = n
            gen_params['t'] = t
            options = get_options_real_world(file_paths, gen_params)
            kwargs = {'store_graphs': True,
                      'graph_storage_root': storage_root,
                      'game_type': options["game_type"],
                      'enforce_connected': True,
                      'institution': options['institution'],
                      'tax': gen_params['t']}

            # Create real world graphs
            g1 = KarateClub(**kwargs)
            g2 = SawMill(**kwargs)
            _, _, test_graph_seeds = NetworkGenerator.construct_network_seeds(0, 0, 100)
            test_graphs_karate = g1.generate_many(gen_params, test_graph_seeds)
            test_graphs_saw = g2.generate_many(gen_params, test_graph_seeds)

            # Karate model
            agent.model_identifier_prefix = \
                f"karate_{gen_params['n']}_{gen_params['m']}_{gen_params['type']}_{gen_params['t']}"
            agent.update_test_history()
            test_perf = agent.eval(test_graphs_karate, test_set=True)
            rand_agent = RandomAgent(target_env)
            rand_agent.setup(get_rand_options(file_paths), None)
            baseline_perf = rand_agent.eval(test_graphs_karate, test_set=True, baseline=True)
            agent.log_test_performance(test_perf,
                                       baseline_perf,
                                       agent.initial_obj_values,
                                       agent.final_obj_values,
                                       rand_agent.final_obj_values)

            # Saw mill model
            agent.model_identifier_prefix = \
                f"saw_mill_{gen_params['n']}_{gen_params['m']}_{gen_params['m']}_{gen_params['type']}_{gen_params['t']}"
            agent.update_test_history()
            test_perf = agent.eval(test_graphs_saw, test_set=True)
            rand_agent = RandomAgent(target_env)
            rand_agent.setup(get_rand_options(file_paths), None)
            baseline_perf = rand_agent.eval(test_graphs_saw, test_set=True, baseline=True)
            agent.log_test_performance(test_perf,
                                       baseline_perf,
                                       agent.initial_obj_values,
                                       agent.final_obj_values,
                                       rand_agent.final_obj_values)
