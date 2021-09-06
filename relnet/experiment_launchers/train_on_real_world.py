import sys
from pathlib import Path

sys.path.append('/relnet')
from relnet.agent.rnet_dqn.rnet_dqn_agent import RNetDQNAgent
from relnet.agent.baseline.baseline_agent import RandomAgent
from relnet.environment.graph_edge_env import GraphEdgeEnv
from relnet.evaluation.file_paths import FilePaths
from relnet.state.network_generators import NetworkGenerator, SawMill, KarateClub
from relnet.objective_functions.social_welfare import MaxContribution
from relnet.experiment_launchers.run_rnet_dqn import get_file_paths, get_rand_options

def get_gen_params_rw():
    gp = {}
    gp['t'] = 0.1
    return gp

def rw_identifier_prefix(gen_params, karate=True):

    if karate:
        return f"MaxContribution_institution_karate_{gen_params['t']}"
    else:
        return f"MaxContribution_institution_saw_mill_{gen_params['t']}"


def get_options(file_paths, gen_params, karate=True):
    game_type = 'majority'
    institution = True

    options = {"log_progress": True,
               "log_filename": str(file_paths.construct_log_filepath()),
               "log_tf_summaries": True,
               "random_seed": 42,
               "models_path": file_paths.models_dir,
               "game_type": game_type,
               "institution": institution,
               'model_identifier_prefix': rw_identifier_prefix(gen_params, karate=karate),
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
    num_training_steps = 400
    num_train_graphs = 100
    num_validation_graphs = 100
    num_test_graphs = 100
    file_paths = get_file_paths()
    storage_root = Path('/experiment_data/real_world_graphs')
    original_dataset_dir = Path('/experiment_data/real_world_graphs/processed_data')
    gen_params = get_gen_params_rw()
    edge_percentage = 10.0
    target_env = GraphEdgeEnv(MaxContribution, edge_percentage)

    for t in [0.0, 0.001, 0.1]:
        gen_params['t'] = t
        options = get_options(file_paths, gen_params, karate=True)
        kwargs = {'store_graphs': True,
                  'graph_storage_root': storage_root,
                  'game_type': options["game_type"],
                  'enforce_connected': False,
                  'institution': options['institution'],
                  'tax': gen_params['t']}

        g1 = KarateClub(**kwargs)
        g2 = SawMill(**kwargs)
        train_graph_seeds, validation_graph_seeds, test_graph_seeds = \
            NetworkGenerator.construct_network_seeds(num_train_graphs, num_validation_graphs, num_test_graphs)

        train_graphs = g1.generate_many(gen_params, train_graph_seeds)
        validation_graphs = g1.generate_many(gen_params, validation_graph_seeds)
        test_graphs = g1.generate_many(gen_params, test_graph_seeds)

        agent = RNetDQNAgent(target_env)
        agent.setup(options, agent.get_default_hyperparameters())
        agent.train(train_graphs, validation_graphs, num_training_steps)
        test_perf = agent.eval(test_graphs, test_set=True)
        rand_agent = RandomAgent(target_env)
        rand_agent.setup(get_rand_options(file_paths), None)
        baseline_perf = rand_agent.eval(test_graphs, test_set=True, baseline=True)
        agent.log_test_performance(test_perf,
                                   baseline_perf,
                                   agent.initial_obj_values,
                                   agent.final_obj_values,
                                   rand_agent.final_obj_values)

        options = get_options(file_paths, gen_params, karate=False)
        train_graphs = g2.generate_many(gen_params, train_graph_seeds)
        validation_graphs = g2.generate_many(gen_params, validation_graph_seeds)
        test_graphs = g2.generate_many(gen_params, test_graph_seeds)
        agent = RNetDQNAgent(target_env)
        agent.setup(options, agent.get_default_hyperparameters())
        agent.train(train_graphs, validation_graphs, num_training_steps)
        test_perf = agent.eval(test_graphs, test_set=True)
        rand_agent = RandomAgent(target_env)
        rand_agent.setup(get_rand_options(file_paths), None)
        baseline_perf = rand_agent.eval(test_graphs, test_set=True, baseline=True)
        agent.log_test_performance(test_perf,
                                   baseline_perf,
                                   agent.initial_obj_values,
                                   agent.final_obj_values,
                                   rand_agent.final_obj_values)
