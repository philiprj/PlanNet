from copy import deepcopy

from relnet.utils.config_utils import local_seed

from itertools import product
import numpy as np


def generate_search_space(parameter_grid,
                          random_search=False,
                          random_search_num_options=20,
                          random_search_seed=42):
    """
    Performs a grid search for parameters - not used here for computational purposes.
    :return: reduced search space for parameter tuning
    """

    combinations = list(product(*parameter_grid.values()))
    search_space = {i: combinations[i] for i in range(len(combinations))}

    if random_search:
        if not random_search_num_options > len(search_space):
            reduced_space = {}
            with local_seed(random_search_seed):
                random_indices = np.random.choice(len(search_space), random_search_num_options, replace=False)
                for random_index in random_indices:
                    reduced_space[random_index] = search_space[random_index]
            search_space = reduced_space
    return search_space


def get_values_for_g_list(agent, g_list, initial_obj_values, validation, make_action_kwargs):
    """
    Runs the action selection until terminal state and gets objective function values.
    """
    if initial_obj_values is None:
        obj_values = agent.environment.get_objective_function_values(g_list)
    else:
        obj_values = initial_obj_values
    agent.environment.setup(g_list, obj_values, training=False)

    t = 0
    while not agent.environment.is_terminal():
        # print(f"making actions at time {t}.")

        action_kwargs = (make_action_kwargs or {})
        list_at = agent.make_actions(t, **action_kwargs)
        list_a_types = [g.action_type for g in g_list]

        agent.environment.step(list_at, list_a_types)
        t += 1
    final_obj_values = agent.environment.get_final_values()
    return obj_values, final_obj_values


def eval_on_dataset(initial_objective_function_values, final_objective_function_values, mean=True):
    # Return the mean error or array of error values.
    if mean:
        return np.mean(final_objective_function_values - initial_objective_function_values)
    else:
        return final_objective_function_values - initial_objective_function_values


def record_episode_histories(agent, g_list):
    # Records the state-action histories over the graphs until terminal.
    states, actions, rewards, initial_values = [], [], [], []

    nets = [deepcopy(g) for g in g_list]
    initial_values = agent.environment.get_objective_function_values(nets)

    agent.environment.setup(nets, initial_values, training=False)
    t = 0
    while not agent.environment.is_terminal():
        list_st = deepcopy(agent.environment.g_list)
        list_at = agent.make_actions(t, **{})

        states.append(list_st)
        actions.append(list_at)
        rewards.append([0] * len(list_at))

        _, _, _, a_types = zip(*list_st)

        agent.environment.step(list_at, a_types)
        t += 1

    final_states = deepcopy(agent.environment.g_list)
    states.append(final_states)
    final_acts = [None] * len(final_states)
    actions.append(final_acts)

    final_obj_values = agent.environment.get_final_values()
    rewards.append(final_obj_values - initial_values)

    return states, actions, rewards, initial_values
