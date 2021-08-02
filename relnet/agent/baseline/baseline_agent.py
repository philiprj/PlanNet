from abc import abstractmethod, ABC

import networkx as nx
import numpy as np

from relnet.agent.base_agent import Agent
from relnet.spectral_properties import *


class BaselineAgent(Agent, ABC):
    is_trainable = False

    def __init__(self, environment):
        super().__init__(environment)
        self.future_actions = None

    def setup(self, options, hyperparams):
        super().setup(options, hyperparams)

    def make_actions(self, t, **kwargs):
        if t % 3 == 0:
            first_actions, second_actions, third_actions = [], [], []
            for i in range(len(self.environment.g_list)):
                first_node, second_node = self.pick_actions_using_strategy(t, i)
                first_actions.append(first_node)
                second_actions.append(second_node)

            self.future_actions = second_actions
            chosen_actions = first_actions

        else:
            chosen_actions = self.future_actions
            self.future_actions = None

        return chosen_actions

    def finalize(self):
        pass

    @abstractmethod
    def pick_actions_using_strategy(self, t, i):
        pass


class RandomAgent(BaselineAgent):
    """
    Selects actions randomly from the available actions.
    """
    algorithm_name = 'random'
    is_deterministic = False

    def __init__(self, environment):
        super().__init__(environment)

    def pick_actions_using_strategy(self, t, i):
        return self.pick_random_actions(i)

    def agent_exploration_policy(self, i):
        """
        :param i: Index of the graph under consideration
        :return: Returns a random action
        """
        return self.pick_random_actions(i)

    def make_actions(self, t, **kwargs):
        if t % 3 == 0:
            action_type, flag = self.environment.exploratory_actions(self.agent_exploration_policy)
            self.next_exploration_actions = flag
            return action_type

        elif t % 3 == 1:
            # If random value less than epsilon then select exploration action
            if self.next_exploration_actions is not None:
                # Selects and returns current and next exploratory actions
                exploration_actions_t0, exploration_actions_t1 = \
                    self.environment.exploratory_actions(self.agent_exploration_policy)
                self.next_exploration_actions = exploration_actions_t1
                return exploration_actions_t0

        # If not decay value then do not decay - return exploration if previous action was greedy? Seem odd
        else:
            if self.next_exploration_actions is not None:
                return self.next_exploration_actions


class GreedyAgent(BaselineAgent):
    """
    Selects the highest value action at all points - no exploration policy
    """
    algorithm_name = 'greedy'
    is_deterministic = True

    def __init__(self, environment):
        super().__init__(environment,)

    def pick_actions_using_strategy(self, t, i):
        first_node, second_node = None, None

        g = self.environment.g_list[i]
        non_edges = list(self.environment.get_graph_non_edges(i))
        if len(non_edges) == 0:
            return (-1, -1)
        elif len(non_edges) == 1:
            return non_edges[0][0], non_edges[0][1]

        initial_value = self.environment.objective_function_values[0, i]
        best_difference = float("-inf")

        for first, second in non_edges:
            g_copy = g.copy()
            next_g, _ = g_copy.add_edge(first, second)
            next_value = self.environment.get_objective_function_value(next_g)

            diff = next_value - initial_value
            if diff > best_difference:
                best_difference = diff
                first_node, second_node = first, second

        return first_node, second_node
