from abc import abstractmethod, ABC
import networkx as nx
import numpy as np
from relnet.agent.base_agent import Agent


class BaselineAgent(Agent, ABC):
    is_trainable = False

    def __init__(self, environment):
        super().__init__(environment)
        self.future_actions = None

    def setup(self, options, hyperparams):
        super().setup(options, hyperparams)

    def make_actions(self, t, **kwargs):
        if t % 3 == 0:
            action_type, flag = self.environment.exploratory_actions(self.pick_actions_using_strategy)
            self.next_exploration_actions = flag
            return action_type

        elif t % 3 == 1:
            if self.next_exploration_actions is not None:
                exploration_actions_t0, exploration_actions_t1 = \
                    self.environment.exploratory_actions(self.pick_actions_using_strategy)
                self.next_exploration_actions = exploration_actions_t1
                return exploration_actions_t0

        else:
            if self.next_exploration_actions is not None:
                return self.next_exploration_actions

    def finalize(self):
        pass

    @abstractmethod
    def pick_actions_using_strategy(self, i):
        pass


class RandomAgent(BaselineAgent):
    """
    Selects actions randomly from the available actions.
    """
    algorithm_name = 'random'
    is_deterministic = False

    def __init__(self, environment):
        super().__init__(environment)

    def pick_actions_using_strategy(self, i):
        return self.pick_random_actions(i)


class EdgeConnectAgent(BaselineAgent):
    """
    Connect nodes with lowest reward, playing non-dominant action, to a node with high reward playing dominant action.
    """
    algorithm_name = 'edge_connect'
    is_deterministic = True

    def __init__(self, environment):
        super().__init__(environment,)

    def pick_actions_using_strategy(self, i):
        first_node, second_node = None, None
        g = self.environment.g_list[i]

        if g.action_type is None:
            first_node = 'add'
            second_node = -1.
            return first_node, second_node

        # Get rewards and action dict, node degree array
        all_nodes = g.all_nodes_set
        rewards = g.rewards
        actions = g.actions
        banned_first_nodes = g.banned_actions
        first_valid_acts = self.environment.get_valid_actions(g, banned_first_nodes)
        if len(first_valid_acts) == 0:
            return -1, -1

        # Get the dominant action
        if (sum(list(actions.values())) / len(list(actions.values()))) < 0.5:
            dominant_action = 0.
        else:
            dominant_action = 1.

        # Get agents playing non-dominant action
        non_dom_nodes = []
        for node, action in actions.items():
            if (action != dominant_action) and (node not in banned_first_nodes):
                non_dom_nodes.append(node)
        # Find agent with lowest reward
        if len(non_dom_nodes) > 0:
            curr_val = 1.
            for node in non_dom_nodes:
                if rewards[node] < curr_val:
                    curr_val = rewards[node]
                    first_node = node
            # Get dominant nodes
            dom_nodes = list(all_nodes - set(non_dom_nodes))
            curr_val = 0.
            banned_second_nodes = g.get_invalid_edge_ends(first_node, self.environment.get_remaining_budget(i))
            for node in dom_nodes:
                if (rewards[node] > curr_val) and (node not in banned_second_nodes):
                    curr_val = rewards[node]
                    second_node = node

        # If no nodes satisfy the criteria, then pick randomly
        if (first_node is None) or (second_node is None):
            first_node, second_node = self.pick_random_actions(i)

        return first_node, second_node


class LowDegreeTarget(BaselineAgent):
    """
    Flip node playing contributes with few connections
    """
    algorithm_name = 'Low_Degree'
    is_deterministic = True

    def __init__(self, environment):
        super().__init__(environment,)

    def pick_actions_using_strategy(self, i):
        first_node, second_node = None, None

        g = self.environment.g_list[i]

        if g.action_type is None:
            first_node = 'flip'
            second_node = -1.
            return first_node, second_node

        degrees = g.node_degrees
        actions = g.actions

        # Get agents playing non-dominant action
        contributes = []
        for node, action in actions.items():
            if action == 1:
                contributes.append(node)
        # Find agent with lowest degree
        curr_min = float(np.finfo(np.float32).max)
        curr_min2 = float(np.finfo(np.float32).max)
        for node in contributes:
            if degrees[node] < curr_min:
                curr_min = degrees[node]
                first_node = node
            elif degrees[node] < curr_min2:
                curr_min2 = degrees[node]
                second_node = node

        return first_node, second_node
