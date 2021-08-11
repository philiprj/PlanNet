from copy import deepcopy
import numpy as np
from relnet.state.network_generators import NetworkGenerator, random_action_init


class GraphEdgeEnv(object):
    # Environment for the graph
    def __init__(self,
                 objective_function,
                 edge_budget_percentage):
        """
        Initialises the Graph Class
        :param objective_function: Objective function used to value state
        :param objective_function_kwargs: objective function keyword arguments in a dict (random seed, num mc sims)
        :param edge_budget_percentage: budget as a percentage of the number of edges
        """
        # Init the parameters
        self.objective_function = objective_function
        self.edge_budget_percentage = edge_budget_percentage

        """
        Number of steps in the MDP 
            - Step 1 select type of action {'add', 'remove', 'flip'}
            - step 2 selected 1st node 
            - step 3 selected 2nd node
        """
        self.num_mdp_substeps = 3
        # Parameters for the reward
        self.reward_eps = 1e-6
        self.reward_scale_multiplier = 100

    def setup(self, g_list, initial_objective_function_values, training=False, intermediate_scale=True):
        """
        Sets up the class
        :param g_list: Graph list
        :param initial_objective_function_values: start values for the objective function
        :param training: Bool for training or validation/testing
        :return:
        """
        # Graphs in a list and the number of steps taken
        self.g_list = g_list

        self.n_steps = 0
        # Total budget for making changes to each graph, and count for budget used
        self.edge_budgets = self.compute_edge_budgets(self.g_list, self.edge_budget_percentage)
        self.used_edge_budgets = np.zeros(len(g_list), dtype=np.float)
        self.exhausted_budgets = np.zeros(len(g_list), dtype=np.bool)
        # Loop through graphs and set first node to None
        for i in range(len(self.g_list)):
            g = g_list[i]
            g.first_node = None
            # Set banned actions to be more expensive than the budget
            g.populate_banned_actions(self.edge_budgets[i], None)
        # Set training or testing
        self.training = training
        # Define the objective function initial values for each graph
        self.objective_function_values = np.zeros((2, len(self.g_list)), dtype=np.float)
        self.objective_function_values[0, :] = initial_objective_function_values
        # Sets the rewards for each graph
        self.rewards = np.zeros(len(g_list), dtype=np.float)
        # If training then scale the objective values by the reward scales
        if self.training:
            self.objective_function_values[0, :] = \
                np.multiply(self.objective_function_values[0, :], self.reward_scale_multiplier)

        self.intermediate_scale = intermediate_scale
        if intermediate_scale:
            self.scale_value = 1.0
        else:
            self.scale_value = 0.0

    def pass_logger_instance(self, logger):
        # Sets current log
        self.logger_instance = logger

    def get_final_values(self):
        # Gets the values for the final state of each graph
        return self.objective_function_values[-1, :]

    def get_objective_function_value(self, s2v_graph):
        # Gets the objective_function values using the embedded graph
        obj_function_value = self.objective_function(s2v_graph)
        return obj_function_value

    def get_objective_function_values(self, s2v_graphs):
        # Gets the objective function values for each graph in the dataset
        return np.array([self.get_objective_function_value(g) for g in s2v_graphs])

    def get_graph_non_edges(self, i):
        """
        Find the edges that are valid additions with current budget which are not already connected
        :param i: Index of the graph in the dataset
        :return: Set of non-connected valid edges
        """
        # Get the current graph
        g = self.g_list[i]
        # Get the current budget for moves and init set
        budget = self.get_remaining_budget(i)
        # Populate banned actions for adding an edge
        g.populate_banned_actions(budget, 'add')
        # Get the banned first nodes
        banned_first_nodes = g.banned_actions
        # Get the valid first actions
        valid_acts = self.get_valid_actions(g, banned_first_nodes)
        non_edges = set()
        # Loop through valid initial nodes
        for first in valid_acts:
            # Get the invalid end nodes and the valid end nodes
            banned_second_nodes = g.get_invalid_edge_ends(first, budget)
            valid_second_nodes = self.get_valid_actions(g, banned_second_nodes)
            # Loop through 2nd nodes that are not already connected and valid connections
            for second in valid_second_nodes:
                # Add to the non connected valid edge set (non directed so only add once)
                if (first, second) in non_edges or (second, first) in non_edges:
                    continue
                non_edges.add((first, second))
        # Return the non connected valid edge set
        return non_edges

    def get_graph_edges(self, i):
        g = self.g_list[i]
        budget = self.get_remaining_budget(i)
        g.populate_banned_actions(budget, 'remove')
        banned_first_nodes = g.banned_actions
        valid_acts = self.get_valid_actions(g, banned_first_nodes)
        non_edges = set()
        for first in valid_acts:
            # Get the invalid end - ones that would disconnect the graph
            banned_second_nodes = g.invalid_edge_ends_removal(first, budget)
            valid_second_nodes = self.get_valid_actions(g, banned_second_nodes)
            # Loop through 2nd nodes and add if not in set already
            for second in valid_second_nodes:
                if (first, second) in non_edges or (second, first) in non_edges:
                    continue
                non_edges.add((first, second))
        return non_edges

    def get_remaining_budget(self, i):
        # Gets remaining budged (initials budget minus used budget) for this graph
        return self.edge_budgets[i] - self.used_edge_budgets[i]

    @staticmethod
    def compute_edge_budgets(g_list, edge_budget_percentage, of_max=False):
        """
        Computes budget for making changes for each graph
        :param g_list: List of graphs
        :param edge_budget_percentage: Budget as a percentage of the number of edges
        :return: Edge budget list for each graph (array)
        """
        # Init empty array of zeros
        edge_budgets = np.zeros(len(g_list), dtype=np.float)
        # Loop through graphs
        for i in range(len(g_list)):
            # Get number of nodes for the graph, and compute budget using this and relative budget parameter
            g = g_list[i]
            if of_max:
                n = g.num_nodes
                edge_budgets[i] = NetworkGenerator.compute_number_edges(n, edge_budget_percentage)
            else:
                m = g.num_edges
                edge_budgets[i] = NetworkGenerator.compute_edges_changes(m, edge_budget_percentage)
        # Return budget for each graph
        return edge_budgets

    @staticmethod
    def get_valid_actions(g, banned_actions):
        """
        Gets the valid actions for a given graph
        :param g: Graph
        :param banned_actions: banned actions for this graph
        :return: Valid first nodes to add edge from
        """
        # Start with all nodes
        all_nodes_set = g.all_nodes_set
        # Remove the banned nodes and return remaining set
        valid_first_nodes = all_nodes_set - banned_actions
        return valid_first_nodes

    @staticmethod
    def apply_action(g, action, remaining_budget, action_type=None, copy_state=False):
        """
        Applies the action to the graph to update the graph
        :param g: graph
        :param action: selected node (1st node then 2nd node)
        :param remaining_budget: budget for changes to the graph
        :param action_type: type of action being applied
        :param copy_state: bool for copying or changing graph directly
        :return: Graph with change and remaining budget
        """
        # If we are on the first step picking the action then update the internal state and continue
        if action in ['add', 'remove', 'flip']:
            g_ref = g
            g_ref.action_type = action
            g_ref.populate_banned_actions(remaining_budget, g_ref.action_type)
            return g_ref, remaining_budget

        # If the action is for the first node
        if g.first_node is None:
            # If copy state then make a copy
            if copy_state:
                g_ref = g.copy()
            # otherwise just point to same graph
            else:
                g_ref = g
            # Set the first node of edge as action
            g_ref.first_node = action

            # Update the banned action using the remaining budget for actions
            g_ref.populate_banned_actions(remaining_budget, action_type)
            # Return the graph and budget
            return g_ref, remaining_budget
        # If the action is the 2nd node then add the full edge
        else:
            if action_type == 'add':
                # Add edge from first to 2nd node and set first node to None
                new_g, edge_cost = g.add_edge(g.first_node, action)
            elif action_type == 'remove':
                new_g, edge_cost = g.remove_edge(g.first_node, action)
            elif action_type == 'flip':
                # 2nd action may be None
                new_g, edge_cost = g.flip_node(g.first_node, action)

            new_g.first_node = None
            # Update the budget with cost of move and update banned actions
            updated_budget = remaining_budget - edge_cost
            # Will ban removal action if no nodes to remove
            new_g.populate_banned_actions(updated_budget)
            # Return updated graph and budget
            return new_g, updated_budget

    def step(self, actions, action_types):
        """
        Takes a step in the environment given the actions
        :param actions: List of nodes - corresponds to actions for each graph
        :param action_types: List of action types for each graph - [{'add', 'remove', 'flip'}]
        """
        # Loop through each graph
        curr_objective_function_value = self.objective_function_values[0, :]
        for i in range(len(self.g_list)):
            # If still has budget for action
            if not self.exhausted_budgets[i]:
                # If action is a dummy action then try to apply
                if actions[i] == -1:
                    # Check if we are logging, log dummy action to get remaining budget and current first node
                    if self.logger_instance is not None:
                        self.logger_instance.warn("budget not exhausted but trying to apply dummy action!")
                        self.logger_instance.error(f"the remaining budget: {self.get_remaining_budget(i)}")
                        g = self.g_list[i]
                        self.logger_instance.error(f"first_node selection: {g.first_node}")
                # Get remaining budget
                remaining_budget = self.get_remaining_budget(i)
                # Apply the action and get updated graph
                self.g_list[i], updated_budget = \
                    self.apply_action(self.g_list[i], actions[i], remaining_budget, action_type=action_types[i])
                # Update the budget
                self.used_edge_budgets[i] += (remaining_budget - updated_budget)

                # If odd step and no budget then get final result
                if (self.n_steps % 3 == 2) and (self.g_list[i].banned_actions == self.g_list[i].all_nodes_set):
                    self.exhausted_budgets[i] = True

                # Get final objective function value
                objective_function_value = self.get_objective_function_value(self.g_list[i])
                # If still training then scale the reward
                if self.training:
                    objective_function_value = objective_function_value * self.reward_scale_multiplier
                # Set final value
                self.objective_function_values[-1, i] = objective_function_value

                # Controls reward scaling and intermediate rewards
                if self.exhausted_budgets[i]:
                    reward = \
                        (objective_function_value - self.objective_function_values[0, i]) + \
                        (objective_function_value - curr_objective_function_value[i]) * self.scale_value
                else:
                    reward = \
                        (objective_function_value - curr_objective_function_value[i]) * self.scale_value

                # Defines a pure intermediate objective (including for terminal states)
                # reward = objective_function_value - curr_objective_function_value[i]

                # If change is too small then just set to the 0
                if abs(reward) < self.reward_eps:
                    reward = 0.
                # Set reward in the results array
                self.rewards[i] = reward
                # Update the current objective
                curr_objective_function_value[i] = objective_function_value

        # Increment steps after full loop
        self.n_steps += 1

    def exploratory_actions(self, agent_exploration_policy):
        """
        Selects the exploration actions according the policy
        :param agent_exploration_policy: Exploration policy (function)
        :return: Actions for 1st node and 2nd node
        """
        # Init empty lists for actions
        act_list_t0, act_list_t1 = [], []
        # Loop through graphs
        for i in range(len(self.g_list)):
            # Select first and 2nd node according to the policy on the graph
            first_node, second_node = agent_exploration_policy(i)
            # Append the nodes to the lists
            act_list_t0.append(first_node)
            act_list_t1.append(second_node)
        # Return the explorations actions
        return act_list_t0, act_list_t1

    def get_max_graph_size(self):
        # return: Largest graph size in datasets
        max_graph_size = np.max([g.num_nodes for g in self.g_list])
        return max_graph_size

    def is_terminal(self):
        # Checks if all graphs have finished updating and returns bool
        return np.all(self.exhausted_budgets)

    def get_state_ref(self):
        # Gets the first_node for each graph, and the banned nodes
        cp_first = [g.first_node for g in self.g_list]
        b_list = [g.banned_actions for g in self.g_list]
        at_list = [g.action_type for g in self.g_list]
        # Return the lists in an iterable list
        return zip(self.g_list, cp_first, b_list, at_list)

    def clone_state(self, indices=None):
        # Returns a copy of the states
        if indices is None:
            cp_first = [g.first_node for g in self.g_list][:]
            b_list = [g.banned_actions for g in self.g_list][:]
            return list(zip(deepcopy(self.g_list), cp_first, b_list))
        # Only gets the states for the graphs listed in the indices
        else:
            cp_g_list = []
            cp_first = []
            b_list = []
            at_list = []
            for i in indices:
                cp_g_list.append(deepcopy(self.g_list[i]))
                cp_first.append(deepcopy(self.g_list[i].first_node))
                b_list.append(deepcopy(self.g_list[i].banned_actions))
                at_list.append(deepcopy(self.g_list[i].action_type))
            # Return the zip for the graphs
            return list(zip(cp_g_list, cp_first, b_list, at_list))
