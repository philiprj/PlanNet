import math
from copy import deepcopy
import networkx as nx
from networkx.algorithms.connectivity import is_locally_k_edge_connected, is_k_edge_connected
import random
import numpy as np
import xxhash
import warnings
from relnet.objective_functions.social_welfare import SocialWelfare
# Budget for making changes to the graph
budget_eps = 1e-5


def random_action_init(g):
    """
    takes a graph and initialises the node actions to random in {0,1} and sets rewards = 0 for all nodes
    :param g: NetworkX graph
    :return: NetworkX graph with action and rewards attributes for all nodes
    """
    # Loop elements in the graph
    for i in range(g.number_of_nodes()):
        # Init the actions and agent rewards to zero
        g.nodes[i]['action'] = random.randint(0, 1)
        g.nodes[i]['reward'] = 0.
    return g


class S2VGraph(object):
    # Takes a NetworkX graph and converts to a S2V object
    def __init__(self, g, game_type='majority', br_order=None, enforce_connected=True, institution=False, tax=0.1):
        """
        :param g: NetworkX graph
        """
        # Get the number of nodes, labels and set for quick comparison
        self.num_nodes = g.number_of_nodes()
        self.node_labels = np.arange(self.num_nodes)
        self.all_nodes_set = set(self.node_labels)
        # Unzip the start and end nodes in edges iterable
        x, y = zip(*g.edges())
        # Get the number of edges
        self.num_edges = len(x)
        # Set the edge pais in array where column 0 is 1st, 2 is end node
        self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
        self.edge_pairs[:, 0] = x
        self.edge_pairs[:, 1] = y
        # Converts the 2D array to a 1D - where each row is appended, e.g. [edge1, edge2, ..., edge_n]
        self.edge_pairs = np.ravel(self.edge_pairs)
        # Get the node degree for each node same sorted order as edge_pairs
        self.node_degrees = np.array([deg for (node, deg) in sorted(g.degree(), key=lambda deg_pair: deg_pair[0])])
        # Init node to none
        self.first_node = None

        # Flag for institution
        self.use_inst = institution
        self.tax = tax

        # Set bool for enforcing a connected graph when removing edges
        self.enforce_connected = enforce_connected
        # Init game type
        assert game_type in ['majority', 'bspgg', 'pgg']
        self.game_type = game_type
        # Set actions type {'add', 'remove', 'flip', None} - will be None given S2V only called after applying action
        self.action_type = None
        # Create a dict to store current actions and rewards
        self.actions = {}
        self.rewards = {}
        # Get max rewards - e.g all players get reward of 1.
        self.max_reward = g.number_of_nodes()
        # Set the cost for contributing to the public good - [0, 1] - here homogeneous but we could add variation
        self.cost = [0.7 * 1. for _ in range(self.num_nodes)]
        self.contribution_cost = dict(zip(self.node_labels, self.cost))
        # Set reward scalar for Public Goods Game
        self.reward_scale = 4.
        if game_type == 'pgg':
            self.max_reward = self.max_reward * self.reward_scale
        # Set the order for agents
        if br_order is None:
            self.br_order = list(range(self.num_nodes))
            random.shuffle(self.br_order)
        else:
            self.br_order = br_order

        # Populate actions and find nash equilibrium
        self.get_actions_from_nx(g)
        self.update_rewards(g)
        self.find_nash()

    def add_edge(self, first_node, second_node):
        """
        Adds an edge between two specified nodes.
        :param first_node: Start node
        :param second_node: end node
        :return: S2V graph object with added edge
        """
        # Convert to NetworkX for change - then add edge using inbuilt function
        nx_graph = self.to_networkx()
        nx_graph.add_edge(first_node, second_node)
        # Apply current actions to graph
        nx_graph = self.apply_actions_2_nx(nx_graph)
        # Convert back to S2V object and return
        s2v_graph = S2VGraph(nx_graph,
                             game_type=self.game_type,
                             br_order=self.br_order,
                             enforce_connected=self.enforce_connected,
                             institution=self.use_inst,
                             tax=self.tax)
        return s2v_graph, 1

    def populate_banned_actions(self, budget=None, action=None):
        """
        Populates a set of banned actions, either due to budget or unavailable nodes
        :param budget: budget for making change
        :param action: {'add', 'remove', 'flip'}
        """
        assert action in ['add', 'remove', 'flip', None]
        self.action_type = action
        # If a budget is given and the current budget is less than minimum budget then set all actions to banned
        if budget is not None:
            if budget < budget_eps:
                self.banned_actions = self.all_nodes_set
                return
        if action is None:
            self.banned_actions = set()
            # If all nodes degrees one then can't remove edge
            if len(self.invalid_first_node_removal(budget)) == self.num_nodes:
                self.banned_actions.update('remove')
            return
        # otherwise, if no first node selected
        if self.first_node is None:
            # Get unavailable first nodes
            if action == 'add':
                self.banned_actions = self.get_invalid_first_nodes(budget)
            elif action == 'remove':
                self.banned_actions = self.invalid_first_node_removal(budget)
            # If flip then empty set
            elif action == 'flip':
                self.banned_actions = set()
        # If a first node is selected then get unavailable end nodes
        else:
            # Get unavailable first nodes
            if action == 'add':
                self.banned_actions = self.get_invalid_edge_ends(self.first_node, budget)
            elif action == 'remove':
                self.banned_actions = self.invalid_edge_ends_removal(self.first_node, budget)
            # If flip then add the first node to the set
            elif action == 'flip':
                self.banned_actions = set([self.first_node])

    def get_invalid_first_nodes(self, budget=None):
        """
        :param budget:
        :return: Set of unavailable start nodes which are already connected to all other nodes
        """
        return set([node_id for node_id in self.node_labels if self.node_degrees[node_id] == (self.num_nodes - 1)])

    def get_invalid_edge_ends(self, query_node, budget=None):
        """
        :param query_node: first node selected to add edge from
        :param budget: Unused
        :return: Set of invalid nodes to connect the first node to
        """
        # Create set of invalid nodes and add current 1st node to it
        results = set()
        results.add(query_node)
        # Reshape node list into array
        existing_edges = self.edge_pairs.reshape(-1, 2)
        # Add the already connected nodes to the set of unavailable nodes
        existing_left = existing_edges[existing_edges[:, 0] == query_node]
        results.update(np.ravel(existing_left[:, 1]))
        existing_right = existing_edges[existing_edges[:, 1] == query_node]
        results.update(np.ravel(existing_right[:, 0]))
        # return the complete set
        return results

    def remove_edge(self, first_node, second_node):
        """
        Removes an edge between two specified nodes.
        :param first_node: Start node
        :param second_node: end node
        :return: S2V graph object with added edge
        """
        # Convert to NetworkX for change - then add edge using inbuilt function
        nx_graph = self.to_networkx()
        nx_graph.remove_edge(first_node, second_node)
        # Apply current actions to graph
        nx_graph = self.apply_actions_2_nx(nx_graph)
        # Convert back to S2V object and return
        s2v_graph = S2VGraph(nx_graph,
                             game_type=self.game_type,
                             br_order=self.br_order,
                             enforce_connected=self.enforce_connected,
                             institution=self.use_inst,
                             tax=self.tax)
        return s2v_graph, 1

    def flip_node(self, first_node, second_node=None):
        """
        Flips the actions of a given node
        :param first_node: Start node
        :param second_node: Optional to change two nodes in 1 move
        :return: S2V graph object with flipped action/s
        """
        # Converts to NetworkX for change
        nx_graph = self.to_networkx()
        nx_graph = self.apply_actions_2_nx(nx_graph)
        # Flip the binary action - if 1 -> 0, if 0 -> abs(-1) = 1
        nx_graph.nodes[first_node]['action'] = abs(nx_graph.nodes[first_node]['action'] - 1)
        if second_node is not None:
            nx_graph.nodes[second_node]['action'] = abs(nx_graph.nodes[second_node]['action'] - 1)
        s2v_graph = S2VGraph(nx_graph,
                             game_type=self.game_type,
                             br_order=self.br_order,
                             enforce_connected=self.enforce_connected,
                             institution=self.use_inst,
                             tax=self.tax)
        return s2v_graph, 1

    def invalid_first_node_removal(self, budget):
        """
        Node is invalid for removal if removing it would disconnect the graph
        :return: set of invalid nodes with only one edge
        """
        # If we are enforcing connection then remove nodes with only one edge, or connected to nodes with 1 edge
        if self.enforce_connected:
            # invalid = set([node_id for node_id in self.node_labels if self.node_degrees[node_id] == 1])
            # valid = self.all_nodes_set - invalid
            # # Update list to include ids with no valid end nodes
            # invalid.update([node_id for node_id in valid if len(self.invalid_edge_ends_removal(node_id)) == self.num_nodes])

            existing_edges = self.edge_pairs.reshape(-1, 2)
            g = self.to_networkx()
            # Get a set of valid nodes for removal
            valid = set()
            for i in range(existing_edges.shape[0]):
                s, t = existing_edges[i, 0], existing_edges[i, 1]
                if (s in valid) and (t in valid):
                    continue
                elif is_locally_k_edge_connected(g, s, t, k=2):
                    valid.update([s, t])
            invalid = self.all_nodes_set - valid
            return invalid
        else:
            invalid = set([node_id for node_id in self.node_labels if self.node_degrees[node_id] == 0])
        return invalid

    def invalid_edge_ends_removal(self, query_node, budget=None):
        """
        :param query_node: first node selected to remove edge from
        :param budget: Unused
        :return: Set of invalid nodes to connect the first node to
        """
        # Create set of invalid nodes and add current 1st node to it
        valid = set()
        # Reshape node list into array
        existing_edges = self.edge_pairs.reshape(-1, 2)
        # Add the connected nodes to the set of available nodes
        existing_left = existing_edges[existing_edges[:, 0] == query_node]
        valid.update(np.ravel(existing_left[:, 1]))
        existing_right = existing_edges[existing_edges[:, 1] == query_node]
        valid.update(np.ravel(existing_right[:, 0]))
        results = self.all_nodes_set - valid
        results.add(query_node)

        if self.enforce_connected:
            # Don't remove node if would disconnect from the graph
            # results.update([node_id for node_id in self.node_labels if self.node_degrees[node_id] == 1])
            g = self.to_networkx()
            results.update([node for node in valid if not is_locally_k_edge_connected(g, query_node, node, k=2)])
        return results

    def to_networkx(self):
        # Takes the stores edges and makes the NetworkX graph from the edges
        edges = self.convert_edges()
        g = nx.Graph()
        # Add nodes then add edges
        g.add_nodes_from(self.node_labels)
        g.add_edges_from(edges)
        return g

    def convert_edges(self):
        # Convert edges array to 2D array of [node 1, node2]
        return np.reshape(self.edge_pairs, (self.num_edges, 2))

    def display(self, ax=None):
        # Draws the graphs from the S2V object using NetworkX
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            nx_graph = self.to_networkx()
            nx.draw_shell(nx_graph, with_labels=True, ax=ax)

    def display_with_positions(self, node_positions, ax=None):
        # Draws with node positions
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            nx_graph = self.to_networkx()
            nx.draw(nx_graph, pos=node_positions, with_labels=True, ax=ax)

    def display_with_actions(self, ax=None):
        # Draws the graphs with actions from the S2V object using NetworkX
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            nx_graph = self.to_networkx()
            nx_graph = self.apply_actions_2_nx(nx_graph)
            label_dict = deepcopy(self.actions)
            # nx.draw_shell(nx_graph, labels=label_dict, with_labels=True, ax=ax, font_weight='bold')
            # nx.draw(nx_graph, labels=label_dict, with_labels=True, ax=ax, font_weight='bold')
            my_pos = nx.spring_layout(nx_graph, seed=42)
            nx.draw(nx_graph, pos=my_pos, labels=label_dict, with_labels=True, ax=ax, font_weight='bold')

    def draw_to_file(self, filename, display_actions=True):
        # Draws and saves to file
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Set figure parameters and save figure
        fig_size_length = self.num_nodes / 5
        figsize = (fig_size_length, fig_size_length)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_title("Social Welfare: {}".format(SocialWelfare(self)))
        if display_actions:
            self.display_with_actions(ax=ax)
        else:
            self.display(ax=ax)
        fig.savefig(filename)
        plt.close()

    def get_adjacency_matrix(self):
        # Catches the warnings that come NetworkX can cause
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # Converts to networkx and generates adjacency matrix
            nx_graph = self.to_networkx()
            adj_matrix = np.asarray(nx.convert_matrix.to_numpy_matrix(nx_graph, nodelist=self.node_labels))
        return adj_matrix

    def apply_actions_2_nx(self, g=None):
        """
        Applies the actions to a NetworkX graph
        :param g: Takes a NetworkX graph to make changes to - if None then applies changes to self.
        :return: NetworkX graph with actions and reward attributes applied
        """
        if g is None:
            g = self.to_networkx()
        # Loop elements in the graph
        for i in self.node_labels:
            # Set the actions and rewards from the dict of node labels
            g.nodes[i]['action'] = self.actions[i]
            g.nodes[i]['reward'] = self.rewards[i]
        return g

    def get_actions_from_nx(self, g):
        """
        Takes a Networkx graph with node attributes {action, reward} and gets the attributes
        :param g: Networkx graph with node attributes {action, reward}
        """
        assert ('action' in g.nodes[self.node_labels[0]]) and ('reward' in g.nodes[self.node_labels[0]])

        for i in self.node_labels:
            self.actions[i] = g.nodes[i]['action']
            self.rewards[i] = g.nodes[i]['reward']

    def update_rewards(self, g):
        """
        Updates the rewards for the internal state
        :param g: NetworkX graph
        """
        # Find the rewards for each agent
        for i in g.nodes:
            neighbors_actions = [g.nodes[n]['action'] for n in g.neighbors(i)]
            act = g.nodes[i]['action']
            # Majority game
            if self.game_type == 'majority':
                # Reward is proportional to the number of agents playing the same action
                try:
                    self.rewards[i] = \
                        len([a for a in neighbors_actions if a == act]) / len([a for a in neighbors_actions])
                except ZeroDivisionError:
                    self.rewards[i] = 0.0
                # If using institution then take tax from defectors
                if self.use_inst:
                    if act == 0.0:
                        self.rewards[i] = self.rewards[i] - self.tax
            # Best shot public goods game
            elif self.game_type == 'bspgg':
                # Max reward is when action is 0 otherwise is 1 - action/2 (arbitrary cost - could change later)
                if act == 1.:
                    self.rewards[i] = (1. - self.contribution_cost[i])
                elif 1 not in neighbors_actions:
                    self.rewards[i] = 0.
                else:
                    self.rewards[i] = 1.
            # Public goods game
            elif self.game_type == 'pgg':
                # Reward is proportional to the number of contributors minus cost of agent contribution
                pgg_c = sum([g.nodes[n]['action'] * self.contribution_cost[n] for n in g.neighbors(i)])
                self.rewards[i] = (((pgg_c + act) * self.reward_scale) / (len(neighbors_actions) + 1)) - \
                                  (self.contribution_cost[i] * act)

    def find_nash(self):
        """
        Find the nash equilibrium by playing best response in rounds
        :param game: Game we are playing - currently implemented {majority, pgg}
        :return: NetworkX graph in equilibrium state
        """
        # Convert to NetworkX
        g = self.apply_actions_2_nx()
        # Set bool for monitoring if an equilibrium has been reached
        equilibrium = False
        # Get the actions for each player
        curr_actions = deepcopy(self.actions)
        # Loop while not converged
        c = 0

        while not equilibrium:
            # Loop through each node and get new actions
            for i in self.br_order:
                # Get the neighbors actions
                neighbors_actions = [g.nodes[n]['action'] for n in g.neighbors(i)]
                # Get the proportion of contributors and defectors
                contri = len([a for a in neighbors_actions if a == 1])
                defect = len(neighbors_actions) - contri
                neighborhood_size = contri + defect + 1

                # Select the action based on the game being played
                if self.game_type == 'majority':
                    if self.use_inst:
                        action = self.institution(len(neighbors_actions), contri)
                        self.actions[i], g.nodes[i]['action'] = action, action
                    else:
                        # If more are playing 1 then pick 1, Otherwise pick 0
                        if contri > defect:
                            # Updates the S2V state and NX graph
                            self.actions[i], g.nodes[i]['action'] = 1., 1.
                        # If less then select less
                        elif contri < defect:
                            self.actions[i], g.nodes[i]['action'] = 0., 0.
                        # If equal then random tie breaker
                        else:
                            self.actions[i] = random.randint(0, 1)
                            g.nodes[i]['action'] = deepcopy(self.actions[i])
                elif self.game_type == 'bspgg':
                    # If any neighboring agents are contributing then chose to defect
                    if 1 in neighbors_actions:
                        self.actions[i], g.nodes[i]['action'] = 0., 0.
                    else:
                        self.actions[i], g.nodes[i]['action'] = 1., 1.
                elif self.game_type == 'pgg':
                    pgg_c = sum([g.nodes[n]['action'] * self.contribution_cost[n] for n in g.neighbors(i)])
                    # If more valuable to contribute then contribute, otherwise do not contribute
                    act1 = (((pgg_c + self.contribution_cost[i]) * self.reward_scale) / neighborhood_size) - \
                           self.contribution_cost[i]
                    act0 = (pgg_c * self.reward_scale) / neighborhood_size
                    if act1 >= act0:
                        self.actions[i], g.nodes[i]['action'] = 1., 1.
                    else:
                        self.actions[i], g.nodes[i]['action'] = 0., 0.

            # If we have reached equilibrium then end the loop
            if self.actions == curr_actions:
                equilibrium = True
            # Otherwise set the current actions to be the selected actions and update the graph
            else:
                curr_actions = deepcopy(self.actions)
                # g = self.apply_actions_2_nx()
                c += 1
            # keeps track of a counter to ensure we don't spend too long attempting to reach equilibrium
            if c >= 20000:
                print("No equilibrium found: continuing with the program.")
                break
        # Update the rewards before returning
        self.update_rewards(g)
        # Return the updated graph with actions and rewards
        return self.apply_actions_2_nx()

    def institution(self, n_neighbours: int, n_contributors: int) -> float:
        """
        :param n_neighbours: int number of neighbors
        :param n_contributors: int number of contributing neighbors
        :param tax: float number of tax
        :return: {1, 0} action depending on which is more valuable
        """
        # Get the fraction of contributing agents
        c_reward = n_contributors / n_neighbours
        d_reward = 1 - c_reward - self.tax
        if c_reward > d_reward:
            return 1.
        elif c_reward < d_reward:
            return 0.
        else:
            return random.randint(0, 1)

    def randomise_actions(self):
        # Takes the graph structure and applies random actions again - initialising a new S2V state object
        nx_graph = self.to_networkx()
        nx_graph = random_action_init(nx_graph)
        s2v_graph = S2VGraph(nx_graph,
                             game_type=self.game_type,
                             br_order=self.br_order,
                             enforce_connected=self.enforce_connected,
                             institution=self.use_inst,
                             tax=self.tax)
        return s2v_graph

    def copy(self):
        # Performs deepcopy of self
        return deepcopy(self)

    def __repr__(self):
        # __repr__ is used to represent a class’s objects as a string - e.g. hash - returned when calling print
        gh = get_graph_hash(self, size=32, include_first=True)
        return f"Graph State with hash {gh}"


def get_graph_hash(g, size=32, include_first=False, include_actions=True):
    """
    Get a hash code for the graph
    :param g: Graph
    :param size: Hash representation bit-size
    :param include_first: Includes first node in the hash
    :return: Return the hash code
    """
    # If 32 use 32 bit storage, otherwise 64, or throw error
    if size == 32:
        hash_instance = xxhash.xxh32()
    elif size == 64:
        hash_instance = xxhash.xxh64()
    else:
        raise ValueError("only 32 or 64-bit hashes supported.")
    # If we want to include first node - selected for action - in hash
    if include_first:
        # Update with either first selected node in one hot, or just zeros
        if g.first_node is not None:
            hash_instance.update(np.array([g.first_node]))
        else:
            hash_instance.update(np.zeros(g.num_nodes))
    # Update with edge pairs
    hash_instance.update(g.edge_pairs)
    if include_actions:
        hash_instance.update(np.array([g.actions.values]))
    # Returns a hash integer value based on input values
    graph_hash = hash_instance.intdigest()
    return graph_hash
