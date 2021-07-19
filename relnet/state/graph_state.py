import math
from copy import deepcopy
import networkx as nx
import random
import numpy as np
import xxhash
import warnings
# Budget for making changes to the graph
budget_eps = 1e-5


class S2VGraph(object):
    # Takes a NetworkX graph and converts to a S2V object
    def __init__(self, g, game_type='majority'):
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

        # Init game type
        self.game_type = game_type
        # Create a dict to store current actions and rewards
        self.actions = {}
        self.rewards = {}
        # Get max rewards - e.g all players get reward of 1.
        self.max_reward = g.number_of_nodes()
        # Set the cost for contributing to the public good - [0, 1]
        self.contribution_cost = 0.2
        # Populate actions and find nash equilibrium
        self.get_actions_from_nx(g)
        g = self.find_nash(game=game_type)

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
        s2v_graph = S2VGraph(nx_graph, game_type=self.game_type)
        return s2v_graph, 1

    def populate_banned_actions(self, budget=None):
        # If a budget is given and the current budget is less than minimum budget then set all actions to banned
        if budget is not None:
            if budget < budget_eps:
                self.banned_actions = self.all_nodes_set
                return
        # otherwise, if no first node selected
        if self.first_node is None:
            # Get unavailable first nodes
            self.banned_actions = self.get_invalid_first_nodes(budget)
        # If a first node is selected then get unavailable end nodes
        else:
            self.banned_actions = self.get_invalid_edge_ends(self.first_node, budget)

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

    def to_networkx(self):
        # Takes the stores edges and makes the NetworkX graph from the edges
        edges = self.convert_edges()
        g = nx.Graph()
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

    def draw_to_file(self, filename):
        # Draws and saves to file
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Set figure parameters and save figure
        fig_size_length = self.num_nodes / 5
        figsize = (fig_size_length, fig_size_length)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
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

    def find_nash(self, game='majority'):
        """
        Find the nash equilibrium by playing best response in rounds
        :param game: Game we are playing - currently implemented {majority, pgg}
        :return: NetworkX graph in equilibrium state
        """
        assert game in ['majority', 'bspgg']
        # Convert to NetworkX
        g = self.apply_actions_2_nx()
        # Set bool for monitoring if an equilibrium has been reached
        equilibrium = False
        # Get the actions for each player
        curr_actions = deepcopy(self.actions)
        # Loop while not converged
        c = 0
        while not equilibrium:
            # Loop through each node and get new actions - in random order each loop
            node_list = list(range(self.num_nodes))
            random.shuffle(node_list)
            for i in node_list:
            # for i in g.nodes:
                # Get the neighbors actions
                neighbors_actions = [g.nodes[n]['action'] for n in g.neighbors(i)]
                # Select the action based on the game being played
                if game == 'majority':
                    # If more are playing 1 then pick 1, Otherwise pick 0
                    if len([a for a in neighbors_actions if a == 1]) > len([a for a in neighbors_actions if a == 0]):
                        # Updates the S2V state and NX graph
                        self.actions[i], g.nodes[i]['action'] = 1., 1.
                    # If less then select less
                    elif len([a for a in neighbors_actions if a == 1]) < len([a for a in neighbors_actions if a == 0]):
                        self.actions[i], g.nodes[i]['action'] = 0., 0.
                    # If equal then random tie breaker
                    else:
                        self.actions[i] = random.randint(0,1)
                        g.nodes[i]['action'] = deepcopy(self.actions[i])
                elif game == 'bspgg':
                    # If any neighboring agents are contributing then chose to defect
                    if 1 in neighbors_actions:
                        self.actions[i], g.nodes[i]['action'] = 0., 0.
                    else:
                        self.actions[i], g.nodes[i]['action'] = 1., 1.
            # If we have reached equilibrium then end the loop
            if self.actions == curr_actions:
                equilibrium = True
            # Otherwise set the current actions to be the selected actions and update the graph
            else:
                curr_actions = deepcopy(self.actions)
                g = self.apply_actions_2_nx()
                c += 1
            # keeps track of a counter to ensure we don't spend too long attempting to reach equilibrium
            if c >= 10000:
                print("No equilibrium found: continuing with the program.")
                break
        # Find the rewards for each agent
        for i in g.nodes:
            neighbors_actions = [g.nodes[n]['action'] for n in g.neighbors(i)]
            act = g.nodes[i]['action']
            if game == 'majority':
                # Reward is proportional to the number of agents playing the same action
                try:
                    self.rewards[i] = (1 / len([a for a in neighbors_actions if a != act])) / self.max_reward
                except:
                    self.rewards[i] = 1. / self.max_reward
            elif game == 'bspgg':
                # Max reward is when action is 0 otherwise is 1 - action/2 (arbitrary cost - could change later)
                if act == 1.:
                    self.rewards[i] = (1. - self.contribution_cost) / self.max_reward
                else:
                    self.rewards[i] = 1. / self.max_reward
        # Return the updated graph with actions and rewards
        return self.apply_actions_2_nx()

    def copy(self):
        # Performs deepcopy of self
        return deepcopy(self)

    def __repr__(self):
        # __repr__ is used to represent a class’s objects as a string - e.g. hash - returned when calling print
        gh = get_graph_hash(self, size=32, include_first=True)
        return f"Graph State with hash {gh}"


def get_graph_hash(g, size=32, include_first=False):
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
    # Returns a hash integer value based on input values
    graph_hash = hash_instance.intdigest()
    return graph_hash
