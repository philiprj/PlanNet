import math
from copy import deepcopy
import networkx as nx
import numpy as np
import xxhash
import warnings
# Budget for making changes to the graph
budget_eps = 1e-5


class S2VGraph(object):
    # Takes a NetworkX graph and converts to a S2V object
    def __init__(self, g):
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
        # Unknown variable atm
        self.dynamic_edges = None

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
        # Convert back to S2V object and return
        s2v_graph = S2VGraph(nx_graph)
        return s2v_graph, 1

    def add_edge_dynamically(self, first_node, second_node):
        """
        TODO: what is this for?
        Adds edge between two specified nodes without changing the underlying NetworkX object
        :param first_node: Start node
        :param second_node: End Node
        :return: 1 to conf
        """
        # Adds to list of dynamic_edges and increments degree count
        self.dynamic_edges.append((first_node, second_node))
        self.node_degrees[first_node] += 1
        self.node_degrees[second_node] += 1
        return 1

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
        :param budget: TODO: this is unused?
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
        # Compare to the list of dynamic_edges (the ones not on NetworkX graph) and update set
        if self.dynamic_edges is not None:
            dynamic_left = [entry[0] for entry in self.dynamic_edges if entry[0] == query_node]
            results.update(dynamic_left)
            dynamic_right = [entry[1] for entry in self.dynamic_edges if entry[1] == query_node]
            results.update(dynamic_right)
        # return the complete set
        return results

    def init_dynamic_edges(self):
        # Init dynamic_edges to empty list
        self.dynamic_edges = []

    def apply_dynamic_edges(self):
        # Applies the stored dynamic_edges to the networkX graph then converts back to S2V object
        nx_graph = self.to_networkx()
        for edge in self.dynamic_edges:
            nx_graph.add_edge(edge[0], edge[1])
        return S2VGraph(nx_graph)

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
