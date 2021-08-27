import json
import math
# abc - Abstract Base Class -
from abc import ABC, abstractmethod
from pathlib import Path
import networkx as nx
from relnet.evaluation.file_paths import FilePaths
from relnet.state.graph_state import S2VGraph, random_action_init
from relnet.utils.config_utils import get_logger_instance


class NetworkGenerator(ABC):
    """
    Abstract parent class for graph generators
    """
    # Forces no unconnected elements
    enforce_connected = True

    def __init__(self, store_graphs=False,
                 graph_storage_root=None,
                 logs_file=None,
                 game_type='majority',
                 enforce_connected=True,
                 institution=False,
                 tax=0.1):
        """
        :param store_graphs: Bool - Stores the generated graph instances
        :param graph_storage_root: Path for storing the graphs
        :param logs_file: File for logging graph data
        """
        super().__init__()
        # Path for storing graphs
        self.store_graphs = store_graphs
        if self.store_graphs:
            self.graph_storage_root = graph_storage_root
            self.graph_storage_dir = graph_storage_root / game_type / self.name
            self.graph_storage_dir.mkdir(parents=True, exist_ok=True)
        # Log file
        if logs_file is not None:
            self.logger_instance = get_logger_instance(logs_file)
        else:
            self.logger_instance = None

        # Set the game type
        self.game_type = game_type
        self.enforce_connected = enforce_connected
        self.use_inst = institution
        self.tax = tax

    def generate(self, gen_params, random_seed):
        """
        Generate a single graph
        :param gen_params: Parameters for the graph - number of nodes/edges
        :param random_seed: Random seed for generating the graph - does this make it deterministic
        :return: A graph of desired type
        """
        # Store if required
        if self.store_graphs:
            # Get graphs from stored graphs
            filename = self.get_data_filename(gen_params, random_seed)
            filepath = self.graph_storage_dir / filename
            should_create = True
            # If the file path exists then instance from graphml and get state from S2V - if not then create the graph
            if filepath.exists():
                try:
                    instance = self.read_graphml_with_ordered_int_labels(filepath)
                    state = self.post_generate_instance(instance)
                    should_create = False
                except Exception:
                    should_create = True
            # If we need to create the graph then create using parameters and get the state
            if should_create:
                instance = self.generate_instance(gen_params, random_seed)
                state = self.post_generate_instance(instance)
                # Write the networkX graph to graphml file
                nx.readwrite.write_graphml(instance, filepath.resolve())
                # Draws S2V graph to file
                drawing_filename = self.get_drawing_filename(gen_params, random_seed)
                drawing_path = self.graph_storage_dir / drawing_filename
                state.draw_to_file(drawing_path)
        # If not using a file then generate graph instance and convert to S2V
        else:
            instance = self.generate_instance(gen_params, random_seed)
            state = self.post_generate_instance(instance)
        # Return S2V state
        return state

    @staticmethod
    def read_graphml_with_ordered_int_labels(filepath):
        """
        Reads the graphml file and generates graph instance
        :param filepath: Path to graphml file
        :return: NetworkX graph
        """
        # Read the file, get the number of nodes and label map - relabel nodes
        instance = nx.readwrite.read_graphml(filepath.resolve())
        num_nodes = len(instance.nodes)
        relabel_map = {str(i): i for i in range(num_nodes)}
        nx.relabel_nodes(instance, relabel_map, copy=False)
        # Initialises the graph, populate it with nodes then edges and return
        G = nx.Graph()
        G.add_nodes_from(sorted(instance.nodes(data=True)))
        G.add_edges_from(instance.edges(data=True))
        G = random_action_init(G)
        return G

    def generate_many(self, gen_params, random_seeds):
        # Returns many graphs using a list of random seeds, one for each graph
        return [self.generate(gen_params, random_seed) for random_seed in random_seeds]

    @abstractmethod
    def generate_instance(self, gen_params, random_seed):
        pass

    @abstractmethod
    def post_generate_instance(self, instance):
        pass

    @staticmethod
    def get_data_filename(gen_params, random_seed):
        # Creates a file name from the parameters used
        n = gen_params['n']

        filename = f"{n}-{random_seed}.graphml"
        return filename

    @staticmethod
    def get_drawing_filename(gen_params, random_seed):
        # Creates a drawing file name from the parameters used
        n = gen_params['n']
        filename = f"{n}-{random_seed}.png"
        return filename

    @staticmethod
    def compute_number_edges(n, edge_percentage):
        # Returns the integer number of edges from the total number of edges available and the edge percentage
        total_possible_edges = (n * (n - 1)) / 2
        return int(math.ceil((total_possible_edges * edge_percentage / 100)))

    @staticmethod
    def compute_edges_changes(m, edge_percentage):
        # Returns the integer number of edges from the total number of edges available and the edge percentage
        return int(math.ceil((m * edge_percentage / 100)))

    @staticmethod
    def construct_network_seeds(num_train_graphs, num_validation_graphs, num_test_graphs):
        # Random seeds for network generation, each unique in ascending order
        train_seeds = list(range(0, num_train_graphs))
        validation_seeds = list(range(num_train_graphs, num_train_graphs + num_validation_graphs))
        offset = num_train_graphs + num_validation_graphs
        test_seeds = list(range(offset, offset + num_test_graphs))
        return train_seeds, validation_seeds, test_seeds


class OrdinaryGraphGenerator(NetworkGenerator, ABC):
    # Extends the NetworkGenerator class to create a S2V object
    def post_generate_instance(self, instance):
        """
        Takes the networkX instance of a graph and converts to a S2V object
        :param instance: NetworkX graph instance
        :return: S2V object for easier manipulation with S2V
        """
        state = S2VGraph(instance,
                         self.game_type,
                         enforce_connected=self.enforce_connected,
                         institution=self.use_inst,
                         tax=self.tax)
        # Builds a list of banned actions for the graph for quick access
        state.populate_banned_actions()
        return state


class GNMNetworkGenerator(OrdinaryGraphGenerator):
    """
    Creates a random network with a set number of nodes and edges
    """
    name = 'erdos_renyi'
    num_tries = 10000

    def generate_instance(self, gen_params, random_seed):
        """
        Creates Graph
        :param gen_params: Parameters for creating the graph in a dict
        :param random_seed: Integer value for random_seed to deterministically create graphs
        :return: NetworkX graph
        """
        # Get edges and vertices from parameters
        number_vertices = gen_params['n']
        number_edges = gen_params['m']
        # If we can have disconnections then simply generate the required networks
        if not self.enforce_connected:
            random_graph = nx.generators.random_graphs.gnm_random_graph(number_vertices, number_edges, seed=random_seed)
            return random_action_init(random_graph)
        # Otherwise attempt to make graphs with no breaks, abort if break and try again
        else:
            for try_num in range(0, self.num_tries):
                random_graph = nx.generators.random_graphs.gnm_random_graph(number_vertices,
                                                                            number_edges,
                                                                            seed=(random_seed + (try_num * 1000)))
                if nx.is_connected(random_graph):
                    return random_action_init(random_graph)
                else:
                    continue
            raise ValueError("Maximum number of tries exceeded, giving up...")

    @staticmethod
    def get_data_filename(gen_params, random_seed):
        # Creates a file name from the parameters used
        n, m = gen_params['n'], gen_params['m']
        filename = f"{n}-{m}-{random_seed}.graphml"
        return filename

    @staticmethod
    def get_drawing_filename(gen_params, random_seed):
        # Creates a drawing file name from the parameters used
        n, m = gen_params['n'], gen_params['m']
        filename = f"{n}-{m}-{random_seed}.png"
        return filename


class BANetworkGenerator(OrdinaryGraphGenerator):
    # Name of approach
    name = 'barabasi_albert'

    def generate_instance(self, gen_params, random_seed):
        """
        Creates Barabasi-Albert Graph
        :param gen_params: Parameters for creating the graph in a dict
        :param random_seed: Integer value for random_seed to deterministically create graphs
        :return: NetworkX graph using Barabasi Albert structure
        """
        # Gets parameters and creates the graph
        n, m = gen_params['n'], gen_params['m_ba']
        ba_graph = nx.generators.random_graphs.barabasi_albert_graph(n, m, seed=random_seed)
        # Init the actions and rewards and return
        ba_graph = random_action_init(ba_graph)
        return ba_graph

    @staticmethod
    def get_data_filename(gen_params, random_seed):
        # Creates a file name from the parameters used
        n, m = gen_params['n'], gen_params['m_ba']
        filename = f"{n}-{m}-{random_seed}.graphml"
        return filename

    @staticmethod
    def get_drawing_filename(gen_params, random_seed):
        # Creates a drawing file name from the parameters used
        n, m = gen_params['n'], gen_params['m_ba']
        filename = f"{n}-{m}-{random_seed}.png"
        return filename


class WSNetworkGenerator(OrdinaryGraphGenerator):
    # Name of approach
    name = 'watts_strogatz'

    def generate_instance(self, gen_params, random_seed):
        """
        Creates Watts Strogatz Graph
        :param gen_params: Parameters for creating the graph in a dict
        :param random_seed: Integer value for random_seed to deterministically create graphs
        :return: NetworkX graph using  Watts Strogatz structure - captures small-world property found in many social
        and biological networks, which generates networks with high clustering coefficient
        """
        # Gets parameters and creates the graph
        n, k, p = gen_params['n'], gen_params['k_ws'], gen_params['p_ws']
        ws_graph = nx.generators.random_graphs.connected_watts_strogatz_graph(n, k, p, seed=random_seed)
        # Init the actions and rewards and return
        ws_graph = random_action_init(ws_graph)
        return ws_graph

    @staticmethod
    def get_data_filename(gen_params, random_seed):
        # Creates a file name from the parameters used
        n, k, p = gen_params['n'], gen_params['k_ws'], gen_params['p_ws']
        filename = f"{n}-{k}-{p}-{random_seed}.graphml"
        return filename

    @staticmethod
    def get_drawing_filename(gen_params, random_seed):
        # Creates a drawing file name from the parameters used
        n, k, p = gen_params['n'], gen_params['k_ws'], gen_params['p_ws']
        filename = f"{n}-{k}-{p}-{random_seed}.png"
        return filename
