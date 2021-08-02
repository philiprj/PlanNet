import sys
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from relnet.agent.fnapprox.gnn_regressor import GNNRegressor
from relnet.utils.config_utils import get_device_placement

# Add the structure2vec to the path and imports pytoch_util - weight_init from it
sys.path.append('/usr/lib/pytorch_structure2vec/s2v_lib')
from pytorch_util import weights_init


def jmax(arr, prefix_sum):
    """
    # Selects the action with the highest value for each graph
    :param arr: Array of action values
    :param prefix_sum: A prefix sum is a sequence of partial sums of a given sequence.
    For example, the cumulative sums of the sequence {a, b, c, …} are a, a+b, a+b+c and so on. this is used to track
    the position in the sequence for indexing.
    :return: Tensor containing the selected actions_tensor and values tensor
    """
    # List for storing actions and respective value
    actions = []
    values = []
    # Loop through range of prefix sum
    for i in range(len(prefix_sum[0:, ])):
        # If first index then start at 0 and get first prefix sum value
        if i == 0:
            start_index = 0
            end_index = prefix_sum[i]
        # otherwise start at i-1 and move forward one value in prefix sum - corresponds to number of nodes in graph
        else:
            start_index = prefix_sum[i - 1]
            end_index = prefix_sum[i]
        # Index the array values between the start and end points
        arr_vals = arr[start_index:end_index]
        # Select the action with the highest value
        act = np.argmax(arr_vals)
        # Index its value
        val = arr_vals[act]
        # Append the action and value to the list
        actions.append(act)
        values.append(val)
    # Convert to a torch tensor
    actions_tensor = torch.LongTensor(actions)
    values_tensor = torch.Tensor(values)
    # Return these tensors
    return actions_tensor, values_tensor


def greedy_actions(q_values, v_p, banned_list):
    """
    Takes the q values and non-eligible actions and returns the
    :param q_values: Q values for each action in the given state
    :param v_p: Prefix sum telling us the position
    :param banned_list: List of banned actions
    :return: Return the tensor of selected actions and values
    """
    # Define the offset to be 0
    offset = 0
    # Init banned actions to be empty list
    banned_acts = []
    # Convert prefix sum to numpy -> CPU
    prefix_sum = v_p.data.cpu().numpy()
    # Loop through prefix sum and converts banned actions to positions in list
    for i in range(len(prefix_sum)):
        # Check the banned list is not empty or current value is not empty for this graph
        if(banned_list is not None) and (banned_list[i] is not None):
            # Loop values in the banned list for this graph
            for j in banned_list[i]:
                # Append the banned actions plus the current offset to account for the graph position
                banned_acts.append(offset + j)
        # Update offset for current position in the graph list
        offset = prefix_sum[i]
    # Data removes the computation history/graph and clones to a new place in memory
    q_values = q_values.data.clone()
    # Resize the array to be the len of q values (1D)
    q_values.resize_(len(q_values))
    # Convert the banned acts to tensor and get GPU/CPU details
    banned = torch.LongTensor(banned_acts)
    device_placement = get_device_placement()
    # Send to GPU if available
    if device_placement == 'GPU':
        banned = banned.cuda()
    # If there are banned actions
    if len(banned_acts):
        # Tells us the smallest number representation in the tensor
        min_tensor = torch.tensor(float(np.finfo(np.float32).min))
        if device_placement == 'GPU':
            min_tensor = min_tensor.cuda()
        # Fills the tensor with min values in the banned indices
        q_values.index_fill_(0, banned, min_tensor)
    # Send to CPU and Numpy and return the Greedy actions
    q_vals_cpu = q_values.data.cpu().numpy()
    return jmax(q_vals_cpu, prefix_sum)


class QNet(GNNRegressor, nn.Module):
    def __init__(self, hyperparams, s2v_module, action_selector=False):
        super().__init__(hyperparams, s2v_module)
        # Takes the graph embedding dimensions
        embed_dim = hyperparams['latent_dim']
        # Defined a 2 layer Fully Connected Network with a single output
        self.linear_1 = nn.Linear(embed_dim * 2, hyperparams['hidden'])
        self.linear_out = nn.Linear(hyperparams['hidden'], 1)

        # Flag - true if this net used for selecting actions
        self.action_selector = action_selector
        # Defines a 4 output layer for action selection, tweaked input to only take graph embedding, no node embedding
        self.linear_action_1 = nn.Linear(embed_dim, hyperparams['hidden'])
        self.linear_action_out = nn.Linear(hyperparams['hidden'], 3)

        # Initialises the weights
        weights_init(self)
        # Features per nodes for S2V - no feats for edges - added features for actions and rewards - one hot for
        # action and node selection
        self.num_node_feats = 8
        self.num_edge_feats = 0
        # If no preset S2V module then define one
        if s2v_module is None:
            # Inherits from GNNRegressor which has a self.model parameter creating the S2V model
            self.s2v = self.model(latent_dim=embed_dim,
                                  output_dim=0,
                                  num_node_feats=self.num_node_feats,
                                  num_edge_feats=self.num_edge_feats,
                                  max_lv=hyperparams['max_lv'])
        else:
            self.s2v = s2v_module

    @staticmethod
    def add_offset(actions, v_p):
        """
        Offsets the actions values by a set amount determined by prefix_sum
        :param actions: List of available actions
        :param v_p: Prefix sum keeps a running tally of nodes in each graph in batch for indexing
        :return: Shifted array of actions
        """
        # Sends to the cpu and create list for storing shifted actions
        prefix_sum = v_p.data.cpu().numpy()
        shifted = []
        # Loops through values - if 1st value set offset to 0, otherwise offset is prefix_sum value?
        for i in range(len(prefix_sum)):
            if i > 0:
                offset = prefix_sum[i - 1]
            else:
                offset = 0
            shifted.append(actions[i] + offset)
        return shifted

    @staticmethod
    def rep_global_embed(graph_embed, v_p):
        """
        Takes the graph embedding and alters?
        :param graph_embed: Graph embedding
        :param v_p: Torch tensor containing running sum of nodes in batch
        :return: Altered graph embedding with one embedding for each node in each graph
        """
        prefix_sum = v_p.data.cpu().numpy()
        rep_idx = []
        # Offsets n_nodes by diff in prefix_sum then
        for i in range(len(prefix_sum)):
            if i == 0:
                n_nodes = prefix_sum[i]
            else:
                n_nodes = prefix_sum[i] - prefix_sum[i - 1]
            # Appends n_nodes times i value to rep index
            rep_idx += [i] * n_nodes
        # Converts to torch tensor with autograd - sends to GPU if available
        rep_idx = Variable(torch.LongTensor(rep_idx))
        if get_device_placement() == 'GPU':
            rep_idx = rep_idx.cuda()
        # Tensor indexes graph_embed along dimension 0 using entries in rep_idx
        graph_embed = torch.index_select(graph_embed, 0, rep_idx)
        return graph_embed

    def prepare_node_features(self, batch_graph, picked_nodes, action_types=None):
        """
        Gets tensor representation of node features - [1,0] if available for selection, [0,1] if already selected
        :param batch_graph: graphs in tuple, sampling from buffer then is sample size tuple
        :param picked_nodes: The nodes that have been picked on the fist step - if first step then none picked
        :param action_types: The list of actions selected for each graph - if step 1 the actions are None
        :return: Note feature tensor and prefix_sum?
        """
        # Number of nodes counter, prefix_sum list, picked nodes
        n_nodes = 0
        prefix_sum = []
        # Loop through the batch of graphs
        for i in range(len(batch_graph)):
            # Increment the current number of nodes seen
            n_nodes += batch_graph[i].num_nodes
            # Append the num nodes ongoing sum to the list
            prefix_sum.append(n_nodes)

        # Init a tensor of zeros of size total number of nodes x number of features
        node_feat = torch.zeros(n_nodes, self.num_node_feats)

        # Create a array of Nones if list is None
        if picked_nodes is None:
            picked_nodes = [None for _ in range(len(batch_graph))]
        # counter for adding values to Array using graph size
        cum_sum = 0
        # Loops through each graph in the tuple - keeping track of the batch value
        for i, graph in enumerate(batch_graph):
            graph_size = graph.num_nodes
            """
            Get the action type from the list 
                - set all initial values in one-hot for action to 1
                - set the selected value to 0
                - set the next value to 1 to show it is selected
                - If we have not selected an action for this graph then keep all as 0
            """
            if action_types[i] == 'add':
                node_feat.numpy()[cum_sum:cum_sum + graph_size, 0] = 1.0
                if picked_nodes[i] is not None:
                    assert (picked_nodes[i] >= 0) and (picked_nodes[i] < graph.num_nodes)
                    node_feat.numpy()[picked_nodes[i], 0] = 0.0
                    node_feat.numpy()[picked_nodes[i], 1] = 1.0
            elif action_types[i] == 'remove':
                node_feat.numpy()[cum_sum:cum_sum + graph_size, 2] = 1.0
                if picked_nodes[i] is not None:
                    assert (picked_nodes[i] >= 0) and (picked_nodes[i] < graph.num_nodes)
                    node_feat.numpy()[picked_nodes[i], 2] = 0.0
                    node_feat.numpy()[picked_nodes[i], 3] = 1.0
            elif action_types[i] == 'flip':
                node_feat.numpy()[cum_sum:cum_sum + graph_size, 4] = 1.0
                if picked_nodes[i] is not None:
                    assert (picked_nodes[i] >= 0) and (picked_nodes[i] < graph.num_nodes)
                    node_feat.numpy()[picked_nodes[i], 4] = 0.0
                    node_feat.numpy()[picked_nodes[i], 5] = 1.0
            # If the action is none of these {None} then keep the first 6 values as 0s
            # Loop through nodes in the graph
            for n in range(graph_size):
                # Index into graph in array using cumulative sum, then index into the node position
                # Set column 2 to actions and 3 to rewards
                node_feat.numpy()[cum_sum + n, 6] = graph.actions[n]
                node_feat.numpy()[cum_sum + n, 7] = graph.rewards[n]
            cum_sum += graph_size

        # Return the features and running count of graph nodes in the batch
        return node_feat, torch.LongTensor(prefix_sum)

    def forward(self, states, actions, greedy_acts=False):
        """
        :param states: Tuple of (graph states, selected nodes, banned action list, action_types)
        :param actions: Tensor of actions
        :param greedy_acts: Bool to tell us if greedy actions are used
        :return: Actions, raw prediction, sum list
        """
        # Unzip the states
        batch_graph, picked_nodes, banned_list, action_types = zip(*states)
        # Gets the prefix _sum and node features for the batch of graphs
        node_feat, prefix_sum = self.prepare_node_features(batch_graph, picked_nodes, action_types)
        # Gets the graph embeddings and prefix sum
        embed, graph_embed, prefix_sum = self.run_s2v_embedding(batch_graph, node_feat, prefix_sum)
        # Convert prefix sum to torch tensor
        prefix_sum = Variable(prefix_sum)
        # If we are not working with first action selection layer.
        if not self.action_selector:
            if actions is None:
                # Repeats the state embeddings for number of actions - e.g. number of nodes
                # Stack embeddings so we have one for each action available
                graph_embed = self.rep_global_embed(graph_embed, prefix_sum)
            else:
                # adds an offset to pick the nodes from the graph we are looking at
                shifted = self.add_offset(actions, prefix_sum)
                embed = embed[shifted, :]
            # Concatenate the embedded and graph embedded along 1st dim
            embed_s_a = torch.cat((embed, graph_embed), dim=1)
            # Apply layers and relu activation
            embed_s_a = F.relu(self.linear_1(embed_s_a))
            # Estimate the Q(s,a) for each available action - batches each available action together
            raw_pred = self.linear_out(embed_s_a)
            # If taking greedy actions then take raw prediction and pick actions
            if greedy_acts:
                actions, _ = greedy_actions(raw_pred, prefix_sum, banned_list)
            # Returns the actions, raw prediction, and prefix sum
            return actions, raw_pred, prefix_sum
        # If we are on the first layer selecting actions then use 4 outputs (or 3/2 if using add_remove/flip)
        else:
            if actions is None:
                # Seems the embed gives embedding for each node - graph embed is for whole - since action choice
                embed_s_a = F.relu(self.linear_action_1(graph_embed))

                raw_pred = self.linear_action_out(embed_s_a)
                q_values = F.softmax(raw_pred, dim=1)

                # New
                actions = torch.argmax(q_values, dim=1)
                action_indices = actions.reshape(-1, 1)
                raw_pred = torch.squeeze(torch.gather(q_values, 1, action_indices))

                # Old
                # # Convert torch values to numpy
                # q_vals_cpu = q_values.data.cpu().numpy()
                # raw_pred = raw_pred.data.cpu().numpy()
                # # Get the index of max values for each graph, and associated values
                # actions = np.argmax(q_vals_cpu, axis=1)
                # # raw_pred = np.max(q_vals_cpu, axis=1)
                # action_indices = actions.reshape(-1, 1)
                # raw_pred = np.take_along_axis(raw_pred, action_indices, axis=1)
                # raw_pred = np.squeeze(raw_pred)
                # actions = torch.LongTensor(actions)
                # raw_pred = torch.Tensor(raw_pred)

                return actions, raw_pred, prefix_sum
            # If we have given actions then select Q values for the actions
            else:
                # Get the graph embeddings
                embed_s_a = F.relu(self.linear_action_1(graph_embed))

                # q_values = F.softmax(self.linear_action_out(embed_s_a), dim=1)
                q_values = self.linear_action_out(embed_s_a)

                # Convert given actions to {0,1,2} then set to torch tensor
                for i in range(0, len(actions)):
                    if (actions[i] == 'add') or (actions[i] == 0):
                        actions[i] = 0
                    elif actions[i] == 'remove' or (actions[i] == 1):
                        actions[i] = 1
                    else:
                        actions[i] = 2
                actions = torch.LongTensor(actions)
                # Reshape to column tensor for indexing the correct values in output - use gather to index
                action_indices = actions.reshape(-1, 1)
                raw_pred = torch.gather(q_values, 1, action_indices)
                return actions, raw_pred, prefix_sum


class NStepQNet(nn.Module):
    """
    Is num_steps = 2, with step 0 being the first action and step 1 being the 2nd action - passed to 2 networks
    Defines the N step Deep Q-Network
    """
    def __init__(self, hyperparams, num_steps=3, action_space_flag='all'):
        """
        :param hyperparams: training hyperparameters
        :param num_steps: Number of steps in the MDP: 1 for action choice, 2 for node selection
        :param action_space_flag: {'all', 'add_remove', 'flip'}
        """
        super(NStepQNet, self).__init__()

        # Creates a list of Torch modules - which can be iterated like normal list - acts similar to sequential
        list_mod = [QNet(hyperparams, None, action_selector=True)]
        # Fill the list with modules for each step - use the same S2V module for all
        for i in range(1, num_steps):
            list_mod.append(QNet(hyperparams, list_mod[0].s2v))

        # Convert to the torch ModuleList class - Pytorch is “aware” of the nn.Module's inside an nn.ModuleList
        self.list_mod = nn.ModuleList(list_mod)
        self.num_steps = num_steps

    def forward(self, time_t, states, actions, greedy_acts=False):
        """
        Defines the forward pass of the network
        :param time_t: Current time step (0 or 1)
        :param states: States in the batch -
        :param actions: Actions in the batch - seems to be None in the call
        :param greedy_acts: Bool for greedy actions
        :return: Output of the torch ModuleList
        """
        # Check time is greater than or equal to 0 and less than the number of steps
        assert (time_t >= 0) and (time_t < self.num_steps)
        # Return the module indexed at time t, using state, actions, greedy actions as inputs
        return self.list_mod[time_t](states, actions, greedy_acts)
