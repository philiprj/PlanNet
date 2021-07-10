from __future__ import print_function

import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from relnet.utils.config_utils import get_device_placement

# Setting a path for the pytorch_structure2vec directory to use the S2V modules
sys.path.append('/usr/lib/pytorch_structure2vec/s2v_lib')
from s2v_lib import S2VLIB
from pytorch_util import weights_init


class MySpMM(torch.autograd.Function):
    """
    TODO: What is the purpose of this class?
    Creates a custom autograd function
    """
    @staticmethod
    def forward(ctx, sp_mat, dense_mat):
        """
        Defines the forward pass using the inputs
        :param ctx: Context object used to store information for the backward pass.
        :param sp_mat: Sparse matrix
        :param dense_mat: Dense matrix
        :return: Matmul of spare and dense matrix
        """
        # Send to GPU if available
        if get_device_placement() == 'GPU':
            sp_mat = sp_mat.cuda()
            dense_mat = dense_mat.cuda()
        # Save the variables
        ctx.save_for_backward(sp_mat, dense_mat)
        # Return a matmul of the matrices
        return torch.mm(sp_mat, dense_mat)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Defines the backward pass
        :param ctx: saved values from the forward pass
        :param grad_output: Loss with respect to the output
        :return: Grad for each matrix
        """
        # Get the input matrices
        sp_mat, dense_mat = ctx.saved_tensors
        # Set the gradients to None
        grad_matrix1 = grad_matrix2 = None
        # If first matrix required grad
        if ctx.needs_input_grad[0]:
            # Matmul of grad output and dense matrix
            grad_matrix1 = Variable(torch.mm(grad_output.data, dense_mat.data.t()))
        # If second matrix required grad
        if ctx.needs_input_grad[1]:
            # Matmul of grad output and dense matrix
            grad_matrix2 = Variable(torch.mm(sp_mat.data.t(), grad_output.data))
        # Return the grad outputs
        return grad_matrix1, grad_matrix2


def gnn_spmm(sp_mat, dense_mat):
    # Passes the matrices through the autograd function
    return MySpMM.apply(sp_mat, dense_mat)


class EmbedMeanField(nn.Module):
    """
    EmbedMeanField uses a Mean Field approach to generate an embedding for the graph
    """
    def __init__(self, latent_dim, output_dim, num_node_feats, num_edge_feats, max_lv=3):
        """
        :param latent_dim: Number of hidden nodes
        :param output_dim: Output dimension of embedding
        :param num_node_feats: Number of features present at each node
        :param num_edge_feats: Number of features for each edge
        :param max_lv: Max number of message passage levels
        """
        super(EmbedMeanField, self).__init__()
        # Init the parameters
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.max_lv = max_lv  
        # Liner layer of node node features to
        self.w_n2l = nn.Linear(num_node_feats, latent_dim)
        # If we have edge features then define the edge layer
        if num_edge_feats > 0:
            self.w_e2l = nn.Linear(num_edge_feats, latent_dim)
        # If the output dimension is greater then 0 then define output layer
        if output_dim > 0:
            self.out_params = nn.Linear(latent_dim, output_dim)
        # Define the hidden layer
        self.conv_params = nn.Linear(latent_dim, latent_dim)
        # Initialise the weights
        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat, pool_global=True, n2n_grad=False, e2n_grad=False):
        """
        Defines the forward pass of the graph network
        :param graph_list: List of graphs in dataset
        :param node_feat: Node features
        :param edge_feat: Edge features
        :param pool_global: bool - final outputs from the the node/edge embeddings are pooled to get output
        :param n2n_grad: Bool for node to node gradient
        :param e2n_grad: Bool for edge to node gradient
        :return: Hidden output and sparse tensor values in a dict
        """
        # Prepare the mean field inputs - node 2 node, edge 2 node, subgraph - sparse tensor types
        n2n_sp, e2n_sp, subg_sp = S2VLIB.PrepareMeanField(graph_list)
        # If using GPU then send to cuda
        if type(node_feat) is torch.cuda.FloatTensor:
            n2n_sp = n2n_sp.cuda()
            e2n_sp = e2n_sp.cuda()
            subg_sp = subg_sp.cuda()
            subg_sp = subg_sp.cuda()
        # Convert to torch tensor
        node_feat = Variable(node_feat)
        # If using edge features
        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
        # Convert to tesnsor with gradient if required
        n2n_sp = Variable(n2n_sp, requires_grad=n2n_grad)
        e2n_sp = Variable(e2n_sp, requires_grad=e2n_grad)
        subg_sp = Variable(subg_sp)
        # Get the hidden output from the network
        h = self.mean_field(node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, pool_global)
        # If the gradient is required for either edges or nodes then also return a dict with variable values
        if n2n_grad or e2n_grad:
            sp_dict = {'n2n': n2n_sp, 'e2n': e2n_sp}
            return h, sp_dict
        else:
            return h

    def mean_field(self, node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, pool_global):
        """
        Performs mean field inference to learn function mappings
        :param node_feat: Node features
        :param edge_feat: Edge features
        :param n2n_sp: Node 2 node sparse tensor
        :param e2n_sp: Edge 2 node sparse tensor
        :param subg_sp: Subgraph tensor
        :param pool_global: Bool for pooling the global embeddings
        :return: output of mean field message passing and the global pooling
        """
        # Gets the hidden output from the node input
        input_node_linear = self.w_n2l(node_feat)
        input_message = input_node_linear
        # If we are using edge features then get the edge hidden values, pool the values and increment message
        if edge_feat is not None:
            input_edge_linear = self.w_e2l(edge_feat)
            # TODO: what does this do?
            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
            input_message += e2npool_input
        # Take ReLU activation
        input_potential = F.relu(input_message)
        # Set the current level to 0
        lv = 0
        # Set current message on this layer to the input
        cur_message_layer = input_potential
        # Loop while the level is less than max level
        while lv < self.max_lv:
            # Pool the node features TODO: confirm this function
            n2npool = gnn_spmm(n2n_sp, cur_message_layer)
            # Hidden layer 2
            node_linear = self.conv_params(n2npool)
            # Combine the node and input message
            merged_linear = node_linear + input_message
            # Apply ReLU function
            cur_message_layer = F.relu(merged_linear)
            # Increment level
            lv += 1
        # Once we have passed through a few layers continue
        # If we have more than one output later then pass to the output layer and apply ReLU
        if self.output_dim > 0:
            out_linear = self.out_params(cur_message_layer)
            reluact_fp = F.relu(out_linear)
        # otherwise the action is our current message layer
        else:
            reluact_fp = cur_message_layer
        # IF we are pooling all embeddings then combine and return the values
        if pool_global:
            y_potential = gnn_spmm(subg_sp, reluact_fp)
            return reluact_fp, F.relu(y_potential)
        # Otherwise just return the output of the message passing layers
        else:
            return reluact_fp


class EmbedLoopyBP(nn.Module):
    def __init__(self, latent_dim, output_dim, num_node_feats, num_edge_feats, max_lv=3):
        """
        Performs inference using a loop belief propagation approach
        :param latent_dim: hidden dimension
        :param output_dim: Output dimension
        :param num_node_feats: Number of node features
        :param num_edge_feats: Number of edge features
        :param max_lv: Max number of passes through network
        """
        super(EmbedLoopyBP, self).__init__()
        # Initialises the network parameters
        self.latent_dim = latent_dim
        self.max_lv = max_lv
        self.w_n2l = nn.Linear(num_node_feats, latent_dim)
        self.w_e2l = nn.Linear(num_edge_feats, latent_dim)
        self.out_params = nn.Linear(latent_dim, output_dim)
        self.conv_params = nn.Linear(latent_dim, latent_dim)
        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat):
        """
        Defines the forward pass
        :param graph_list: List of training graphs
        :param node_feat: Node features
        :param edge_feat: Edge features
        :return: Hidden state
        """
        # Prepares the inputs for the graphs - sparse tensor for node to node/edge and edge to node/edge, and subgraph
        n2e_sp, e2e_sp, e2n_sp, subg_sp = S2VLIB.PrepareLoopyBP(graph_list)
        # Send to GPU if available
        if type(node_feat) is torch.cuda.FloatTensor:
            n2e_sp = n2e_sp.cuda()
            e2e_sp = e2e_sp.cuda()
            e2n_sp = e2n_sp.cuda()
            subg_sp = subg_sp.cuda()
        # Convert to torch variables with gradients
        node_feat = Variable(node_feat)
        edge_feat = Variable(edge_feat)
        n2e_sp = Variable(n2e_sp)
        e2e_sp = Variable(e2e_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)
        # Propagate through the networks and return
        h = self.loopy_bp(node_feat, edge_feat, n2e_sp, e2e_sp, e2n_sp, subg_sp)
        return h

    def loopy_bp(self, node_feat, edge_feat, n2e_sp, e2e_sp, e2n_sp, subg_sp):
        """
        Performs loopy belief propagation on the graphs for a given number of layers
        :param node_feat: Node features
        :param edge_feat: Edge features
        :param n2e_sp: sparse tensor for node to edge
        :param e2e_sp: sparse tensor for edge to edge
        :param e2n_sp: sparse tensor for edge to node
        :param subg_sp: sparse tensor for subgraph
        :return: Returns the output after a few given layers
        """
        # Get the inputs from the node/edge features
        input_node_linear = self.w_n2l(node_feat)
        input_edge_linear = self.w_e2l(edge_feat)
        # Pools the node to edge values
        n2epool_input = gnn_spmm(n2e_sp, input_node_linear)
        # Combine the node and edge features into an input message and applies ReLU
        input_message = input_edge_linear + n2epool_input
        input_potential = F.relu(input_message)
        # Initialises the level
        lv = 0
        cur_message_layer = input_potential
        # Loop while the level is less than max levels
        while lv < self.max_lv:
            # Combine the values and pass to the hidden layer
            e2epool = gnn_spmm(e2e_sp, cur_message_layer)
            edge_linear = self.conv_params(e2epool)
            # Combine the outputs
            merged_linear = edge_linear + input_message
            # Apply ReLU and Increment the layers
            cur_message_layer = F.relu(merged_linear)
            lv += 1
        # Pool the output and apply ReLU, get the final output of layer
        e2npool = gnn_spmm(e2n_sp, cur_message_layer)
        hidden_msg = F.relu(e2npool)
        out_linear = self.out_params(hidden_msg)
        reluact_fp = F.relu(out_linear)
        # Pool the final layer with the subgraph and return
        y_potential = gnn_spmm(subg_sp, reluact_fp)
        return F.relu(y_potential)
