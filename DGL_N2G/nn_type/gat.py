import dgl
import dgl.nn as dglnn
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from __init__ import device


class KLayerGAT(nn.Module):
    def __init__(self,
                 num_layer,
                 in_dim,
                 hidden_dim,
                 n_classes,
                 num_head=None,
                 graph=True):
        super().__init__()
        if num_head is None:
            num_head = [1, 1]
        self.graph = graph
        self.num_layers = num_layer
        self.num_heads = num_head
        self.layers = nn.ModuleList()

        self.layers.append(
            dglnn.GATv2Conv(in_feats=in_dim,
                            out_feats=hidden_dim,
                            num_heads=self.num_heads[0],
                            activation=F.elu))
        for l in range(1, num_layer - 1):
            self.layers.append(
                dglnn.GATv2Conv(in_feats=hidden_dim * self.num_heads[l - 1],
                                out_feats=hidden_dim,
                                num_heads=self.num_heads[l],
                                activation=F.elu))
        if graph:
            self.layers.append(
                dglnn.GATv2Conv(in_feats=hidden_dim * self.num_heads,
                                out_feats=hidden_dim,
                                num_heads=self.num_heads))
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, n_classes))
        else:
            self.layers.append(
                dglnn.GATv2Conv(in_feats=hidden_dim * self.num_heads[-2],
                                out_feats=n_classes,
                                num_heads=self.num_heads[-1],
                                activation=None))

    def forward(self, g, inputs):

        if self.graph:
            # inputs are features of nodes
            feat = F.leaky_relu(self.layers[0](g, inputs))
            feat = feat.reshape((-1, feat.shape[-1]))

            for i in range(1, len(self.layers) - 3):
                #h = h.softmax(dim=-2).sum(dim=-2)
                feat = F.leaky_relu(self.layers[i](g, feat))
                feat = feat.reshape((-1, feat.shape[-1]))

            with g.local_scope():
                g.ndata['h'] = feat
                # Calculate graph representation by average readout.
                hg = dgl.mean_nodes(g, 'h')
                hg = F.leaky_relu(self.layers[-2](hg))
                hg = self.layers[-1](hg)
                return hg
        else:
            feat = inputs
            for l in range(self.num_layers - 1):
                feat = self.layers[l](g, feat).flatten(1)
            # output projection
            feat = self.layers[-1](g, feat).mean(1)

            return feat


class HeteroGAT(nn.Module):
    """
    Heterograph GAT Layer Building Block
    """
    def __init__(self, in_dim, out_dim, graph_relation_list):
        super(HeteroGAT, self).__init__()

        # relation weight matrix
        self.relation_weight_matrix = nn.ModuleDict({
            ''.join(relation): nn.Linear(in_dim, out_dim).to(device)
            for relation in graph_relation_list
        })

        # attention weight matrix
        self.relation_attn_fc = nn.ModuleDict({
            ''.join(relation): nn.Linear(2 * out_dim, 1, bias=False).to(device)
            for relation in graph_relation_list
        })

    def forward(self, graph, node_feature_dict, eweight=None):
        """
        per relation message passing/reduction function dict
        relation => (message passing func, message reduction func)

        :param graph:
        :param node_feature_dict:
        :return:
        """

        graph = graph.to(device)
        rel_func_dict = {}

        for relation in graph.canonical_etypes:
            # zero edges of this relation? we move onto next relation
            if not graph.num_edges(relation):
                continue

            relation_str = ''.join(relation[1])
            src = relation[0]
            dst = relation[2]

            udfFunctions = RelationUDF(relation_str, self.relation_attn_fc)

            # compute W_r * h
            wh_src = self.relation_weight_matrix[relation_str](
                node_feature_dict[src])
            wh_dst = self.relation_weight_matrix[relation_str](
                node_feature_dict[dst])

            # save in graph for message passing (z = whh'_)
            graph.nodes[src].data[f'wh_{relation_str}'] = wh_src
            graph.nodes[dst].data[f'wh_{relation_str}'] = wh_dst

            if eweight is not None:
                graph.edges[relation].data['w'] = eweight[relation].view(-1, 1)
            else:
                graph.edges[relation].data['w'] = th.ones(
                    [graph.number_of_edges(relation), 1], device=graph.device)

            # equation (2)
            graph.apply_edges(udfFunctions.edge_attention, etype=relation)

            rel_func_dict[relation] = (udfFunctions.message_func,
                                       udfFunctions.reduce_func)

        # equation (3) & (4)
        # self.g.update_all(self.message_func, self.reduce_func)
        # trigger message passing & aggregation
        graph.multi_update_all(rel_func_dict, 'sum')

        # return self.g.ndata.pop('h')
        return {
            ntype: graph.nodes[ntype].data['h']
            for ntype in graph.ntypes if graph.num_nodes(ntype)
        }


# Used for GAT Edge Attention, Message, and Reduce UDF Functions
class RelationUDF:
    def __init__(self, relation_str, attention_weight_matrix):
        self.relation_str = relation_str
        self.attention_weight_matrix = attention_weight_matrix

    def edge_attention(self, edges):
        """
        edge UDF for equation (2)

        :param edges:
        :param relation:
        :return:
        """

        wh2 = th.cat([
            edges.src[f'wh_{self.relation_str}'],
            edges.dst[f'wh_{self.relation_str}']
        ],
                     dim=1)

        a = self.attention_weight_matrix[self.relation_str](wh2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        """
        message UDF for equation (3) & (4)

        :param edges:
        :param relation:
        :return:
        """

        return {
            f'wh_{self.relation_str}': edges.src[f'wh_{self.relation_str}'],
            'e': edges.data['e']
        }

    def reduce_func(self, nodes):
        """
        reduce UDF for equation (3) & (4)

        :param nodes:
        :param relation:
        :return:
        """

        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = th.sum(alpha * nodes.mailbox[f'wh_{self.relation_str}'], dim=1)
        return {'h': h}


class KLayerHeteroGAT(nn.Module):
    """
    K-Layer RGAT for Hetero-graph
    If num layers == 1, then hidden layer size does not matter and can be some arbitrary value

    structural: if we are training on graph structure & not considering any explicit node features
    """
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim,
                 graph_relation_list):
        super(KLayerHeteroGAT, self).__init__()
        """
        embedding mapping for featureless heterograph
        graph memory location => embedding dict

        :param in_dim: input dimension
        :param hidden_dim: hidden dimension
        :param out_dim: output dimension (# of classes)
        """

        assert num_layers > 0, 'Number of layers in RGCN must be greater than 0!'

        self.input_feature_size = in_dim

        self.layers = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(HeteroGAT(in_dim, out_dim, graph_relation_list))
        else:
            # Add first layer
            self.layers.append(
                HeteroGAT(in_dim, hidden_dim, graph_relation_list))

            # Add intermediate layers
            for i in range(0, num_layers - 2):
                self.layers.append(
                    HeteroGAT(hidden_dim, hidden_dim, graph_relation_list))

            # Add last (output layer)
            self.layers.append(nn.Linear(hidden_dim, out_dim).to(device))

    def forward(self, g, feat=None, eweight=None):

        g = g.to(device)

        if not feat:
            feat = {
                ntype: th.zeros((g.num_nodes(ntype), self.input_feature_size),
                                device=device)
                for ntype in g.ntypes
            }

        h = self.layers[0](g, feat, eweight=eweight)
        h = {node_type: F.elu(output) for node_type, output in h.items()}

        # Apply intermed layers
        for i in range(1, len(self.layers) - 1):
            h = self.layers[i](g, h, eweight=eweight)
            h = {node_type: F.elu(output) for node_type, output in h.items()}

        with g.local_scope():
            g.ndata['h'] = h

            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))

            # Calculate graph representation by average readout.
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)

            return th.sigmoid(self.layers[-1](hg))
