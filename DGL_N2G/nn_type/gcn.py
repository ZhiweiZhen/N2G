# Name: Kunal Mukherjee
# Email: kunmukh@gmail.com
# Date: 9/23/2022
# Project:

import torch
import dgl
import dgl.function as fn
import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F

from __init__ import device


class KLayerGCN(nn.Module):
    def __init__(self, num_layer, in_dim, hidden_dim, n_classes, graph=True):
        super().__init__()
        self.graph = graph
        self.layers = nn.ModuleList()

        self.layers.append(
            dglnn.GraphConv(in_feats=in_dim,
                            out_feats=hidden_dim,
                            activation=F.relu))
        for _ in range(1, num_layer - 1):
            self.layers.append(
                dglnn.GraphConv(in_feats=hidden_dim,
                                out_feats=hidden_dim,
                                activation=F.relu))

        if graph:
            self.layers.append(
                dglnn.GraphConv(in_feats=hidden_dim, out_feats=hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, n_classes))
        else:
            self.layers.append(
                dglnn.GraphConv(in_feats=hidden_dim,
                                out_feats=n_classes,
                                activation=None))

    def forward(self, g, inputs):

        if self.graph:
            # inputs are features of nodes
            feat = F.leaky_relu(self.layers[0](g, inputs))
            feat = feat.reshape((-1, feat.shape[-1]))

            for i in range(1, len(self.layers) - 3):
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
            feat = self.layers[0](g, inputs)
            for i in range(1, len(self.layers)):
                feat = self.layers[i](g, feat)

            return feat


# Define a Heterograph Conv model
class HeteroGCN(nn.Module):
    def __init__(self, num_layers, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(
                dglnn.HeteroGraphConv(
                    {
                        rel: dglnn.GraphConv(in_feats, out_feats).to(device)
                        for rel in rel_names
                    },
                    aggregate='sum'))
        else:
            self.layers.append(
                dglnn.HeteroGraphConv(
                    {
                        rel: dglnn.GraphConv(in_feats, hid_feats).to(device)
                        for rel in rel_names
                    },
                    aggregate='sum'))

            self.layers.append(
                dglnn.HeteroGraphConv(
                    {
                        rel: dglnn.GraphConv(hid_feats, hid_feats).to(device)
                        for rel in rel_names
                    },
                    aggregate='sum'))

            self.layers.append(
                dglnn.HeteroGraphConv(
                    {
                        rel: dglnn.GraphConv(hid_feats, out_feats).to(device)
                        for rel in rel_names
                    },
                    aggregate='sum'))

    def forward(self, graph, feat):
        graph = graph.to(device)

        # inputs are features of nodes
        with graph.local_scope():

            if len(self.layers) == 1:
                feat = self.layers[0](graph, feat)
                return feat

            # Apply intermediate layers
            for i in range(len(self.layers) - 1):
                feat = self.layers[i](graph, feat)
                feat = {k: F.normalize(v) for k, v in feat.items()}
                feat = {k: F.leaky_relu(v) for k, v in feat.items()}

            # Apply last layer w/o activation function
            feat = self.layers[-1](graph, feat)

            graph.ndata['h'] = feat

            return graph.ndata['h']


class KLayerHeteroRGCN(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()
        self.in_dim = in_dim
        self.rgcn = HeteroGCN(num_layers, in_dim, hidden_dim, hidden_dim,
                              rel_names)
        self.linear = nn.Linear(hidden_dim, n_classes).to(device)

    def forward(self, g, feat=None, eweight=None):

        g = g.to(device)

        if not feat:
            feat = {
                ntype: torch.zeros((g.num_nodes(ntype), self.in_dim),
                                   device=device)
                for ntype in g.ntypes
            }

        h = feat
        h = self.rgcn(g, h)

        with g.local_scope():
            g.ndata['h'] = h

            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))

            # Calculate graph representation by average readout.
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)

            return torch.sigmoid(self.linear(hg))
