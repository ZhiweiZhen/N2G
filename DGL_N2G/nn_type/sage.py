import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F


class KLayerSAGE(nn.Module):
    def __init__(self,
                 num_layer,
                 in_dim,
                 hidden_dim,
                 n_classes,
                 agg_type='mean',
                 graph=True):
        super().__init__()
        self.graph = graph
        self.layers = nn.ModuleList()
        self.agg_type = agg_type

        self.layers.append(
            dglnn.SAGEConv(in_feats=in_dim,
                           out_feats=hidden_dim,
                           aggregator_type=self.agg_type,
                           activation=F.elu))
        for _ in range(1, num_layer - 1):
            self.layers.append(
                dglnn.SAGEConv(in_feats=hidden_dim,
                               out_feats=hidden_dim,
                               aggregator_type=self.agg_type,
                               activation=F.elu))
        if self.graph:
            self.layers.append(
                dglnn.SAGEConv(in_feats=hidden_dim,
                               out_feats=hidden_dim,
                               aggregator_type=self.agg_type))
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, n_classes))
        else:
            self.layers.append(
                dglnn.SAGEConv(in_feats=hidden_dim,
                               out_feats=n_classes,
                               aggregator_type=self.agg_type,
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
