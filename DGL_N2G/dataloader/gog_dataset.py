import dgl
from dgl.data import DGLDataset
import torch
import numpy as np
from pathlib import Path


# ref: https://docs.dgl.ai/tutorials/blitz/6_load_data.html
class GoGDataset(DGLDataset):
    @property
    def name(self):
        return self._name

    def __init__(self, name, path, distance_type, threshold, split, log=None):
        self.name = name
        self.path = Path(path)
        self.distance_type = distance_type
        self.threshold = threshold
        self.split = split
        self.log = log

        super(GoGDataset, self).__init__(name=name)

    def process(self):

        adj_matrix_path = self.path / self.name / self.distance_type / f'distance_{self.threshold}.txt'
        feat_path = self.path / self.name / 'feature.txt'
        label_path = self.path / self.name / 'LABEL.txt'

        self.log.info(adj_matrix_path)
        self.log.info(feat_path)
        self.log.info(label_path)

        adj_matrix = np.loadtxt(adj_matrix_path)
        src, dst = np.nonzero(adj_matrix)

        self.graph = dgl.graph((src, dst))
        self.graph.ndata['feat'] = torch.from_numpy(
            np.loadtxt(feat_path, dtype=np.float32).transpose())
        
        # # If your dataset is a node classification dataset, you will need to assign
        # # masks indicating whether a node belongs to training, validation, and test set.
        n_nodes = adj_matrix.shape[0]
        n_val = int(n_nodes*0.1)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        # randomly select a node
        np.random.seed(1)
        test_random_node = np.random.randint(0, adj_matrix.shape[0])
        # run bfs from the node to select subgraph
        bfs = dgl.bfs_nodes_generator(self.graph, test_random_node)
        test_nodes = torch.cat(bfs)[:n_val]

        val_random_node = torch.cat(bfs)[-1]
        _graph = self.graph.clone()
        in_edges = _graph.in_edges(test_nodes, form='eid')
        _graph.remove_edges(in_edges)
        bfs = dgl.bfs_nodes_generator(_graph, val_random_node)
        val_nodes = torch.cat(bfs)[:n_val]

        test_mask[test_nodes] = True
        #val_mask[val_nodes] = True
        train_mask = ~(test_mask)

        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        
        labels =  torch.from_numpy(
            np.loadtxt(label_path, dtype=np.float32)).type(torch.LongTensor)
        index = np.where(train_mask==True)
        np.random.shuffle(index)
        #print(labels[index]==0)
        index_bias= np.where(labels[index]==0)
        index_bias = index_bias[:int(len(index)*0.2)]
        #labels[index_bias] = 1- labels[index_bias]
       
        self.graph.ndata['label'] = labels
        self.num_classes = torch.tensor(
            [torch.unique(self.graph.ndata['label']).size(dim=0)])
        

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

    @name.setter
    def name(self, value):
        self._name = value
