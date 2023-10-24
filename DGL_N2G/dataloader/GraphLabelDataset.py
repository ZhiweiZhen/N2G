from .BaseDataloader import ProvDataset
import os
import torch as th
import pickle
import pathlib
from collections import Counter


class GraphLabelDataset(ProvDataset):
    _ProvDataset__cache_file_name = '_multiclass_processed_prov_dataset.bin'

    def __init__(self,
                 input_dir,
                 label_types,
                 node_attributes_map,
                 relation_attributes_map,
                 graph_labels_pickled_path,
                 bidirection=False,
                 force_reload=False,
                 verbose=False):
        self.input_dir = input_dir
        self.label_types = label_types

        if graph_labels_pickled_path:
            with open(graph_labels_pickled_path, 'rb') as file:
                self.graph_labels = pickle.load(file)

        super(GraphLabelDataset,
              self).__init__(name='Graph Label Provenance Graph',
                             input_dir=input_dir,
                             node_attributes_map=node_attributes_map,
                             relation_attributes_map=relation_attributes_map,
                             bidirection=bidirection,
                             force_reload=force_reload,
                             verbose=verbose)

    def process(self):
        subfolders = [f.path for f in os.scandir(self.input_dir) if f.is_dir()]

        label_counter = Counter()

        for graph_subfolder_path in subfolders:
            graph_name = pathlib.Path(graph_subfolder_path).resolve().name

            if graph_name not in self.graph_labels:
                continue

            graph_label = self.label_types.index(self.graph_labels[graph_name])

            label_counter[graph_label] += 1

            self._ProvDataset__processGraph(graph_subfolder_path, graph_label)

        self.labels = th.tensor(
            self.labels,
            dtype=th.int64)  # convert label list to tensor for saving

        label_processed_counter = Counter()
        for unique_label in self.label_types:
            if label_counter[unique_label]:
                label_processed_counter[unique_label] = sum(
                    label == unique_label for label in self.labels)
                print(
                    f'Processed {label_processed_counter[unique_label]}/{label_counter[unique_label]} graphs with label: {unique_label} ({float(label_processed_counter[unique_label]) / label_counter[unique_label] * 100:.2f}%)'
                )
