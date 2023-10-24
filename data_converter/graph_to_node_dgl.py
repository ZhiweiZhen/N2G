import logging
import os
import socket
from datetime import datetime
import argparse
import sys

from dgl.data.tu import TUDataset
import networkx as nx
import n2gnetcomp as nc
import dgl
import numpy as np
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append('../')
from DGL_GoG.gnnUtils import get_multi_class_train_val_test_datasets

# set up logging variables
log = logging.getLogger(__name__)
unique_str = datetime.now().strftime('%b%d_%H-%M-%S')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s\t| %(message)s',
                    handlers=[
                        logging.FileHandler(os.path.join(
                            os.getcwd(), 'logs',
                            f'{unique_str}_{socket.gethostname()}.log'),
                                            mode='w',
                                            encoding='utf-8'),
                        logging.StreamHandler()
                    ])


# get the arguments
def getArgs():
    '''
    helper function to get the script arguments
    :return:
    '''

    parser = argparse.ArgumentParser(
        description=
        'Runner script for converting graphs to nodes using distances.')

    cmd_parser = parser.add_subparsers(title='command', dest='service_command')
    converter_parser = cmd_parser.add_parser(
        name='gog', help='choose graph to node conversion')

    for psr in [converter_parser]:
        psr.add_argument('-d',
                         '--dataset',
                         type=str,
                         default='REDDIT-BINARY',
                         help='get the datasets available at '
                         'https://chrsmrrs.github.io/datasets/docs/datasets/',
                         required=True)
        psr.add_argument('-o',
                         '--output',
                         type=str,
                         default='/tmp/result/',
                         help='get the output directory',
                         required=True)
        psr.add_argument('-dis',
                         '--distance',
                         type=str,
                         default='laplacian',
                         choices=['laplacian', 'vertex_edge', 'deltance0'],
                         help='choose which distance metric',
                         required=True)
        psr.add_argument(
            '-t',
            '--threshold',
            type=float,
            default='0.75',
            help='choose the threshold to generate the adjacent matrix',
            required=True)
        psr.add_argument('-k',
                         '--k',
                         type=int,
                         default='10',
                         help='choose the k for laplacian',
                         required=False)

    ARGS = parser.parse_args()

    if ARGS.service_command in ['gog']:
        dataset = ARGS.dataset
        output = ARGS.output
        distance = ARGS.distance
        k = ARGS.k
        threshold = ARGS.threshold
    else:
        exit(-2)

    return dataset, output, distance, threshold, k


def main():
    dataset, output, distance_choice, threshold, k = getArgs()
    log.info(f'Dataset: {dataset} Output: {output}')

    if dataset == 'python_train':
        bidirection = False
        dataset_dir_path = '/data/gnn_data/python/25k/'
        graph_label_pickle_path = dataset_dir_path + 'graph_labels_new.pickle'

        # label_types = [
        #     'AppXSvc', 'wlidsvc', 'camsvc', 'WaaSMedicSvc', 'wuauserv',
        #     'ClipSVC', 'gpsvc', 'BITS'
        # ]
        # label_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        label_types = ['A', 'D', 'E', 'F', 'H']

        node_attributes = {
            'ProcessNode': [],
            'SocketChannelNode': [],
            'FileNode': []
        }

        edge_attributes = {
            ('ProcessNode', 'PROC_CREATE', 'ProcessNode'): [],
            ('ProcessNode', 'READ', 'FileNode'): [],
            ('ProcessNode', 'WRITE', 'FileNode'): [],
            ('ProcessNode', 'FILE_EXEC', 'FileNode'): [],
            ('ProcessNode', 'WRITE', 'FileNode'): [],
            ('ProcessNode', 'WRITE', 'SocketChannelNode'): [],
            ('ProcessNode', 'READ', 'SocketChannelNode'): [],
            ('ProcessNode', 'IP_CONNECTION_EDGE', 'ProcessNode'): [],
            ('ProcessNode', 'IP_CONNECTION_EDGE', 'FileNode'): [],
        }

        train_dataset, val_dataset, test_dataset = get_multi_class_train_val_test_datasets(
            dataset_dir_path,
            label_types,
            node_attributes,
            edge_attributes,
            graph_label_pickle_path,
            bidirection=bidirection,
            verbose=True)

        data = train_dataset
    else:
        data = TUDataset(dataset)

    true_labels = [l.item() for idx, (_, l) in enumerate(data)]

    if distance_choice == 'laplacian':
        assert k is not None

    distance_dir_path = os.path.join(output, dataset, distance_choice)

    os.makedirs(distance_dir_path, exist_ok=True)

    with open(os.path.join(distance_dir_path, 'LABEL.txt'), 'w') as file:
        [file.write(str(label) + '\n') for label in true_labels]

    get_distance(data, true_labels, distance_choice, distance_dir_path, k=k)

    feature_generate(true_labels, data, distance_dir_path)

    create_gog_threshold_dataset(distance_dir_path, threshold)


def feature_generate(true_labels, data, distance_dir_path):

    feature = np.zeros((7, len(true_labels)))

    for i in range(len(true_labels)):
        graph = dgl.to_networkx(dgl.to_homogeneous(data[i][0]))

        feature[0][i] = sum(nx.degree_centrality(graph).values()) / len(
            nx.degree_centrality(graph))
        try:
            feature[1][i] = sum(
                nx.eigenvector_centrality(
                    graph, max_iter=10000).values()) / len(
                        nx.eigenvector_centrality(graph, max_iter=10000))
        except nx.exception.NetworkXException:
            feature[1][i] = 0
        feature[2][i] = sum(nx.closeness_centrality(graph).values()) / len(
            nx.closeness_centrality(graph))
        try:
            feature[3][i] = sum(
                nx.current_flow_closeness_centrality(graph).values()) / len(
                    nx.current_flow_closeness_centrality(graph))
        except nx.exception.NetworkXException:
            feature[3][i] = 0
        feature[4][i] = sum(nx.betweenness_centrality(graph).values()) / len(
            nx.betweenness_centrality(graph))
        try:
            feature[5][i] = sum(
                nx.current_flow_betweenness_centrality(graph).values()) / len(
                    nx.current_flow_betweenness_centrality(graph))
        except nx.exception.NetworkXException:
            feature[5][i] = 0
        try:
            feature[6][i] = sum(nx.subgraph_centrality(graph).values()) / len(
                nx.subgraph_centrality(graph))
        except AttributeError:
            feature[6][i] = 0
        except nx.exception.NetworkXNotImplemented:
            feature[6][i] = 0

    np.savetxt(os.path.join(distance_dir_path, 'feature.txt'), feature)


def create_gog_threshold_dataset(distance_dir_path, t):

    data = np.genfromtxt(os.path.join(distance_dir_path, 'distance.txt'))
    idx = data < np.quantile(data, t)
    data[~idx] = 0
    data[idx] = 1
    np.savetxt(os.path.join(distance_dir_path, f'distance_{t}.txt'), data)


# define the function that will be run in a separate thread
def work_distance_matrix(data, true_labels, row, distance_matrix,
                         distance_type, k):

    # print(f'WORKING on row: {row}')
    for col in range(row, len(true_labels) - 1):
        A1, A2 = [
            nx.adjacency_matrix(G) for G in [
                dgl.to_networkx(dgl.to_homogeneous(data[row][0])),
                dgl.to_networkx(dgl.to_homogeneous(data[col][0]))
            ]
        ]

        if distance_type == 'laplacian':
            distance_matrix[row][col] = nc.lambda_dist(A1,
                                                       A2,
                                                       kind='laplacian',
                                                       k=k)
        if distance_type == 'vertex_edge':
            distance_matrix[row][col] = 1 - nc.vertex_edge_distance(A1, A2)
        if distance_type == 'deltance0':
            distance_matrix[row][col] = nc.deltacon0(A1, A2)
        # print(f'WORKING on row: {row}{col}: val:{distance_matrix[row][col]}')
    # print(f'FINSIHED on row: {row}')


def get_distance(data, true_labels, distance_type, distance_dir_path, k=10):
    distance_matrix = [[0.] * len(true_labels)
                       for _ in range(len(true_labels))]

    # single thread implementation
    # for i in tqdm(range(len(true_labels))):
    #     for j in tqdm(range(i, len(true_labels) - 1), leave=False):
    #         # print(f'i={i}/{len(true_labels)} j={j}/{len(true_labels)} ')
    #         A1, A2 = [
    #             nx.adjacency_matrix(G) for G in [
    #                 dgl.to_networkx(dgl.to_homogeneous(data[i][0])),
    #                 dgl.to_networkx(dgl.to_homogeneous(data[j][0]))
    #             ]
    #         ]

    #         if distance_type == 'laplacian':
    #             distance_matrix[i][j] = nc.lambda_dist(A1,
    #                                                    A2,
    #                                                    kind='laplacian',
    #                                                    k=k)
    #         if distance_type == 'vertex_edge':
    #             distance_matrix[i][j] = 1 - nc.vertex_edge_distance(A1, A2)
    #         if distance_type == 'deltance0':
    #             distance_matrix[i][j] = nc.deltacon0(A1, A2)

    # multi thread implementation
    # create a thread pool with a maximum of 8 threads
    with tqdm(total=len(true_labels)) as pbar:
        with ThreadPoolExecutor(max_workers=8) as executor:
            # for i in range(len(true_labels)):
            results = [
                executor.submit(work_distance_matrix, data, true_labels, i,
                                distance_matrix, distance_type, k)
                for i in range(len(true_labels))
            ]
            for result in as_completed(results):
                r = result.result()
                pbar.update(1)

    distance_matrix = np.array(distance_matrix).astype(np.float32)
    distance_matrix = distance_matrix + distance_matrix.T - np.diag(
        np.diag(distance_matrix))

    np.savetxt(os.path.join(distance_dir_path, 'distance.txt'),
               distance_matrix)


if __name__ == '__main__':
    main()
