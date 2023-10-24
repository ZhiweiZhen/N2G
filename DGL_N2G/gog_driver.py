# import logging
import logging
import os
import sys
import socket
from datetime import datetime
import argparse
import yaml

from nn_type.gcn import KLayerGCN
from nn_type.gat import KLayerGAT
from nn_type.sage import KLayerSAGE

sys.path.append('../')
from DGL_GoG.dataloader.gog_dataset import GoGDataset
from gnnUtils import visualize_graph, printDatasetStats, train_GoG

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
        description='Runner script for converting json to NetworkX.')

    cmd_parser = parser.add_subparsers(title='command', dest='service_command')
    gog_parser = cmd_parser.add_parser(name='gog', help='choose gog')

    for psr in [gog_parser]:
        psr.add_argument('-c',
                         '--config',
                         type=str,
                         default='config.yaml',
                         help='get the configuration file for training',
                         required=True)

    ARGS = parser.parse_args()

    if ARGS.service_command in ['gog']:
        config = ARGS.config
    else:
        exit(-2)

    log.info('CONFIG PATH: {}'.format(config))

    return config


def main():
    fname = getArgs()
    log.info(fname)

    with open(fname) as configFile:
        config = yaml.load(configFile, Loader=yaml.FullLoader)

        log.info(f'config: {config}')

        data_name = config['data']
        distance_type = config['distanceType']
        threshold = config['threshold']
        dataset_path = config['datasetPath']

        nn_type = config['nnType']  # gcn, gat, sage
        num_layers = config['numLayers']
        hidden_feat_size = config['hiddenFeatureSize']
        epochs = config['epochs']
        loss_rate = config['lossRate']
        weight_decay = config['weightDecay']

        train_test_split = config['trainTestSplit']
        device = config['device']

        if nn_type == 'gat':
            num_heads = config['numHeads']
            heads = ([num_heads] * num_layers) + [num_heads]
        if nn_type == 'sage':
            agg_type = config['aggregationType']

    log.info(
        f'data:{data_name} dataset path:{dataset_path} nn:{nn_type} # of layers:{num_layers} '
        f'hidden feat size:{hidden_feat_size} epochs{epochs} lr:{loss_rate} '
        f'weighted decay:{weight_decay} split:{train_test_split}')

    data = GoGDataset(data_name, dataset_path, distance_type, threshold,
                      train_test_split, log)
    graph = data[0]
    visualize_graph(graph)

    graph = graph.to(device)

    printDatasetStats(log, data, graph)

    input_feat_size = graph.ndata['feat'].shape[1]
    num_classes = data.num_classes.item()

    if nn_type == 'gcn':
        model = KLayerGCN(num_layers,
                          input_feat_size,
                          hidden_feat_size,
                          num_classes,
                          graph=False).to(device)
    elif nn_type == 'gat':
        model = KLayerGAT(num_layers,
                          input_feat_size,
                          hidden_feat_size,
                          num_classes,
                          heads,
                          graph=False).to(device)
    elif nn_type == 'sage':
        model = KLayerSAGE(num_layers,
                           input_feat_size,
                           hidden_feat_size,
                           num_classes,
                           agg_type,
                           graph=False).to(device)

    train_GoG(graph, model, epochs, loss_rate, weight_decay, log)


if __name__ == '__main__':
    main()
