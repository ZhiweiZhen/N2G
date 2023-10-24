import os
import dgl
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F

from DGL_GoG.dataloader.GraphLabelDataset import GraphLabelDataset

from __init__ import device


def get_multi_class_train_val_test_datasets(dataset_dir_path,
                                            label_types,
                                            node_attributes_map,
                                            relation_attributes_map,
                                            graph_labels_pickled_path,
                                            bidirection=False,
                                            verbose=True):
    train_dataset = GraphLabelDataset(os.path.join(dataset_dir_path, 'train'),
                                      label_types,
                                      node_attributes_map,
                                      relation_attributes_map,
                                      graph_labels_pickled_path,
                                      bidirection=bidirection,
                                      verbose=verbose)
    val_dataset = GraphLabelDataset(os.path.join(dataset_dir_path,
                                                 'validation'),
                                    label_types,
                                    node_attributes_map,
                                    relation_attributes_map,
                                    graph_labels_pickled_path,
                                    bidirection=bidirection,
                                    verbose=verbose)
    test_dataset = GraphLabelDataset(os.path.join(dataset_dir_path, 'test'),
                                     label_types,
                                     node_attributes_map,
                                     relation_attributes_map,
                                     graph_labels_pickled_path,
                                     bidirection=bidirection,
                                     verbose=verbose)

    return train_dataset, val_dataset, test_dataset


def printDatasetStats(log, data, g):

    features = g.ndata['feat']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    num_feats = features.shape[1]
    n_classes = data.num_classes.item()
    n_edges = g.number_of_edges()
    log.info("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Features %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" % (
        n_edges,
        n_classes,
        num_feats,
        train_mask.int().sum().item(),
        val_mask.int().sum().item(),
        test_mask.int().sum().item(),
    ))


def visualize_graph(graph):

    options = {
        'node_color': 'black',
        'node_size': 20,
        'width': 1,
    }
    G = dgl.to_networkx(graph)
    nx.draw(G, **options)
    plt.show()


def train(log, model, train_dataloader, test_dataloader, epochs, loss_rate,
          weight_decay):

    opt = torch.optim.Adam(model.parameters(),
                           lr=loss_rate,
                           weight_decay=weight_decay)

    for epoch in range(epochs):
        training_loss = []
        training_acc = []
        testing_loss = []
        testing_acc = []

        for batched_graph, labels in train_dataloader:

            batched_graph = batched_graph.to(device)
            feats = batched_graph.ndata['node_labels'].float()
            labels = labels.reshape(-1).long().to(device)

            model = model.to(device)
            logits = model(batched_graph, feats)

            y_pred = logits.argmax(-1)

            acc = y_pred.eq(labels).sum() / labels.shape[0]
            loss = F.cross_entropy(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            training_loss.append(loss.detach().item())
            training_acc.append(acc.detach().item())

        for batched_graph, labels in test_dataloader:
            batched_graph = batched_graph.to(device)
            feats = batched_graph.ndata['node_labels'].float()
            labels = labels.reshape(-1).long().to(device)

            logits = model(batched_graph, feats)
            y_pred = logits.argmax(-1)

            acc = y_pred.eq(labels).sum() / labels.shape[0]
            loss = F.cross_entropy(logits, labels)

            testing_loss.append(loss.cpu().detach())
            testing_acc.append(acc.detach().item())

        if epoch % 100 == 0:
            log.info(
                f'Epoch {epoch}, Train Loss: {np.mean(training_loss):.3f} accuracy: {np.mean(training_acc)*100:.3f}, Test Loss: {np.mean(testing_loss):.3f} accuracy: {np.mean(testing_acc)*100:.3f}'
            )

    return testing_acc


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(g, model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)


def train_GoG(g, model, epochs, loss_rate, weight_decay, log):

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=loss_rate,
                                 weight_decay=weight_decay)

    features = g.ndata['feat']
    labels = g.ndata['label'].type(torch.LongTensor).to(device)

    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    for e in range(epochs):
        # Forward
        logits = model(g, features)
        # print(labels.shape)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits[train_mask], labels[train_mask])
        #val_acc = accuracy(logits[val_mask], labels[val_mask])
        test_acc = accuracy(logits[test_mask], labels[test_mask])

        # if train_acc < 0.3:
        #     print(logits)

        if e % 100 == 0:
            log.info(
                'In epoch {}, loss: {:.3f}, train_acc: {:.3f}, test_acc: {:.3f})'
                .format(e, loss, train_acc,  test_acc))

    acc = evaluate(g, model, features, labels, test_mask)
    print(acc)
    log.info('Test Accuracy {:.4f}'.format(acc))
