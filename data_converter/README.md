# N2G Generation

## Files

* `graph_to_nodes.py` -
* `graph_to_node_dgl.py` -

## Setup

```shell
mkdir logs
```
## Run

```shell
usage: graph_to_node_dgl.py gog [-h] -d DATASET -dis
                                {laplacian,vertex_edge,deltance0} -t THRESHOLD
                                [-k K]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        get the REDDIT-BINARY dataset
  -dis {laplacian,vertex_edge,deltance0}, --distance {laplacian,vertex_edge,deltance0}
                        choose which distance metric
  -t THRESHOLD, --threshold THRESHOLD
                        choose the threshold to generate the adjacent matrix
  -k K, --k K           choose the k for laplacian
```

```commandline
python3 graph_to_node_dgl.py gog -d REDDIT-BINARY -dis laplacian -t 0.75 -k 10
```

## Dir

* `results` -
