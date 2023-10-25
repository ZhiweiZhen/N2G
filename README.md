# United We Stand, Divided We Fall: Networks to Graph (N2G) Abstraction for Robust Graph Classification under Graph Label Corruption

> **_NOTE:_**
> To work in the N2G repository, please make sure you have installed the yapf python
> formatter and the attached pre-commit hook. You can do this by running
> `pip install pre-commit (https://pypi.org/project/pre-commit/) and pre-commit install` in your command line inside the repository.
> If your development environemnt is in Windows, you may need to add
> `C:\Users\NAME\AppData\Local\Programs\Python\PythonXXX\Scripts` to your PATH.

Implementation of N2G method for GCN, GAT and GRAPHSAGE.

To run each method, go to 'DGL/GoG' and run gog_driver.py file, for example

```
!python3 gog_driver.py gog -c config.yaml
```

In each method, the folder 'data' contains
1. Differnent distance matrix generated using netcomp
2. Feature matrix
3. Class label

For distance matrix, we have a $N \times N$ matrix whose values are pairwise distance for n graphs. Here we have lambda distance, deltance0 distance and vertex-edge overlop distance.

Then we use the quantile threshold to generate the unweighted 0-1 adjacent matrix.

For each N2G, we have 7 features: average degree centrality, betweenness centrality, closeness centrality, eigenvector centrality, current flow betweenness cen-trality, subgraph centrality, and current flow closeness centrality.

Regarding how to generate distance matrix in tud_benchamrk, please refer to '/data_converter/graph_to_node_dgl.py' file.
