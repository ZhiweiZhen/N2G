# United We Stand, Divided We Fall: Networks to Graph (N2G) Abstraction for Robust Graph Classification under Graph Label Corruption

> **_NOTE:_**
> To work in the N2G repository, please make sure you have installed the yapf python
> formatter and the attached pre-commit hook. You can do this by running
> `pip install pre-commit (https://pypi.org/project/pre-commit/) and pre-commit install` in your command line inside the repository.
> If your development environemnt is in Windows, you may need to add
> `C:\Users\...\Python\PythonXXX\Scripts` to your PATH.

Implementation of N2G method for GCN, GAT and GRAPHSAGE.

To run each method, go to 'DGL_N2G' and run gog_driver.py file, for example

```
!python3 gog_driver.py gog -c config.yaml
```

For each method, the folder 'data' contains
1. Different distance matrices generated using n2gnetcomp;
2. Feature matrix;
3. Class label.

For the distance matrix, we have a $N \times N$ matrix whose values are pairwise distance for n graphs. Here we consider using lambda distance, deltance0 distance and vertex-edge overlop distance. Afer that, we use the quantile threshold to generate the corresponding unweighted adjacent matrix.

For each N2G, we have 7 features: average degree centrality, betweenness centrality, closeness centrality, eigenvector centrality, current flow betweenness cen-trality, subgraph centrality, and current flow closeness centrality.

Regarding how to generate distance matrix in tud_benchamrk, please refer to '/data_converter/graph_to_node_dgl.py' file.

Regarding corrupted labels, there are 2 kinds of corrupted labels : (i) Uniformly corrupted and (ii) Biased corrupted labels. 
The corruption type and rate can be adjusted in 'config.yaml' file in DGL_N2G folder

