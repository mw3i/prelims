import numpy as np 

import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
clean = lambda ax: [ax.set_xticks([]),ax.set_yticks([])]

import seaborn as sns

import networkx as nx 

##__Plot
fig = plt.figure(figsize = [10,14])
gs = GridSpec(6,3)

# plot adj mat

adj_mats = {
    '2': np.genfromtxt('samples/2.csv', delimiter=','),
    '50': np.genfromtxt('samples/50.csv', delimiter=','),
    '100': np.genfromtxt('samples/100.csv', delimiter=','),
}
for _, [M_val, data] in enumerate(adj_mats.items()):

    # plot network
    graph_ax = plt.subplot(gs[0,_])
    clean(graph_ax)
    G = nx.Graph()
    for base in range(data.shape[0]):
        G.add_node(base)
    for base in range(data.shape[0]):
        for target in range(base, data.shape[0]):
            if data[base,target] == 1:
                G.add_edge(base, target)


    pos = nx.layout.kamada_kawai_layout(G)
    # pos = nx.layout.random_layout(G)
    # pos = nx.layout.spectral_layout(G)
    # pos = nx.layout.spring_layout(G)
    nx.draw_networkx_edges(
        G, 
        pos, 
        alpha = .1,
        ax = graph_ax,
    )
    nx.draw_networkx_nodes(
        G, 
        pos, 
        node_size = 50,
        ax = graph_ax,
        alpha = .5
    )


    adj_ax = plt.subplot(gs[1,_])
    adj_ax.imshow(data, cmap = 'binary', vmin = 0, vmax = 1)
    clean(adj_ax)

    # print('Cluster Coeff:', nx.average_clustering(G))
    # print('Diameter:', nx.diameter(G))
    # print('AvShortPath:', nx.average_shortest_path_length(G))


    ## P(k)
    pk_ax = plt.subplot(gs[2,_])
    sns.distplot(
        data.sum(axis = 0),
        ax = pk_ax,
        hist = False,
    )
    pk_ax.set_xlim([data.sum(axis = 0).min(),data.sum(axis = 0).max()])
    pk_ax.set_yticks([])


    ## style
    if _ == 0:
        graph_ax.set_ylabel('Network\nVisualization', fontsize = 20, fontweight = 'bold')
        adj_ax.set_ylabel('Adjacency\nMatrix', fontsize = 20, fontweight = 'bold')
        pk_ax.set_ylabel('Degree\nDistribution', fontsize = 20, fontweight = 'bold')
    graph_ax.set_title('M = ' + str(M_val), fontsize = 25, fontweight = 'bold')


## divider ax
d = plt.subplot(gs[3,:]); d.imshow(np.zeros([1,400]), cmap = 'binary'); clean(d) # <-- make | divide | clean


## Plot Walk Results
walk_ax = plt.subplot(gs[4:,:])
walk_ax.plot(
    np.arange(2,101,4),
    np.genfromtxt('walk_results.csv'),
    # np.genfromtxt('saves/1 network per 10000 walks.csv'),
    linewidth = 4,
    color = 'orange',
)
walk_ax.set_title('Mean Walk Distance of\nRandom Walks Between 2 Random Nodes', fontsize = 20, fontweight = 'bold')
walk_ax.set_ylabel('Mean\nWalk Distance', fontsize = 20, fontweight = 'bold'); walk_ax.set_xlabel('M', fontsize = 20, fontweight = 'bold')
walk_ax.set_xticks(np.arange(2,101,8))

plt.tight_layout()
plt.savefig('results_figure.png')

















