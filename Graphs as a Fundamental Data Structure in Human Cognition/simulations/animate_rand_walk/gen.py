import numpy as np 

n = 50
M_vals = [2,50,100] # <-- sparse
# M = 50 # <-- dense

for M in M_vals:
    nodes = list(range(M))
    init_adj = np.ones([M,M])
    adj = np.zeros([n,n])

    adj[:M,:M] += init_adj

    ## Model A from Steyvers & Tenenbaum (2005)
    for step in range(len(nodes),n):

        # 1: choose an existing node i
        Pi_dist = adj[:step-1,:step-1].sum(axis = 0) / adj[:step-1,:step-1].sum(axis = 0).sum()
        node_i = np.random.choice(np.arange(step-1), p = Pi_dist)

        # 2: add a new node, and connect it randomly to i's neighborhood
        neighborhood_sample = np.random.choice(np.arange(step)[adj[node_i,:step].astype(bool)], size = M, replace = False) # uniform sample from node_i's neighborhood
        adj[neighborhood_sample,step] = 1
        adj[step,neighborhood_sample] = 1


    np.savetxt(str(M)+'.csv', adj, delimiter = ',')
    exit()
