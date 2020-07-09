import numpy as np 

def randwalk(i,j,adj):
    loc = i
    c = 0
    while loc != j:
        loc = np.random.choice(np.argwhere(adj[:,loc] == 1)[:,0])
        c += 1
    return c


n = 200
M_vals = np.arange(2,100,10)
net_iters = 10
walk_iters = 1000

hist = []
for M in M_vals:
    print('M:',M)

    net_mean = 0
    for net in range(net_iters):

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


        ## simulate random walk for # of iters
        mean_time = 0
        for i in range(walk_iters):

            # choose starting node
            start = np.random.randint(adj.shape[0])

            # choose neighbor of path length 3 (or however many times stop = ... is run)
            stop = np.random.choice(np.argwhere(adj[:,start] == 1)[:,0])
            # stop = np.random.choice(np.argwhere(adj[:,stop] == 1)[:,0])
            # stop = np.random.choice(np.argwhere(adj[:,stop] == 1)[:,0])

            mean_time += randwalk(
                start,
                stop,                
                adj
            )

        net_mean += mean_time / walk_iters
        # print('\t',net)
    hist.append(net_mean)
    print(hist)
    print('\t--')

np.savetxt('walk_results.csv',hist)

# import matplotlib.pyplot as plt 
# plt.plot(
#     M_vals,
#     hist,
# )
# plt.xticks(M_vals)
# plt.savefig('walk_results.png')
