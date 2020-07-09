import numpy as np 
import networkx as nx 

def randwalk(i,j,adj):
    loc = i
    c = 0
    while loc != j:
        loc = np.random.choice(np.argwhere(adj[:,loc] == 0)[:,0])
        c += 1
    return c


if __name__ == '__main__':
    import pandas as pd 
    import matplotlib.pyplot as plt 
    from matplotlib.gridspec import GridSpec
    clean = lambda ax: [ax.set_xticks([]),ax.set_yticks([])]
    import seaborn as sns

    adj_mats = {
        'adj2': np.genfromtxt('2.csv', delimiter=','),
        'adj10': np.genfromtxt('10.csv', delimiter=','),
        'adj100': np.genfromtxt('100.csv', delimiter=','),
    }


    iters = 10000
    all_data = {}
    for name, data in adj_mats.items():
        times = [
            randwalk(
                np.random.randint(data.shape[0]),
                np.random.randint(data.shape[1]),
                data
            )
            for i in range(iters)
        ]
        print(name, ':', np.mean(times))
        all_data[name] = times



    for name, data in all_data.items():
        sns.distplot(
            data, label = name
        )

    plt.legend()
    plt.savefig('times.png')




