import numpy as np 
import networkx as nx 

# def randwalk(i,j,adj):
#     loc = i
#     c = 0
#     while loc != j:
#         loc = np.random.choice(np.argwhere(adj[:,loc] == 0)[:,0])
#         c += 1
#     return c


if __name__ == '__main__':
    import pandas as pd 

    import matplotlib.pyplot as plt 
    from matplotlib.gridspec import GridSpec
    clean = lambda ax: [ax.set_xticks([]),ax.set_yticks([])]


    adj = np.genfromtxt('2.csv', delimiter=',')    # plot network
    G = nx.Graph()
    for base in range(adj.shape[0]):
        G.add_node(base)
    for base in range(adj.shape[0]):
        for target in range(base, adj.shape[0]):
            if adj[base,target] == 1:
                G.add_edge(base, target)


    pos = nx.layout.kamada_kawai_layout(G)

    i,j = [np.random.randint(adj.shape[0]), np.random.randint(adj.shape[1])]


    def plot_net(i,j, frame, first = False, last = False):
        cols = ['blue'] * adj.shape[0]
        cols[i] = 'red'
        cols[j] = 'red'

        edgecols = ['black'] * adj.shape[0]
        edgecols[i] = 'green'

        nx.draw_networkx_edges(
            G, 
            pos, 
            alpha = .2,
        )
        
        nx.draw_networkx_nodes(
            G, 
            pos, 
            node_size = 75,
            alpha = .65,
            node_color = cols,
            edgecolors = edgecols,
            edgewidth = 1.5,
        )

        if first == True:
            plt.annotate(
                'start', 
                xy = pos[i], 
                xytext = [pos[i][0]+.2, pos[i][1]+.2],
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize = 20, fontweight = 'bold',
            )
            
            plt.annotate(
                'stop', 
                xy = pos[j], 
                xytext = [pos[j][0]+.2, pos[j][1]+.2],
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize = 20, fontweight = 'bold',
            )

        if last == True:
            plt.annotate(
                'Done', 
                xy = pos[i], 
                xytext = [pos[i][0]+.2, pos[i][1]+.2],
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize = 20, fontweight = 'bold',
            )

        plt.savefig('frames/' + str(frame) + '.png')
        plt.close()

    for f in range(10):
        plot_net(i, j, f, first = True)

    loc = i
    c = 10
    while loc != j:
    # for n in range(10):
        loc = np.random.choice(np.argwhere(adj[:,loc] == 1)[:,0])

        plot_net(loc, j, c)


        c += 1
        print(loc)


    def animate(folder_with_frames, output_filename='animation.mp4', num_frames = 1, fps=1, forward_reverse=False):
        import os
        import imageio

        # imageio.plugins.ffmpeg.download() # <-- you may need to run this the first time you use this code
        with imageio.get_writer(os.path.normpath(output_filename), fps=fps) as writer:


            # for frame in sorted(os.listdir(folder_with_frames)):
            for i in range(num_frames):
                frame = str(i) + '.png'
                if frame.endswith('.png') or frame.endswith('.jpg') or frame.endswith('.jpeg'):
                    image = imageio.imread(os.path.normpath(os.path.join(folder_with_frames, frame)))
                    writer.append_data(image)


            if forward_reverse == True:

                # for frame in sorted(os.listdir(folder_with_frames), reverse=True):
                for i in range(num_frames-1,0,-1):
                    frame = str(i) + '.png'
                    if frame.endswith('.png') or frame.endswith('.jpg') or frame.endswith('.jpeg'):
                        image = imageio.imread(os.path.normpath(os.path.join(folder_with_frames, frame)))
                        writer.append_data(image)

    for f in range(c,c+10):
        plot_net(loc, j, f, last = True)
        # c += 1

    animate(
        'frames/',
        num_frames = c,
        fps = 8
    )

