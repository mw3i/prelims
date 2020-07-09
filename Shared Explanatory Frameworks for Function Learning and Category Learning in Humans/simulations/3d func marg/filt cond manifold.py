'''
notes: i think this is going to have to be a sheet to work
	^ need to switch to meshgrid
'''
import autograd.numpy as np 
from autograd import grad

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
# plt.style.use('classic')

fig = plt.figure(figsize = [8,3])
gs = GridSpec(2,5)
n = 50

## Filtration
m1 = 1
b1 = 0
# x1 = np.linspace(0,1,n)

m2 = 0
b2 = 0
# x2 = np.linspace(0,1,n)

l = np.linspace(0,1,n)
x1, x2 = np.meshgrid(l,l)

w1 = 1
w2 = 1
y = w1 * (m1 * x1 + b1) + w2 * (m2 * x2 + b2)


ax00 = plt.subplot(gs[0,0], projection = '3d')
ax00.plot_surface(
	x1,
	x2, 
	y,
	facecolors = cm.viridis(np.linspace(0,1,n*n).reshape(n,n)),
)

ax01 = plt.subplot(gs[0,1])
ax01.scatter(x1, y, c = np.linspace(0,1,n*n).reshape(n,n), alpha = .3, cmap = 'viridis', marker = 's')

ax10 = plt.subplot(gs[1,0])
ax10.scatter(x2, y, c = np.linspace(0,1,n*n).reshape(n,n), cmap = 'viridis', marker = 's')

ax11 = plt.subplot(gs[1,1])
ax11.scatter(x1, x2, c = np.linspace(0,1,n*n).reshape(n,n), cmap = 'viridis', marker = 's')


# format
ax00.set_xticks([]); ax00.set_yticks([]); ax00.set_zticks([])
ax01.set_xticks([]); ax01.set_yticks([])
ax10.set_xticks([]); ax10.set_yticks([])
ax11.set_xticks([]); ax11.set_yticks([])
ax00.set_xlabel(''); ax00.set_ylabel(''); ax00.set_zlabel('')
ax00.set_xlim([0,1]); ax00.set_ylim([0,1]); ax00.set_zlim([0,1])
# - - - - 

## Condensation
m1 = 1
b1 = 0
# x1 = np.linspace(0,1,n)

m2 = 1
b2 = 0
# x2 = np.linspace(0,1,n)

l = np.linspace(0,1,n)
x1, x2 = np.meshgrid(l,l)

w1 = .5
w2 = .5
y = w1 * (m1 * x1 + b1) + w2 * (m2 * x2 + b2)


ax00 = plt.subplot(gs[0,3], projection = '3d')
ax00.plot_surface(
	x1,
	x2, 
	y,
	facecolors = cm.viridis(np.linspace(0,1,n*n).reshape(n,n)),
)

ax01 = plt.subplot(gs[1,3])
ax01.scatter(x1, y, c = np.linspace(0,1,n*n).reshape(n,n), cmap = 'viridis', marker = 's')

ax10 = plt.subplot(gs[0,4])
ax10.scatter(x2, y, c = np.linspace(0,1,n*n).reshape(n,n), cmap = 'viridis', marker = 's')

ax11 = plt.subplot(gs[1,4])
ax11.scatter(x1, x2, c = np.linspace(0,1,n*n).reshape(n,n), cmap = 'viridis', marker = 's')


# format
ax00.set_xticks([]); ax00.set_yticks([]); ax00.set_zticks([])
ax01.set_xticks([]); ax01.set_yticks([])
ax10.set_xticks([]); ax10.set_yticks([])
ax11.set_xticks([]); ax11.set_yticks([])
ax00.set_xlabel(''); ax00.set_ylabel(''); ax00.set_zlabel('')
ax00.set_xlim([0,1]); ax00.set_ylim([0,1]); ax00.set_zlim([0,1])

# plt.show()
plt.savefig('test.png')














