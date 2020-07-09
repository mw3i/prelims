import autograd.numpy as np 
from autograd import grad

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure()
gs = GridSpec(2,2)

filtCL = plt.subplot(gs[0,0])

d = np.array([
	[]
])

filtCL.set_xticks([]); filtCL.set_yticks([])
filtCL.set_xlabel('x1'); filtCL.set_ylabel('x2')





condCL = plt.subplot(gs[0,1])

condCL.set_xticks([]); condCL.set_yticks([])
condCL.set_xlabel('x1'); condCL.set_ylabel('x2')

# - - - -

filtFL = plt.subplot(gs[1,1], projection = '3d')
x1 = np.linspace(0,1,15)
x2 = np.linspace(0,1,15)
y = (1 * x1 + 0) + (1 * x2 + 0)
filtFL.scatter(
	x1, x2, y
)
filtFL.set_xticks([]); filtFL.set_yticks([]); filtFL.set_zticks([])
filtFL.set_xlabel('x1'); filtFL.set_ylabel('x2'); filtFL.set_zlabel('y')


condFL = plt.subplot(gs[1,0], projection = '3d')
x1 = np.linspace(0,1,15)
x2 = np.zeros(15)
y = (1 * x1 + 0) + (1 * x2 + 0)
condFL.scatter(
	x1, x2, y
)
condFL.set_xticks([]); condFL.set_yticks([]); condFL.set_zticks([])
condFL.set_xlabel('x1'); condFL.set_ylabel('x2'); condFL.set_zlabel('y')

plt.show()
# plt.savefig('test.png')















