import autograd.numpy as np 
from autograd import grad

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure()
gs = GridSpec(2,2)

m1 = .5
b1 = .5
x1 = np.sin(10*np.linspace(0,1,100))

m2 = 2
b2 = 0
x2 = np.linspace(0,1,100) ** 2

w1 = 1
w2 = 1
y = w1 * (m1 * x1 + b1) + w2 * (m2 * x2 + b1)




ax00 = plt.subplot(gs[0,0], projection = '3d')
ax00.plot(x1, x2, y)

ax01 = plt.subplot(gs[0,1])
ax01.plot(x1, y)

ax10 = plt.subplot(gs[1,0])
ax10.plot(x2, y)


plt.show()
# plt.savefig('test.png')














