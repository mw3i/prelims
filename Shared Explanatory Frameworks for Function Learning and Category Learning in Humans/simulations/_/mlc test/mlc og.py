import autograd.numpy as np 
from autograd import grad

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,2, figsize = [8,4])

n = 20
es = 20
lr = .001
# x,y = np.random.multivariate_normal(
# 	[.5,.5],
# 	[ [1,.9],
# 	  [.9,1] ],
# 	n,
# ).T
x = np.linspace(0.01,2,100)
y = (x * 1 + 0) ** (1/2)
ax[0].scatter(x,y)

h = 0
def fl(args):
	w,x,b,y = args
	y_ = (x * w + b) ** (1/2)
	return np.sum((y - y_) ** 2)
g = grad(fl)

w, b = 0.01, 0.0

## Train
n_ord = []
for e in range(es):
	for i in range(n):
		grads = g([w,x[i],b,y[i]])
		w -= lr * grads[0]
		b -= lr * grads[2]
	n_ord.append(fl([w,x,b,y]))

ax[0].plot(
	x,
	(x * w + b) ** (1/2)
)

ax[1].plot(n_ord, alpha = .5, linewidth = 3)


## Train rand
w, b = 0.01, 0.0

n_rand = []
for e in range(es):
	for i in np.random.permutation(n):
		grads = g([w,x[i],b,y[i]])
		w -= lr * grads[0]
		b -= lr * grads[2]
	n_rand.append(fl([w,x,b,y]))

ax[0].plot(
	x,
	(x * w + b) ** (1/2)
)

ax[1].plot(n_rand, alpha = .5, linewidth = 3)




plt.savefig('test.png')














