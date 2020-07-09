import autograd.numpy as np 
from autograd import grad

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,2, figsize = [8,4])

n = 20
es = 20
lr = .001
ups = 20
p = (2)
# x,y = np.random.multivariate_normal(
# 	[.5,.5],
# 	[ [1,.9],
# 	  [.9,1] ],
# 	n,
# ).T
x = np.linspace(0.01,2,n)
y = (x * 1 + 0) ** (p)
ax[0].scatter(x,y)

## Model
def fl(args):
	w,x,b,y,w2,b2,h = args
	y_ = ((x * w + b) + (h * w2 + b)) ** (p)
	h = (x * w + b)
	return np.sum((y - y_) ** 2)

g = grad(fl)


## Train
w, w2, b, b2 = 0.01, .01, 0.0, .0
h = 0.0
n_ord = []
for e in range(es):
	for i in range(n):
		grads = g([w,x[i],b,y[i],w2,b2,h])
		for u in range(ups):
			w -= lr * grads[0] 
			w2 -= lr * grads[-3]
			b -= lr * grads[2]
			b2 -= lr * grads[-2]
	n_ord.append(fl([w,x,b,y,w2,b2,h]))

ax[0].plot(
	x,
	(x * w + b) ** (p)
)
ax[1].plot(n_ord, alpha = .5, linewidth = 3)


## Train rand
w, w2, b, b2 = 0.01, .01, 0.0, .0
h = 0.0
n_rand = []
for e in range(es):
	for i in np.random.permutation(n):
		grads = g([w,x[i],b,y[i],w2,b2,h])
		for u in range(ups):
			w -= lr * grads[0]
			w2 -= lr * grads[-3]
			b -= lr * grads[2]
			b2 -= lr * grads[-2]
	n_rand.append(fl([w,x,b,y,w2,b2,h]))

ax[0].plot(
	x,
	(x * w + b) ** (p)
)
ax[1].plot(n_rand, alpha = .5, linewidth = 3)




plt.savefig('test.png')














