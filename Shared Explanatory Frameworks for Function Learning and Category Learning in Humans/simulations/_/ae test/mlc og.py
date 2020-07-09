import autograd.numpy as np 
from autograd import grad

import matplotlib.pyplot as plt

sigmoid = lambda x: 1/(1+np.exp(-x))
linear = lambda x: x

## DATA
n = 20
es = 1000
lr = .005
ups = 10

mm = .5
mp = 1.5
x = np.linspace(mm,mp,n).reshape(n,1)
# y = x
# y = (x * 1 + 0) ** (1/2)
y = -((x-.5) ** 2) + .7
# y = x ** (1/2)
plt.scatter(x,y)


## MODEL
acth = sigmoid
acto = linear
def f(p, x):
	h = acth((x**(1/2)) @ p[0]['w'] + p[0]['b'])
	o = acto(h @ p[1]['w'] + p[1]['b'])
	return o

def l(p, x, y):
	return np.sum((y - f(p, x)) ** 2)

g = grad(l)

h = 1
par = {
	0: {
		'w': np.random.normal(0,.1,[1,h]),
		'b': np.random.normal(0,.1,[1,h]),
	},
	1: {
		'w': np.random.uniform(0,1,[h,1]),
		'b': np.random.uniform(0,1,[1,1]),
	},
}

## Train
for e in range(es):
	for i in np.random.permutation(n):
		grads = g(par, x[i:i+1], y[i:i+1])
		for u in range(ups):
			par[0]['w'] -= lr * grads[0]['w']
			par[0]['b'] -= lr * grads[0]['b']
			par[1]['w'] -= lr * grads[1]['w']
			par[1]['b'] -= lr * grads[1]['b']

xx = np.linspace(mm-.5,mp+.5,1000).reshape(1000,1)
plt.plot(
	xx,
	f(par,xx)
)

plt.xlim([mm-.5,mp+.5]); plt.ylim([mm-.5,mp+.5])
plt.savefig('test.png')














