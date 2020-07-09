import autograd.numpy as np 
from autograd import grad
s = lambda x: 1 / (1 + np.exp(-x))

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
fig = plt.figure(
	figsize = [8,4]
)
gs = GridSpec(4,8)
behavioral = np.genfromtxt('data/behavioral.csv', delimiter = ',')

structures = {
	'shj1': np.genfromtxt('data/shj1.csv', delimiter = ','),
	'shj2': np.genfromtxt('data/shj2.csv', delimiter = ','),
	'shj3': np.genfromtxt('data/shj3.csv', delimiter = ','),
	'shj4': np.genfromtxt('data/shj4.csv', delimiter = ','),
	'shj5': np.genfromtxt('data/shj5.csv', delimiter = ','),
	'shj6': np.genfromtxt('data/shj6.csv', delimiter = ','),
}

types_ax = plt.subplot(gs[:,:4])
for c, category in enumerate(structures):
	types_ax.plot(behavioral[:,c], label = 'Type ' + str(c+1))


## Model
def f(p,x):
	h = s(x @ p['input'] + p['bias']).round(2)
	print(h)
	exit()
	# h = s((x * p['input']) @ p['attn']).T
	o = h @ p['assoc']
	return o

def l(p,x,y):
	return np.sum(np.square(np.subtract(f(p,x),y)))

g = grad(l)

phi = 3
results = np.zeros([8,25,6])
for c, category in enumerate(structures):
	params = {
		# 'input': np.random.normal(0,.5,[3,3]),
		'input': np.array([
			[6,-6,0,0,0,0],
			[0,0,6,-6,0,0],
			[0,0,0,0,6,-6],
		]).astype(float),
		'bias': np.array([
			[10,-2,10,-2,10,-2],
		]),
		'attn': np.array([[.3,.3,.3]]).T,
		'assoc': np.full([3,2], 1/3)
	}

	data = structures[category]
	for e in range(25):
		for i in np.random.permutation(data.shape[0]):
			i = 0
			# set up 
			X = data[i:i+1,:-1]
			print(X)
			Y = [[-1,-1]]
			Y[0][int(data[i,-1]-1)] = 1

			# probs
			# p = f(params,X)
			p = f(params,data[:,:-1])
			exit()
			probs = np.exp(phi*p) / np.sum(np.exp(phi*p), axis = 1)
			results[i,e,c] = probs[0,int(data[i,-1]-1)]

			# update
			gradients = g(params,X,Y)
			params['input'] -= .2 * gradients['input']
			# params['attn'] -= .2 * gradients['attn']
			params['assoc'] -= .01 * gradients['assoc']
			exit()

print(results.mean(axis=0))
# exit()
res_ax = plt.subplot(gs[:,4:])
for c, category in enumerate(structures):
	res_ax.plot(
		1 - results[:,:,c].mean(axis = 0), 
		linestyle = '--',
		alpha = .5, 
	)





for a in [types_ax, res_ax]: [a.set_yticks([0,.5])]
types_ax.legend()
plt.tight_layout()
plt.savefig('test.png')