import autograd.numpy as np 
from autograd import grad

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,4, figsize = [8,2])

s = lambda x: 1 / (1 + np.exp(-x))

def f(p,x):
	return s(x @ p['iw'] + p['ib']) @ p['ow'] + p['ob']

def l(p,x,y):
	return np.sum(np.square(f(p,x) - y))

def upd(p,x,y):
	gradients = g(p,X[i:i+1],Y[i:i+1])
	p['iw'] -= lr * gradients['iw']
	p['ib'] -= lr * gradients['ib']
	p['ow'] -= lr * gradients['ow']
	p['ob'] -= lr * gradients['ob']
	return p

g = grad(l)

p = {
	'iw': np.array([
		[5,]
	]),
	'ib': np.array([
		[-2.5,],
	]),
	'ow': np.array([
		[1.]
	]),
	'ob': np.array([
		[0]
	]),
}

X = np.linspace(.3,.7,10).reshape(-1,1)
T = np.concatenate([np.linspace(0,.25,5),np.linspace(.75,1,5)]).reshape(-1,1)
Tt = np.linspace(0,1,1000).reshape(-1,1)

lr = .2
Y = X
yT = Tt
# for e in range(100):
	# for i in np.random.permutation(X.shape[0]):
	# for i in np.random.permutation(X.shape[0]):
		# p = upd(p,X[i:i+1],Y[i:i+1])

ax[0].plot(Tt,yT, c = 'black', alpha = .8); ax[0].scatter(X,Y, alpha = .5)

ax[0].scatter(X,f(p,X), alpha = .5, label = 'training')
ax[0].scatter(T,f(p,T), alpha = .5, label = 'transfer')
ax[0].legend()

#--------------------------------
#-------------------------------------------
#--------------------------------

p = {
	'iw': np.array([
		[3.]
	]),
	'ib': np.array([
		[-1.],
	]),
	'ow': np.array([
		[1]
	]),
	'ob': np.array([
		[-.2]
	]),
}

lr = .2
# Y = (X ** (1/2)) - .1
# yT = (T ** (1/2)) - .1
Y = - np.exp(-X) + 1
yT = - np.exp(-Tt) + 1
# Y = 200 * (1 - np.exp(-X/25))
# yT = 200 * (1 - np.exp(-T/25))
# for e in range(100):
	# for i in range(X.shape[0]):
	# for i in np.random.permutation(X.shape[0]):
		# p = upd(p,X[i:i+1],Y[i:i+1])

ax[1].plot(Tt,yT, c = 'black', alpha = .8); ax[1].scatter(X,Y, alpha = .5)

ax[1].scatter(X,f(p,X), alpha = .5)
ax[1].scatter(T,f(p,T), alpha = .5)

#--------------------------------
#-------------------------------------------
#--------------------------------


p = {
	'iw': np.array([
		[1.]
	]),
	'ib': np.array([
		[0.],
	]),
	'ow': np.array([
		[1.]
	]),
	'ob': np.array([
		[-.1]
	]),
}

lr = .2
Y = -((X-.5) ** 2) + .5 # <-- change all these to lambdas
yT = -((Tt-.5) ** 2) + .5
# for e in range(100):
	# for i in range(X.shape[0]):
	# for i in np.random.permutation(X.shape[0]):
		# p = upd(p,X[i:i+1],Y[i:i+1])

ax[2].plot(Tt,yT, c = 'black', alpha = .8); ax[2].scatter(X,Y, alpha = .5)

ax[2].scatter(X,f(p,X), alpha = .5)
ax[2].scatter(T,f(p,T), alpha = .5)

#--------------------------------
#-------------------------------------------
#--------------------------------


p = {
	'iw': np.array([
		[10,-10]
	]),
	'ib': np.array([
		[-1.,9],
	]),
	'ow': np.array([
		[.5],
		[.5]
	]),
	'ob': np.array([
		[-.41]
	]),
}

T = np.linspace(.3,.7,10).reshape(-1,1)
X = np.concatenate([np.linspace(0,.25,5),np.linspace(.75,1,5)]).reshape(-1,1)
Tt = np.linspace(0,1,1000).reshape(-1,1)

lr = .2
Y = -np.abs(X-.5) + .75 # <-- change all these to lambdas
yT = -np.abs(Tt-.5) + .75

# for e in range(100):
	# for i in range(X.shape[0]):
	# for i in np.random.permutation(X.shape[0]):
		# p = upd(p,X[i:i+1],Y[i:i+1])

ax[3].plot(np.linspace(0,.4,50),-np.abs(np.linspace(0,.4,50)-.5)+.75, c = 'black', alpha = .8)
ax[3].plot(np.linspace(.6,1,50),-np.abs(np.linspace(.6,1,50)-.5)+.75, c = 'black', alpha = .8)

ax[3].scatter(X,Y, alpha = .5)

ax[3].scatter(X,f(p,X), alpha = .5)
ax[3].scatter(T,f(p,T), alpha = .5)
#--------------------------------
#-------------------------------------------
#--------------------------------


for a in ax.flatten(): [a.set_xlim([0,1]), a.set_ylim([0,1])]
for a in ax.flatten(): [a.set_xticks([]), a.set_yticks([])]
plt.tight_layout()
plt.savefig('test.png')














