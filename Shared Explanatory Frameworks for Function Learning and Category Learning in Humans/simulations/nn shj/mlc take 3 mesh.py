import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec

fig = plt.figure()
gs = GridSpec(20,6)

ax = plt.subplot(gs[:-8,:], projection = '3d')
s = lambda x: 1 / (1 + np.exp(-x))

l = np.linspace(-3,3,10)
grid = np.array(np.meshgrid(l,l)).reshape(2,10*10).T
XX, YY = np.meshgrid(l,l)

weights = np.array([
	[1,-1,0,0],
	[0,0,1,-1]
])

bias = np.array([
	[1,0,1,0]
])

b = 0
print(s(grid @ weights + bias).sum(axis=-1))
activation = s(grid @ weights + bias).sum(axis = -1) - b

print(activation.shape)
ax.plot_surface(
	XX, YY, activation.reshape(10,10),
)

h11 = plt.subplot(gs[-8,:]); h11.set_yticks([]); h11.set_xticks([])
h12 = plt.subplot(gs[-7,:]); h12.set_yticks([]); h12.set_xticks([])
h21 = plt.subplot(gs[-6,:]); h21.set_yticks([]); h21.set_xticks([])
h22 = plt.subplot(gs[-5,:]); h22.set_yticks([]); h22.set_xticks([])

b1 = plt.subplot(gs[-4,:]); b1.set_yticks([]); b1.set_xticks([])
b2 = plt.subplot(gs[-3,:]); b2.set_yticks([]); b2.set_xticks([])
b3 = plt.subplot(gs[-2,:]); b3.set_yticks([]); b3.set_xticks([])
b4 = plt.subplot(gs[-1,:]); b4.set_yticks([]); b4.set_xticks([])


sh11 = Slider(h11, '11', -5, 5, valinit=1, valstep=.1)
sh12 = Slider(h12, '12', -5, 5, valinit=-1, valstep=.1)
sh21 = Slider(h21, '21', -5, 5, valinit=1, valstep=.1)
sh22 = Slider(h22, '22', -5, 5, valinit=-1, valstep=.1)

sb1 = Slider(b1, 'b1', -5, 5, valinit=1, valstep=.1)
sb2 = Slider(b2, 'b2', -5, 5, valinit=0, valstep=.1)
sb3 = Slider(b3, 'b3', -5, 5, valinit=1, valstep=.1)
sb4 = Slider(b4, 'b4', -5, 5, valinit=0, valstep=.1)


def update(val):
    weights[0,0] = sh11.val
    weights[0,1] = sh12.val
    weights[1,2] = sh21.val
    weights[1,3] = sh22.val
    
    bias[0,0] = sb1.val
    bias[0,1] = sb2.val
    bias[0,2] = sb3.val
    bias[0,3] = sb4.val

    ax.clear()
    activation = s(grid @ weights + bias).sum(axis = -1) - b
    ax.plot_surface(
    	XX, YY,
    	activation.reshape(10,10),
    )

    fig.canvas.draw_idle()


sh11.on_changed(update)
sh12.on_changed(update)
sh21.on_changed(update)
sh22.on_changed(update)

sb1.on_changed(update)
sb2.on_changed(update)
sb3.on_changed(update)
sb4.on_changed(update)





plt.show()
# plt.savefig('test.png')