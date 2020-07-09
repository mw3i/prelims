import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(linewidth=10000)

def shuffle(img):
    newimg = []
    ## 16 pixel; chunk size 2
    cuts = 4
    cutX = img.shape[-1] // cuts
    cutY = img.shape[-2] // cuts
    for idxX in np.random.permutation(cuts):
        img_cols = []
        for idxY in np.random.permutation(cuts):
            img_cols.append(img[:,idxY*cutY:(idxY+1)*cutY,idxX*cutX:(idxX+1)*cutX])

        newimg.append(img_cols[:])

    for r in range(len(newimg)):
        newimg[r] = torch.cat(newimg[r],-1)

    return torch.cat(newimg, -2)


# - - - - -


from cifar10_models import googlenet, vgg11_bn, inception_v3
model_name = 'vgg11'
model = googlenet(pretrained = True)


import cv2
img = cv2.imread("imgs/img.jpg")
imgtens = torch.tensor(np.array([img.transpose()]), dtype = torch.float)

from matplotlib.gridspec import GridSpec
fig, ax = plt.subplots(
    1,2,
    figsize = [10,5]
)

r = 3
gs = GridSpec(r * 2,r)


ax[0].imshow(img, cmap = 'binary')
[ax[0].set_xticks([]), ax[0].set_yticks([])]
ax[0].set_title('original')

conv1_activation = model.conv1(imgtens).detach().numpy()
ax[1].imshow(-conv1_activation[0,0].T, cmap = 'binary')
[ax[1].set_xticks([]), ax[1].set_yticks([])]
ax[1].set_title('filtered')

plt.tight_layout()
plt.savefig("filtered_single.png")

exit()
