import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torchvision.models import AlexNet

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


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=20,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=20,
                                         shuffle=False, num_workers=2)

classes = ['plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# - - - - -


from cifar10_models import *
model_name = 'vgg11'
model = googlenet(pretrained = True)




# exit()
dataiter = iter(trainloader)
images, labels = dataiter.next()

# make black and white
# images = images.mean(axis = 1, keepdims = True)
# images = torch.cat([images, images, images], 1)

# take away center of image
blocked_images = images.clone()
blocked_images[:,:,5:-5,5:-5] = -1


fig, ax = plt.subplots(3,4)

pred = F.softmax(model.forward(images),1)
bestitems = torch.argsort(torch.max(pred, -1)[0]).detach().numpy().tolist()

shuffed = []
for img in images:
    shuffed.append(shuffle(img).reshape(-1,*img.shape))
shuffed = torch.cat(shuffed,0)

for _, i in enumerate(bestitems[-4:]):
    img = images[i]
    img = img / 2 + .5
    img = np.transpose(img, (1,2,0))
    ax[0,_].imshow(img)

    prediction = F.softmax(model.forward(images),1)[i:i+1]
    ax[0,_].set_title('True Label: ' + classes[labels[i].detach().numpy()])
    ax[0,_].set_xlabel('Prediction: ' + classes[torch.argmax(prediction,1)] + '\nconf: ' + str(torch.max(prediction).detach().numpy().round(2)))

    ## shuffle image
    shuffled_img = shuffed[i]

    img = shuffled_img / 2 + .5
    img = np.transpose(img, (1,2,0))
    ax[1,_].imshow(img)

    prediction = F.softmax(model.forward(shuffed),1)[i:i+1]
    ax[1,_].set_xlabel('Prediction: ' + classes[torch.argmax(prediction,1)] + '\nconf: ' + str(torch.max(prediction).detach().numpy().round(2)))


    ## blocked image
    blocked_image = blocked_images[i]

    img = blocked_image / 2 + .5
    img = np.transpose(img, (1,2,0))
    ax[2,_].imshow(img)

    prediction = F.softmax(model.forward(blocked_images),1)[i:i+1]
    ax[2,_].set_xlabel('Prediction: ' + classes[torch.argmax(prediction,1)] + '\nconf: ' + str(torch.max(prediction).detach().numpy().round(2)))




ax[0,0].set_ylabel('original\nimage')
ax[1,0].set_ylabel('shuffled\nimage')
ax[2,0].set_ylabel('blocked\nimage')



for a in ax.flatten(): [a.set_xticks([]), a.set_yticks([])]
# plt.suptitle(model_name)
plt.tight_layout()
plt.savefig('test.png')


exit()





