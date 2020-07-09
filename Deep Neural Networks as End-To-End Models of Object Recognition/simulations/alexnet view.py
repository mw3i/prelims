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


from cifar10_models import googlenet, vgg11_bn, inception_v3
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

num_imgs = 5
num_filters = 5

fig, ax = plt.subplots(num_imgs, num_filters + 1)

pred = F.softmax(model.forward(images),1)
bestitems = torch.argsort(torch.max(pred, -1)[0]).detach().numpy().tolist()



# print(dir(model.conv1))
conv1_activation = model.conv1(images)
for i in range(num_imgs):
    for f in range(num_filters):
        ax[i,f+1].imshow(
            conv1_activation[i,f].detach().numpy(),
            cmap = 'binary',
        )

for i in range(num_imgs):
    img = images[i]
    img = img / 2 + .5
    img = np.transpose(img, (1,2,0))
    ax[i,0].imshow(img)


# ax[0,0].set_title('original')


for a in ax.flatten(): [a.set_xticks([]), a.set_yticks([])]
plt.tight_layout()
plt.savefig("filtered.png")

exit()
