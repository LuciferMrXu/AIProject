import torchvision
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch import nn
import torch
import matplotlib.pyplot as plt
from icecream import ic
from PIL import Image
import numpy as np


def visualize_model(model, input_, output):
    width = 8
    fig, ax = plt.subplots(output[0].shape[0] // width, width, figsize=(20, 20))

    for i in range(output[0].shape[0]):
        ix = np.unravel_index(i, ax.shape)
        plt.sca(ax[ix])
        ax[ix].title.set_text('filter-{}'.format(i))
        plt.imshow(output[0][i].detach())

    plt.show()


preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

resnet = torchvision.models.resnet18(pretrained=True) # transfer step 1: load pretrained model

conv_model = [m for _, m in resnet.named_modules() if isinstance(m, torch.nn.Conv2d)]

for m in conv_model:
    m.register_forward_hook(visualize_model)

myself = preprocess(Image.open('dataset/today.jpg'))

with torch.no_grad():
    resnet(myself.unsqueeze(0)) # un-squeeze for convert myself to [ [myself] ]


