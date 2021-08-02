import torchvision
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch import nn
import torch
import matplotlib.pyplot as plt
from icecream import ic

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

cifar_10 = torchvision.datasets.CIFAR10('.', download=True, transform=preprocess)

train_loader = torch.utils.data.DataLoader(cifar_10, batch_size=128, shuffle=True)

resnet = torchvision.models.resnet18(pretrained=True) # transfer step 1: load pretrained model

for param in resnet.parameters():
    param.requires_grad = False  # frozen weights

feature_num = resnet.fc.in_features
resnet.fc = nn.Linear(feature_num, 10)  # rewrite fc classifier

ic(resnet(cifar_10[0][0].unsqueeze(0)))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet.parameters(), lr=1e-3, momentum=0.9)

epochs = 2

losses = []

for epoch in range(epochs):
    epoch_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        ic(epoch, i)
        output = resnet(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        if i > 0:
            print('Epoch: {} batch:{}, loss ==> {}'.format(epoch, i, epoch_loss / i))

    losses.append(epoch_loss / i)

plt.plot(losses)
plt.show()

