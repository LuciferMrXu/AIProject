import torchvision
import torch.nn.functional as F
from torchvision import models
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

cifar_10 = torchvision.datasets.CIFAR10('.', download=True, transform=preprocess)
train_loader = torch.utils.data.DataLoader(cifar_10,
                                          batch_size=512,
                                          shuffle=True)

res_net = models.resnet18(pretrained=True)


plt.imshow(cifar_10[10][0].permute(1, 2, 0))


for param in res_net.parameters():
    # 冻结反向传播
    param.requires_grad = False

# ResNet: CNN(with residual)-> CNN(with residual)-CNN(with residual)-Fully Connected

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = res_net.fc.in_features

res_net.fc = nn.Linear(num_ftrs, 10) # only update this part parameters 

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = torch.optim.SGD(res_net.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochslosses = []

epochs = 10
losses = []

for epoch in range(epochs):
    loss_train = 0
    for i, (imgs, labels) in enumerate(train_loader):        
        print(i)
        outputs = res_net(imgs)
        
        loss = criterion(outputs, labels)
        
        optimizer_conv.zero_grad()
        
        loss.backward() # -> only update fully connected layer
        
        optimizer_conv.step()
        
        loss_train += loss.item()
        
        if i > 0 and i % 10 == 0:
            print('Epoch: {}, batch: {}'.format(epoch, i))
            print('-- loss: {}'.format(loss_train / i))
            
    losses.append(loss_train / len(train_loader))


