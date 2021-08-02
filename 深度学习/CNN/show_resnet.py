import cv2
import numpy as np
import os
import torch
from torchvision import transforms
from torchvision.models import resnet18
from torchsummary import summary
import matplotlib.pyplot as plt


def show_one_model(model, input_, output):
    width = 8

    fig, ax = plt.subplots(output[0].shape[0] // width, width, figsize=(20, 20))

    for i in range(output[0].shape[0]):
        ix = np.unravel_index(i, ax.shape)
        plt.sca(ax[ix])
        ax[ix].title.set_text('Filter-{}'.format(i))
        plt.imshow(output[0][i].detach())
        # plt.pause(0.05)

    input('this is conv: {}, received a {} tensor,  press any key to continue: '.format(model, input_[0].shape))

    plt.show()


def main(img):
    """
    前向传播，在传的过程中打印特征图
    """

    # 定义device、transforms
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    ])
    # 处理图片、定义模型
    img = transform(img).unsqueeze(0)
    model = resnet18(pretrained=True)

    # 打印模型summary，可用于卷积层对照
    summary(model, (3, 224, 224))

    for p in model.parameters():
        print(p)

    conv_models = [m for _, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)]

    for conv in conv_models:
        conv.register_forward_hook(show_one_model)

    # 冻结所有参数
    with torch.no_grad():
        model(img)

    # conv_models = [m for _, m in model.named_modules() if isinstance(m, torch.nn.Conv2d)]
    
    # first_conv = conv_models[0]
    
    # show_one_model(first_conv, img, output=first_conv(img))


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.realpath(__file__))
    img = cv2.imdecode(np.fromfile(os.path.join(BASE_DIR,'data\Face1.png'),dtype=np.uint8),-1)
    # img = cv2.imread(os.path.join(BASE_DIR,'data\Face1.png'))
    # 读取完毕，进行通道分离（四通道）：
    b,g,r,a = cv2.split(img)
    img = cv2.merge([b,g,r])
    main(img)
