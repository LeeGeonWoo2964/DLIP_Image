**Date:**  2025-June-03

**Author:**  Geonwoo Lee 21900489

**Github:** https://github.com/LeeGeonWoo2964/DLIP_Image/tree/main/final%20Project/final%20code

**Demo Video:** https://youtu.be/1PkqeVDRZsQ

---
# Introduction
- This tutorial is about designing an image processing system to identify and select high-value recyclable materials from waste that is randomly coming in on a conveyor belt. Both the training and test datasets were created manually. In this project, the goal was to design an image processing system with over 90% precision and accuracy. To achieve this while also ensuring real-time performance, YOLOv8m was used, as it offers high accuracy and precision with fewer parameters and lower A100 TensorRT FP16 latency.

## Problem Statement
- As environmental problems are worsening day by day, the importance of resource recycling is increasing. However, the recycling rate still remains at just 6.9% (fig 1). This is because the sorting process is done manually, so recycled products are not price-competitive enough (fig 3). As shown in fig 4 below, the price of recycled aluminum, which can be sorted automatically, is only 62.5% of the price of virgin products. In contrast, the price of recycled PET, which cannot be automatically sorted, is 98% of the price of virgin products—showing little difference. Therefore, in this final project, I aim to develop an automatic waste classification system to reduce costs and improve accuracy in the recycling process. 
- Furthermore, while searching for datasets to train YOLO, I realized that datasets containing multiple objects suitable for YOLO training in a single image are rare, and labeling such datasets is difficult. Therefore, in this project, we aim to build a highly accurate model, even with a small YOLO dataset, by using YOLOv8n to determine the location of objects and a CNN model to classify the object's class.

| <figure> <p align = "center"> <img src="https://github.com/LeeGeonWoo2964/DLIP_Image/blob/main/final%20Projecct/Recycled%20Material%20Usage%20Rate.png?raw=true" width="250"> </figure> | <figure> <p align = "center"> <img src="https://github.com/LeeGeonWoo2964/DLIP_Image/blob/main/final%20Projecct/Price%20ratio%20of%20Recycled%20Product%20to%20Virgin%20Product.png?raw=true" width="250"> </figure> |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|                                                                         figure 1. Recycled Material Usage Rate                                                                          | figure 2. Price Ratio of Recycled Product to Virgin Product                                                                                                                                                          |

| <figure> <p align = "center"> <img src="https://github.com/LeeGeonWoo2964/DLIP_Image/blob/main/final%20Projecct/%EC%9A%B4%EC%98%81%20%EC%98%88%EC%82%B0%20%EB%B9%84%EC%A4%91.png?raw=true" width="400"> </figure> |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                                  figure 3. Personnel Cost of Total Annual Budget                                                                                  |


| <figure> <p align = "center"> <img src="https://github.com/LeeGeonWoo2964/DLIP_Image/blob/main/final%20Projecct/Yolo_CNN_Floechart.png?raw=true" width="800"> </figure> |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                            figure 4. Flowchart of Final LAB Code (YOLO+CNN)                                                             |


<div class="page-break" style="page-break-before: always;"></div>

---
## Expected Output
- The goal of this project is to sort randomly incoming waste items on a conveyor belt. Therefore, in this project, the waste will be mixed randomly, and I will develop an image processing system that can identify and select high-value recyclable materials such as glass, plastic, cardboard, metal. 

| <figure> <p align = "center"> <img src="https://github.com/LeeGeonWoo2964/DLIP_Image/blob/main/final%20Projecct/Expected_Output.jpg?raw=true" width="300"> </figure> |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------: |

## Evaluation
- Currently, trained workers achieve an accuracy of about 70–85% when sorting waste manually. Therefore, in this project, the target accuracy is set at over 90%. In addition, since the goal of this project is to create economic value through recycling, there should be no misclassified recyclable waste among the sorted items. For this reason, the precision is also set to a target of over 90% for various waste.
$$Accuracy=\frac{TN+TP}{ALL}$$
$$Precision=\frac{TP}{TP+FP}$$

| <figure> <p align = "center"> <img src="https://github.com/LeeGeonWoo2964/DLIP_Image/blob/main/final%20Projecct/confusion-matrix.jpeg?raw=true" width="400"> </figure> |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                            Confusion Matrix                                                                            |

<div class="page-break" style="page-break-before: always;"></div>


---
# Requirement 

## Hardware
- Conveyor Belt
- Smart Phone Camera(Galaxy S21)
- NVDIA GeForce RTX 4060 Laptop GPU
## Software 
- Python 3.9.21
- pytorch version 2.1.2
- cudatoolkit 11.8.0
- DarkLabel
- YOLOv8m
- ***CNN(이후 수정 필요)***
- [calflops](https://github.com/MrYxJ/calculate-flops.pytorch): 모델의 Flops 계산
### Software Installation
- If you are taking 'Image Processing with Deep Learning' at Handong Global University, you should have already downloaded Python 3.9.21, PyTorch 2.1.2, and CUDA Toolkit 11.8.0. If you have not yet downloaded them, please refer to the link below.
- [python](https://ykkim.gitbook.io/dlip/image-processing/tutorial/tutorial-installation-for-py-opencv)
- [pytorch](https://ykkim.gitbook.io/dlip/installation-guide/installation-guide-for-deep-learning)
- [Cudatoolkit](https://ykkim.gitbook.io/dlip/installation-guide/cuda-installation/cuda-10.2#bf6e)
#### YOLOv8
- 본 프로젝트에서는 객체를 인식하여 CNN 모델의 ROI를 설정하기 위해 YOLOv8m을 사용한다. 그렇기에 YOLOv8을 설치하여야한다. 본 문단을 작성하기 위해 한동대학교 김영근 교수님의 gitbook을 참고하였다.
- 참고: [HGU Prof. y.k.kim: Tutorial Yolov8 in Pytorch](https://ykkim.gitbook.io/dlip/deep-learning-for-perception/dp-tutorial/tutorial-yolo-in-pytorch/tutorial-yolov8-in-pytorch)
1. Creat New python environment for Yolov8
	- yolov8을 설치할 python 환경을 구축하기 위해 먼저 아나콘다 프롬포트를 사용해 yolov8이라는 이름의 새로운 python환경을 생성하고 기존에 python 환경을 복사한다. 본 프로젝트에서는 기존에 사용하던 python환경의 이름이 py39였으므로 아래의 Code 1의 명령문을 입력하였다.  
	- 만약 기존에 사용하던 python환경이 존재하지 않는다면 Code 2의 명령문을 입력하여 새로운 python 환경을 생성할 수 있다.
- Code 1
```anaconda prompt
conda create --name yolov8 --clone py39
```
- Code 2
```anaconda prompt
conda create -n yolov8 python=3.9.12
conda activate yolov8

conda install -c anaconda numpy==1.26
pip install opencv-python matplotlib

conda install pytorch=2.1 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

pip install torchsummary
```
2. Install Library including Yolov8 
	- yolo은 ultralytics 라이브러리 내부에 포함된 모델이므로 ultralytics 라이브러리만 설치하면 yolo를 사용할 수 있다. 
```anaconda prompt
conda activate yolov8

pip install ultralytics
pip install onnx  
```

---
####  DarkLabel
- [How to use DarkLabel](https://coddingjiwon.tistory.com/13)
- To train YOLO, a dataset of images with multiple mixed objects must be created. In this project, DarkLabel was used.
# Dataset
- Since there was no free dataset with mixed waste, I created one myself using DarkLabel for Yolov8.
	- [Download here](https://github.com/LeeGeonWoo2964/DLIP_Image/tree/main/final%20Project/DarkLabel_Dataset)
- Since we use YOLOv8n for object detection and a CNN model for object classification, we need a dataset for the CNN model. These data don't need to be mixed, so we'll use a trash dataset from Kaggle. For this project, we're classifying trash into only four categories: glass, metal, cardboard, and plastic. Therefore, we'll process the data from Kaggle into a final CNN model dataset with a train/validation/test split of 0.7, 0.15, and 0.15 respectively.
	- [CNN Dataset]
		- Garbage Classification
			- [Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) : download here
		- Garbage Dataset
			- [Garbage Dataset](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2): download here
---
# Method

## Deeplearning Model
### Yolov8m
- Since YOLOv8 is used only for object detection, it does not need to classify the object's class. Therefore, in this project, we compared the lightest and fastest models in the YOLOv8 series, YOLOv8n and YOLOv8m, to select the YOLO model to use. As YOLOv8n exhibited significant overfitting issues due to the small dataset, YOLOv8m was ultimately chosen.  
- However, as mentioned earlier, we were unable to obtain an image dataset with multiple objects together, which is necessary for training the YOLO model. As a result, we created approximately 30 YOLO training data points using DarkLabel. The extremely small number of training data points made overfitting unavoidable. Consequently, the model showed high performance when classifying objects included in the training data, but low performance when classifying objects not included. Specifically, as seen in the figure below, the picnic box, which was part of the training, was correctly identified as cardboard, while the milk carton, which was not trained, was incorrectly recognized as glass instead of cardboard.
<figure> <p align = "center"> <img src="https://github.com/LeeGeonWoo2964/DLIP_Image/blob/main/final%20Projecct/YOLOv8m.png?raw=true" width="700"> </figure>

|  <figure> <p align = "center"> <img src="https://github.com/LeeGeonWoo2964/DLIP_Image/blob/main/final%20Projecct/Only_Yolo_Pretrained_Output.png?raw=true" width="200"> </figure>   | <figure> <p align = "center"> <img src="https://github.com/LeeGeonWoo2964/DLIP_Image/blob/main/final%20Projecct/Only_Yolo_Pretrained_confusion_Mat.png?raw=true" width="200"> </figure>    |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
|                                                                               Pretrained waste Output                                                                               | Pretrained waste Confusion Matrix                                                                                                                                                          |
| <figure> <p align = "center"> <img src="https://github.com/LeeGeonWoo2964/DLIP_Image/blob/main/final%20Projecct/Only_Yolo_Nonpretrained_Output.png?raw=true" width="200"> </figure> | <figure> <p align = "center"> <img src="https://github.com/LeeGeonWoo2964/DLIP_Image/blob/main/final%20Projecct/Only_Yolo_Nonpretrained_confusion_Mat.png?raw=true" width="200"> </figure> |
|                                                                             Non-Pretrained waste Output                                                                             | Non-Pretrained waste Confusion Matrix                                                                                                                                                      |


| Only Yolov8m | Pretrained Waste | Non-Pretrained Waste |
| :----------: | :---------------: | :--------------------: |
|   Accracy    | 0.8              |         0.46         |
|    Recall    | 0.746            |         0.44         |
|  Precision   | 0.817            |        0.458         |
|   F1-score   | 0.77             |        0.414         |
|    FLOPs     | 92.4685 GFLOPs   |    92.4685 GFLOPs    |

### CNN model
- To determine the class of objects detected by the YOLO model, a CNN model was employed. Unlike the scarce YOLO training data, CNN training data containing only one object per image is readily available in large quantities. Therefore, using a CNN model with low computational overhead can lead to a highly accurate model while ensuring real-time performance. In this project, we selected the CNN model by comparing ResNet18 and MobileNet V2.

<figure> <p align = "center"> <img src="https://github.com/LeeGeonWoo2964/DLIP_Image/blob/main/final%20Projecct/Yolo_CNN_Floechart.png?raw=true" width="600"> </figure>
#### ResNet-18
- Before ResNet model deeper neural networks suffered from a severe vanishing gradient problem, which often led to a decrease in accuracy. ResNet addressed this issue by introducing the concept of residual learning. Residual learning involves adding the input to the output of the learned function, using this sum as the basis for further learning. These shortcut connections ensure that input information is continuously propagated through the network, thereby alleviating the vanishing gradient problem.
	- Additionally, among the ResNet series, ResNet-18 was chosen for its low computational complexity, boasting the fewest FLOPs at 1.8 FLOPs, which helps ensure real-time performance.
	- <figure> <p align = "center"> <img src="https://github.com/LeeGeonWoo2964/DLIP_Image/blob/main/final%20Projecct/ResNet%20Diagram.png?raw=true" width="500"> </figure>

#### MobileNet V2
- I also considered using MobileNet V2 in this project because real-time performance is a crucial factor. MobileNet V2 offers excellent performance despite being a lightweight model, thanks to its inverted residual blocks, linear bottlenecks, and the depthwise separable convolutions from MobileNet V1.
- An inverted residual block differs from a typical residual block in that the number of channels inside the block is greater than the number of channels outside. It still retains the fundamental characteristic of a residual block, which is to use the sum of the input and the output of the previous learning step for further training.
- A linear bottleneck structure is used within the inverted residual block to compress the number of channels when deriving the final feature map. It consolidates important information from each internal channel into a single feature map. Instead of using a common non-linear function like ReLU, a linear function is employed here. This is because using non-linear functions such as ReLU would lead to significant information loss when compressing the characteristics of the preceding feature map.
	- <figure> <p align = "center"> <img src="https://github.com/LeeGeonWoo2964/DLIP_Image/blob/main/final%20Projecct/Inverted%20Residual%20Block.png?raw=true" width="500"> </figure>
- Depthwise separable convolution can be divided into Depthwise Convolution and Pointwise Convolution steps.
	- Depthwise Convolution means applying filters independently to each input channel of the feature map, unlike standard convolution where filters are convolved across all input channels.
	- Pointwise Convolution is a convolution that combines information between the results of Depthwise Convolution and adjusts the number of channels. Since Depthwise Convolution performs filtering for each channel independently, information such as inter-channel interactions is not reflected. Additionally, Depthwise Convolution cannot adjust the number of channels in the output feature map. To compensate for this, Pointwise Convolution is used to reflect inter-channel interactions and adjust the number of channels in the output feature map.


## Test Environment

- The waste items passing on the conveyor belt will be recorded with a camera from a certain height for testing purposes.
<figure> <p align = "center"> <img src="https://github.com/LeeGeonWoo2964/DLIP_Image/blob/main/final%20Projecct/conveyor%20belt.jpg?raw=true" width="400"> </figure>
---
# Tutorial Procedure
- 본 프로젝트를 진행할 때 모델을 먼저 학습시키는 Pretrain Model 단계와 학습시킨 모델을 사용해 실제로 사용하는 YOLO+CNN Experiment 단계로 나눌 수 있다. 
## Pretrain Model

### Make YOLO Dataset
1. Using DarkLabel, I label each object's boundary and class within a single image. Then, I divide the data into training data and validation data.
<figure> <p align = "center"> <img src="https://github.com/LeeGeonWoo2964/DLIP_Image/blob/main/final%20Projecct/DarkLabel_example.png?raw=true" width="400"> </figure>
### Make CNN Dataset
1. Download the two datasets below from Kaggle and combine their metal, cardboard, plastic, and glass folders.
	-  Garbage Classification
			- [Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) : download here
	- Garbage Dataset
			- [Garbage Dataset](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2): download here
2. The data then needs to be split into training, validation, and test sets. Since these datasets are large, manual splitting like with the YOLO dataset isn't feasible. Therefore, use the code below to perform the split.
```python
import os
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, random_split
from PIL import Image
from torchvision.transforms.functional import to_pil_image

# 데이터가 우리가 T3_2에서 사용한 구조가 아니기에 T3_2의 구조로 변환환
data_dir = "./CNN_dataset"
if not (os.path.exists("./CNN_dataset/train") or os.path.exists("./CNN_dataset/test")):
    # 1. 전체 데이터셋을 들고오기
    # 2. label 별로 나누기
    # 3. label 별로 data split(train=0.8, test=0.2)
    # 4. 이를 지정된 경로에 폴더를 생성해 저장

    # 1. 전체 데이터셋 불러오기
    Dataset_total = ImageFolder(data_dir)
    print('load data')
    # 2. label 별로 나누기
    from torch.utils.data import Subset

    print('labeling data')

    cardboard_index = [i for i, (_, label) in enumerate(Dataset_total) if label == Dataset_total.class_to_idx['cardboard']]
    glass_index = [i for i, (_, label) in enumerate(Dataset_total) if label == Dataset_total.class_to_idx['glass']]
    metal_index = [i for i, (_, label) in enumerate(Dataset_total) if label == Dataset_total.class_to_idx['metal']]
    plastic_index = [i for i, (_, label) in enumerate(Dataset_total) if label == Dataset_total.class_to_idx['plastic']]
    
    cardboard_dataset = Subset(Dataset_total, cardboard_index)
    glass_dataset = Subset(Dataset_total, glass_index)
    metal_dataset = Subset(Dataset_total, metal_index)
    plastic_dataset = Subset(Dataset_total, plastic_index)

    from torch.utils.data import random_split
    def split_dataset(subset, train_ratio=0.7,val_ratio=0.15,test_ratio=0.15):
        train_size = int(len(subset) * train_ratio)
        val_size=int(len(subset) * val_ratio)
        test_size = len(subset) - train_size-val_size

        return random_split(subset, [train_size, val_size,test_size])

    print('split data')
    cardboard_train, cardboard_val, cardboard_test = split_dataset(cardboard_dataset,0.7,0.15,0.15)
    glass_train, glass_val,glass_test = split_dataset(glass_dataset,0.7,0.15,0.15)
    metal_train, metal_val,metal_test = split_dataset(metal_dataset,0.7,0.15,0.15)
    plastic_train, plastic_val,plastic_test = split_dataset(plastic_dataset,0.7,0.15,0.15)
    
    # 4. 이를 지정된 경로에 폴더를 생성해 저장
    from PIL import Image
    from torchvision.transforms.functional import to_pil_image
    save_root=data_dir
    # 폴더 생성 함수
    def save_images(subset, target_label, subset_type):
        save_dir = os.path.join(save_root, subset_type, target_label)
        os.makedirs(save_dir, exist_ok=True)

        for i in range(len(subset)):
            img, label = subset[i]  
            if isinstance(img, Image.Image): 
                img = img.convert("RGB")
            else:
                img = to_pil_image(img)

            save_path = os.path.join(save_dir, f"{target_label.lower()}_{i}.jpg")
            img.save(save_path)

    # 저장
    print('saving data')
    save_images(cardboard_train, 'cardboard', 'train')
    save_images(cardboard_val, 'cardboard', 'val')
    save_images(cardboard_test, 'cardboard', 'test')
    
    save_images(glass_train, 'glass', 'train')
    save_images(glass_val, 'glass', 'val')
    save_images(glass_test, 'glass', 'test')

    save_images(metal_train, 'metal', 'train')
    save_images(metal_val, 'metal', 'val')
    save_images(metal_test, 'metal', 'test')

    save_images(plastic_train, 'plastic', 'train')
    save_images(plastic_val, 'plastic', 'val')
    save_images(plastic_test, 'plastic', 'test')


else: 
    print("already exist")
    pass
```

### YOLOv8m
- YOLO Pretrain
	- After separating the data for YOLO training, I proceeded with the training. For this, I loaded and trained a pre-trained YOLOv8m model.
```python
from ultralytics import YOLO

def train():
    # Load a pretrained YOLO model
    model = YOLO('yolov8m.pt')
    results = model.train(data='YOLO.yaml', epochs=100)
    
if __name__ == '__main__':
    train()
    print("end")
```
- YOLO.yaml
	- The YAML code below is for training the YOLO model. For both the training and validation sets, image data and corresponding `.txt` data must be present together.
```python
train: ../YOLO_dataset/images/training/
val: ../YOLO_dataset/images/validation/

# number of classes
nc: 7

# class names
names: ['battery','biological','cardboard', 'glass','metal','paper', 'plastic']
```
### CNN 
- To classify the objects extracted using the YOLO model, I compared ResNet18 and MobileNet V2, as mentioned earlier.
- For training these models, I referenced Professor Y.K. Kim's [python tutorial T3_2](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/Tutorial_PyTorch_T3_2_Transfer_Learning_using_Pre_trained_Models_(classification).ipynb) from Handong Global University.
- Pretrain Code
	- CNN_Pretrain_resnet.ipynb
	- CNN_Pretrain_MobileNetV2.ipynb
#### Declare Model
- ResNet18
```python
# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception*]
model_name = "resnet18" 

# Number of classes in the dataset

num_classes = 4

feature_extract = True   # True: only update the reshaped layer params, False: finetune the whole model,

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)  

model_ft = model_ft.to(device)
  
from torchsummary import summary

summary(model_ft, (3,input_size,input_size))
```

- MobileNet V2
	- I brought the MobileNet V2 architecture from the [MobileNet V2 GitHub repository](https://github.com/d-li14/mobilenetv2.pytorch/blob/master/models/imagenet/mobilenetv2.py).
```python
"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch.nn as nn
import math

__all__ = ['mobilenetv2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs)

```

#### Train Model
- After declaring the train, validation, and test functions, I used a loop to train the model within a range that avoids overfitting. To prevent overfitting, I added an Early Stopping feature. If the validation loss increases for 5 consecutive times within the max_epochs range, it's considered the starting point of overfitting, and the best_model before this increase is saved.

```python
def train(dataloader, model, loss_fn, optimizer, device, print_freq=15):
    model.train()
    running_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch % print_freq == 0:
            print(f"Batch {batch}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss 

    
def val(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y_pred=pred.argmax(1)
            val_loss += loss_fn(pred, y).item()
            correct += (y_pred == y).type(torch.float).sum().item()
            
    val_loss /= num_batches
    correct /= size
    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")
    return val_loss

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y_pred=pred.argmax(1);
            test_loss += loss_fn(pred, y).item()
            correct += (y_pred == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

max_epochs = 100

train_losses = []
val_losses=[]
pre_val_loss=10
early_stopping_flag=0

for t in range(max_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_epoch_loss = train(train_dataloader, model_ft, loss_fn, optimizer, device, 15)
    train_losses.append(train_epoch_loss) 
    
    val_epoch_losses=val(val_dataloader, model_ft, loss_fn, device)
    val_losses.append(val_epoch_losses) 

    if early_stopping_flag<=5: 
        if val_epoch_losses<pre_val_loss: 
            pre_val_loss=val_epoch_losses 
            best_model=model_ft 
            early_stopping_flag=0
        else:
            early_stopping_flag=early_stopping_flag+1
    elif early_stopping_flag>5:
        break
```

#### Copmare ResNet18 and MobileNet V2
- As shown in the table below, while MobileNet V2 had higher accuracy than ResNet18, after comprehensively considering precision, recall, and F1-score, I determined that the ResNet18 model was superior and therefore used it.

|           | ResNet18 | MobileNet V2 |
| :-------: | -------- | ------------ |
| Accuracy  | 0.8195   | 0.8244       |
| Precision | 0.8560   | 0.8456       |
|  Recall   | 0.8596   | 0.8174       |
| F1-score  | 0.8576   | 0.8175       |

| <figure> <p align = "center"> <img src="https://github.com/LeeGeonWoo2964/DLIP_Image/blob/main/final%20Projecct/ResNet.png?raw=true" width="200"> </figure> | <figure> <p align = "center"> <img src="https://github.com/LeeGeonWoo2964/DLIP_Image/blob/main/final%20Projecct/MobileNet.png?raw=true" width="200"> </figure> |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|                                                                  ResNet18 Confusion Matrix                                                                  | MobileNet V2 Confusion Matrix                                                                                                                                  |

---
## YOLO + CNN Experiment 
- In this stage, I classify trash using the model trained in the Pretrain Model phase. First, I'll explain the YOLO_CNN function, then move on to the main code.

<figure> <p align = "center"> <img src="https://github.com/LeeGeonWoo2964/DLIP_Image/blob/main/final%20Projecct/YOLO_CNN_Flowchart.png?raw=true" width="500"> </figure>

### YOLO_CNN
- As I've mentioned, in this project, I extract objects with YOLO and then classify the extracted objects using ResNet.

#### YOLO
- In the YOLO part stage, I recognize and extract objects from images.
```python
#YOLO=====================
    # Perform object detection on an image using the model
    result = yolo_model.predict(source=frame, iou=0.6)
    yolo_result=result[0]

    img=yolo_result.plot()
    boxes = result[0].boxes
    # Get Rectangle Point
    box_point=result[0].boxes.xyxy.cpu().numpy() #(x1,y1,x2,y2)

    if boxes is None or len(boxes) == 0:
        return frame     

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
```

#### Crop Img
- A problem can occur where a single object is incorrectly recognized as two or more objects. Therefore, I added code to ignore an object if the distance between the center points of its boundary boxes is less than a certain threshold.
```python
    Crop_tensor=[]
    pre_center_x=[]
    pre_center_y=[]
    pre_center_x.append(0)
    pre_center_y.append(0)
    CNN_src=CNN_frame #Gray Scale

    for i in range(len(box_point)):
        dx=min(abs(x-int(((box_point[i,2])+(box_point[i,0]))/2)) for x in pre_center_x)
        dy=min(abs(y-int(((box_point[i,3])+(box_point[i,1]))/2)) for y in pre_center_y)
        if dx >=80 and dy >=80:
            pre_center_x.append(int(((box_point[i,2])+(box_point[i,0]))/2))
            pre_center_y.append(int(((box_point[i,3])+(box_point[i,1]))/2))
        # 1. 인식된 객체를 자르고 tensor로 바꿔 CNN에 넣을 수 있도록 처리
            Temp_Crop=CNN_src[int(box_point[i,1]):int(box_point[i,3]),int(box_point[i,0]):int(box_point[i,2])]
            if isinstance(Temp_Crop,np.ndarray):
                Temp_tensor=Image.fromarray(Temp_Crop.astype('uint8')).convert('RGB')
                Temp_tensor=transform(Temp_tensor)
                Crop_tensor.append(Temp_tensor)
    if not Crop_tensor:
        return frame

    CNN_input_batch=torch.stack(Crop_tensor).to(device)
    #YOLO=====================
```

#### CNN - ResNet18
```python
    # Class
    classes = ['cardboard', 'glass','metal', 'plastic']
    #class마다 색깔 지정
    cls_color=[(255,0,0),(0,255,0),(0,0,255),(255,255,0)]

    # CNN
    with torch.no_grad():
        output = cnn_model(CNN_input_batch)
        _, predicted = torch.max(output, 1)
        for i, pred in enumerate(predicted):
            print(f"[{i}] Predicted class: {classes[pred.item()]}")
            probs=torch.softmax(output[i],dim=0)
            object_class=classes[pred.item()]
            object_color=cls_color[pred.item()]
            if probs[pred.item()].item()<=0.0:
                continue
            #src에 사각형 그리고 class 이름 및 로봇 팔이 잡을 지점 표시
            pt1=[int(box_point[i,0]),int(box_point[i,1])]
            pt2=[int(box_point[i,2]),int(box_point[i,3])]
            circle_x=int(((box_point[i,2])+(box_point[i,0]))/2)
            circle_y=int(((box_point[i,3])+(box_point[i,1]))/2)
            text_pt=(int(box_point[i,0])+20,int(box_point[i,1]+20))
            cv.rectangle(result_img, pt1, pt2, object_color, thickness=10, lineType=None, shift=None)
            cv.circle(result_img,(circle_x,circle_y),10,object_color,-1,cv.LINE_AA)
            cv.putText(result_img,f"({object_class} {probs[pred.item()].item():.2f})",text_pt,cv.FONT_HERSHEY_SIMPLEX,3,object_color,5,cv.LINE_AA)
 
```

### Main

<figure> <p align = "center"> <img src="https://github.com/LeeGeonWoo2964/DLIP_Image/blob/main/final%20Projecct/YOLO_CNN_Main_Flowchart.png?raw=true" width="500"> </figure>

### Connect Camera
- As previously mentioned, the camera I used in this project is a Galaxy S21. To use this smartphone model as a webcam, I installed the IP Webcam application on the smartphone and then connected its IP address within the Python code. It's important to note that the IP address varies for each phone and can sometimes change. 
- <figure> <p align = "center"> <img src="https://github.com/LeeGeonWoo2964/DLIP_Image/blob/main/final%20Projecct/IP_webcam_phone.png?raw=true" width="400"> </figure>
```python
url = "http://172.17.209.76:8080/video" 
cam = cv.VideoCapture(url)
```

- Main Code
```python 
from urllib.request import urlopen

url = "http://172.17.209.76:8080/video" # Report Connect Camera
cam = cv.VideoCapture(url)

if not cam.isOpened():
    raise RuntimeError("No Video")

check, frame = cam.read()
height, width, channels = frame.shape
height=frame.shape[0]
width=frame.shape[1]

# ROI=np.zeros((height,width,3), np.uint8)
kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('Final Demo Video.avi', fourcc, 30.0, (width, height))

i=0
while True:
    i=i+1

    check, frame = cam.read()

    frame_processed=Yolo_cnn(frame)

    # 녹화 시작
    out.write(frame)

    cv.namedWindow('frame', flags=cv.WINDOW_NORMAL)
    cv.resizeWindow('frame', width=500, height=400)
    cv.imshow('frame',frame_processed)

    key=cv.waitKey(1)
    if  key == ord("q"):
        break      
    elif key==ord("s"):
        cv.imwrite(f"{i}.jpg",frame)        


# 녹화 종료
cam.release()
out.release()
cv.destroyAllWindows()
```

****
# Result and Analysis
- The objective of this project was to implement an Automatic Waste Classifier Using an RGB Camera with an accuracy of 0.85 or higher. To address the difficulty in acquiring suitable YOLO datasets, I extracted objects using a YOLOv8m model and classified them with ResNet18.
- As a result, I was able to somewhat overcome the challenge of YOLO's susceptibility to overfitting due to the scarcity of training data. However, due to inadequate preprocessing and poorly configured shooting conditions, I could not achieve the target accuracy. Furthermore, I observed that the object's class continuously changed as the conveyor belt moved.

| <figure> <p align = "center"> <img src="https://github.com/LeeGeonWoo2964/DLIP_Image/blob/main/final%20Projecct/Demo_confusion_matrix.png?raw=true" width="400"> </figure> |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                           Demo Confusion Matrix                                                                            |

|           | Only Yolov8m <br>Pretrained Waste | Only Yolov8m<br>Non-Pretrained Waste | YOLOv8 + ResNet18 |
| :-------: | :-------------------------------: | :----------------------------------: | :-----------------: |
|  Accracy  |                0.8                |                 0.46                 | 0.7000            |
|  Recall   |               0.746               |                 0.44                 | 0.8563            |
| Precision |               0.817               |                0.458                 | 0.7833            |
| F1-score  |               0.77                |                0.414                 | 0.7953            |
- Nevertheless, the idea of using a YOLO model augmented with a CNN has a wide range of potential applications. As I mentioned earlier, YOLO is undoubtedly a powerful model, but creating its datasets is labor-intensive. However, datasets used for training CNNs are significantly easier to create and are readily available on platforms like Kaggle. Therefore, the approach of using YOLO to recognize objects and then a CNN to determine their class will remain useful as long as the method of acquiring YOLO datasets isn't automated.

---
# Reference

1. Yudin, D., Zakharenko, N., Smetanin, A., Filonov, R., Kichik, M., Kuznetsov, V., Larichev, D., Gudov, E., Budennyy, S., & Panov, A. (2024). Hierarchical waste detection with weakly supervised segmentation in images from recycling plants. _Engineering Applications of Artificial Intelligence, 128_, 107542. [https://doi.org/10.1016/j.engappai.2023.107542](https://doi.org/10.1016/j.engappai.2023.107542)
2. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 4510-4520). https://doi.org/10.1109/CVPR.2018.00474