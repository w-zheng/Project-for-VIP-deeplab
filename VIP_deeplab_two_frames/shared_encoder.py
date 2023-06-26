import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights

# # Resnet50 as backbone
# class Backbone0(nn.Module): 
#     def __init__(self) :
#         super().__init__()
#         # load pretrained resnet-50
#         self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
#     def forward(self,x):
#         # low level extractor 
#         x=self.resnet.conv1(x)
#         print(x.size())
#         x=self.resnet.bn1(x)
#         x=self.resnet.relu(x)
#         low_level_features =self.resnet.maxpool(x)
#         # mid level extractor
#         x=self.resnet.layer1(low_level_features )
#         mid_level_features =self.resnet.layer2(x)
#         # high level extractor
#         high_level_features =self.resnet.layer3(mid_level_features)

#         # low_level_features=>1/4    mid_level_features=>1/8    high_level_features=>1/16
#         return low_level_features ,mid_level_features ,high_level_features 
    

# def convrelu(in_channels, out_channels, kernel, padding):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
#         nn.ReLU(inplace=True),
#     )

# class Backbone_ResNet50(nn.Module):

#     def __init__(self):
#         super().__init__()

#         self.base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
#         self.base_layers = list(self.base_model.children())
#         self.layer0 = nn.Sequential(*self.base_layers[:3])
#         self.layer1 = nn.Sequential(*self.base_layers[3:5])
#         self.layer2 = self.base_layers[5]
#         self.layer3 = self.base_layers[6]
#         self.layer4 = self.base_layers[7]
#         self.conv_original_size0 = convrelu(3, 64, 3, 1)
#         self.conv_original_size1 = convrelu(64, 64, 3, 1)


#     def forward(self, input):
#         # backbone
#         # x_original = self.conv_original_size0(input)
#         # print('0',x_original.size())

#         # x_original = self.conv_original_size1(x_original)
#         # print('1',x_original.size())

#         layer0 = self.layer0(input)
#         # print('2',layer0.size())

#         layer1 = self.layer1(layer0)
#         print('3',layer1.size())

#         layer2 = self.layer2(layer1)
#         # print('4',layer2.size())

#         layer3 = self.layer3(layer2)
#         print('5',layer3.size())

#         layer4 = self.layer4(layer3)
#         # print('6',layer4.size())

#         return layer4

class Backbone(nn.Module):
    def __init__(self, dilation=2) :
        super().__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT) # ImageNet-pretrained

        self.layer1_output = nn.Sequential(
            self.resnet.conv1, 
            self.resnet.bn1, 
            self.resnet.relu, 
            self.resnet.maxpool, 
            self.resnet.layer1
        )
        self.layer2_output = self.resnet.layer2
        self.layer3_output = self.resnet.layer3
        self.layer4_output = nn.Conv2d(1024, 1024, kernel_size=3, padding=dilation, dilation=dilation)

    def forward(self, input):
        output_layer1 = self.layer1_output(input)
        output_layer2 = self.layer2_output(output_layer1)
        output_layer3 = self.layer3_output(output_layer2)
        output_layer4 = self.layer4_output(output_layer3)

        return output_layer1, output_layer2, output_layer4


# def test_backbone_ResNet50():
#     # Backbone_ResNet50.train()
#     dtype = torch.float32
#     y = torch.zeros((32, 3, 64, 64), dtype=dtype)  # minibatch size 64, image size [3, 32, 32]
#     print('input', y.size())
#     model = Backbone_ResNet50()
#     scores = model(y)
#     print(scores.size())  

# def test_backbone():
#     dtype = torch.float32
#     y = torch.zeros((32, 3, 64, 64), dtype=dtype)  # minibatch size 64, image size [3, 32, 32]
#     print('input:', y.size())
#     model = Backbone0()
#     scores = model(y)
#     print(scores[0].size(),scores[1].size(),scores[2].size()) 

def test():
    print("testing")
    dtype = torch.float32
    y = torch.zeros((32, 3, 64, 64), dtype=dtype)  
    print('input: ', y.size())
    model = Backbone()
    scores = model(y)
    # output:  torch.Size([32, 256, 16, 16]) torch.Size([32, 512, 8, 8]) torch.Size([32, 1024, 4, 4])
    print('output: ', scores[0].size(), scores[1].size(), scores[2].size()) 

# test_backbone()
# # test_backbone_ResNet50()
# test()
