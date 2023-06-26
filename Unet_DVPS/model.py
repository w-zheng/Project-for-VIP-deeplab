import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class Model(nn.Module):

    def __init__(self):
        super().__init__()

        # self.base_model = torchvision.models.resnet50(pretrained=True)

        # Backbone
        self.base_model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)

        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer1_1x1 = convrelu(256, 256, 1, 0)
        self.layer2 = self.base_layers[5]
        self.layer2_1x1 = convrelu(512, 512, 1, 0)
        self.layer3 = self.base_layers[6]
        self.layer3_1x1 = convrelu(1024, 1024, 1, 0)
        self.layer4 = self.base_layers[7]
        self.layer4_1x1 = convrelu(2048, 2048, 1, 0)

        # Semantic Seg Head
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(1024 + 2048, 1024, 3, 1)
        self.conv_up2 = convrelu(512 + 1024, 512, 3, 1)
        self.conv_up1 = convrelu(256 + 512, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 64, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, 20, 1)

        # Depth Estimation Head
        self.layer0_1x1_depth = convrelu(64, 64, 1, 0)
        self.layer2_1x1_depth = convrelu(512, 512, 1, 0)
        self.layer3_1x1_depth = convrelu(1024, 1024, 1, 0)
        self.layer4_1x1_depth = convrelu(2048, 2048, 1, 0)


        self.upsample_depth = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3_depth = convrelu(1024 + 2048, 1024, 3, 1)
        self.conv_up2_depth = convrelu(512 + 1024, 512, 3, 1)
        self.conv_up1_depth = convrelu(256 + 512, 256, 3, 1)
        self.conv_up0_depth = convrelu(64 + 256, 64, 3, 1)

        self.conv_original_size0_depth = convrelu(3, 64, 3, 1)
        self.conv_original_size1_depth = convrelu(64, 64, 3, 1)
        self.conv_original_size2_depth = convrelu(128, 64, 3, 1)

        self.conv_last_depth = nn.Conv2d(64, 1, 1)

    def forward(self, input):
        # backbone
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # Semantic Seg Head
        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)

        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)
        x = self.upsample(x)

        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)
        x = self.upsample(x)

        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)
        x = self.upsample(x)

        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)
        out = torch.sigmoid(self.conv_last(x))
        
        ## Depth Estimation Head
        layer4_depth = self.layer4_1x1(layer4)
        x_depth = self.upsample_depth(layer4_depth)

        layer3_depth = self.layer3_1x1_depth(layer3)
        x_depth = torch.cat([x_depth, layer3_depth], dim=1)
        x_depth = self.conv_up3(x_depth)
        x_depth = self.upsample_depth(x_depth)

        layer2_depth = self.layer2_1x1_depth(layer2)
        x_depth = torch.cat([x_depth, layer2_depth], dim=1)
        x_depth = self.conv_up2(x_depth)
        x_depth = self.upsample_depth(x_depth)

        layer1_depth = self.layer1_1x1(layer1)
        x_depth = torch.cat([x_depth, layer1_depth], dim=1)
        x_depth = self.conv_up1(x_depth)
        x_depth = self.upsample_depth(x_depth)

        layer0_depth = self.layer0_1x1(layer0)
        x_depth = torch.cat([x_depth, layer0_depth], dim=1)
        x_depth = self.conv_up0_depth(x_depth)
        x_depth = self.upsample_depth(x_depth)
        x_depth = torch.cat([x_depth, x_original], dim=1)
        x_depth = self.conv_original_size2_depth(x_depth)

        out_depth = torch.sigmoid(self.conv_last_depth(x_depth))*20000
        # print(out.shape)
        return out, out_depth
