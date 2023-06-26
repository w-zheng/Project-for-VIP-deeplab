import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from functools import partial
from conv_layers import stacked_conv
from shared_encoder import Backbone 

# instance segmentation module

###############################ASPP module###########################################

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
   
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)  

class ASPP_segmantic(nn.Module):
    def __init__(self, in_channels, atrous_rates=[6,12,18], out_channels=256):
        super(ASPP_segmantic, self).__init__()
        modules = []
   
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
    
############################################ASPP module#####################################



class SemanticDecoder(nn.Module):
    def __init__(self, in_ch=1024, in_decoder=256, in_level1=256,
                 in_low1=512, out_low1=64, in_level2=256, out_level2=32, out_decoder=256):
        super(SemanticDecoder, self).__init__()

        self.aspp = ASPP_segmantic(in_ch)   # (32,256,4,4)

        fuse_in1 = in_level1 + out_low1
        fuse_in2 = in_level1 + out_level2

        self.fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2, conv_type='depthwise_separable_conv')
        self.fuse_conv1 = self.fuse_conv(fuse_in1, out_decoder)
        self.fuse_conv2 = self.fuse_conv(fuse_in2, out_decoder)

        self.conv0 = nn.Conv2d(in_decoder, in_level1, 1, bias=False)
        self.norm0 = nn.BatchNorm2d(in_level1)
        self.relu0 = nn.ReLU()

        self.conv1 = nn.Conv2d(in_low1, out_low1, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_low1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_level2, out_level2, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_level2)
        self.relu2 = nn.ReLU()

    def forward(self, out_1_16, out_1_8, out_1_4): #  torch.Size([32, 256, 16, 16]) torch.Size([32, 512, 8, 8]) torch.Size([32, 1024, 4, 4])
        x = self.aspp(out_1_16) # (32,256,4,4)

        conv0 = self.conv0(x)
        norm0 = self.norm0(conv0)
        activated0 = self.relu0(norm0)
        x0 = F.interpolate(activated0, size=out_1_8.size()[2:], mode='bilinear', align_corners=True)

        conv1 = self.conv1(out_1_8)
        norm1 = self.norm1(conv1)
        activated1 = self.relu1(norm1)
        x1 = torch.cat((x0, activated1), dim=1)
        x1 = self.fuse_conv1(x1)

        conv2 = self.conv2(out_1_4)
        norm2 = self.norm2(conv2)
        activated2 = self.relu2(norm2)
        x2 = F.interpolate(x1, size=out_1_4.size()[2:], mode='bilinear', align_corners=True)
        x2 = torch.cat((x2, activated2), dim=1)
        x2 = self.fuse_conv2(x2)

        return x2
    


class Semantic_head(nn.Module):
    def __init__(self, num_class=20, decoder_ch=256, head_ch=256):
        super(Semantic_head, self).__init__()
        self.fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                                 conv_type='depthwise_separable_conv')
        self.fuse_con = self.fuse_conv(decoder_ch, head_ch)
        self.conv = nn.Conv2d(head_ch, num_class, 1, bias=False)

    def forward(self, x):
        h1 = self.fuse_con(x)
        pred = self.conv(h1)
        pred = F.interpolate(pred, scale_factor=4, mode='bilinear', align_corners=True)
        return pred

    
class Depth_head(nn.Module):
    def __init__(self, num_class=1, decoder_ch=256, head_ch_1=32, head_ch_2 = 64):
        super(Depth_head, self).__init__()
        self.fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                                 conv_type='depthwise_separable_conv')
        self.fuse_conv_depth = partial(stacked_conv, kernel_size=3, stride=2, num_stack=1, padding=1,
                                 conv_type='depthwise_separable_conv')

        self.fuse_con = self.fuse_conv(decoder_ch, head_ch_1)
        self.fuse_con_depth = self.fuse_conv_depth(head_ch_1, head_ch_2)
        self.conv = nn.Conv2d(head_ch_2, num_class, 1, bias=False)

    def forward(self, x):
        h1 = self.fuse_con(x)
        # print('h1',h1.shape)
        h2 = self.fuse_con_depth(h1)
        h2 = F.interpolate(h2, scale_factor=8, mode='bilinear', align_corners=True)
        # print('h2',h1.shape)
        pred = self.conv(h2)
        pred = torch.sigmoid(pred) * 88
        return pred

class FinalSemanticDecoder(nn.Module):
    def __init__(self):
        super(FinalSemanticDecoder, self).__init__()
        self.backbone = Backbone()
        self.semantic_decoder = SemanticDecoder()
        self.semantic_pre = Semantic_head()  
        self.depth_head = Depth_head()

    def forward(self, features):
        low_level_features ,mid_level_features ,high_level_features  = self.backbone(features)
        semantic_feature = self.semantic_decoder(high_level_features, mid_level_features, low_level_features)

        depth_prediction = self.depth_head(semantic_feature)
        semantic_pre = self.semantic_pre(semantic_feature)

        return semantic_pre, depth_prediction
