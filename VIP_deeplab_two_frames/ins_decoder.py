import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from functools import partial
from conv_layers import stacked_conv
from shared_encoder import Backbone 


# instance segmentation for current framemodule

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

class ASPP_instance(nn.Module):
    def __init__(self, in_channels, atrous_rates=[6,12,18], out_channels=256):
        super(ASPP_instance, self).__init__()
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


class InstanceDecoder(nn.Module):
    def __init__(self, in_aspp=1024, in_decoder=256, in_level1=256,
                 in_low1=512, out_low1=32, in_level2=128,
                 in_mid2=256, out_mid2=16, out_decoder=128):
        super(InstanceDecoder, self).__init__()

        self.aspp = ASPP_instance(in_aspp) #  torch.Size([32, 256, 16, 16]) torch.Size([32, 512, 8, 8]) torch.Size([32, 1024, 4, 4])

        fuse_in1 = in_level1 + out_low1
        fuse_in2 = in_level2 + out_mid2

        self.fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2, conv_type='depthwise_separable_conv')
        self.fuse_conv1 = self.fuse_conv(fuse_in1, out_decoder)
        self.fuse_conv2 = self.fuse_conv(fuse_in2, out_decoder)

        self.conv_aspp = nn.Conv2d(in_decoder, in_level1, 1, bias=False)
        self.norm_aspp = nn.BatchNorm2d(in_level1)
        self.relu_aspp = nn.ReLU()

        self.conv_level1 = nn.Conv2d(in_low1, out_low1, 1, bias=False)
        self.norm_level1 = nn.BatchNorm2d(out_low1)
        self.relu_level1 = nn.ReLU()

        self.conv_level2 = nn.Conv2d(in_mid2, out_mid2, 1, bias=False)
        self.norm_level2 = nn.BatchNorm2d(out_mid2)
        self.relu_level2 = nn.ReLU()

    def forward(self, out_16, out_8, out_4):    #   torch.Size([32, 1024, 4, 4]) torch.Size([32, 512, 8, 8]) torch.Size([32, 256, 16, 16]) 
        x = self.aspp(out_16)   # 256 in, 256 out

        x_level1 = self.conv_aspp(x)
        x_level1 = self.norm_aspp(x_level1)
        x_level1 = self.relu_aspp(x_level1)
        x_level1 = F.interpolate(x_level1, size=out_8.size()[2:], mode='bilinear', align_corners=True) # -> 1/8,256

        x_level2 = self.conv_level1(out_8)
        x_level2 = self.norm_level1(x_level2)
        x_level2 = self.relu_level1(x_level2)
        x_level2 = torch.cat((x_level1, x_level2), dim=1)
        x_level2 = self.fuse_conv1(x_level2)

        x_output = self.conv_level2(out_4)
        x_output = self.norm_level2(x_output)
        x_output = self.relu_level2(x_output)
        x_output_ = F.interpolate(x_level2, size=out_4.size()[2:], mode='bilinear', align_corners=True)
        x_output = torch.cat((x_output_, x_output), dim=1)
        x_output = self.fuse_conv2(x_output)

        return x_output


class Instance_center_prediction_Head(nn.Module):
    def __init__(self, num_class, 
                 decoder_channel = 128, 
                 head_channel = 32 
                 ):
        super(Instance_center_prediction_Head, self).__init__()
        self.fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                            conv_type='depthwise_separable_conv') # 5X5 Conv
        self.fuse_con = self.fuse_conv(decoder_channel, head_channel)

        self.conv = nn.Conv2d(head_channel, num_class, 1, bias=False) #1X1 conv

    def forward(self, x):
        h1= self.fuse_con(x)
        pred = self.conv(h1)
        pred = F.interpolate(pred, scale_factor=4, mode='bilinear', align_corners=False)
        # pred = torch.sigmoid(pred)
        return pred
    
class Instance_center_regression_Head(nn.Module):
    def __init__(self, num_class, 
                 decoder_channel = 128, 
                 head_channel = 32 
                 ):
        super(Instance_center_regression_Head, self).__init__()
        self.fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                            conv_type='depthwise_separable_conv') # 5X5 Conv
        self.fuse_con = self.fuse_conv(decoder_channel, head_channel)

        self.conv = nn.Conv2d(head_channel, num_class, 1, bias=False) #1X1 conv

    def forward(self, x):
        h1= self.fuse_con(x)
        pred = self.conv(h1)
        pred = F.interpolate(pred, scale_factor=4, mode='bilinear', align_corners=False)
        return pred
    


class FinalInstanceDecoder(nn.Module):
    def __init__(self,):
        super(FinalInstanceDecoder, self).__init__()
        #Build Instance Decoder
        self.backbone = Backbone()
        self.instance_decoder = InstanceDecoder()
        self.instance_head_pre = Instance_center_prediction_Head(num_class=1, decoder_channel = 128)
        self.instance_head_reg = Instance_center_regression_Head(num_class=2, decoder_channel = 128)

    def forward(self, input):

        low_level_features ,mid_level_features ,high_level_features = self.backbone(input)
        instance = self.instance_decoder(high_level_features, mid_level_features, low_level_features)
        center_prediction = self.instance_head_pre(instance)
        center_regression = self.instance_head_reg(instance)

        return center_prediction, center_regression, high_level_features
    

    def loss_instance(self, center_heatmap_pred, center_heatmap_gt, offset_pred, offset_gt, instance_mask, mask_center, mask_offset):
        # Compute MSE loss for center heatmap prediction
        center_loss = F.mse_loss(center_heatmap_pred[mask_center], center_heatmap_gt[mask_center], reduction='none')
        center_loss = (center_loss * instance_mask[mask_center]).mean()

        # Compute L1 loss for offset prediction
        offset_loss = F.l1_loss(offset_pred, offset_gt, reduction='none')
        offset_loss = (offset_loss * instance_mask)[mask_offset].mean()

        return center_loss, offset_loss




