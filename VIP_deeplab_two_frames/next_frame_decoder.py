from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from ins_decoder import FinalInstanceDecoder
from shared_encoder import Backbone
from conv_layers import stacked_conv

###############################ASPP module###########################################

class ASPP_instance_next(nn.Module):  # for different instance might use different channels.
    def __init__(self, in_channels,out_channels_1x1=32,out_channels_3x3_1=32,out_channels_3x3_2=32,out_channels_3x3_3=32,out_channels_pool=32):
        super().__init__()
        #  calculate final out channels
        self.final_channels=out_channels_1x1+out_channels_3x3_1+out_channels_3x3_2+out_channels_3x3_3+out_channels_pool
        
        
        # 1x1 convolution layer
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_1x1, kernel_size=1),
            nn.BatchNorm2d(out_channels_1x1),
            nn.ReLU(inplace=True))
        
        # 3x3 dilated convolution layers
        self.conv3x3_1 = nn.Sequential(nn.Conv2d(in_channels, out_channels_3x3_1, kernel_size=3, padding=6, dilation=6,bias=False),
                                       nn.BatchNorm2d(out_channels_3x3_1),
                                       nn.ReLU(inplace=True))
        self.conv3x3_2 = nn.Sequential(nn.Conv2d(in_channels, out_channels_3x3_2, kernel_size=3, padding=12, dilation=12,bias=False),
                                       nn.BatchNorm2d(out_channels_3x3_2),
                                       nn.ReLU(inplace=True))
        self.conv3x3_3 = nn.Sequential(nn.Conv2d(in_channels, out_channels_3x3_3, kernel_size=3, padding=18, dilation=18,bias=False),
                                       nn.BatchNorm2d(out_channels_3x3_3),
                                       nn.ReLU(inplace=True))

        # Image pooling
        self.image_pooling = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                           nn.Conv2d(in_channels, out_channels_pool, kernel_size=1,bias=False),
                                           nn.BatchNorm2d(out_channels_pool),
                                           nn.ReLU(inplace=True))    
#self.conv_image_pool =
    def forward(self, x):
        # Apply 1x1 convolution
        feat_1x1 = self.conv1x1(x)

        # Apply dilated convolutions
        feat_dilated_1 = self.conv3x3_1(x)
        feat_dilated_2 = self.conv3x3_2(x)
        feat_dilated_3 = self.conv3x3_3(x)

        # Apply image pooling
        feat_image_pool = self.image_pooling(x)
        feat_image_pool = torch.nn.functional.interpolate(feat_image_pool, size=x.size()[2:], mode='bilinear', align_corners=False)

        # Concatenate features along the channel dimension
        out = torch.cat((feat_1x1, feat_dilated_1, feat_dilated_2, feat_dilated_3, feat_image_pool), dim=1)
        return out
############################################ASPP module#####################################

# Cascade-ASPP 4X
class CascadeASPP(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.aspp1=ASPP_instance_next(in_channels=in_channels  ,out_channels_1x1=16,out_channels_3x3_1=32,out_channels_3x3_2=32,out_channels_3x3_3=32,out_channels_pool=16)
        self.aspp2=ASPP_instance_next(in_channels=self.aspp1.final_channels  ,out_channels_1x1=50,out_channels_3x3_1=32,out_channels_3x3_2=32,out_channels_3x3_3=32,out_channels_pool=16)
        self.aspp3=ASPP_instance_next(in_channels=self.aspp1.final_channels+self.aspp2.final_channels  ,out_channels_1x1=16,out_channels_3x3_1=32,out_channels_3x3_2=32,out_channels_3x3_3=32,out_channels_pool=16)
        self.aspp4=ASPP_instance_next(in_channels=2*self.aspp1.final_channels+self.aspp2.final_channels+self.aspp3.final_channels   ,out_channels_1x1=16,out_channels_3x3_1=32,out_channels_3x3_2=32,out_channels_3x3_3=32,out_channels_pool=16)

        #cause concate in the end , the fianl channels will be like: 4*temp1+2*temp2+temp3+temp4
        self.final_channels = 4*self.aspp1.final_channels+2*self.aspp2.final_channels+self.aspp3.final_channels+self.aspp4.final_channels

    def forward(self,x):
        temp1=self.aspp1(x)
        temp2=self.aspp2(temp1)

        c1=torch.cat((temp1,temp2),dim=1)
        temp3=self.aspp3(c1)

        c2=torch.cat((temp1,c1,temp3),dim=1)
        temp4=self.aspp4(c2)

        out=torch.cat((temp1,c1,c2,temp4),dim=1)
        return out

############################################CascadeASPP module#####################################

class NextInstanceDecoder(nn.Module):
    def __init__(self, in_aspp=2048, in_decoder=1092, in_level1=256,
                 in_low1=512, out_low1=32, in_level2=128,
                 in_mid2=256, out_mid2=16, out_decoder=128):
        super(NextInstanceDecoder, self).__init__()

        self.aspp = CascadeASPP(in_aspp) #  torch.Size([32, 256, 16, 16]) torch.Size([32, 512, 8, 8]) torch.Size([32, 1024, 4, 4])

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


class Next_Instance_center_regression_Head(nn.Module):
    def __init__(self, num_class, 
                 decoder_channel = 128, 
                 head_channel = 32 
                 ):
        super(Next_Instance_center_regression_Head, self).__init__()
        self.fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                            conv_type='depthwise_separable_conv') # 5X5 Conv
        self.fuse_con = self.fuse_conv(decoder_channel, head_channel)

        self.conv = nn.Conv2d(head_channel, num_class, 1, bias=False) #1X1 conv

    def forward(self, x):
        h1= self.fuse_con(x)
        pred = self.conv(h1)
        pred = F.interpolate(pred, scale_factor=4, mode='bilinear', align_corners=False)
        return pred



class InstanceNextDecoder(nn.Module):
    def __init__(self,):
        super(InstanceNextDecoder, self).__init__()

        self.backbone = Backbone()
        self.instance_last = FinalInstanceDecoder()
        self.instance_decoder = NextInstanceDecoder()
        self.instance_head_reg = Next_Instance_center_regression_Head(num_class=2, decoder_channel = 128)

    def forward(self, input, input_last):
        _,_,high_level_features_last = self.instance_last(input_last)
        low_level_features ,mid_level_features ,high_level_features = self.backbone(input)

        high_level_features = torch.cat((high_level_features_last,high_level_features), dim=1)
        instance = self.instance_decoder(high_level_features, mid_level_features, low_level_features)
        center_regression_next = self.instance_head_reg(instance)

        return center_regression_next















# class GenerateHead(nn.Module):
#     def __init__(self, num_class, # should be determined
#                  decoder_channel = 256, #channel number from decoder
#                  head_channel = 256 # Middle channel number
#                  ):
#         super(GenerateHead, self).__init__()
#         self.fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
#                                  conv_type='depthwise_separable_conv')  # 5X5 Conv
#         self.fuse_con = self.fuse_conv(decoder_channel, head_channel)

#         self.conv = nn.Conv2d(head_channel, num_class, 1, bias=False)  # 1X1 conv

#     def forward(self, x):
#         x = self.fuse_con(x)
#         x = F.interpolate(x, size=(4*x.shape[2], 4*x.shape[3]), mode='bilinear',
#                            align_corners=True)  # torch.Size([2, 32, 1024, 1024])
#         x = self.conv(x)

#         return x

# # decoder_channel = 256 #channel number from decoder
# # head_channel = 256 # Middle channel number

# # instance_1 = SingleSemanticDecoder()
# # instance_2 = GenerateHead(num_class = 2) # should be determined
# #
# # x = torch.randn(2, 3, 1024, 2048)
# # z_low, z_mid, z_high = Backbone(x)
# # xx=instance_1(z_high, z_mid, z_low)
# # out=instance_2(xx)
# # print(out.shape) # torch.Size([2, 2, 1024, 2048])


# class DepthPredictionHead(nn.Module):
#     def __init__(self,
#                  decoder_channel = 256, # The input channel size from Semantic decoder
#                  head_channel_1 = 32, # The first layer after 5X5 Conv
#                  head_channel_2 = 64, # The second layer after 3X3 Conv
#                  Dout_channel = 1, # The last output of Depth Prediction
#                  upsample_depth = (1024, 1024)
#                  ):
#         super(DepthPredictionHead, self).__init__()

#         self.fuse_conv1 = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
#                                   conv_type='depthwise_separable_conv')  # 5X5 Conv
#         self.fuse_con1 = self.fuse_conv1(decoder_channel, head_channel_1)

#         self.fuse_conv2 = partial(stacked_conv, kernel_size=3, num_stack=1, padding=1,
#                                   conv_type='depthwise_separable_conv')  # 3X3 Conv Should be discussed
#         self.fuse_con2 = self.fuse_conv2(head_channel_1, head_channel_2)

#         self.conv = nn.Conv2d(head_channel_2, Dout_channel, 1, bias=False)  # 1X1 conv

#     def forward(self, x):
#         x = self.fuse_con1(x)
#         x = F.interpolate(x, size=(4*x.shape[2], 4*x.shape[3]), mode='bilinear',
#                            align_corners=True)  # torch.Size([2, 32, 1024, 1024])
#         x = self.fuse_con2(x)  # torch.Size([2, 64, 1024, 1024])
#         x = self.conv(x)

#         return x


# # instance_1 = SingleSemanticDecoder()
# # instance_3 = DepthPredictionHead()
# #
# # x = torch.randn(2, 3, 1024, 2048)
# # z_low, z_mid, z_high = Backbone(x)
# # xx=instance_1(z_high, z_mid, z_low) # torch.Size([2, 256, 512, 512])
# # out=instance_3(xx) # torch.Size([2, 1, 1024, 2048])
# # # print(out.shape)


# '''
#     Designed three different decoder class for  training.
# '''

# class SemanticDecoder(nn.Module):
#     def __init__(self,):
#         super(SemanticDecoder, self).__init__()
#         self.semantic_decoder = SingleSemanticDecoder()
#         self.semantic_head = GenerateHead(num_class = 19) # Need to discuss and Mind! ignore index 32, form 0 to 31= 32 classes
#         self.depth_head = DepthPredictionHead()

#     def forward(self, z_high, z_mid, z_low):
#         # Semantic Depth
#         semantic = self.semantic_decoder(z_high, z_mid, z_low)
#         depth_prediction = self.depth_head(semantic)
#         semantic_prediction = self.semantic_head(semantic)
#         pred_depth = depth_prediction
#         pred_sematic = semantic_prediction

#         return pred_depth, pred_sematic

# # x = torch.randn(5, 3, 1024, 2048)
# # z_low, z_mid, z_high = Backbone(x)
# # instance_4 = SemanticDecoder()
# # depth_prediction, semantic_prediction = instance_4(z_high, z_mid, z_low)
# # print(depth_prediction.shape) # torch.Size([5, 1, 1024, 2048])
# # print(semantic_prediction.shape) # torch.Size([5, 32, 1024, 2048])


# class InstanceDecoder(nn.Module):
#     def __init__(self,):
#         super(InstanceDecoder, self).__init__()
#         #Build Instance Decoder
#         # self.aspp_channels = ASPP(in_channels=1024)
#         self.instance_decoder = SingleInstanceDecoder(low_level_channel_output = 32, mid_level_channel_output = 16)
#         self.instance_head_pre = GenerateHead(num_class=1, decoder_channel = 128)
#         self.instance_head_reg = GenerateHead(num_class=2, decoder_channel=128)

#     def forward(self, z_high, z_mid, z_low):
#         # instance center
#         instance = self.instance_decoder(z_high, z_mid, z_low)
#         instance_prediction = self.instance_head_pre(instance)
#         instance_regression = self.instance_head_reg(instance)

#         return instance_prediction, instance_regression

# # x = torch.randn(2, 3, 1024, 2048)

# # instance_4 = InstanceDecoder()
# # instance_prediction, instance_regression = instance_4(x)
# # print(instance_prediction.shape) #torch.Size([2, 1, 1024, 2048])
# # print(instance_regression.shape) #torch.Size([2, 2, 1024, 2048])

# class NextFrameDecoder(nn.Module):
#     def __init__(self,fuse_in_channel = 2048, fuse_out_channel = 1024):
#         super(NextFrameDecoder, self).__init__()
#         #Build Instance Decoder
#         # self.aspp_channels = ASPP(in_channels=1024)
#         self.Backbone = Backbone
#         self.instance_decoder = NextInstanceDecoder(low_level_channel_output=32, mid_level_channel_output=16)
#         self.instance_head = GenerateHead(num_class=2, decoder_channel=128)

#     def forward(self, z_high, z_mid1, z_low1):
#         # Next-frame regression
#         instance = self.instance_decoder(z_high, z_mid1, z_low1)
#         next_regression = self.instance_head(instance)

#         return next_regression

# # x = torch.randn(5, 3, 1024, 2048)
# # xx = torch.randn(5, 3, 1024, 2048)
# #
# # instance_14 = NextFrameDecoder()
# # next_instance_regression = instance_14(x, xx)
# # print(next_instance_regression.shape) # torch.Size([5, 2, 1024, 2048])


# class DecoderArch(nn.Module):
#     def __init__(self,):
#         super(DecoderArch, self).__init__()
#         self.Backbone = Backbone
#         self.Semantic = SemanticDecoder()
#         self.Instance = InstanceDecoder()
#         self.NextFrameInstance = NextFrameDecoder()

#     def forward(self, featuresT0, featuresT1):
#         z_low0, z_mid0, z_high0 = self.Backbone(featuresT0)  # z_high0 = torch.Size([5, 1024, 64, 128])
#         z_low1, z_mid1, z_high1 = self.Backbone(featuresT1)  # z_high1 = torch.Size([5, 1024, 64, 128])
#         z_high = torch.cat((z_high0, z_high1), dim=1)  # torch.Size([5, 2048, 64, 128])


#         depth_prediction, semantic_prediction = self.Semantic(z_high0, z_mid0, z_low0)
#         instance_prediction, instace_regression = self.Instance(z_high0, z_mid0, z_low0)
#         next_frame_instance = self.NextFrameInstance(z_high, z_mid1, z_low1)

#         return depth_prediction, semantic_prediction, instance_prediction, instace_regression, next_frame_instance

# '''
# Decoder Arch:
# Input: Frame T data, Frame T+1 data
# Output: 
#         Depth prediction, 
#         Semantic prediction, 
#         Instace center prediction, 
#         Instance center regression, 
#         Next-frame instance center regression
# '''

# # x = torch.randn(2, 3, 1024, 2048)
# # xx = torch.randn(2, 3, 1024, 2048)
# #
# # instance_15 = DecoderArch()
# #
# # depth_prediction, semantic_prediction, instance_prediction, instace_regression, next_frame_instance = instance_15(x, xx)
# #
# # print(depth_prediction.shape) # torch.Size([5, 1, 1024, 2048])
# # print(semantic_prediction.shape) # torch.Size([5, 19, 1024, 2048])
# # print(instance_prediction.shape) # torch.Size([5, 1, 1024, 2048])
# # print(instace_regression.shape) # torch.Size([5, 2, 1024, 2048])
# # print(next_frame_instance.shape) # torch.Size([5, 2, 1024, 2048])