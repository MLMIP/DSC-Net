import logging
import torch.nn as nn
import torch._utils
from .GatedSpatialConv import GatedSpatialConv2d
BatchNorm2d = torch.nn.BatchNorm2d
BatchNorm2d_class = nn.BatchNorm2d
relu_inplace =True
ALIGN_CORNERS = True
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from .convnext import convnext_small,convnext_tiny
from .decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationHead, ClassificationHead,initialization

class DUpsampling(nn.Module):
    """DUsampling module"""

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(DUpsampling, self).__init__()
        self.scale_factor = scale_factor
        self.conv_w = nn.Conv2d(in_channels, out_channels * scale_factor * scale_factor, 1, bias=False)

    def forward(self, x):
        x = self.conv_w(x)
        n, c, h, w = x.size()

        # N, C, H, W --> N, W, H, C
        x = x.permute(0, 3, 2, 1).contiguous()

        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, h * self.scale_factor, c // self.scale_factor)

        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()

        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, h * self.scale_factor, w * self.scale_factor, c // (self.scale_factor * self.scale_factor))

        # N, H * scale, W * scale, C // (scale ** 2) -- > N, C // (scale ** 2), H * scale, W * scale
        x = x.permute(0, 3, 1, 2)

        return x

class Decoder_DUp(nn.Module):
    def __init__(self, in_channels, in_copy_channels, med_channels, out_channels, scale_factor=1):
        super(Decoder_DUp, self).__init__()
        self.flag = med_channels
        self.up = DUpsampling(in_channels,in_channels,scale_factor)

        self.up_conv = nn.Sequential(
            nn.Conv2d(in_channels + med_channels + in_copy_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x,  x_copy, x_med, interpolate=True): # x:h,x_med:b2t,x_copy=x
        x = self.up(x)
        if interpolate:
            # Iterpolating instead of padding gives better results
            x = F.interpolate(x, size=(x_copy.size(2), x_copy.size(3)),
                              mode="bilinear", align_corners=True)
        else:
            # Padding in case the incomping volumes are of different sizes
            diffY = x_copy.size()[2] - x.size()[2]
            diffX = x_copy.size()[3] - x.size()[3]
            x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2))
        # Concatenate
        if self.flag != 0:
            x = torch.cat([x, x_copy, x_med], dim=1)
            x = self.up_conv(x)
        else:
            x = torch.cat([x, x_copy], dim=1)
            x = self.up_conv(x)
        return x

class GFF(nn.Module):
    def __init__(self,high_feats1_channel,high_feats2_channel):
        super(GFF, self).__init__()
        self.high_feats1_channel = high_feats1_channel
        self.high_feats2_channel = high_feats2_channel
        self.low_trans = nn.Conv2d(64,2,1)
        self.Gate_Fusion_high = GatedSpatialConv2d(64, 64)  # 输入和输出先进行concate,然后进行一个CNN后用sigmoid激活得到一个attention系数矩阵

        if self.high_feats1_channel != 0:
            self.Gate_Fusion_high1 = GatedSpatialConv2d(64, 64)

        if self.high_feats2_channel != 0:
            self.Gate_Fusion_high2 = GatedSpatialConv2d(64, 64)
        # self.Gate_Fusion4 = GatedSpatialConv2d(8, 8)
        if self.high_feats1_channel != 0 and self.high_feats2_channel !=0:
            self.out_fusion = nn.Sequential(nn.Conv2d(64*3,64*3,kernel_size=3,padding=1),
                                            nn.BatchNorm2d(64*3),
                                            nn.ReLU(),
                                            nn.Conv2d(64*3,64,1))
        else:
            self.out_fusion = nn.Sequential(nn.Conv2d( 64* 2, 64*2,kernel_size=3,padding=1),
                                            nn.BatchNorm2d(64 * 2),
                                            nn.ReLU(),
                                            nn.Conv2d(64 * 2, 64, 1))

        self.squeeze_body_edge = SqueezeBodyEdge(64, nn.BatchNorm2d)
        self.edge_fusion = nn.Conv2d(64*2,64,1)
        self.seg_out = nn.Conv2d(64*2,64,1)
    def forward(self, high_feats, high_feats1, high_feats2, low_feats):
        low_feats_trans = self.low_trans(low_feats)
        low_feats_size = low_feats.size()
        high_feats = F.interpolate(high_feats, low_feats_size[2:],
                                   mode='bilinear', align_corners=True)
        if self.high_feats1_channel != 0:
            high_feats1 = F.interpolate(high_feats1, low_feats_size[2:],
                                   mode='bilinear', align_corners=True)
        if self.high_feats2_channel != 0:
            high_feats2 = F.interpolate(high_feats2, low_feats_size[2:],
                                   mode='bilinear', align_corners=True)

        fusion_high = self.Gate_Fusion_high(high_feats, low_feats_trans)
        if self.high_feats1_channel != 0:
            fusion_high1 = self.Gate_Fusion_high1(high_feats1, low_feats_trans)
        if self.high_feats2_channel != 0:
            fusion_high2 = self.Gate_Fusion_high2(high_feats2, low_feats_trans)

        if self.high_feats2_channel == 0:
            fusion_feats = torch.cat([fusion_high, fusion_high1], dim=1)
            out = self.out_fusion(fusion_feats)
        elif self.high_feats1_channel == 0:
            fusion_feats = torch.cat([fusion_high, fusion_high2], dim=1)
            out = self.out_fusion(fusion_feats)
        else:
            fusion_feats = torch.cat([fusion_high,fusion_high1,fusion_high2],dim=1)
            out = self.out_fusion(fusion_feats)
        # return out
        seg_body, seg_edge = self.squeeze_body_edge(low_feats)
        seg_edge = self.edge_fusion(torch.cat([seg_edge, out], dim=1))
        # seg_edge_out = self.edge_out(seg_edge)
        # seg_body_out = self.body_out(seg_body)

        seg_out = seg_edge + seg_body
        seg_out = self.seg_out(torch.cat([seg_out,low_feats],dim=1))
        return seg_out

class DF(nn.Module):
    def __init__(self,high_feats1_channel,high_feats2_channel):
        super(DF, self).__init__()
        self.high_feats1_channel = high_feats1_channel
        self.high_feats2_channel = high_feats2_channel
        self.low_trans = nn.Conv2d(64,2,1)
        self.Gate_Fusion_high = GatedSpatialConv2d(64, 64)  # 输入和输出先进行concate,然后进行一个CNN后用sigmoid激活得到一个attention系数矩阵

        if self.high_feats1_channel != 0:
            self.Gate_Fusion_high1 = GatedSpatialConv2d(64, 64)

        if self.high_feats2_channel != 0:
            self.Gate_Fusion_high2 = GatedSpatialConv2d(64, 64)
        # self.Gate_Fusion4 = GatedSpatialConv2d(8, 8)
        if self.high_feats1_channel != 0 and self.high_feats2_channel !=0:
            self.out_fusion = nn.Sequential(nn.Conv2d(64*3,64*3,kernel_size=3,padding=1),
                                            nn.BatchNorm2d(64*3),
                                            nn.ReLU(),
                                            nn.Conv2d(64*3,64,1))
        else:
            self.out_fusion = nn.Sequential(nn.Conv2d( 64* 2, 64*2,kernel_size=3,padding=1),
                                            nn.BatchNorm2d(64 * 2),
                                            nn.ReLU(),
                                            nn.Conv2d(64 * 2, 64, 1))

        self.squeeze_body_edge = SqueezeBodyEdge(64, nn.BatchNorm2d)
        self.edge_fusion = nn.Conv2d(64*2,64,1)
        self.seg_out = nn.Conv2d(64*2,64,1)
    def forward(self, high_feats, high_feats1, high_feats2, low_feats):
        low_feats_trans = self.low_trans(low_feats)
        low_feats_size = low_feats.size()
        high_feats = F.interpolate(high_feats, low_feats_size[2:],
                                   mode='bilinear', align_corners=True)
        if self.high_feats1_channel != 0:
            high_feats1 = F.interpolate(high_feats1, low_feats_size[2:],
                                   mode='bilinear', align_corners=True)
        if self.high_feats2_channel != 0:
            high_feats2 = F.interpolate(high_feats2, low_feats_size[2:],
                                   mode='bilinear', align_corners=True)

        fusion_high = self.Gate_Fusion_high(high_feats, low_feats_trans)
        if self.high_feats1_channel != 0:
            fusion_high1 = self.Gate_Fusion_high1(high_feats1, low_feats_trans)
        if self.high_feats2_channel != 0:
            fusion_high2 = self.Gate_Fusion_high2(high_feats2, low_feats_trans)

        if self.high_feats2_channel == 0:
            fusion_feats = torch.cat([fusion_high, fusion_high1], dim=1)
            out = self.out_fusion(fusion_feats)
        elif self.high_feats1_channel == 0:
            fusion_feats = torch.cat([fusion_high, fusion_high2], dim=1)
            out = self.out_fusion(fusion_feats)
        else:
            fusion_feats = torch.cat([fusion_high,fusion_high1,fusion_high2],dim=1)
            out = self.out_fusion(fusion_feats)
        return out

class DS(nn.Module):
    def __init__(self,low_channel=64):
        super(DS, self).__init__()
        self.squeeze_body_edge = SqueezeBodyEdge(64, nn.BatchNorm2d)
        self.edge_fusion = nn.Conv2d(64*2,64,1)
        self.seg_out = nn.Conv2d(64*2,64,1)
    def forward(self, sup_feats, low_feats):

        seg_body, seg_edge = self.squeeze_body_edge(low_feats)
        seg_edge = self.edge_fusion(torch.cat([seg_edge, sup_feats], dim=1))

        seg_out = seg_edge + seg_body
        seg_out = self.seg_out(torch.cat([seg_out,low_feats],dim=1))
        return seg_out

class SqueezeBodyEdge(nn.Module):
    def __init__(self, inplane, norm_layer):
        """
        implementation of body generation part
        :param inplane:
        :param norm_layer:
        """
        super(SqueezeBodyEdge, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            norm_layer(inplane),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=2),
            norm_layer(inplane),
            nn.ReLU(inplace=True)
        )

        self.flow_make = nn.Conv2d(inplane *2 , 2, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        size = x.size()[2:]
        seg_down = self.down(x)
        seg_down = F.upsample(seg_down, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([x, seg_down], dim=1))
        seg_flow_warp = self.flow_warp(x, flow, size)
        seg_edge = x - seg_flow_warp
        return seg_flow_warp, seg_edge

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)

        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output

class SPBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,dilation=1, downsample=None):
        super(SPBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,dilation=dilation,
                               padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        # self.ca = ChannelAttention(planes * 4)
        # self.sa = SpatialAttention(kernel_size=3)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # out = self.ca(out) * out
        # out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class SANet(nn.Module):

    def __init__(self):
        super(SANet, self).__init__() # 2222,3463
        High_Extractor = convnext_tiny(pretrained=True)
        # High_Extractor.init_weights()

        self.encoder = High_Extractor
        del High_Extractor
        # dims = [96, 192, 384, 768]
        self.decoder = UnetDecoder(
            encoder_channels=(3, 96, 192, 384, 768),
            decoder_channels=(384, 192, 96, 64),
            n_blocks=4,
            use_batchnorm=True,
            attention_type=None,
            center=True,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=64,
            out_channels=1,
            activation="sigmoid",
            kernel_size=3,
            upsampling=2
        )

        self.initialize()
        # self.decoder2 = Decoder_DUp(512, 256, 0, 256, 2)
        #
        # self.decoder1 = Decoder_DUp(256, 128, 0, 128, 2)

        # self.decoder3 = Decoder_DUp(1024, 512, 0, 512, 2)
        #
        # self.decoder2 = Decoder_DUp(512, 256, 0, 256, 2)
        #
        # self.decoder1 = Decoder_DUp(256, 128, 0, 128, 2)

        # self.out_high = nn.Sequential(
        #     nn.Conv2d(128, 64, kernel_size=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 1, kernel_size=1, bias=False))
    def initialize(self):
        initialization.initialize_decoder(self.decoder)
        initialization.initialize_head(self.segmentation_head)

    def forward(self, x):
        x_size = x.size()
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        out_high = self.segmentation_head(decoder_output)

        out_seg = []

        out_high = F.interpolate(out_high, x_size[2:],
                               mode='bilinear', align_corners=True)
        # print(out_high.size())
        # out_high = torch.sigmoid(out_high)
        out_seg.append(out_high)
        return out_seg


def get_seg_model(cfg, **kwargs):
    model = SANet()
    return model
