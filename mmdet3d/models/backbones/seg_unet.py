import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import load_checkpoint
from torch import nn as nn

from mmdet.models import BACKBONES

import ipdb

def conv1x3(in_filters, out_filters, conv_cfg, stride=1):
    return build_conv_layer(conv_cfg,
                            in_filters,
                            out_filters,
                            (1, 3),
                            stride=1,
                            padding=(0, 1))

def conv3x1(in_filters, out_filters, conv_cfg, stride=1):
    return build_conv_layer(conv_cfg,
                            in_filters,
                            out_filters,
                            (3, 1),
                            stride=1,
                            padding=(1, 0))

def conv3x3(in_filters, out_filters, conv_cfg, stride=1, padding=1):
    return build_conv_layer(conv_cfg,
                            in_filters,
                            out_filters,
                            3,
                            stride=stride,
                            padding=padding)



class Asym_ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, conv_cfg, norm_cfg, pooling=False, pool_padding=1):
        super(Asym_ResBlock, self).__init__()
        self.pooling = pooling

        self.l_conv0 = conv1x3(in_filters, out_filters, conv_cfg)
        self.l_bn0 = build_norm_layer(norm_cfg, out_filters)[1]
        self.l_act0 = nn.ReLU(inplace=True)

        self.l_conv1 = conv3x1(out_filters, out_filters, conv_cfg)
        self.l_bn1 = build_norm_layer(norm_cfg, out_filters)[1]
        #self.l_act1 = nn.ReLU(inplace=True)

        self.r_conv0 = conv3x1(in_filters, out_filters, conv_cfg)
        self.r_bn0 = build_norm_layer(norm_cfg, out_filters)[1]
        self.r_act0 = nn.ReLU(inplace=True)

        self.r_conv1 = conv1x3(out_filters, out_filters, conv_cfg)
        self.r_bn1 = build_norm_layer(norm_cfg, out_filters)[1]

        self.act = nn.ReLU(inplace=True)

        if self.pooling:
            self.pool = conv3x3(out_filters, out_filters, conv_cfg, stride=2, padding=pool_padding)

    def forward(self, x):
        shortcut = self.l_conv0(x)
        shortcut = self.l_bn0(shortcut)
        shortcut = self.l_act0(shortcut)

        shortcut = self.l_conv1(shortcut)
        shortcut = self.l_bn1(shortcut)

        feat = self.r_conv0(x)
        feat = self.r_bn0(feat)
        feat = self.r_act0(feat)

        feat = self.r_conv1(feat)
        feat = self.r_bn1(feat)

        #out = self.act(feat + shortcut)
        lateral = feat + shortcut

        if self.pooling:
            out_pool = self.act(lateral)
            out_pool = self.pool(out_pool)
            return out_pool, lateral
        else:
            return self.act(lateral)


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, conv_cfg, norm_cfg, upsample_cfg, up_output_padding=0):
        super(UpBlock, self).__init__()
        # self.drop_out = drop_out
        self.trans_conv = conv3x3(in_filters, out_filters, conv_cfg)
        self.trans_bn = build_norm_layer(norm_cfg, out_filters)[1]
        self.trans_act = nn.ReLU(inplace=True)

        self.conv0 = conv1x3(out_filters, out_filters, conv_cfg)
        self.bn0 = build_norm_layer(norm_cfg, out_filters)[1]
        self.act0 = nn.ReLU(inplace=True)

        self.conv1 = conv3x1(out_filters, out_filters, conv_cfg)
        self.bn1 = build_norm_layer(norm_cfg, out_filters)[1]
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_filters, out_filters, conv_cfg)
        self.bn2 = build_norm_layer(norm_cfg, out_filters)[1]
        self.act2 = nn.ReLU(inplace=True)

        self.up_conv = build_upsample_layer(
                           upsample_cfg,
                           in_channels=out_filters,
                           out_channels=out_filters,
                           kernel_size=2,
                           stride=2,
                           padding=0,
                           output_padding=up_output_padding)

    def forward(self, x, lateral):
        up_feat = self.trans_conv(x)
        up_feat = self.trans_bn(up_feat)
        up_feat = self.trans_act(up_feat)

        ## upsample
        up_feat = self.up_conv(up_feat)

        up_feat = up_feat + lateral

        up_feat = self.conv0(up_feat)
        up_feat = self.bn0(up_feat)
        up_feat = self.act0(up_feat)

        up_feat = self.conv1(up_feat)
        up_feat = self.bn1(up_feat)
        up_feat = self.act1(up_feat)

        up_feat = self.conv2(up_feat)
        up_feat = self.bn2(up_feat)
        up_feat = self.act2(up_feat)

        return up_feat


class ReconBlock(nn.Module):
    def __init__(self, in_filters, out_filters, conv_cfg, norm_cfg):
        super(ReconBlock, self).__init__()
        self.conv0 = conv3x1(in_filters, out_filters, conv_cfg)
        self.bn0 = build_norm_layer(norm_cfg, out_filters)[1]
        self.act0 = nn.Sigmoid()

        self.conv1 = conv1x3(in_filters, out_filters, conv_cfg)
        self.bn1 = build_norm_layer(norm_cfg, out_filters)[1]
        self.act1 = nn.Sigmoid()

    def forward(self, x):
        att_feat0 = self.conv0(x)
        att_feat0 = self.bn0(att_feat0)
        att_feat0 = self.act0(att_feat0)

        att_feat1 = self.conv1(x)
        att_feat1 = self.bn1(att_feat1)
        att_feat1 = self.act1(att_feat1)

        att_feat = att_feat0 + att_feat1

        out = att_feat * x

        return out



@BACKBONES.register_module()
class SEG_UNET(nn.Module):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self,
                 in_channels=16,
                 out_channels=[128, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 feat_channels=64,
                 upsample_cfg=dict(type='deconv', bias=False)):
        super(SEG_UNET, self).__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.

        self.ContextBlock = Asym_ResBlock(in_channels,      feat_channels, conv_cfg, norm_cfg, pooling=False)
        self.Block0 = Asym_ResBlock(feat_channels,      2 * feat_channels, conv_cfg, norm_cfg, pooling=True)
        self.Block1 = Asym_ResBlock(2 * feat_channels,  4 * feat_channels, conv_cfg, norm_cfg, pooling=True)
        self.Block2 = Asym_ResBlock(4 * feat_channels,  8 * feat_channels, conv_cfg, norm_cfg, pooling=True)
        self.Block3 = Asym_ResBlock(8 * feat_channels, 8 * feat_channels, conv_cfg, norm_cfg, pooling=True, pool_padding=(0, 1))

        self.upBlock0 = UpBlock(8 * feat_channels, 8 * feat_channels, conv_cfg, norm_cfg, upsample_cfg, up_output_padding=(1, 0))
        self.upBlock1 = UpBlock(8 * feat_channels, 8 * feat_channels, conv_cfg, norm_cfg, upsample_cfg)
        self.upBlock2 = UpBlock(8 * feat_channels, 4 * feat_channels, conv_cfg, norm_cfg, upsample_cfg)
        self.upBlock3 = UpBlock(4 * feat_channels, 2 * feat_channels, conv_cfg, norm_cfg, upsample_cfg)

        self.trans_conv = nn.Sequential(
            conv3x3(2 * feat_channels, feat_channels, conv_cfg),
            build_norm_layer(norm_cfg, feat_channels)[1],
            nn.ReLU(inplace=True)
        )

        #self.ReconBlock = ReconBlock(2 * feat_channels, 2 * feat_channels, conv_cfg, norm_cfg)
        self.ReconBlock = ReconBlock(feat_channels, feat_channels, conv_cfg, norm_cfg)

    def init_weights(self, pretrained=None):
        """Initialize weights of the 2D backbone."""
        # Do not initialize the conv layers
        # to follow the original implementation
        if isinstance(pretrained, str):
            from mmdet3d.utils import get_root_logger
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    def forward(self, x):

        x = self.ContextBlock(x)
        down0, lateral0 = self.Block0(x)
        down1, lateral1 = self.Block1(down0)
        down2, lateral2 = self.Block2(down1)
        down3, lateral3 = self.Block3(down2)

        up3 = self.upBlock0(down3, lateral3)
        up2 = self.upBlock1(up3, lateral2)
        up1 = self.upBlock2(up2, lateral1)
        up0 = self.upBlock3(up1, lateral0)

        up0 = self.trans_conv(up0)

        out = self.ReconBlock(up0)

        out = torch.cat((out, up0), dim=1)
        
        return out
