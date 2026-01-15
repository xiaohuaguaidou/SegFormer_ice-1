# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from .car import ClassAwareRegularization
#线性嵌入模块，将输入特征进行维度转换和映射
class ProgressiveClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, dropout_ratio=0.1):
        super().__init__()
        # 第一层：1x1卷积快速降维 (768 → 384)
        self.conv1 = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim // 2, kernel_size=1),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(inplace=True),
        )
        # 第二层：3x3卷积在低维空间提取特征 (384 → 192)
        self.conv2 = nn.Sequential(
            nn.Conv2d(embedding_dim // 2, embedding_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(embedding_dim // 4),
            nn.ReLU(inplace=True),
        )
        # 第三层：1x1卷积输出 (192 → num_classes)
        self.conv3 = nn.Conv2d(embedding_dim // 4, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, x):
        x = self.conv1(x)  # 快速降维，保留主要信息
        x = self.conv2(x)  # 在低维空间进行特征提取
        x = self.dropout(x)
        x = self.conv3(x)  # 最终分类
        return x

class LightweightKANLayer(nn.Module):
    """轻量级KAN，大幅降低计算量"""

    def __init__(self, input_dim, embed_dim, num_bases=3):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_bases = num_bases

        # 大幅减少参数：从(input_dim, embed_dim, spline_size)到(input_dim, num_bases)
        self.bases = nn.Parameter(torch.randn(input_dim, num_bases))
        self.weights = nn.Linear(input_dim * num_bases, embed_dim)

        # 添加基础路径的投影层，确保输出维度一致
        self.base_proj = nn.Linear(input_dim, embed_dim)

        self.base_activation = nn.SiLU()

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # 基础激活 + 投影到目标维度
        base_out = self.base_activation(x_flat)  # (B, N, C)
        base_out = self.base_proj(base_out)  # (B, N, embed_dim)

        # 简化版样条：使用学习的基函数
        bases = self.bases.unsqueeze(0)  # (1, C, num_bases)
        x_expanded = x_flat.unsqueeze(-1)  # (B, N, C, 1)
        spline_out = torch.sigmoid(x_expanded * bases)  # 简化计算

        # 合并
        spline_flat = spline_out.reshape(B, -1, C * self.num_bases)
        spline_proj = self.weights(spline_flat)  # (B, N, embed_dim)

        # 残差连接 - 现在维度一致了
        output = base_out + spline_proj
        return output
class LightweightKANMLP(nn.Module):
    """
    修复版KANMLP
    """

    def __init__(self, input_dim, embed_dim, grid_size=5, k=3):
        super().__init__()
        self.kan_layer = LightweightKANLayer(input_dim, embed_dim)

    def forward(self, x):
        return self.kan_layer(x)


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1, car_cfg: dict = None, lambda_car: float = 0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = LightweightKANMLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = LightweightKANMLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = LightweightKANMLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = LightweightKANMLP(input_dim=c1_in_channels, embed_dim=embedding_dim)


        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )

        #self.classifier1 = ProgressiveClassifier(embedding_dim, num_classes, dropout_ratio)
        '''self.classifier = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(embedding_dim // 2, num_classes, kernel_size=1)
        )'''
        self.dropout        = nn.Dropout2d(dropout_ratio)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.car = None
        self.lambda_car = float(lambda_car)
        if car_cfg is not None:
            car_cfg = dict(car_cfg)  # copy to avoid mutating caller dict
            car_cfg.setdefault("filters", embedding_dim)  # features after fuse/dropout have embedding_dim channels
            car_cfg.setdefault("num_class", num_classes)
            # device can be left None; will use features device in forward
            self.car = ClassAwareRegularization(**car_cfg)
    def forward(self, inputs, labels=None):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))


        x = self.dropout(_c)
        car_out = None
        if self.car is not None and self.training and labels is not None:
            # CAR expects features [B, C, H, W] and labels [B, H, W]
            # CAR will internally resize labels to match any pooling rates it uses.
            # Ensure labels dtype is long
            if labels.dtype != torch.long:
                labels = labels.long()
            car_out = self.car(x, labels, training=True)
            # attach weighted loss for convenience
            if 'loss' in car_out:
                car_out['weighted_loss'] = car_out['loss'] * self.lambda_car

        # prediction
        x = self.linear_pred(x)

        # preserve original simple API when CAR disabled / labels not provided
        if car_out is None:
            return x  # logits Tensor as before
        else:
            return x, car_out  # logits, car dict

class SegFormer(nn.Module):
    def __init__(self, num_classes = 21, phi = 'b0', pretrained = False,car_cfg: dict = None, lambda_car: float = 0.1):
        super(SegFormer, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone   = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim   = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim, car_cfg=car_cfg, lambda_car=lambda_car)

    def forward(self, inputs, labels: torch.Tensor = None):
        """
        If labels is provided (training), we forward it into the decode_head so CAR can compute loss.
        Returns:
          - if head returns logits only: logits (upsampled to input H,W)
          - if head returns (logits, car_out): (logits_upsampled, car_out) where car_out is the CAR dict
        """
        H, W = inputs.size(2), inputs.size(3)

        x = self.backbone.forward(inputs)
        head_out = self.decode_head.forward(x, labels=labels)

        # head_out may be logits or (logits, car_out)
        if isinstance(head_out, tuple) or (isinstance(head_out, list) and len(head_out) == 2):
            logits, car_out = head_out
        else:
            logits, car_out = head_out, None

        # upsample logits to input size
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)

        if car_out is None:
            return logits
        else:
            return logits, car_out
