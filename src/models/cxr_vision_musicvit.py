"""
Covers useful modules referred in the paper
All dimensions in comments are induced from 224 x 224 x 3 inputs
and CMT-S
Created by Kunhong Yu
Date: 2021/07/14
"""


import torch as t
import torch.nn as nn
from torch.nn import functional as F

from .ACMBlock import ACMBlock
from .CMT_utils import IRFFN, LMHSA, LPU, CMTBlock, PatchAggregation, Stem


def ACM_loss(logit):
    return 2-(2*logit)

class CXRSiamese_CMT_ACM(t.nn.Module):
    """Define CMT model"""

    def __init__(self,
                 in_channels = 3,
                 stem_channels = 16,
                 cmt_channelses = [46, 92, 184, 368],
                 pa_channelses = [46, 92, 184, 368],
                 R = 3.6,
                 repeats = [2, 2, 10, 2],
                 input_size = 224,
                 sizes = [64, 32, 16, 8],
                 patch_ker=2,
                 patch_str=2,
                 num_classes = None,
                 num_label = 1000,
                 disease_classes = 1000
                 ):
        """
        Args :
            --in_channels: default is 3
            --stem_channels: stem channels, default is 16
            --cmt_channelses: list, default is [46, 92, 184, 368]
            --pa_channels: patch aggregation channels, list, default is [46, 92, 184, 368]
            --R: expand ratio, default is 3.6
            --repeats: list, to specify how many CMT blocks stacked together, default is [2, 2, 10, 2]
            --input_size: default is 224
            --num_classes: default is 1000 for ImageNet
        """
        super(CXRSiamese_CMT_ACM, self).__init__()

        if input_size == 224:
            sizes = [56, 28, 14, 7]
        elif input_size == 160:
            sizes = [40, 20, 10, 5]
        elif input_size == 192:
            sizes = [48, 24, 12, 6]
        elif input_size == 256:
            sizes = [64, 32, 16, 8]
        elif input_size == 288:
            sizes = [72, 36, 18, 9]
        elif input_size == 512:
            sizes = sizes
            # sizes = [127, 62, 30, 14]
        else:
            raise Exception('No other input sizes!')

        # 1. Stem
        self.stem = Stem(in_channels = in_channels, out_channels = stem_channels)

        # 2. Patch Aggregation 1
        self.pa1 = PatchAggregation(in_channels = stem_channels, out_channels = pa_channelses[0], patch_ker=patch_ker, patch_str=patch_str)
        self.pa2 = PatchAggregation(in_channels = cmt_channelses[0], out_channels = pa_channelses[1], patch_ker=patch_ker, patch_str=patch_str)
        self.pa3 = PatchAggregation(in_channels = cmt_channelses[1], out_channels = pa_channelses[2], patch_ker=patch_ker, patch_str=patch_str)
        self.pa4 = PatchAggregation(in_channels = cmt_channelses[2], out_channels = pa_channelses[3], patch_ker=patch_ker, patch_str=patch_str)

        # 3. CMT block
        cmt1 = []
        for _ in range(repeats[0]):
            cmt_layer = CMTBlock(input_size = sizes[0],
                                 kernel_size = 8,
                                 d_k = cmt_channelses[0],
                                 d_v = cmt_channelses[0],
                                 num_heads = 1,
                                 R = R, in_channels = pa_channelses[0])
            cmt1.append(cmt_layer)
        self.cmt1 = t.nn.Sequential(*cmt1)

        cmt2 = []
        for _ in range(repeats[1]):
            cmt_layer = CMTBlock(input_size = sizes[1],
                                 kernel_size = 4,
                                 d_k = cmt_channelses[1] // 2,
                                 d_v = cmt_channelses[1] // 2,
                                 num_heads = 2,
                                 R = R, in_channels = pa_channelses[1])
            cmt2.append(cmt_layer)
        self.cmt2 = t.nn.Sequential(*cmt2)

        cmt3 = []
        for _ in range(repeats[2]):
            cmt_layer = CMTBlock(input_size = sizes[2],
                                 kernel_size = 2,
                                 d_k = cmt_channelses[2] // 4,
                                 d_v = cmt_channelses[2] // 4,
                                 num_heads = 4,
                                 R = R, in_channels = pa_channelses[2])
            cmt3.append(cmt_layer)
        self.cmt3 = t.nn.Sequential(*cmt3)

        cmt4 = []
        for _ in range(repeats[3]):
            cmt_layer = CMTBlock(input_size = sizes[3],
                                 kernel_size = 1,
                                 d_k = cmt_channelses[3] // 8,
                                 d_v = cmt_channelses[3] // 8,
                                 num_heads = 8,
                                 R = R, in_channels = pa_channelses[3])
            cmt4.append(cmt_layer)
        self.cmt4 = t.nn.Sequential(*cmt4)

        # 4. Global Avg Pool
        self.avg = t.nn.AdaptiveAvgPool2d(1)

        # 5. FC
        self.fc = t.nn.Sequential(
            t.nn.Linear(cmt_channelses[-1], 1280),
            t.nn.ReLU(inplace = True) # we use ReLU here as default
        )

        # 6. Classifier
        self.classifier = t.nn.Sequential(
            t.nn.Linear(1280*2, num_label)
        )
        
        self.disease_linear = nn.Linear(1280,disease_classes)
        
        # CMT Ti
        self.acm1 = ACMBlock(in_channels=23*2)
        self.acm2 = ACMBlock(in_channels=23*4)
        self.acm3 = ACMBlock(in_channels=23*8)
        self.acm4 = ACMBlock(in_channels=23*16)

    def forward(self, x , x2):

        # 1. Stem
        x_stem = self.stem(x)

        # 1. Stem
        x2_stem = self.stem(x2)
        
        # 2. PA1 + CMTb1
        x_pa1 = self.pa1(x_stem)
        x_cmtb1 = self.cmt1(x_pa1)

        # 2. PA1 + CMTb1
        x2_pa1 = self.pa1(x2_stem)
        x2_cmtb1 = self.cmt1(x2_pa1)
        x_cmtb1, x2_cmtb1, orth_loss1 = self.acm1(x_cmtb1, x2_cmtb1)
        
        
        # 3. PA2 + CMTb2
        x_pa2 = self.pa2(x_cmtb1)
        x_cmtb2 = self.cmt2(x_pa2)

        # 3. PA2 + CMTb2
        x2_pa2 = self.pa2(x2_cmtb1)
        x2_cmtb2 = self.cmt2(x2_pa2)
        x_cmtb2, x2_cmtb2, orth_loss2 = self.acm2(x_cmtb2, x2_cmtb2)
        
        
        # 4. PA3 + CMTb3
        x_pa3 = self.pa3(x_cmtb2)
        x_cmtb3 = self.cmt3(x_pa3)

        # 4. PA3 + CMTb3
        x2_pa3 = self.pa3(x2_cmtb2)
        x2_cmtb3 = self.cmt3(x2_pa3)
        
        x_cmtb3, x2_cmtb3, orth_loss3 = self.acm3(x_cmtb3, x2_cmtb3)
        

        # 5. PA4 + CMTb4
        x_pa4 = self.pa4(x_cmtb3)
        x_cmtb4 = self.cmt4(x_pa4)

        # 5. PA4 + CMTb4
        x2_pa4 = self.pa4(x2_cmtb3)
        x2_cmtb4 = self.cmt4(x2_pa4)
        
        x_cmtb4, x2_cmtb4, orth_loss4 = self.acm4(x_cmtb4, x2_cmtb4)

        # 6. Avg
        x_avg = self.avg(x_cmtb4)
        x_avg = x_avg.squeeze()

        # 6. Avg
        x2_avg = self.avg(x2_cmtb4)
        x2_avg = x2_avg.squeeze()

        # 7. Linear + Classifier
        x_fc = self.fc(x_avg)

        # 7. Linear + Classifier
        x2_fc = self.fc(x2_avg)
        
        orth_score = (orth_loss1 + orth_loss2 + orth_loss3 + orth_loss4) / 4
        
        x1 = self.disease_linear(x_fc)
        x2 = self.disease_linear(x2_fc)
        
        if x_fc.shape[0] == 1280:
            x_fc = x_fc.unsqueeze(0)
            x2_fc = x2_fc.unsqueeze(0)
            
        cat = t.cat([x_fc, x2_fc],1)
        out = self.classifier(cat)

        return x1, x2, out, orth_score