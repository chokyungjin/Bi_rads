import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .ACMBlock import ACMBlock
from .CMT_utils import IRFFN, LMHSA, LPU, CMTBlock, PatchAggregation, Stem
from .cxr_text_model import CXRBertForSequenceClassifier


class CXRTextGuidedModel(nn.Module):
    def __init__(
        self,
        num_labels: int,
        output_attentions: bool = None,
        cxr_bert_pretrained: str = None,
        bert_pretrained: str = None,
        vit_pretrained: str = "",
        output_hidden_states: bool = False,
        freeze_bert: bool = None,
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
        disease_classes = 1000,
        p: float = 0,
        **kwargs,
    ):
        super().__init__()
        self.cxr_bert_pretrained = cxr_bert_pretrained
        self.bert_pretrained = bert_pretrained
        self.vit_pretrained = vit_pretrained
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.num_labels = num_labels
        self.bertseq_classifier = CXRBertForSequenceClassifier(pretrained=self.bert_pretrained,
                                              output_attentions = self.output_attentions,
                                              output_hidden_states = self.output_hidden_states,
                                              num_labels=self.num_labels)

        if self.cxr_bert_pretrained is not None:
            pretrained_dict = torch.load(self.cxr_bert_pretrained)['state_dict']
            pretrained_dict = {key.replace("text_model.", ""): value for key, value in pretrained_dict.items()}
            self.bertseq_classifier.load_state_dict(pretrained_dict)
            
        self.bert = self.bertseq_classifier.bert
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        
        if freeze_bert:
            for name, param in self.bert.named_parameters():
                if name not in ['classifier.weight', 'classifier.bias']:
                    param.requires_grad = False
                    
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
        self.cmt1 = torch.nn.Sequential(*cmt1)

        cmt2 = []
        for _ in range(repeats[1]):
            cmt_layer = CMTBlock(input_size = sizes[1],
                                 kernel_size = 4,
                                 d_k = cmt_channelses[1] // 2,
                                 d_v = cmt_channelses[1] // 2,
                                 num_heads = 2,
                                 R = R, in_channels = pa_channelses[1])
            cmt2.append(cmt_layer)
        self.cmt2 = torch.nn.Sequential(*cmt2)

        cmt3 = []
        for _ in range(repeats[2]):
            cmt_layer = CMTBlock(input_size = sizes[2],
                                 kernel_size = 2,
                                 d_k = cmt_channelses[2] // 4,
                                 d_v = cmt_channelses[2] // 4,
                                 num_heads = 4,
                                 R = R, in_channels = pa_channelses[2])
            cmt3.append(cmt_layer)
        self.cmt3 = torch.nn.Sequential(*cmt3)

        cmt4 = []
        for _ in range(repeats[3]):
            cmt_layer = CMTBlock(input_size = sizes[3],
                                 kernel_size = 1,
                                 d_k = cmt_channelses[3] // 8,
                                 d_v = cmt_channelses[3] // 8,
                                 num_heads = 8,
                                 R = R, in_channels = pa_channelses[3])
            cmt4.append(cmt_layer)
        self.cmt4 = torch.nn.Sequential(*cmt4)

        self.conv1by1_1 = nn.Conv2d(144, cmt_channelses[0], kernel_size=1)
        self.conv1by1_2 = nn.Conv2d(cmt_channelses[0], cmt_channelses[1], kernel_size=1)
        self.conv1by1_3 = nn.Conv2d(cmt_channelses[1], cmt_channelses[2], kernel_size=1)
        self.conv1by1_4 = nn.Conv2d(cmt_channelses[2], cmt_channelses[3], kernel_size=1)

        # 4. Global Avg Pool
        self.avg = torch.nn.AdaptiveAvgPool2d(1)

        # 5. FC
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(cmt_channelses[-1], 1280),
            torch.nn.ReLU(inplace = True) # we use ReLU here as default
        )
        self.relu = torch.nn.ReLU(inplace = True)

        # 6. Classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1280*2, self.num_labels)
        )
        
        self.disease_linear = nn.Linear(1280,disease_classes)
        
        # CMT Ti
        self.acm1 = ACMBlock(in_channels=23*2)
        self.acm2 = ACMBlock(in_channels=23*4)
        self.acm3 = ACMBlock(in_channels=23*8)
        self.acm4 = ACMBlock(in_channels=23*16)
                
    def aggregate_tokens(self, caption_ids, token_type_ids, attention_mask):
        
        if caption_ids.shape[0] == 1:
            caption_ids = caption_ids.squeeze(0)
        bert_output = self.bert(
            input_ids=caption_ids,
            attention_mask=attention_mask
        )        
        attentions = bert_output['attentions'] 
        output = torch.cat(attentions, dim=1) 
        
        # B, 144, 128, 128
        x1 = self.conv1by1_1(output)
        x1 = self.relu(x1)
        # B, 52,128, 128 # CMT1

        x2 = self.max_pool(x1)
        x2 = self.conv1by1_2(x2)
        x2 = self.relu(x2)
        # B, 104,64, 64 # CMT2

        x3 = self.max_pool(x2)
        x3 = self.conv1by1_3(x3)
        x3 = self.relu(x3)
        # B, 208,32, 32 # CMT3

        x4 = self.max_pool(x3)
        x4 = self.conv1by1_4(x4)
        x4 = self.relu(x4)
        # B, 416,16, 16 # CMT4
                
        return x1, x2, x3, x4
    
    def forward(self, caption_ids, token_type_ids, attention_mask, x , x2):

        # B, 144, 128, 128
        att1, att2, att3, att4 = self.aggregate_tokens(caption_ids, token_type_ids, attention_mask)
        
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
        x_cmtb1 += att1
        x2_cmtb1 += att1
        
        # 3. PA2 + CMTb2
        x_pa2 = self.pa2(x_cmtb1)
        x_cmtb2 = self.cmt2(x_pa2)

        # 3. PA2 + CMTb2
        x2_pa2 = self.pa2(x2_cmtb1)
        x2_cmtb2 = self.cmt2(x2_pa2)
        x_cmtb2, x2_cmtb2, orth_loss2 = self.acm2(x_cmtb2, x2_cmtb2)
        x_cmtb2 += att2
        x2_cmtb2 += att2
        
        # 4. PA3 + CMTb3
        x_pa3 = self.pa3(x_cmtb2)
        x_cmtb3 = self.cmt3(x_pa3)

        # 4. PA3 + CMTb3
        x2_pa3 = self.pa3(x2_cmtb2)
        x2_cmtb3 = self.cmt3(x2_pa3)
        
        x_cmtb3, x2_cmtb3, orth_loss3 = self.acm3(x_cmtb3, x2_cmtb3)
        x_cmtb3 += att3
        x2_cmtb3 += att3

        # 5. PA4 + CMTb4
        x_pa4 = self.pa4(x_cmtb3)
        x_cmtb4 = self.cmt4(x_pa4)

        # 5. PA4 + CMTb4
        x2_pa4 = self.pa4(x2_cmtb3)
        x2_cmtb4 = self.cmt4(x2_pa4)
        
        x_cmtb4, x2_cmtb4, orth_loss4 = self.acm4(x_cmtb4, x2_cmtb4)

        x_cmtb4 += att4
        x2_cmtb4 += att4
        
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
        
        cat = torch.cat([x_fc, x2_fc],1)
        out = self.classifier(cat)

        return x1, x2, out, orth_score