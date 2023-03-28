import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, GPT2LMHeadModel


class CXRBertForSequenceClassifier(nn.Module):
    def __init__(
        self,
        num_labels: int,
        output_attentions: bool = None,
        pretrained: str = "",
        output_hidden_states: bool = False,
        freeze_bert: bool = None,
        **kwargs,
    ):
        super().__init__()
        self.pretrained = pretrained
        self.num_labels = num_labels
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.bert = AutoModel.from_pretrained(self.pretrained,
                                              output_attentions = self.output_attentions,
                                              output_hidden_states = self.output_hidden_states)
        
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        
        self.criterion = nn.CrossEntropyLoss()
        
        if freeze_bert:
            for name, param in self.bert.named_parameters():
                if name not in ['classifier.weight', 'classifier.bias']:
                    param.requires_grad = False

    def forward(self, caption_ids, token_type_ids, attention_mask, change_labels):
        
        if caption_ids.shape[0] == 1:
            caption_ids = caption_ids.squeeze(0)
            
        bert_output = self.bert(
            input_ids=caption_ids,
            attention_mask=attention_mask
        )
        last_hidden_state = bert_output['last_hidden_state']
        pooler_output = bert_output['pooler_output']
        hidden_states = bert_output['hidden_states']
        
        output = self.drop(pooler_output)
        output = self.out(output)
        
        change_loss = self.criterion(output, change_labels)
        pred = F.softmax(output , dim=1)
        return change_loss, output, pred
        