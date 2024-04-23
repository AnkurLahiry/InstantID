from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
from torch.nn import functional as F

class BertSummarizer(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(BertSummarizer, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        probs = torch.sigmoid(logits).squeeze(-1)
        return probs