import torch
import torch.nn as nn
from transformers import BertModel

class BERTTagClassifier(nn.Module):
    """
    BERT-based multi-label tag classifier for music context understanding.
    """
    def __init__(self, model_name='bert-base-uncased', num_tags=16, freeze_layers=8):
        super(BERTTagClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.freeze_layers = freeze_layers
        
        # Freeze the first N layers
        if freeze_layers > 0:
            for name, param in self.bert.named_parameters():
                if name.startswith('embeddings'):
                    param.requires_grad = False
                elif name.startswith('encoder.layer'):
                    layer_num = int(name.split('.')[2])
                    if layer_num < freeze_layers:
                        param.requires_grad = False
                        
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_tags)
        )

    def forward(self, input_ids, attention_mask):
        if hasattr(self, 'freeze_layers') and self.freeze_layers >= 12:
            with torch.no_grad():
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits, cls_embedding, outputs.last_hidden_state

    def get_cls_embedding(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    def get_all_hidden_states(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
