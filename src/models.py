import torch
from torch import nn


class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert_model = bert_model
        self.classifier = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        bert_output = self.bert_model(**x)
        return self.classifier(bert_output.pooler_output).flatten()
