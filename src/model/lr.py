import torch.nn as nn
from torch.nn.init import xavier_normal_

from src.model.general_recommender import GeneralRecommender

class LR(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LR, self).__init__(config, dataset)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        # parameters initialization
        for name, submodule in self.named_modules():
            self._init_weights(name, submodule)
    
    def _init_weights(self, name, module):
        if name not in ['id2afeat', 'id2tfeat', 'id2feature']:
            if isinstance(module, nn.Embedding):
                xavier_normal_(module.weight.data)
    def forward(self, interaction):
        output = self.first_order_linear(interaction)
        return output.squeeze(-1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]

        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))