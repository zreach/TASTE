
r"""
FM
################################################
Reference:
    Steffen Rendle et al. "Factorization Machines." in ICDM 2010.
"""

import torch.nn as nn
from torch.nn.init import xavier_normal_

from src.model.general_recommender import GeneralRecommender
from src.model.layers import BaseFactorizationMachine


class FM(GeneralRecommender):
    """Factorization Machine considers the second-order interaction with features to predict the final score."""

    def __init__(self, config, dataset):
        super(FM, self).__init__(config, dataset)

        # define layers and loss
        self.fm = BaseFactorizationMachine(reduce_sum=True)
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
        fm_all_embeddings = self.concat_embed_input_fields(
            interaction
        )  # [batch_size, num_field, embed_dim]
        y = self.first_order_linear(interaction) + self.fm(fm_all_embeddings)
        return y.squeeze(-1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]

        output = self.forward(interaction)
        return self.loss(output, label)

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))
