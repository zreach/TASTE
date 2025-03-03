from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import pickle

from src.utils import set_color
from src.model.layers import FMEmbedding, FMFirstOrderLinear, FLEmbedding, MLPLayers


class GeneralRecommender(nn.Module):
    def __init__(self, config, dataset):
        self.logger = getLogger()
        super(GeneralRecommender, self).__init__()

        self.field_names = dataset.fields(
                source=config['filed_names']
            )

        self.LABEL = config["LABEL_FIELD"]
        self.embedding_size = config["embedding_size"]
        self.device = config["device"]
        self.numerical_features = config["numerical_features"]
        self.token_field_names = []
        self.token_field_dims = []
        self.float_field_names = []
        self.float_field_dims = []
        self.num_feature_field = 0

        self.token2id = dataset.field2token_id
        self.id2token = {}

        self.use_audio = config['use_audio']
        self.use_text = config['use_text']
        self.content_feature_type = config['content_feature_type']
        self.content_embedding_size = config['content_embedding_size']
        
        if self.use_audio is None:
            self.use_audio = False
        if self.use_text is None:
            self.use_text = False
        if self.content_feature_type is None:
            self.content_feature_type = 'CLAP' #TODO 设置默认参数
        
        if self.use_audio:
            wav_feat_path = config['wav_feat_path']
            with open(wav_feat_path, 'rb') as fp:
                    music_features_array = pickle.load(fp)
        
        if self.use_text:
            wav_feat_path = config['wav_feat_path']
            with open(wav_feat_path, 'rb') as fp:
                    music_features_array = pickle.load(fp)


    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return (
            super().__str__()
            + set_color("\nTrainable parameters", "blue")
            + f": {params}"
        )