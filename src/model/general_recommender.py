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
        self.wav_feature_type = config['wav_feature_type']
        self.wav_embedding_size = config['wav_embedding_size']
        
        if self.use_audio is None:
            self.use_audio = False
        if self.use_text is None:
            self.use_text = False
        if self.wav_feature_type is None:
            self.wav_feature_type = 'CLAP' #TODO 设置默认参数
        
        if self.use_audio:
            wav_feat_path = config['wav_feat_path']
            with open(wav_feat_path, 'rb') as fp:
                wav_features_array = pickle.load(fp)
            wav_features_array['[PAD]'] = np.zeros((1, self.wav_embedding_size))
            music_features = torch.zeros((len(self.token2id['tracks_id']), self.wav_embedding_size ))
            
            for k, v in self.token2id['tracks_id'].items():
                self.id2token[v] = k
                music_features[v] = torch.Tensor(wav_features_array[k])

            self.id2afeat = nn.Embedding.from_pretrained(music_features)
            self.id2afeat.requires_grad_(False)

            size_list = [
                self.wav_embedding_size 
            ] + config['wav_mlp_sizes'] + [self.embedding_size]
            self.wav_mlp = MLPLayers(size_list, 0.2)
            self.num_feature_field += 1

        if self.use_text:
            text_feat_path = config['text_feat_path']
            with open(text_feat_path, 'rb') as fp:
                    text_features_array = pickle.load(fp)
            text_features_array['[PAD]'] = np.zeros((1, self.text_embedding_size))
            text_features = torch.zeros((len(self.token2id['tracks_id']), self.text_embedding_size ))
            
            for k, v in self.token2id['tracks_id'].items():
                self.id2token[v] = k
                text_features[v] = torch.Tensor(text_features_array[k])

            self.id2tfeat = nn.Embedding.from_pretrained(text_features)
            self.id2tfeat.requires_grad_(False)
            size_list = [
                self.text_embedding_size 
            ] + config['text_mlp_sizes'] + [self.embedding_size]
            self.text_mlp = MLPLayers(size_list, 0.2)
            self.num_feature_field += 1


        for field_name in self.field_names:
            if field_name == self.LABEL:
                continue
            if dataset.field2type[field_name] == "token":
                self.token_field_names.append(field_name)
                self.token_field_dims.append(dataset.num(field_name))
            
            elif (
                dataset.field2type[field_name] == "float"
                and field_name in self.numerical_features
            ):
                self.float_field_names.append(field_name)
                self.float_field_dims.append(dataset.num(field_name))
            else:
                continue

            self.num_feature_field += 1
        
        if len(self.token_field_dims) > 0:
            self.token_field_offsets = np.array(
                (0, *np.cumsum(self.token_field_dims)[:-1]), dtype=np.long
            )
            self.token_embedding_table = FMEmbedding(
                self.token_field_dims, self.token_field_offsets, self.embedding_size
            )
        if len(self.float_field_dims) > 0:
            self.float_field_offsets = np.array(
                (0, *np.cumsum(self.float_field_dims)[:-1]), dtype=np.long
            )
            self.float_embedding_table = FLEmbedding(
                self.float_field_dims, self.float_field_offsets, self.embedding_size
            )

        self.first_order_linear = FMFirstOrderLinear(config, dataset)

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