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
    def embed_float_fields(self, float_fields):
        """Embed the float feature columns

        Args:
            float_fields (torch.FloatTensor): The input dense tensor. shape of [batch_size, num_float_field]

        Returns:
            torch.FloatTensor: The result embedding tensor of float columns.
        """
        # input Tensor shape : [batch_size, num_float_field]
        if float_fields is None:
            return None
        # [batch_size, num_float_field, embed_dim]
        float_embedding = self.float_embedding_table(float_fields)

        return float_embedding

    def embed_token_fields(self, token_fields):
        """Embed the token feature columns

        Args:
            token_fields (torch.LongTensor): The input tensor. shape of [batch_size, num_token_field]

        Returns:
            torch.FloatTensor: The result embedding tensor of token columns.
        """
        # input Tensor shape : [batch_size, num_token_field]
        if token_fields is None:
            return None
        # [batch_size, num_token_field, embed_dim]
        token_embedding = self.token_embedding_table(token_fields)

        return token_embedding

    def get_wav_embedding(self, interaction):
        track_ids = interaction['tracks_id']
        wav_features = self.id2afeat(track_ids)
        embed_features = self.wav_mlp(wav_features)
        return embed_features.unsqueeze(1)
    
    def get_text_embedding(self, interaction):
        track_ids = interaction['tracks_id']
        text_features = self.id2tfeat(track_ids)
        # print(wav_features[0])
        embed_features = self.text_mlp(text_features)
        return embed_features.unsqueeze(1)
    
    def concat_embed_input_fields(self, interaction):
        
        sparse_embedding, dense_embedding = self.embed_input_fields(interaction)
        all_embeddings = []
        if self.use_audio:
            wav_embedding = self.get_wav_embedding(interaction) 
            all_embeddings.append(wav_embedding)
            self.a_feat =  wav_embedding
        if self.use_text:
            text_embedding = self.get_text_embedding(interaction) 
            all_embeddings.append(text_embedding)
            self.t_feat = text_embedding
        if sparse_embedding is not None:
            all_embeddings.append(sparse_embedding)
        if dense_embedding is not None and len(dense_embedding.shape) == 3:
            all_embeddings.append(dense_embedding)
        
        return torch.cat(all_embeddings, dim=1)

    def embed_input_fields(self, interaction):
        """Embed the whole feature columns.

        Args:
            interaction (Interaction): The input data collection.

        Returns:
            torch.FloatTensor: The embedding tensor of token sequence columns.
            torch.FloatTensor: The embedding tensor of float sequence columns.
        """
        float_fields = []
        for field_name in self.float_field_names:
            if len(interaction[field_name].shape) == 3:
                float_fields.append(interaction[field_name])
            else:
                float_fields.append(interaction[field_name].unsqueeze(1))
        if len(float_fields) > 0:
            float_fields = torch.cat(
                float_fields, dim=1
            )  # [batch_size, num_float_field, 2]
        else:
            float_fields = None
        # [batch_size, num_float_field] or [batch_size, num_float_field, embed_dim] or None
        float_fields_embedding = self.embed_float_fields(float_fields)

        float_seq_fields = []
        for field_name in self.float_seq_field_names:
            float_seq_fields.append(interaction[field_name])

        float_seq_fields_embedding = self.embed_float_seq_fields(float_seq_fields)

        if float_fields_embedding is None:
            dense_embedding = float_seq_fields_embedding
        else:
            if float_seq_fields_embedding is None:
                dense_embedding = float_fields_embedding
            else:
                dense_embedding = torch.cat(
                    [float_seq_fields_embedding, float_fields_embedding], dim=1
                )

        token_fields = []
        for field_name in self.token_field_names:
            token_fields.append(interaction[field_name].unsqueeze(1))
        if len(token_fields) > 0:
            token_fields = torch.cat(
                token_fields, dim=1
            )  # [batch_size, num_token_field, 2]
        else:
            token_fields = None
        # [batch_size, num_token_field, embed_dim] or None
        token_fields_embedding = self.embed_token_fields(token_fields)

        token_seq_fields = []
        for field_name in self.token_seq_field_names:
            token_seq_fields.append(interaction[field_name])
        # [batch_size, num_token_seq_field, embed_dim] or None
        token_seq_fields_embedding = self.embed_token_seq_fields(token_seq_fields)

        if token_fields_embedding is None:
            sparse_embedding = token_seq_fields_embedding
        else:
            if token_seq_fields_embedding is None:
                sparse_embedding = token_fields_embedding
            else:
                sparse_embedding = torch.cat(
                    [token_seq_fields_embedding, token_fields_embedding], dim=1
                )

        # sparse_embedding shape: [batch_size, num_token_seq_field+num_token_field, embed_dim] or None
        # dense_embedding shape: [batch_size, num_float_field, 2] or [batch_size, num_float_field, embed_dim] or None
        return sparse_embedding, dense_embedding
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