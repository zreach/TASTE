import copy
import pickle
import os
import yaml
from collections import Counter, defaultdict
from logging import getLogger

import numpy as np
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset
from scipy.sparse import coo_matrix

class GeneralDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset_name = config["dataset"]

        self.fieldinfo = {}
        self.alias = {}

        self._get_field()
    
    def _get_field(self):
        """Initialization common field names."""
        self.uid_field = self.config["USER_ID_FIELD"]
        self.iid_field = self.config["ITEM_ID_FIELD"]
        self.label_field = self.config["LABEL_FIELD"]
        self.time_field = self.config["TIME_FIELD"]

    def _get_data(self):
        pass

    def _load_inter(self, dataset_name, dataset_path):
        inter_feat_path = os.path.join(dataset_path, f"{dataset_name}.inter")
        if not os.path.isfile(inter_feat_path):
            raise ValueError(f"File {inter_feat_path} not exist.")
        
    
    def _load_feat(self, filepath, source):

        load_col, unload_col = self._get_load_and_unload_col(source)
        if load_col == set():
            return None

        field_separator = self.config["field_separator"]
        columns = []
        usecols = []
        dtype = {}
        encoding = self.config["encoding"]
        with open(filepath, "r", encoding=encoding) as f:
            head = f.readline()[:-1]
        for field_type in head.split(field_separator):
            field, ftype = field_type.split(":")
            
            if load_col is not None and field not in load_col:
                continue
            if unload_col is not None and field in unload_col:
                continue

            self.field2source[field] = source
            self.field2type[field] = ftype
            if not ftype.value.endswith("seq"):
                self.field2seqlen[field] = 1
            if "float" in ftype.value:
                self.field2bucketnum[field] = 2
            columns.append(field)
            usecols.append(field_type)
            dtype[field_type] = np.float64 if ftype == 'float' else str

        if len(columns) == 0:
            self.logger.warning(f"No columns has been loaded from [{source}]")
            return None
        
        df = pd.read_csv(
            filepath,
            delimiter=field_separator,
            usecols=usecols,
            encoding='utf-8',
        )

        df.columns = columns

        seq_separator = self.config["seq_separator"]
        for field in columns:
            ftype = self.field2type[field]
            if not ftype.value.endswith("seq"):
                continue
            df[field].fillna(value="", inplace=True)
            max_seq_len = max(map(len, df[field].values))
            if self.config["seq_len"] and field in self.config["seq_len"]:
                seq_len = self.config["seq_len"][field]
                df[field] = [
                    seq[:seq_len] if len(seq) > seq_len else seq
                    for seq in df[field].values
                ]
                self.field2seqlen[field] = min(seq_len, max_seq_len)
            else:
                self.field2seqlen[field] = max_seq_len

        return df

class GNNDataset(Dataset):
    pass