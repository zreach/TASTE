import copy
import importlib
import os
import pickle
import warnings
from typing import Literal

def create_dataset(config):
    dataset_module = importlib.import_module("src.data.dataset")
    if hasattr(dataset_module, config["model"] + "Dataset"):
        dataset_class = getattr(dataset_module, config["model"] + "Dataset")
    else:
        model_type = config["MODEL_TYPE"]
        type2class = {
            "General": "GeneralDataset",
            "GNN": "GNNDataset"
        }
        dataset_class = getattr(dataset_module, type2class[model_type])
    
    dataset = dataset_class(config)
    # TODO 存储数据
    return dataset