import copy
import importlib
import os
import pickle
import warnings
from typing import Literal
import torch

from src.utils import set_color
from src.utils.argument_list import dataset_arguments
from src.sampler import Sampler, RepeatableSampler

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

def _create_sampler(
    dataset,
    built_datasets,
    distribution: str,
    repeatable: bool,
    alpha: float = 1.0,
    base_sampler=None,
):
    phases = ["train", "valid", "test"]
    sampler = None
    if distribution != "none":
        if base_sampler is not None:
            base_sampler.set_distribution(distribution)
            return base_sampler
        if not repeatable:
            sampler = Sampler(
                phases,
                built_datasets,
                distribution,
                alpha,
            )
        else:
            sampler = RepeatableSampler(
                phases,
                dataset,
                distribution,
                alpha,
            )
    return sampler

def create_samplers(config, dataset, built_datasets):
    train_neg_sample_args, valid_neg_sample_args, test_neg_sample_args = config["train_neg_sample_args"], config["valid_neg_sample_args"], config["test_neg_sample_args"]
    base_sampler = _create_sampler(
        dataset,
        built_datasets,
        train_neg_sample_args["distribution"],
        repeatable,
        train_neg_sample_args["alpha"],
    )

def data_preparation(config, dataset):
    model_type = config["MODEL_TYPE"]
    built_datasets = dataset.build()

    train_dataset, valid_dataset, test_dataset = built_datasets
    train_sampler, valid_sampler, test_sampler = create_samplers(
        config, dataset, built_datasets
    )