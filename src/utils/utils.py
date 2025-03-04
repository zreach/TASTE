import datetime
import importlib
import os
import random
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from texttable import Texttable

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)



def set_color(log, color, highlight=True):
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"

def init_seed(seed, reproducibility):
    r"""init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

def get_model(model_name):
    model_module = importlib.import_module("src.model")
    if hasattr(model_module, model_name):
        model_class = getattr(model_module, model_name)
    else:
        raise ValueError(f"model {model_name} not found")
    return model_class

def get_trainer(config):
    trainer_module = importlib.import_module("src.trainer")
    if hasattr(trainer_module, config["MODEL_TYPE"]):
        trainer_class = getattr(trainer_module, config["MODEL_TYPE"])
    else:
        raise ValueError(f"trainer {config['MODEL_TYPE']} not found")
    return trainer_class