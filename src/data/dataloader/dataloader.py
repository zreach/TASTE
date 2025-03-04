import math
import copy
from logging import getLogger

import torch

from src.data.interaction import Interaction

class AbstractDataLoader(torch.utils.data.DataLoader):
    def __init__(self, config, dataset, sampler, shuffle=False):
        self.shuffle = shuffle
        self.config = config
        self._dataset = dataset
        self._sampler = sampler
        self._batch_size = self.step = self.model = None
        self._init_batch_size_and_step()
        index_sampler = None
        self.generator = torch.Generator()
        self.generator.manual_seed(config["seed"])
        self.transform = construct_transform(config)
        self.is_sequential = config["MODEL_TYPE"] == ModelType.SEQUENTIAL