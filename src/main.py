import argparse

from logging import getLogger

from src.configuration.config import Config
from src.data import create_dataset, data_preparation
from src.data.transform import construct_transform
from src.utils import (
    init_logger,
    init_seed,
    set_color,

)

def run(
        model_name,
        dataset_name,
        config,
    ):
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    transform = construct_transform(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="LR", help="name of models")
    parser.add_argument(
        "--dataset_name", "-d", type=str, default="lfm1b", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument('--config', type=str, help="Override config parameter, key=value format", nargs='*')

    args = parser.parse_args()
    config = Config(yaml_files=args.config_files, config=args.config)
    
    run(model_name=args.model_name, dataset_name=args.dataset_name, config=config)
    
    # print(config)



