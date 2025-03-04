import argparse

from logging import getLogger

from src.configuration.config import Config
from src.data import create_dataset, data_preparation
from src.data.transform import construct_transform
from src.utils import (
    init_logger,
    init_seed,
    set_color,
    get_model,
    get_trainer,
)

def run(
        model_name='LR',
        dataset_name="lfm1b-filtered",
        config=None,
        save_model=True,
    ):
    init_logger(config)
    logger = getLogger()
    
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(model_name)
    
    transform = construct_transform(config)
    trainer = get_trainer(config)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config["show_progress"]
    )

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    result = {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }
    


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



