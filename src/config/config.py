import yaml
import argparse
import os
import re
from logging import getLogger

from src.utils import InputType

class Config(object):
    def __init__(self, model_name, dataset_name, yaml_files=None, config=None):

        self.config = {}
        self.yaml_loader = self._build_yaml_loader()

        self.model_name = model_name
        self.dataset_name = dataset_name

        if yaml_files:
            for file in yaml_files:
                self._load_yaml(file)


        self.update_config(config) # command line
        self._set_default_parameters()


    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        return loader
    
    def update_config(self, config):
        for item in config:
            key, value = item.split("=")
            # print(value)
            self.config[key] = value
    # def _parse_args(self):
    #     # 使用argparse解析命令行参数
    #     parser = argparse.ArgumentParser(description="Configuration via YAML and command-line arguments.")
    #     parser.add_argument('--config', type=str, help="Override config parameter, key=value format", nargs='*')
        
    #     # 解析命令行参数
    #     args = parser.parse_args()

    #     if args.config:
    #         for item in args.config:
    #             key, value = item.split("=")
    #             # print(value)
    #             self.config[key] = value

    def _load_yaml(self, file):

        if os.path.exists(file):
            with open(file, 'r') as f:
                yaml_data = yaml.safe_load(f, Loader=self.yaml_loader)
                if yaml_data:
                    self.config.update(yaml_data)

    def get(self, key, default=None):

        return self.config.get(key, default)

    def _set_default_parameters(self):
        self.config['dataset'] = self.dataset_name
        self.config["model"] = self.model_name
        self.data_root = self.config['data_root']
        self.data_path = os.path.join(self.data_root, self.dataset_name)
        
        if hasattr(self.model_class, "input_type"):
            self.config["MODEL_INPUT_TYPE"] = self.model_class.input_type
        elif "loss_type" in self.config:
            if self.config["loss_type"] in ["CE"]:
                self.config["MODEL_INPUT_TYPE"] = InputType.POINTWISE
            elif self.config["loss_type"] in ["BPR"]:
                self.config["MODEL_INPUT_TYPE"] = InputType.PAIRWISE
        else:
            raise ValueError(
                "Either Model has attr 'input_type',"
                "or arg 'loss_type' should exist in config."
            )

        metrics = self.config["metrics"]
        if isinstance(metrics, str):
            self.config["metrics"] = [metrics]

        eval_type = set()
        # for metric in self.config["metrics"]:
        #     if metric.lower() in metric_types:
        #         eval_type.add(metric_types[metric.lower()])
        #     else:
        #         raise NotImplementedError(f"There is no metric named '{metric}'")
        if len(eval_type) > 1:
            raise RuntimeError(
                "Ranking metrics and value metrics can not be used at the same time."
            )
        self.config["eval_type"] = eval_type.pop()


        valid_metric = self.config["valid_metric"].split("@")[0]

        topk = self.config["topk"]
        if isinstance(topk, (int, list)):
            if isinstance(topk, int):
                topk = [topk]
            for k in topk:
                if k <= 0:
                    raise ValueError(
                        f"topk must be a positive integer or a list of positive integers, but get `{k}`"
                    )
            self.config["topk"] = topk
        else:
            raise TypeError(f"The topk [{topk}] must be a integer, list")

        if "additional_feat_suffix" in self.config:
            ad_suf = self.config["additional_feat_suffix"]
            if isinstance(ad_suf, str):
                self.config["additional_feat_suffix"] = [ad_suf]

        # train_neg_sample_args checking
        default_train_neg_sample_args = {
            "distribution": "uniform",
            "sample_num": 1,
            "alpha": 1.0,
            "dynamic": False,
            "candidate_num": 0,
        }

        if (
            self.config.get("neg_sampling") is not None
            or self.config.get("training_neg_sample_num") is not None
        ):
            logger = getLogger()
            logger.warning(
                "Warning: Parameter 'neg_sampling' or 'training_neg_sample_num' has been deprecated in the new version, "
                "please use 'train_neg_sample_args' instead and check the API documentation for proper usage."
            )

        if self.config.get("train_neg_sample_args") is not None:
            if not isinstance(self.config["train_neg_sample_args"], dict):
                raise ValueError(
                    f"train_neg_sample_args:[{self.config['train_neg_sample_args']}] should be a dict."
                )
            for op_args in default_train_neg_sample_args:
                if op_args not in self.config["train_neg_sample_args"]:
                    self.config["train_neg_sample_args"][op_args] = (
                        default_train_neg_sample_args[op_args]
                    )

        # eval_args checking
        default_eval_args = {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "order": "RO",
            "group_by": "user",
            "mode": {"valid": "full", "test": "full"},
        }
        if not isinstance(self.config["eval_args"], dict):
            raise ValueError(
                f"eval_args:[{self.config['eval_args']}] should be a dict."
            )

        default_eval_args.update(self.config["eval_args"])

        mode = default_eval_args["mode"]
        # backward compatible
        if isinstance(mode, str):
            default_eval_args["mode"] = {"valid": mode, "test": mode}

        # in case there is only one key in `mode`, e.g., mode: {'valid': 'uni100'} or mode: {'test': 'full'}
        if isinstance(mode, dict):
            default_mode = mode.get("valid", mode.get("test", "full"))
            default_eval_args["mode"] = {
                "valid": mode.get("valid", default_mode),
                "test": mode.get("test", default_mode),
            }

        self.config["eval_args"] = default_eval_args

    def __repr__(self):
        return f"Config({self.config})"


if __name__ == "__main__":

    yaml_files = ['config1.yaml']
    

    config = Config(yaml_files)
    print(config.get('key1', 'default_value'))
    print(config)
