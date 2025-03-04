import yaml
import argparse
import os
import re

class Config(object):
    def __init__(self, yaml_files=None, config=None):
        # 存储配置项的字典
        self.config = {}
        self.yaml_loader = self._build_yaml_loader()

        if yaml_files:
            for file in yaml_files:
                self._load_yaml(file)

        # 解析命令行参数
        self.update_config(config)

        # 如果提供了yaml文件，加载这些文件中的配置项

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

    def __repr__(self):
        return f"Config({self.config})"


if __name__ == "__main__":

    yaml_files = ['config1.yaml']
    

    config = Config(yaml_files)
    print(config.get('key1', 'default_value'))
    print(config)
