import yaml
import argparse
import os

class Config:
    def __init__(self, yaml_files=None, config=None):
        # 存储配置项的字典
        self.config = {}
        if yaml_files:
            for file in yaml_files:
                self._load_yaml(file)

        # 解析命令行参数
        self.update_config(config)

        # 如果提供了yaml文件，加载这些文件中的配置项

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
                yaml_data = yaml.safe_load(f)
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
