import yaml
# 读取配置文件
def load_config(config_path='src/config/config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config