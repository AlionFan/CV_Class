# 定义不同的参数组合进行实验
param_combinations = [
    {'batch_size': 32, 'learning_rate': 1e-3, 'num_epochs': 3},
    {'batch_size': 32, 'learning_rate': 1e-4, 'num_epochs': 3},
    {'batch_size': 64, 'learning_rate': 1e-4, 'num_epochs': 3},
    {'batch_size': 32, 'learning_rate': 1e-3, 'num_epochs': 5},
    {'batch_size': 32, 'learning_rate': 1e-4, 'num_epochs': 5},
    {'batch_size': 32, 'learning_rate': 1e-4, 'num_epochs': 10},
    {'batch_size': 64, 'learning_rate': 1e-4, 'num_epochs': 5},
    {'batch_size': 64, 'learning_rate': 1e-4, 'num_epochs': 10},
]

# 基础配置
base_config = {
    'data_dir': 'data/flower_data',
    'cat_to_name_path': 'data/cat_to_name.json',
    'result_dir': './result',
    'image_size': 224,
    'use_pretrained': True,
    'feature_extract': True,  
    'scheduler_step_size': 7,
    'scheduler_gamma': 0.1
}