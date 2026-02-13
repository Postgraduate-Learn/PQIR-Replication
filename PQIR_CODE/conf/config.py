import simplejson
import os

config_json = os.path.join(os.path.dirname(__file__), 'config.json')


class Config:
    def __init__(self):
        with open(config_json, 'r') as f:
            content = f.read()
        # 将json格式字符串解析成python字典
        config = simplejson.loads(content)
        # 原始xml文件相关
        self.so_url = config['so_url']
        self.so_dir = config['so_dir']
        self.experiment_dir = config['experiment_dir']
        self.post_xml = os.path.join(self.so_dir, config['post_xml'])
        # LLM 相关
        self.api_key = config['api_key']
        self.api_base_url = config['api_base_url']
        self.gpt_model = config['gpt_model']
        # 数据增强相关
        self.augmentation_sample_count = config['augmentation_sample_count']
        self.min_sample_threshold = config['min_sample_threshold']
        self.sample_category_columns = config['sample_category_columns']
        self.augmentation_input_paths = config['augmentation_input_paths']
        self.augmented_output_path = config['augmented_output_path']
        self.summarized_output_path = config['summarized_output_path']

        # 训练和测试相关
        self.train_data_path = config['train_data_paths']
        self.train_model_path = config['train_model_path']
        self.train_output_path = config['train_output_path']
        self.train_logging_dir = config['train_logging_dir']
        self.model_save_path = config['model_save_path']
        self.error_data_path = config['error_data_path']
        self.test_data_paths = config['test_data_paths']

        # 真实数据相关
        self.predicted_file_path = config['predicted_file_path']


