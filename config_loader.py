import yaml
from typing import Dict, Any

class OCRConfigLoader:
    @staticmethod
    def load_config(yaml_path: str) -> Dict[str, Any]:
        """Tải cấu hình từ file YAML"""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return config