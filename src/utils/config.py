"""
Utility functions and configuration management
"""

import yaml
import logging
import os
from typing import Dict, Any


def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return {}


def setup_logging(config: Dict[str, Any] = None):
    """Setup logging configuration"""
    if config is None:
        config = load_config()
    
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_config.get('file', 'logs/anomaly_detection.log')
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


class Config:
    """Configuration class for easy access to config values"""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        self.config = load_config(config_path)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        return self.config.get('models', {}).get(model_name, {})
    
    def get_data_paths(self) -> Dict[str, str]:
        """Get all data paths"""
        return self.config.get('data', {})


if __name__ == '__main__':
    # Example usage
    config = Config()
    print("Data paths:", config.get_data_paths())
    print("Isolation Forest config:", config.get_model_config('isolation_forest'))
