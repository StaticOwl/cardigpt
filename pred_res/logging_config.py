import logging
import logging.config
import os
from datetime import datetime

import yaml


def generate_log_file_name():
    """Generate a log file name with a timestamp."""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_name = os.path.join(log_dir, f'project_{timestamp}.log')
    return log_file_name


def setup_logging(path='logging.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
    """Setup logging configuration."""
    value = os.getenv(env_key, None)
    if value:
        path = value

    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())

        # Set the dynamically generated log filename
        config['handlers']['file_handler']['filename'] = generate_log_file_name()

        # Update console handler to only log error level messages
        for handler in config['handlers'].values():
            if handler['class'] == 'logging.StreamHandler':
                handler['level'] = 'ERROR'

        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
