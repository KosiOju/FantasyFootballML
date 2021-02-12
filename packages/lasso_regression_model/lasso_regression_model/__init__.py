import logging

from lasso_regression_model.config import config as cfg
from lasso_regression_model.config import logging_config


VERSION_PATH = cfg.PACKAGE_ROOT / 'VERSION'

# setup logger for use in package
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # during production this becomes INFO
logger.addHandler(logging_config.get_console_handler())
logger.propagate = False


with open(VERSION_PATH, 'r') as version_file:
	__version__ = version_file.read().strip()