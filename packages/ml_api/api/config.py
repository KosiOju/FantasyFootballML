import logging
from logging.handlers import TimedRotatingFileHandler
import pathlib
import os
import sys

PACKAGE_ROOT = pathlib.path(__file__).resolve().parent.parent

FORMATTER = logging.Formatter(
	"%(asctime)s - %(name)s - %(levelname)s -"
	"%(funcName)s:%(lineo)d - %(message)s"
)
LOG_DIR = PACKAGE_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / 'ml_api.log'


def get_console_handler():
	console_handler = logging.StreamHandler(sys.stdout)
	console_handler.setFormatter(FORMATTER)

	return console_handler


def get_file_handler():
	file_handler = TimedRotatingFileHandler(
		LOG_FILE, when='midnight'
	)
	file_handler.setFormatter(FORMATTER)
	file_handler.setLevel(logging.WARNING)

	return file_handler


def get_logger(*, logger_name):
	# get logger with prepared handlers

	logger = logging.getLogger(logger_name)

	logger.setLevel(logging.DEBUG)

	logger.addHandler(get_console_handler())
	logger.addHandler(get_file_handler())
	logger.propagate = False

	return logger


class Config:
	DEBUG = False
	TESTING = False
	CSRF_ENABLED = True
	SECRET_KEY = 'my_secret_key'
	SERVER_PORT = 5000


class ProductionConfig(Config):
	DEBUG = False
	SERVER_PORT = os.environ.get('PORT', 5000)


class DevelopmentConfig(Config):
	Development = True
	DEBUG = True


class TestingConfig(Config):
	TESTING = True