import os
import yaml
import logging
import logging.config

PACKAGE_LOCATION = os.path.join(os.path.dirname(os.path.realpath(__file__)))
RESOURCES_LOCATION = os.path.join(PACKAGE_LOCATION, "resources")
CONFIG_LOCATION = os.path.join(RESOURCES_LOCATION, "config")

def _get_config_dict(name: str):
	with open(os.path.join(CONFIG_LOCATION, "{name}.yml".format(name=name))) as f:
		configmap = yaml.load(f, Loader=yaml.SafeLoader)
	return configmap if configmap else {}

def _init_logger():
	log_config = _get_config_dict("log_config")
	logging.config.dictConfig(log_config)
_init_logger()