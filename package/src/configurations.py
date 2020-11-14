import os
import yaml

from src import CONFIG_LOCATION


def _get_config_dict(name: str):
	with open(os.path.join(CONFIG_LOCATION, "{name}.yml".format(name=name))) as f:
		configmap = yaml.load(f, Loader=yaml.SafeLoader)
	return configmap if configmap else {}


config = _get_config_dict(name="default")
