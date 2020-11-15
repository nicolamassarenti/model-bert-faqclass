import os
import yaml
from box import Box

from src import CONFIG_LOCATION


def _get_config_dict(name: str) -> dict:
	with open(os.path.join(CONFIG_LOCATION, "{name}.yml".format(name=name))) as f:
		configmap = yaml.load(f, Loader=yaml.SafeLoader)
	return configmap if configmap else {}


config = Box(_get_config_dict(name="default"))
