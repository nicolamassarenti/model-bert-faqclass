import logging
import logging.config
import os

import yaml

PACKAGE_LOCATION = os.path.join(os.path.dirname(os.path.realpath(__file__)))
RESOURCES_LOCATION = os.path.join(PACKAGE_LOCATION, "resources")
CONFIG_LOCATION = os.path.join(RESOURCES_LOCATION, "config")


def _get_config_dict(name: str) -> dict:
    with open(os.path.join(CONFIG_LOCATION, "{name}.yml".format(name=name))) as f:
        configmap = yaml.load(f, Loader=yaml.SafeLoader)
    return configmap if configmap else {}


def _init_logger():
    LOGS_LOCATION = os.path.join(PACKAGE_LOCATION, "logs")

    # Creating folder if not exists
    if not os.path.exists(LOGS_LOCATION):
        os.makedirs(LOGS_LOCATION)

    DEBUG_LOGS_LOCATION = os.path.join(LOGS_LOCATION, "debug.log")
    INFO_LOGS_LOCATION = os.path.join(LOGS_LOCATION, "info.log")
    ERROR_LOGS_LOCATION = os.path.join(LOGS_LOCATION, "error.log")

    # Retrieving dictionary config
    log_config = _get_config_dict("log_config")

    # Upading path
    log_config["handlers"]["debug_rotating_file_handler"][
        "filename"
    ] = DEBUG_LOGS_LOCATION
    log_config["handlers"]["info_rotating_file_handler"][
        "filename"
    ] = INFO_LOGS_LOCATION
    log_config["handlers"]["error_file_handler"]["filename"] = ERROR_LOGS_LOCATION

    # Setting configs
    logging.config.dictConfig(log_config)


_init_logger()
