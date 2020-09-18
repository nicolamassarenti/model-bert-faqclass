import logging
import logging.config
import os
from dotenv import load_dotenv


def settings():
    """
    Runs the required execution to set-up the project.

    :return: None
    """

    # Loading env variables from .env file
    load_dotenv(verbose=True)

    # Setting up the logging
    logging.config.fileConfig(os.getenv("PATH_LOGS_CONFIG"), disable_existing_loggers=False)
