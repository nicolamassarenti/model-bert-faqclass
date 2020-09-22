import logging
import os

from src.handlers.bertHandler import BertHandler

logger = logging.getLogger(__name__)


def train(X_train, Y_train, X_validation, Y_validation, X_test, Y_test, num_classes):
    """
    Executes the training of BERT. TODO aggiornare parametri

    :param X_train:
    :param Y_train:
    :param X_validation:
    :param Y_validation:
    :param X_test:
    :param Y_test:
    :return:
    """
    # Creating the handler
    bert_handler = BertHandler(
        model_path=os.getenv("PATH_BERT_MODEL"),
        model_name=os.getenv("MODEL_NAME"),
        model_version=os.getenv("MODEL_VERSION"),
        max_sequence_length=int(os.getenv("MAX_SEQ_LENGTH")),
        plot_path=os.getenv("PATH_MODEL_PLOT")
    )
    logger.info("Bert handler successfully created")

    X_train_encoded = bert_handler.encode(X_train["faqs"])
    X_validation_encoded = bert_handler.encode(X_validation["faqs"])
    X_test_encoded = bert_handler.encode(X_test["faqs"])
    logger.info("Encoding executed.")

    X_train_encoded["keywords_ids"] = X_train["keywords_ids"]
    X_validation_encoded["keywords_ids"] = X_validation["keywords_ids"]
    X_test_encoded["keywords_ids"] = X_test["keywords_ids"]

    logger.info("Tokenization completed.")

    model = bert_handler.build_custom_model(
        num_classes=num_classes
    )







