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
        max_sequence_length=int(os.getenv("MAX_SEQ_LENGTH"))
    )
    logger.info("bert handler succesfully created")

    # Creating the tokenizer
    bert_handler.init_tokenizer(
        path_vocab_file=os.getenv("PATH_VOCAB_FILE"),
        do_lower_case=True
    )
    logger.info("bert tokenizer succesfully created")

    X_train["faqs"] = bert_handler.tokenize(X_train["faqs"])
    X_validation["faqs"] = bert_handler.tokenize(X_validation["faqs"])
    X_test["faqs"] = bert_handler.tokenize(X_test["faqs"])
    logger.info("Tokenization completed.")

    model = bert_handler.create_custom_model(
        num_classes=num_classes
    )





