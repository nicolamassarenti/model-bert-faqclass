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
        plot_path=os.getenv("PATH_MODEL_PLOT"),
        checkpoint_path=os.getenv("PATH_MODEL_CHECKPOINT")
    )
    logger.info("Bert handler successfully created")

    X_train_encoded = bert_handler.encode(X_train["faqs"])
    X_validation_encoded = bert_handler.encode(X_validation["faqs"])
    X_test_encoded = bert_handler.encode(X_test["faqs"])
    logger.info("Encoding executed.")

    X_train_encoded["keywords_ids"] = bert_handler.get_feature_from_ids(
        data=X_train["keywords_ids"],
        num_classes=num_classes["keywords"]
    )
    X_validation_encoded["keywords_ids"] = bert_handler.get_feature_from_ids(
        data=X_validation["keywords_ids"],
        num_classes=num_classes["keywords"]
    )
    X_test_encoded["keywords_ids"] = bert_handler.get_feature_from_ids(
        data=X_test["keywords_ids"],
        num_classes=num_classes["keywords"]
    )
    logger.info("Keywords ids converted to categorical.")

    bert_handler.build_custom_model(num_keywords=num_classes["keywords"], output_classes=num_classes["faqs"])
    logger.info("Model built.")

    X_train = [
        X_train_encoded["input_word_ids"],
        X_train_encoded["input_mask"],
        X_train_encoded["input_type_ids"],
        X_train_encoded["keywords_ids"]
    ]
    Y_train = bert_handler.to_categorical_tensor(data=Y_train, num_classes=num_classes["faqs"])

    X_val = [
        X_validation_encoded["input_word_ids"],
        X_validation_encoded["input_mask"],
        X_validation_encoded["input_type_ids"],
        X_validation_encoded["keywords_ids"]
    ]
    Y_val = bert_handler.to_categorical_tensor(data=Y_validation, num_classes=num_classes["faqs"])
    logger.info("Data prepared for training")

    logger.info("Starting training")
    bert_handler.train(
        X_train = X_train,
        y_train=Y_train,
        X_val=X_val,
        y_val=Y_val,
        epochs=int(os.getenv("MODEL_TRAINING_EPOCHS")),
        load_checkpoint=False
    )
    logger.info("Training completed")







