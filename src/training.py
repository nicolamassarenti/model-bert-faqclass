import logging
import os

from src.handlers.bertHandler import BertHandler

logger = logging.getLogger(__name__)


def train(x_train, y_train, x_validation, y_validation, x_test, y_test, num_classes):
    """
    Builds, encodes the dataset and executes the training of the model.
    
    :param x_train: training examples
    :param y_train: training labels
    :param x_validation: validation examples
    :param y_validation: validation labels
    :param x_test: test examples
    :param y_test: validation examples
    :param num_classes: number of classes to classify
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

    # Encoding the faq examples
    x_train_encoded = bert_handler.encode(x_train["kb"])
    x_validation_encoded = bert_handler.encode(x_validation["kb"])
    x_test_encoded = bert_handler.encode(x_test["kb"])
    logger.info("Encoding of the faq examples executed")

    # Obtaining the feature vectors from the keywords ids associated to each example
    x_train_encoded["keywords_ids"] = bert_handler.get_feature_from_ids(
        data=x_train["keywords_ids"],
        num_classes=num_classes["keywords"]
    )
    x_validation_encoded["keywords_ids"] = bert_handler.get_feature_from_ids(
        data=x_validation["keywords_ids"],
        num_classes=num_classes["keywords"]
    )
    x_test_encoded["keywords_ids"] = bert_handler.get_feature_from_ids(
        data=x_test["keywords_ids"],
        num_classes=num_classes["keywords"]
    )
    logger.info("Computed the features from keyword ids vectors associated to each example")

    bert_handler.build_custom_model(num_keywords=num_classes["keywords"], output_classes=num_classes["kb"])
    logger.info("Model built")

    x_train = [
        x_train_encoded["input_word_ids"],
        x_train_encoded["input_mask"],
        x_train_encoded["input_type_ids"],
        x_train_encoded["keywords_ids"]
    ]
    y_train = bert_handler.to_categorical_tensor(data=y_train, num_classes=num_classes["kb"])

    x_val = [
        x_validation_encoded["input_word_ids"],
        x_validation_encoded["input_mask"],
        x_validation_encoded["input_type_ids"],
        x_validation_encoded["keywords_ids"]
    ]
    y_val = bert_handler.to_categorical_tensor(data=y_validation, num_classes=num_classes["kb"])
    logger.info("Data prepared for training")

    logger.info("Starting training")
    bert_handler.train(
        X_train=x_train,
        y_train=y_train,
        X_val=x_val,
        y_val=y_val,
        epochs=int(os.getenv("MODEL_TRAINING_EPOCHS")),
        load_checkpoint=False
    )
    logger.info("Training completed")
