import logging

from bert_faqclass.configurations import config
from bert_faqclass.connectors.gcloud.firestore.client import (
    connector as firestore_connector,
)
from bert_faqclass.connectors.gcloud.firestore.collections import FirestoreCollections
from bert_faqclass.connectors.gcloud.storage.locations import StorageLocations
from bert_faqclass.handlers.datasetHandler import DatasetHandler
from bert_faqclass.model.model import Model
from bert_faqclass.model.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


def run():
    ####################################################################################################################
    # Constants
    ####################################################################################################################
    logger.info(
        "Knowledge base collection: {name}".format(
            name=FirestoreCollections.KNOWLEDGE_BASE.value
        )
    )
    logger.info(
        "Keywords collection: {name}".format(name=FirestoreCollections.KEYWORDS.value)
    )

    train_split = float(config["model"]["training"]["split"]["train"])
    validation_split = float(config["model"]["training"]["split"]["validation"])
    test_split = float(config["model"]["training"]["split"]["test"])
    logger.info("Train split: {split}".format(split=train_split))
    logger.info("Validation split: {split}".format(split=validation_split))
    logger.info("Test split: {split}".format(split=test_split))
    if train_split + validation_split + test_split != 1.0:
        error_message = "Training splits configuration is inconsistent: train+validation+test != 1, sum is {sum}".format(
            sum=train_split + validation_split + test_split
        )
        logger.critical(error_message)
        exit(1)

    logger.info(
        "Model savings location: {location}".format(
            location=StorageLocations.MODEL.complete_path
        )
    )

    load_checkpoints = config["model"]["training"]["load_checkpoints"]
    logging.info(
        "Load checkpoint option set to {option}".format(option=load_checkpoints)
    )
    is_checkpoint_enabled = config["model"]["training"]["is_checkpoints_enabled"]
    logger.info(
        "Checkpoints location: {location}. Enablement option set to {option}".format(
            location=StorageLocations.CHECKPOINTS.complete_path,
            option=is_checkpoint_enabled,
        )
    )

    bert_url = config["model"]["bert"]["url"]
    logger.info("Bert url: {url}".format(url=bert_url))
    preprocessor_url = config["model"]["preprocessor"]["url"]
    logger.info("Preprocessor url: {url}".format(url=preprocessor_url))

    model_name = config["model"]["name"]
    model_version = config["model"]["version"]
    logger.info(
        "Model name: {name}, version: {version}".format(
            name=model_name, version=model_version
        )
    )

    max_sequence_length = int(round(config["model"]["inputs"]["max_sequence_length"]))
    logger.info("Max sequence length: {length}".format(length=max_sequence_length))

    num_epochs = int(config["model"]["training"]["epochs"])
    logger.info("Epochs: {epochs}".format(epochs=num_epochs))

    batch_size = int(config["model"]["training"]["batch_size"])
    logger.info("Batch size: {batch_size}".format(batch_size=batch_size))

    ####################################################################################################################
    # Services and handlers
    ####################################################################################################################
    logger.info("Creating services, model and utils objects")

    preprocessor = Preprocessor(preprocessor_url=preprocessor_url)
    logger.debug("Preprocessor created")

    dataset_handler = DatasetHandler(
        train_split=train_split,
        val_split=validation_split,
        test_split=test_split,
        preprocessor=preprocessor,
    )
    logger.info("Dataset handler created")

    model = Model(
        base_model_url=bert_url,
        checkpoint_location=StorageLocations.CHECKPOINTS.complete_path,
        fine_tuned_model_location=StorageLocations.MODEL.complete_path,
        model_name=model_name,
        model_version=model_version,
        max_sequence_length=max_sequence_length,
    )
    logger.info("Model created")

    ####################################################################################################################
    # Retrieving data
    ####################################################################################################################
    logger.info("Starting to retrieve data")
    kb = firestore_connector.get_all_data(FirestoreCollections.KNOWLEDGE_BASE.value)
    logger.info("Retrieved knowledge base")

    keywords = firestore_connector.get_all_data(FirestoreCollections.KEYWORDS.value)
    logger.info("Retrieved keywords")

    num_faqs = len(kb)
    logger.info("Number of KB classes: {num_classes}".format(num_classes=num_faqs))

    num_keywords = len(keywords)
    logger.info(
        "Number of keywords classes: {num_classes}".format(num_classes=num_keywords)
    )

    ####################################################################################################################
    # Preparing training, validation and test sets
    ####################################################################################################################
    logger.info("Starting to process kb and keyword and splitting into ")
    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    ) = dataset_handler.get_train_val_test_splits(kb=kb, keywords=keywords)

    ####################################################################################################################
    # Building model
    ####################################################################################################################
    logger.info("Starting to build model")
    model.build(num_keywords=num_keywords, output_classes=num_faqs)
    logger.info("Model built")

    ####################################################################################################################
    # Training
    ####################################################################################################################
    logger.info("Starting training")
    model.train(
        X_train=x_train,
        y_train=y_train,
        X_val=x_val,
        y_val=y_val,
        epochs=num_epochs,
        load_checkpoint=load_checkpoints,
    )
    logger.info("Training completed")

    logger.info(
        "Saving model at path {path}".format(path=StorageLocations.MODEL.complete_path)
    )
    model.save()
    logger.info("Model saved")

    ####################################################################################################################
    # Testing performance
    ####################################################################################################################
    model.test(x=x_test, y=y_test)


if __name__ == "__main__":
    run()
