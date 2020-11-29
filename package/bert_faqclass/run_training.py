import logging
import os

from bert_faqclass.configurations import config
from bert_faqclass.connectors.gcloud.storage.locations import StorageLocations
from bert_faqclass.connectors.gcloud.firestore.client import connector as firestore_connector
from bert_faqclass.connectors.gcloud.firestore.collections import FirestoreCollections
from bert_faqclass.handlers.datasetHandler import DatasetHandler
from bert_faqclass.handlers.modelWrapper import ModelWrapper


logger = logging.getLogger(__name__)


def run_training():
    ####################################################################################################################
    # Constants
    ####################################################################################################################
    logger.info("Knowledge base collection: {name}".format(name=FirestoreCollections.KNOWLEDGE_BASE))
    logger.info("Keywords collection: {name}".format(name=FirestoreCollections.KEYWORDS))

    train_split = float(config.model.training.split.train)
    validation_split = float(config.model.training.split.validation)
    test_split = float(config.model.training.split.test)
    logger.info("Train split: {split}".format(split=train_split))
    logger.info("Validation split: {split}".format(split=validation_split))
    logger.info("Test split: {split}".format(split=test_split))
    if train_split + validation_split + test_split != 1.0:
        error_message = "Training splits configuration is inconsistent: train+validation+test != 1, sum is {sum}".format(sum=train_split + validation_split + test_split)
        logger.critical(error_message)
        exit(1)

    model_savings_location = os.path.join(
        config.gcloud.storage.prefix,
        config.gcloud.storage.model.model_savings.bucket,
        config.gcloud.storage.model.model_savings.folders
    )
    logger.info("Model savings location: {location}".format(location=model_savings_location))

    is_tensorboard_enabled = config.model.training.is_tensorboard_enabled
    tensorboard_location = os.path.join(
        config.gcloud.storage.prefix,
        config.gcloud.storage.model.tensorboard.bucket,
        config.gcloud.storage.model.tensorboard.folders
    )
    logger.info("Tensorboard location: {location}. Enablement option set to {option}".format(
        location=tensorboard_location,
        option=is_tensorboard_enabled
    ))
    load_checkpoints = config.model.training.load_checkpoints
    logging.info("Load checkpoint option set to {option}".format(option=load_checkpoints))
    is_checkpoint_enabled = config.model.training.is_checkpoints_enabled
    checkpoints_location = os.path.join(
        config.gcloud.storage.prefix,
        config.gcloud.storage.model.checkpoints.bucket,
        config.gcloud.storage.model.checkpoints.folders
    )
    logger.info("Checkpoints location: {location}. Enablement option set to {option}".format(
        location=checkpoints_location,
        option=is_checkpoint_enabled
    ))

    bert_url = config.model.bert.url
    logger.info("Bert url: {url}".format(url=bert_url))

    model_name = config.model.name
    model_version = config.model.version
    logger.info("Model name: {name}, versione: {version}".format(name=model_name, version=model_version))

    max_sequence_length = int(round(config.model.inputs.max_sequence_length))
    logger.info("Max sequence length: {length}".format(length=max_sequence_length))
    num_epochs = int(config.model.training.epochs)
    logger.info("Epochs: {epochs}".format(epochs=num_epochs))
    batch_size = int(config.model.training.batch_size)
    logger.info("Batch size: {batch_size}".format(batch_size=batch_size))

    ####################################################################################################################
    # Services and handlers
    ####################################################################################################################
    logger.info("Database service set-up done")

    dataset_handler = DatasetHandler(
        train_split=train_split,
        val_split=validation_split,
        test_split=test_split
    )
    logger.info("Dataset handler set-up done")

    bert_handler = ModelWrapper(
        base_model_url=bert_url,
        checkpoint_location=checkpoints_location,
        model_name=model_name,
        model_version=model_version,
        max_sequence_length=max_sequence_length,
    )
    logger.info("Model handler set-up done.")

    ####################################################################################################################
    # Retrieving data
    ####################################################################################################################
    logger.info("Starting to retrieve data from database service")
    kb = firestore_connector.get_all_data(FirestoreCollections.KNOWLEDGE_BASE)
    logger.info("Retrieved knowledge base")
    keywords = firestore_connector.get_all_data(FirestoreCollections.KEYWORDS)
    logger.info("Retrieved keywords")
    logger.info("Data retrieved")

    num_keywords = len(keywords)
    num_faqs = len(kb)
    logger.info("Number of KB classes: {num_classes}".format(num_classes=num_faqs))
    logger.info("Number of keywords classes: {num_classes}".format(num_classes=num_keywords))

    # Getting only the DisplayText field and making all the keywords lowercase
    keywords = dataset_handler.get_keywords_from_raw_data(keywords)
    logger.info("Processed the keywords from raw data. Keywords are {keywords}".format(keywords=keywords))

    ####################################################################################################################
    # Preparing training, validation and test sets
    ####################################################################################################################
    x, y = dataset_handler.get_examples_and_labels(kb=kb, keywords=keywords)
    logger.info("Dataset retrieved")
    logger.debug("Dataset is x: `{x}`, y: `{y}`".format(x=x, y=y))

    # Splitting the dataset into train, validation and test sets
    x_train, y_train, x_val, y_val, x_test, y_test = dataset_handler.get_train_validation_test_sets(
        x=x,
        y=y,
        train_split=train_split,
        validation_split=validation_split,
        test_split=test_split
    )
    logger.info("Dataset split into train, validation and test sets.")
    logger.debug("x_train: `{x_train}'".format(x_train=x_train))
    logger.debug("y_train: `{y_train}'".format(y_train=y_train))
    logger.debug("x_val: `{x_val}'".format(x_val=x_val))
    logger.debug("y_val: `{y_val}'".format(y_val=y_val))
    logger.debug("x_test: `{x_test}'".format(x_test=x_test))
    logger.debug("y_test: `{y_test}'".format(y_test=y_test))

    # Encoding the faq examples
    x_train_encoded = bert_handler.encode(x_train["kb"])
    logger.info("Encoded knowledge base training set")

    x_validation_encoded = bert_handler.encode(x_val["kb"])
    logger.info("Encoded knowledge base validation set")

    x_test_encoded = bert_handler.encode(x_test["kb"])
    logger.info("Encoded knowledge base test set")

    # Obtaining the feature vectors wrt the keywords ids associated to each example
    x_train_encoded["keywords_ids"] = bert_handler.get_features_tensor_from_ids(
        ids=x_train["keywords_ids"],
        num_classes=num_keywords
    )
    logger.info("Obtained features tensor with ids for training examples")
    x_validation_encoded["keywords_ids"] = bert_handler.get_features_tensor_from_ids(
        ids=x_val["keywords_ids"],
        num_classes=num_keywords
    )
    logger.info("Obtained features tensor with ids for validation examples")
    x_test_encoded["keywords_ids"] = bert_handler.get_features_tensor_from_ids(
        ids=x_test["keywords_ids"],
        num_classes=num_keywords
    )
    logger.info("Obtained features tensor with ids for test examples")

    ####################################################################################################################
    # Building model
    ####################################################################################################################
    bert_handler.build_custom_model(
        num_keywords=num_keywords,
        output_classes=num_faqs
    )
    logger.info("Model built")

    x_train = [
        x_train_encoded["input_word_ids"],
        x_train_encoded["input_mask"],
        x_train_encoded["input_type_ids"],
        x_train_encoded["keywords_ids"]
    ]
    logger.info("Training inputs structured as required by the model to be valid inputs")
    y_train = bert_handler.to_categorical_tensor(data=y_train, num_classes=num_faqs)
    logger.info("Training labels categorical tensor created")
    x_val = [
        x_validation_encoded["input_word_ids"],
        x_validation_encoded["input_mask"],
        x_validation_encoded["input_type_ids"],
        x_validation_encoded["keywords_ids"]
    ]
    logger.info("Validation inputs structured as required by the model to be valid inputs")
    y_val = bert_handler.to_categorical_tensor(data=y_val, num_classes=num_faqs)
    logger.info("Validation labels categorical tensor created")

    ####################################################################################################################
    # Training
    ####################################################################################################################
    logger.info("Starting training")
    bert_handler.train(
        X_train=x_train,
        y_train=y_train,
        X_val=x_val,
        y_val=y_val,
        epochs=num_epochs,
        load_checkpoint=load_checkpoints
    )
    logger.info("Training completed")


if __name__ == "__main__":
    run_training()