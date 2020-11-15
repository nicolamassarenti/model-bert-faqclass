import logging

from src.training import train
from src.configurations import config
from src.services.firestoreService import FirestoreService
from src.handlers.datasetHandler import DatasetHandler


logger = logging.getLogger(__name__)


def run_training():
    # Constants
    knowledge_base_collection = config.gcloud.database.collections.knowledge_base.name
    keywords_collection = config.gcloud.database.collections.keywords.name

    train_split = float(config.model.training.split.train)
    validation_split = float(config.model.training.split.validation)
    test_split = float(config.model.training.split.test)
    if train_split + validation_split + test_split != 1.0:
        error_message = "Training splits configuration is inconsistent: train+validation+test != 1, sum is {sum}".format(sum=train_split + validation_split + test_split)
        logger.critical(error_message)
        exit(1)

    # Services and handlers
    database_service = FirestoreService()
    logger.info("Database service set-up done")

    dataset_handler = DatasetHandler(
        train_split=train_split,
        val_split=validation_split,
        test_split=test_split
    )
    logger.info("Dataset handler set-up done")

    # Retrieving data
    logger.info("Starting to retrieve data from database service")
    kb = database_service.get_all_data(knowledge_base_collection)
    logger.info("Retrived knowledge base")
    keywords = database_service.get_all_data(keywords_collection)
    logger.info("Retrieved keywords")
    logger.info("Data retrieved")

    num_classes_keywords = len(keywords)
    num_classes_kb = len(kb)
    logger.info("Number of KB classes: {num_classes}".format(num_classes=num_classes_kb))
    logger.info("Number of keywords classes: {num_classes}".format(num_classes=num_classes_keywords))

    # Getting only the DisplayText field and making all the keywords lowercase
    keywords = dataset_handler.get_keywords_from_raw_data(keywords)
    logger.info("Processed the keywords from raw data. Keywords are {keywords}".format(keywords=keywords))

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

    # Running training
    train(
        x_train=x_train,
        y_train=Y_train,
        x_validation=X_val,
        y_validation=Y_val,
        x_test=X_test,
        y_test=Y_test,
        num_classes={"keywords": num_classes_keywords, "kb": num_classes_kb}
    )
    logger.info("Training executed.")


if __name__ == "__main__":
    run_training()