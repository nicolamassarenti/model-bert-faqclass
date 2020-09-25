import logging
import os

from settings import settings
from src.training import train
from src.services.firestoreService import FirestoreService
from src.handlers.datasetHandler import DatasetHandler

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    settings()
    logger.info("Settings set up correctly")

    # Setting up firestore service
    firestore_service = FirestoreService()
    logger.info("Firestore service set up done.")

    # Setting up dataset handler
    dataset_handler = DatasetHandler(
        database_service=firestore_service,
        knowledge_base_collection=os.getenv("FIRESTORE_COLLECTION_KB"),
        keywords_collection=os.getenv("FIRESTORE_COLLECTION_KEYWORDS"),
        train_split=float(os.getenv("MODEL_TRAINING_SET_PERCENTAGE")),
        val_split=float(os.getenv("MODEL_VALIDATION_SET_PERCENTAGE"))
    )
    logger.info("Dataset handler set up done.")

    # Retrieving data
    X, Y = dataset_handler.get_data()
    num_classes_keywords = dataset_handler.get_num_classes_keywords()
    num_classes_kb = dataset_handler.get_num_classes_faqs()
    logger.info("Dataset retrieved")

    # Splitting the dataset into train, validation and test sets
    X_train, Y_train, X_val, Y_val, X_test, Y_test = dataset_handler.get_train_validation_test_sets(X, Y)
    logger.info("Dataset split into train, validation and test sets.")

    # Running training
    train(
        X_train=X_train,
        Y_train=Y_train,
        X_validation=X_val,
        Y_validation=Y_val,
        X_test=X_test,
        Y_test=Y_test,
        num_classes={"keywords": num_classes_keywords, "kb": num_classes_kb}
    )
    logger.info("Training executed.")
