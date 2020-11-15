import logging

from src.training import train
from src.configurations import config
from src.services.firestoreService import FirestoreService
from src.handlers.datasetHandler import DatasetHandler

if __name__ == "__main__":
    # Setting up firestore service
    firestore_service = FirestoreService()
    logging.info("Firestore service set up done.")

    # Setting up dataset handler
    dataset_handler = DatasetHandler(
        database_service=firestore_service,
        knowledge_base_collection=config["gcloud"]["database"]["collections"]["knowledge_base"]["name"],
        keywords_collection=config["gcloud"]["database"]["collections"]["keywords"]["name"],
        train_split=float(config["model"]["training"]["split_percentages"]["train"]),
        val_split=float(config["model"]["training"]["split_percentages"]["validation"])
    )
    logging.info("Dataset handler set up done.")

    # Retrieving data
    X, Y = dataset_handler.get_data()
    num_classes_keywords = dataset_handler.get_num_classes_keywords()
    num_classes_kb = dataset_handler.get_num_classes_kb()
    logging.info("Dataset retrieved")

    # Splitting the dataset into train, validation and test sets
    X_train, Y_train, X_val, Y_val, X_test, Y_test = dataset_handler.get_train_validation_test_sets(X, Y)
    logging.info("Dataset split into train, validation and test sets.")

    # Running training
    train(
        x_train=X_train,
        y_train=Y_train,
        x_validation=X_val,
        y_validation=Y_val,
        x_test=X_test,
        y_test=Y_test,
        num_classes={"keywords": num_classes_keywords, "kb": num_classes_kb}
    )
    logging.info("Training executed.")
