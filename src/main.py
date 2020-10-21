import logging
import os
import argparse

from src.settings import settings
from src.training import train
from src.services.firestoreService import FirestoreService
from src.handlers.datasetHandler import DatasetHandler

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("bert-faqclass")
    parser.add_argument("--google_application_credential",
                        help="Path to the service account", type=str)
    parser.add_argument("--bert_url",
                        help="Tensorflow hub url to download bert model.", type=str)
    parser.add_argument("--model_checkpoint_path",
                        help="Path to the folder with the model checkpoints.", type=str)
    parser.add_argument("--logs_config_path",
                        help="Path to the logs config file.", type=str)

    args = parser.parse_args()

    print(args.google_application_credential)
    print(args.bert_url)
    print(args.model_checkpoint_path)
    print(args.logs_config_path)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.google_application_credential
    os.environ["PATH_BERT_MODEL"] = args.bert_url
    os.environ["PATH_MODEL_CHECKPOINT"] = args.model_checkpoint_path
    os.environ["PATH_LOGS_CONFIG"] = args.logs_config_path


    os.environ["MODEL_NAME"] = "bert-faqclass"
    os.environ["MODEL_VERSION"] = "0.0.2"

    os.environ["MODEL_TRAINING_EPOCHS"] = "1000"
    os.environ["MODEL_TRAINING_SET_PERCENTAGE"] = "0.7"
    os.environ["MODEL_VALIDATION_SET_PERCENTAGE"] = "0.2"
    os.environ["MAX_SEQ_LENGTH"] = "128"

    os.environ["PLOT_MODEL"] = "False"

    os.environ["FIRESTORE_COLLECTION_KB"] = "KnowledgeBase"
    os.environ["FIRESTORE_COLLECTION_KEYWORDS"] = "Keywords"



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
    num_classes_kb = dataset_handler.get_num_classes_kb()
    logger.info("Dataset retrieved")

    # Splitting the dataset into train, validation and test sets
    X_train, Y_train, X_val, Y_val, X_test, Y_test = dataset_handler.get_train_validation_test_sets(X, Y)
    logger.info("Dataset split into train, validation and test sets.")

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
    logger.info("Training executed.")
