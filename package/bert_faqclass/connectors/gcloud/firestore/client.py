import logging

from firebase_admin import firestore
import firebase_admin

logger = logging.getLogger(__name__)


class StorageConnector:
    def __init__(self):
        firebase_admin.initialize_app()
        self.__db = firestore.client()

    def add(self, collection: str, data: dict):
        """
        Adds a document to the collection
        :param collection: the collection
        :param data: the document to add
        :return: None
        """
        self.__db.collection(collection).add(data)
        logger.info("Adding to collection `{}` document: {}".format(collection, data))

    def get_all_data(self, collection: str) -> [dict]:
        """
        Retrieves all the data from a specific `collection`
        :param collection: (string) the name of the collection
        :return: array of documents
        """

        data = []

        # Querying for the documents
        collection_reference = self.__db.collection(collection)

        for doc in collection_reference.stream():
            logging.debug(u"Retrieved doc with id: `{}` from collection `{}`".format(doc.id, collection))
            data.append(doc.to_dict())

        return data

    def empty_collection(self, collection: str):
        # Querying for the documents
        collection_reference = self.__db.collection(collection)

        for doc in collection_reference.stream():
            self.__db.collection(collection).document(doc.id).delete()
            logger.info("Deleted from collection `{}` document: {}".format(collection, doc.id))


connector = StorageConnector()
logging.info("Firestore connector created")
