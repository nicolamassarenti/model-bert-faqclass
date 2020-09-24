import firebase_admin
import logging
from firebase_admin import firestore

logger = logging.getLogger(__name__)


class FirestoreService:
    def __init__(self):
        """
        Creates the service that connects to Firestore

        :return None
        """
        self.__db = None

        self._connect_to_db()

    def _connect_to_db(self):
        """
        Creates the connection to Firestore

        :return: None
        """
        firebase_admin.initialize_app()

        self.__db = firestore.client()

    def get_all_data(self, collection):
        """
        Retrieves all the data from a specific `collection`
        :param collection: (string) the name of the collection
        :return: array of documents
        """

        data = []

        # Querying for the documents
        collection_reference = self.__db.collection(collection)

        for doc in collection_reference.stream():
            logging.debug(u"Retrieved doc with id: `{}`".format(doc.id))
            data.append(doc.to_dict())

        return data

    def add(self, collection, data):
        """
        TODO: aggiungere commenti e log
        :param collection:
        :param data:
        :return:
        """
        self.__db.collection(collection).add(data)