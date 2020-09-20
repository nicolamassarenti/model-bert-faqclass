import logging
import re
import numpy as np

logger = logging.getLogger(__name__)


class DatasetHandler:
    def __init__(self, database_service, knowledge_base_collection=None, keywords_collection=None):
        """
        Initializes the DatasetHandler object

        :param database_service: (object) the object that connects to the database
        :param knowledge_base_collection: (string) the name of the KB collection
        :param keywords_collection: (string) the name of the keywords collection
        """
        if knowledge_base_collection is None:
            message = "knowledge_base_collection is None"
            logger.error(message)
            raise Exception(message)
        if keywords_collection is None:
            message = "keywords_collection is None"
            logger.error(message)
            raise Exception(message)

        self._database_service = database_service

        self._knowledge_base_collection = knowledge_base_collection
        self._keywords_collection = keywords_collection

    def _get_examples_from_faq(self, faq):
        """
        Given an faq returns a list with the examples associated to the faq
        :param faq: the faq
        :return: list of string with the training examples
        """
        examples = [faq['MainExample']]
        for lang in faq['TrainingExamples']:
            examples += lang['Examples']

        return examples

    def _get_example_label_pairs(self, data):
        """
        Returns each training example associated to the label.

        :param data: all the knowledge base
        :return: the list of training examples associated to the label
        """
        example_label_pairs = []
        label = 0
        for faq in data:
            examples = self._get_examples_from_faq(faq)
            example_label_pairs += list(map(lambda x: {"X": x, "Y": label}, examples))
            label += 1

        return example_label_pairs

    def _get_keyword_id(self, text, keywords):
        """
        TODO: mettere descrizione
        :param text:
        :param keywords:
        :return:
        """
        text = text.lower()
        keywords = [key.lower() for key in keywords]

        regex_pattern = r'(\s|\.|\')({})(\s|\n|\.|[?!])'
        for idx, key in enumerate(keywords):
            if re.search(pattern=regex_pattern.format(key), string=text):
                return idx + 1

        return 0

    def _get_examples_keywords_labels(self, keywords, data):
        """
        TODO: mettere descrizione
        :param keywords:
        :param data:
        :return:
        """
        X_train, X_keywords, Y = [], [], []
        for example in data:
            keyword_id = self._get_keyword_id(example["X"], keywords)
            X_train.append(example["X"])
            X_keywords.append(keyword_id)
            Y.append(example["Y"])

        return X_train, X_keywords, Y

    def _shuffle_and_convert_to_np_array(self, X_text, X_keys, Y):
        """
        TODO: mettere descrizione
        :param X_text:
        :param X_keys:
        :param Y:
        :return:
        """
        X_text = np.array(X_text)
        X_keys = np.array(X_keys)
        Y = np.array(Y)
        indices = np.arange(X_text.shape[0])
        np.random.shuffle(indices)

        X_text = X_text[indices].tolist()
        X_keys = X_keys[indices].tolist()
        Y = Y[indices].tolist()

        return X_text, X_keys, Y

    def get_data(self):
        """
        Returns the data (..... TODO: mettere descrizione)
        :return:
        """

        #TODO: mettere qualche log sia di info sia di debug
        kb = self._database_service.get_all_data(self._knowledge_base_collection)
        keywords = self._database_service.get_all_data(self._keywords_collection)

        keywords = list(map(lambda x : x["DisplayText"], keywords))

        training_examples_and_labels = self._get_example_label_pairs(data=kb)
        X_examples, X_keywords, Y = self._get_examples_keywords_labels(
            keywords=keywords,
            data=training_examples_and_labels
        )

        X_examples, X_keywords, Y = self._shuffle_and_convert_to_np_array(
            X_text=X_examples,
            X_keys=X_keywords,
            Y=Y
        )

        return X_examples, X_keywords, Y