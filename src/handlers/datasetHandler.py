import logging
import re
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class DatasetHandler:
    def __init__(self,
                 database_service,
                 knowledge_base_collection=None,
                 keywords_collection=None,
                 train_split=0.8,
                 val_split=0.1):
        """
        Initializes the DatasetHandler object

        :param database_service: the service that connects to the database
        :param knowledge_base_collection: the name of the collection that contains the knowledge base
        :param keywords_collection: the name of the collection that contains the keywords
        :param train_split: percentage of the training data
        :param val_split: percentage of the validation data
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
        self._keywords = []
        self._num_classes_keywords = 0
        self._num_classes_kb = 0
        self._regex = r'(\s|\.|\')({})(\s|\n|\.|[?!-])'

        self._train_split = train_split
        self._val_split = val_split

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
            example_label_pairs += list(map(lambda x: {"x": x, "y": label}, examples))
            label += 1

        return example_label_pairs

    def _get_keyword_id(self, text):
        """
        Given a text, if a keyword is inside the text, it returns the position of the keyword inside the vector of
        keywords, otherwise, if no keyword is inside the text, returns None.

        :param text: the text
        :return: None OR integer from [0:len(keywords)]
        """
        text = text.lower()

        for idx, key in enumerate(self._keywords):
            if re.search(pattern=self._regex.format(key), string=text):
                return idx

        return None

    def _get_examples_keywords_labels(self, dataset):
        """
        For each example in the dataset, returns the example, the keyword id and the label associated to the example.
        
        :param dataset: the dataset
        :return: list of training examples, list of keywords ids, list of labels
        """
        x_train, x_keywords, y = [], [], []
        for example in dataset:
            keyword_id = self._get_keyword_id(example["x"])
            x_train.append(example["x"])
            x_keywords.append(keyword_id)
            y.append(example["y"])

        return x_train, x_keywords, y

    def _shuffle(self, x_text, x_keys, y):
        """
        Shuffles the data, keeping the indexes the same among the three data.

        :param x_text: list of training examples
        :param x_keys: list of keywords ids
        :param y: list of labels
        :return:
        """
        x_text = np.array(x_text)
        x_keys = np.array(x_keys)
        y = np.array(y)
        indices = np.arange(x_text.shape[0])
        np.random.shuffle(indices)

        x_text = x_text[indices].tolist()
        x_keys = x_keys[indices].tolist()
        y = y[indices].tolist()

        return x_text, x_keys, y


    def get_data(self):
        """
        Returns the data examples, keyword ids and labels shuffled.
        """

        # TODO: mettere qualche log sia di info sia di debug
        kb = self._database_service.get_all_data(self._knowledge_base_collection)
        logger.info("Retrived the kb.")
        keywords = self._database_service.get_all_data(self._keywords_collection)
        logger.info("Retrieved the keywords")

        self._num_classes_keywords = len(keywords)
        self._num_classes_kb = len(kb)
        logger.info("Number of KB classes: {}".format(self._num_classes_kb))
        logger.info("Number of keywords classes: {}".format(self._num_classes_keywords))

        # Getting only the DisplayText field and making all the keywords lowercase
        self._keywords = list(map(lambda x : x["DisplayText"].lower(), keywords))

        # Getting the label associated to each examples
        training_examples_and_labels = self._get_example_label_pairs(data=kb)
        x_kb, x_keywords, y = self._get_examples_keywords_labels(
            dataset=training_examples_and_labels
        )

        # Shuffling keeping consistency between examples and labels
        x_kb, x_keywords, y = self._shuffle(
            x_text=x_kb,
            x_keys=x_keywords,
            y=y
        )
        logger.info("Obtained the faqs examples, the keyword ids and the labels.")

        return {"kb": x_kb, "keywords_ids": x_keywords}, y

    def get_train_validation_test_sets(self, x, y):
        """
        Divides the data and labels into train, validation and test sets.
        :param x: examples
        :param y: labels
        :return:
        """
        num_examples = len(x["kb"])

        limit_train = round(self._train_split * num_examples)
        limit_val = round(self._val_split * num_examples)

        x_kb = x["kb"]
        x_keywords_ids = x["keywords_ids"]

        # Train
        x_train = {
            "kb": x_kb[:limit_train],
            "keywords_ids": x_keywords_ids[:limit_train]
        }
        y_train = y[:limit_train]

        # Validation
        x_val = {
            "kb": x_kb[limit_train:limit_train + limit_val],
            "keywords_ids": x_keywords_ids[limit_train:limit_train + limit_val]
        }
        y_val = y[limit_train:limit_train + limit_val]

        # Test
        x_test = {
            "kb": x_kb[limit_train + limit_val:],
            "keywords_ids": x_keywords_ids[limit_train + limit_val:]
        }
        y_test = y[min(num_examples, limit_train + limit_val):]

        return x_train, y_train, x_val, y_val, x_test, y_test

    def get_num_classes_keywords(self):
        """
        Returns the number of keywords.
        """
        return self._num_classes_keywords

    def get_num_classes_kb(self):
        """
        Returns the number of classes in the kb, i.e. the number of faqs
        """
        return self._num_classes_kb

