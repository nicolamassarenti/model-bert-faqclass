import logging
import re
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class DatasetHandler:
    def __init__(self,
                 train_split: float = 0.8,
                 val_split: float = 0.1,
                 test_split: float = 0.1):
        """
        Initializes the DatasetHandler object

        :param train_split: percentage of the training data
        :param val_split: percentage of the validation data
        """
        # self._keywords = []
        # self._num_classes_keywords = 0
        # self._num_classes_kb = 0
        self._regex = r'(\s|\.|\')({})(\s|\n|\.|[?!-])'

        self._train_split = train_split
        self._val_split = val_split
        self._test_split = test_split

    def _get_examples_from_faq(self, faq: dict) -> [str]:
        """
        Given an faq returns a list with the examples associated to the faq

        :param faq: the faq
        :return: list of string with the training examples
        """
        examples = [faq['MainExample']]
        for lang in faq['TrainingExamples']:
            examples.extend(lang['Examples'])

        return examples

    def _get_example_label_pairs(self, data: dict) -> [dict]:
        """
        Returns each training example associated to the label.

        :param data: all the knowledge base
        :return: the list of training examples associated to the label
        """
        example_label_pairs = []
        label = 0
        for faq in data:
            examples = self._get_examples_from_faq(faq)
            logger.debug("Processed faq `{faq}` and obtained examples `{examples}`".format(faq=faq, examples=examples))
            example_label_pairs.extend(list(map(lambda x: {"x": x, "y": label}, examples)))
            label += 1

        return example_label_pairs

    def _get_keyword_id(self, keywords: [str], text: str) -> int:
        """
        Given a text, if a keyword is inside the text, it returns the position of the keyword inside the vector of
        keywords, otherwise, if no keyword is inside the text, returns None.

        :param text: the text
        :return: None OR integer from [0:len(keywords)]
        """
        text = text.lower()

        for idx, key in enumerate(keywords):
            if re.search(pattern=self._regex.format(key), string=text):
                return idx

        return None


    def _get_examples_keywords_labels(self, dataset: [dict], keywords: [str]) -> ([str], [int], [str]):
        """
        For each example in the dataset, returns the example, the keyword id and the label associated to the example.
        
        :param dataset: the dataset
        :return: list of training examples, list of keywords ids, list of labels
        """
        x_train, x_keywords, y = [], [], []
        for example in dataset:
            keyword_id = self._get_keyword_id(keywords=keywords, text=example["x"])
            x_train.append(example["x"])
            x_keywords.append(keyword_id)
            y.append(example["y"])

        return x_train, x_keywords, y

    def _shuffle(self, x_text: [str], x_keys: [int], y: [int]) -> ([str], [int], [int]):
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

    def get_examples_and_labels(self, kb: [dict] = None, keywords: [str] = None) -> (dict, [str]):
        """
        Returns the data examples, keyword ids and labels shuffled.
        """

        # Getting the label associated to each examples
        training_examples_and_labels = self._get_example_label_pairs(data=kb)
        logger.debug("For each entry, obtain the example and the correspondent label")

        x_kb, x_keywords, y = self._get_examples_keywords_labels(
            dataset=training_examples_and_labels,
            keywords=keywords
        )
        logger.debug("Obtained x_kb=`{x_kb}`, x_keywords=`{x_keywords}`, y=`{y}`".format(
            x_kb=x_kb, x_keywords=x_keywords, y=y
        ))

        # Shuffling keeping consistency between examples and labels
        x_kb, x_keywords, y = self._shuffle(
            x_text=x_kb,
            x_keys=x_keywords,
            y=y
        )
        logger.debug("Shuffled elements. Obtained x_kb=`{x_kb}`, x_keywords=`{x_keywords}`, y=`{y}`".format(
            x_kb=x_kb, x_keywords=x_keywords, y=y
        ))

        return {"kb": x_kb, "keywords_ids": x_keywords}, y

    def get_train_validation_test_sets(self,
                                       x: [dict],
                                       y: [str],
                                       train_split: float = 0.6,
                                       validation_split: float = 0.2,
                                       test_split: float = 0.2
                                       ) -> (dict, [int], dict, [int], dict, [int]):
        """
        Divides the data and labels into train, validation and test sets.
        :param x: examples
        :param y: labels
        :return:
        """
        # TODO: rifare funzione
        num_examples = len(x["kb"])

        limit_train = round(train_split * num_examples)
        limit_val = round(validation_split * num_examples)

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

    def get_keywords_from_raw_data(self, keywords: [dict]) -> [str]:
        return list(map(lambda x: x["DisplayText"].lower(), keywords))
