import logging
import re

import numpy as np
import tensorflow as tf

from bert_faqclass.model.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class DatasetHandler:
    def __init__(
        self,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        preprocessor: Preprocessor = None,
    ):
        """
        Created the dataset handler
        :param train_split: split of dataset for model training
        :param val_split: split of dataset for model validation
        :param test_split: split of dataset for model test
        :param preprocessor: the preprocessor object to format data as required by the model
        """

        self._regex = r"(\s|\.|\')({})(\s|\n|\.|[?!-])"

        self._train_split = train_split
        self._val_split = val_split
        self._test_split = test_split

        self._preprocessor = preprocessor

    def _get_examples_from_faq(self, faq: dict) -> [str]:
        """
        Given an faq returns a list with the examples associated to the input faq

        :param faq: the faq
        :return: list of string with the training examples
        """
        examples = [faq["MainExample"]]
        for lang in faq["TrainingExamples"]:
            examples.extend(lang["Examples"])

        return examples

    def _get_example_label_pairs(self, data: dict) -> [dict]:
        """
        Returns a dictionary with the training example and the corresponding label

        :param data: the knowledge base
        :return: the list of training examples associated to the label
        """
        example_label_pairs = []
        for label_id, faq in enumerate(data):
            examples = self._get_examples_from_faq(faq)
            logger.debug(
                "Processed faq `{faq}` with examples `{examples}`".format(
                    faq=faq, examples=examples
                )
            )
            example_label_pairs.extend(
                list(map(lambda x: {"x": x, "y": label_id}, examples))
            )

        return example_label_pairs

    def _get_keyword_id(self, keywords: [str], text: str) -> int:
        """
        Given a text, if a keyword is inside the text, it returns the position of the keyword inside the vector of
        keywords, otherwise, if no keyword is inside the text, returns None.

        :param text: the training example
        :return: 0 if no keywords otherwise the keyword id, from 1 to len(keyword)+1
        """
        text = text.lower()

        for idx, key in enumerate(keywords):
            if re.search(pattern=self._regex.format(key), string=text):
                return idx + 1

        return 0

    def _get_examples_keywords_labels(self, dataset: [dict], keywords: [str]) -> ([str], [int], [str]):
        """
        For each example in the dataset, returns the example, the keyword id and the corresponding label

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
        Shuffles the data

        :param x_text: list of training examples
        :param x_keys: list of keywords ids
        :param y: list of labels
        :return: the input shuffled
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

    def _get_keywords_from_raw_data(self, keywords: [dict]) -> [str]:
        """
        Returns the keyword
        :param keywords:
        :return:
        """
        return list(map(lambda x: x["DisplayText"].lower(), keywords))

    def get_train_val_test_splits(
            self, kb: [dict], keywords: [dict]
    ) -> (tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor):
        """
        Returns the train, validation and test splits
        :param kb: the knowledge base
        :param keywords: the list of keywords
        :return: the train, validation and test examples and labels
        """
        # Data format transformation
        keywords = self._get_keywords_from_raw_data(keywords)
        logger.info(
            "Processed the keywords from raw data. Keywords are {keywords}".format(
                keywords=keywords
            )
        )

        training_examples_and_labels = self._get_example_label_pairs(data=kb)
        logger.debug("For each entry, obtain the example and the correspondent label")

        x_kb, x_keywords, y = self._get_examples_keywords_labels(
            dataset=training_examples_and_labels, keywords=keywords
        )
        logger.debug(
            "Obtained x_kb=`{x_kb}`, x_keywords=`{x_keywords}`, y=`{y}`".format(
                x_kb=x_kb, x_keywords=x_keywords, y=y
            )
        )

        # Shuffling keeping consistency between examples and labels
        x_kb, x_keywords, y = self._shuffle(x_text=x_kb, x_keys=x_keywords, y=y)
        logger.debug(
            "Shuffled elements. Obtained x_kb=`{x_kb}`, x_keywords=`{x_keywords}`, y=`{y}`".format(
                x_kb=x_kb, x_keywords=x_keywords, y=y
            )
        )

        # Computing constants
        num_keywords = len(keywords)
        logger.debug("Num keywords: {num_keywords}".format(num_keywords=num_keywords))

        num_faqs = len(kb)
        logger.debug("Num faqs: {num_faqs}".format(num_faqs=num_faqs))

        num_examples = len(x_kb)
        logger.debug("Num examples: {num_examples}".format(num_examples=num_examples))

        train_idx = round(self._train_split * num_examples)
        logger.debug("Training set split index: {idx}".format(idx=train_idx))

        val_idx = train_idx + round(self._val_split * num_examples)
        logger.debug("Validation set split index: {idx}".format(idx=val_idx))

        # Splitting into training, validation and test sets and preprocessing
        logger.debug(
            "Starting to split into training, validation and test sets and preprocessing data"
        )
        x_train = [
            self._preprocessor.preprocess(data=x_kb[:train_idx]),
            tf.keras.utils.to_categorical(
                y=x_keywords[:train_idx], num_classes=num_keywords + 1
            ),
        ]
        y_train = tf.keras.utils.to_categorical(y=y[:train_idx], num_classes=num_faqs)

        x_val = [
            self._preprocessor.preprocess(data=x_kb[train_idx:val_idx]),
            tf.keras.utils.to_categorical(
                y=x_keywords[train_idx:val_idx], num_classes=num_keywords + 1
            ),
        ]
        y_val = tf.keras.utils.to_categorical(
            y=y[train_idx:val_idx], num_classes=num_faqs
        )

        x_test = [
            self._preprocessor.preprocess(data=x_kb[val_idx:]),
            tf.keras.utils.to_categorical(
                y=x_keywords[val_idx:], num_classes=num_keywords + 1
            ),
        ]
        y_test = tf.keras.utils.to_categorical(y=y[val_idx:], num_classes=num_faqs)

        return x_train, y_train, x_val, y_val, x_test, y_test
