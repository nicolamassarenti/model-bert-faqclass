import logging
import traceback
import tensorflow as tf
import tensorflow_hub as hub
import src.handlers.tokenizer as tokenizer
import numpy as np

logger = logging.getLogger(__name__)


class BertHandler:

    def __init__(self, model_path, model_name="bert", model_version="default", max_sequence_length=128):
        """
        The constructor of the handler.

        :param model_path: (string) the path where the model is stored
        :param model_name: (string) the name of the model
        :param model_version: (string) the version of the model
        """
        if model_path is None:
            raise Exception("model_path is None")

        self._model_path = model_path
        self.model_version = model_version
        self.model_name = model_name
        self._params = None
        self._bert_layer = None
        self._tokenizer = None
        self._max_sequence_length = max_sequence_length

    def init_bert_layer(self, trainable=False):
        """
        Creates the bert layer

        :param trainable: (bool) true if you want to re-train the layer via transfer learning
        :return: None
        """
        try:
            self._bert_layer = hub.KerasLayer(self._model_path, trainable=trainable)

        except Exception as e:
            stacktrace = traceback.format_exc()

            logger.error("{}".format(stacktrace))
            raise e

    def get_bert_layer(self):
        """
        Returns the bert layer

        :return: object
        """
        return self._bert_layer

    def init_tokenizer(self, path_vocab_file, do_lower_case=True):
        """
        Creates the tokenizer.

        :param path_vocab_file: (string) the path to the .txt vocab file
        :param do_lower_case: (bool) true if the tokenizer must set to lower case all the letters
        :return: None
        """
        try:
            self._tokenizer = tokenizer.FullTokenizer(
                vocab_file=path_vocab_file,
                do_lower_case=do_lower_case)
        except Exception as e:
            stacktrace = traceback.format_exc()

            logger.error("{}".format(stacktrace))
            raise e

    def get_tokenizer(self):
        """
        Returns the tokenizer.

        :return: object
        """
        return self._tokenizer

    def tokenize(self, data):
        """
        TODO: descrivere
        :param data:
        :return:
        """

        data_tokenized = map(self._tokenizer.tokenize, data)
        data_tokenized = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], data_tokenized)
        data_tokenized = list(map(self._tokenizer.convert_tokens_to_ids, data_tokenized))

        data_tokenized = map(lambda tids: tids + [0] * (self._max_sequence_length - len(tids)), data_tokenized)
        return np.array(list(data_tokenized))

    def create_custom_model(self, num_classes):
        """
        TODO commenti e logs
        :param num_classes:
        :return:
        """
        # Model definition
        input_bert = tf.keras.layers.Input(shape=(self._max_sequence_length,), dtype='int32', name='input_bert')
        input_keywords = tf.keras.layers.Input(shape=(6,), name='input_keywords')

        bert_hidden = self._bert_layer(input_bert)
        bert_output = tf.keras.layers.Flatten(name="bert_output_flatten")(bert_hidden)

        hidden = tf.concat([bert_output, input_keywords], -1)
        hidden = tf.keras.layers.Dense(256, activation=tf.nn.relu, name='dense_1')(hidden)
        hidden = tf.keras.layers.BatchNormalization(name='batch_norm_1')(hidden)
        hidden = tf.keras.layers.Dropout(0.5)(hidden)
        hidden = tf.keras.layers.Dense(256, activation=tf.nn.relu, name='dense_2')(hidden)
        hidden = tf.keras.layers.BatchNormalization(name='batch_norm_2')(hidden)
        hidden = tf.keras.layers.Dropout(0.5)(hidden)
        output = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax, name='output')(hidden)
        # End model definition

        model = tf.keras.Model([input_bert, input_keywords], output, name="FAQ_classifier")
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=0.00001), metrics=['accuracy'])
        return model