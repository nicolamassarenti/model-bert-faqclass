import logging
import traceback

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

logger = logging.getLogger(__name__)


class Model:
    def __init__(
        self,
        base_model_url: str = None,
        checkpoint_location: str = None,
        fine_tuned_model_location: str = None,
        model_name: str = "bert-faqclass",
        model_version: str = "1.0.0",
        max_sequence_length: int = 128,
    ):
        """
        Is the constuctor of the handler.

        :param model_path: the url to download the model from tensorflow hub
        :param model_name: the name of the model
        :param model_version: the version of the model
        :param max_sequence_length: the max sequence length accepted
        :param plot_path: True if to plot the model
        :param checkpoint_location: the path to the model
        """

        if base_model_url is None:
            error_message = "Url to download the base model has not been provided"
            logger.critical(error_message)
            exit(1)

        self.model_version = model_version
        self.model_name = model_name

        self._base_model_url = base_model_url
        self._max_sequence_length = max_sequence_length
        self._checkpoint_location = checkpoint_location
        self.fine_tuned_model_location = fine_tuned_model_location

        self._base_model = None
        self._model = None

        self._init_bert_layer(trainable=False)

    def _init_bert_layer(self, trainable: bool = False):
        """
        Creates the bert layer

        :param trainable: (bool) true if you want to re-train the layer via transfer learning
        :return: None
        """
        try:
            self._base_model = hub.KerasLayer(
                self._base_model_url, trainable=trainable, name="BERT"
            )
            logger.debug(
                "Downloaded base model from {url}".format(url=self._base_model_url)
            )

            logger.debug("Tokenizer built")

        except Exception as e:
            stacktrace = traceback.format_exc()
            logger.critical("{}".format(stacktrace))
            exit(1)

    def get_features_tensor_from_ids(self, ids: [int], num_classes: int) -> tf.Tensor:
        """
        Returns the features given the ids of the keywords
        :param ids: the vector of ids
        :param num_classes: number of keywords
        :return:
        """

        # # features = np.zeros(shape=(len(ids), num_classes), dtype=np.bool)
        # features = []
        # for idx, keyword_id in enumerate(ids):
        #     if keyword_id is None:
        #         features.append(0)
        #     else:
        #         features.append(keyword_id + 1)
        #         # features[idx, keyword_id] = True

        return tf.keras.utils.to_categorical(ids)

    def get_bert_layer(self):
        """
        Returns the bert layer

        :return: object
        """
        return self._base_model

    def build_custom_model(self, num_keywords: int, output_classes: int):
        """
        Builds the custom model

        :param num_keywords: number of keywords
        :param output_classes: number of output classes
        :return: None
        """
        # Model definition
        encoder_inputs = dict(
            input_word_ids=tf.keras.layers.Input(
                shape=(self._max_sequence_length,),
                dtype=tf.int32,
                name="input_word_ids",
            ),
            input_mask=tf.keras.layers.Input(
                shape=(self._max_sequence_length,), dtype=tf.int32, name="input_mask"
            ),
            input_type_ids=tf.keras.layers.Input(
                shape=(self._max_sequence_length,), dtype=tf.int32, name="segment_ids"
            ),
        )
        keywords_ids = tf.keras.layers.Input(shape=(num_keywords+1, ), name='keywords_ids')

        bert_output = self._base_model(encoder_inputs)["pooled_output"]

        # hidden = tf.concat([bert_output, keywords_ids], -1)
        hidden = tf.keras.layers.Dense(128, activation=tf.nn.relu, name="dense_1")(
            bert_output
        )
        hidden = tf.keras.layers.BatchNormalization(name="batch_norm_1")(hidden)
        hidden = tf.keras.layers.Dropout(0.5)(hidden)
        hidden = tf.keras.layers.Dense(64, activation=tf.nn.relu, name="dense_2")(
            hidden
        )
        hidden = tf.concat([hidden, keywords_ids], -1)
        hidden = tf.keras.layers.BatchNormalization(name="batch_norm_2")(hidden)
        hidden = tf.keras.layers.Dropout(0.5)(hidden)
        output = tf.keras.layers.Dense(
            output_classes, activation=tf.nn.softmax, name="output"
        )(hidden)

        self._model = tf.keras.Model(
            inputs=[encoder_inputs, keywords_ids], outputs=output, name=self.model_name
        )
        self._model.summary()
        logger.debug("Model defined")

        self._model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.optimizers.Adam(),
            metrics=["accuracy"],
        )
        logger.debug("Model compiled")

    def train(
        self,
        X_train: tf.Tensor,
        y_train: tf.Tensor,
        X_val: tf.Tensor,
        y_val: tf.Tensor,
        epochs: int = 200,
        load_checkpoint: bool = False,
    ):
        """
        Trains the model
        :param X_train: the training set
        :param y_train: the training labels
        :param X_val: the validation set
        :param y_val: the validation labels
        :param epochs: number of epochs
        :param load_checkpoint: True if to load from checkpoint
        :return: None
        """
        callbacks = []
        # if self._checkpoint_location is not None:
        #     cp_callback = tf.keras.callbacks.ModelCheckpoint(
        #         filepath=self._checkpoint_location,
        #         save_weights_only=True,
        #         verbose=1
        #     )
        #     callbacks = [cp_callback]
        #
        # if self._checkpoint_location is not None and load_checkpoint:
        #     self._model.load_weights(self._checkpoint_location)
        #
        # early_stopping = tf.keras.callbacks.EarlyStopping(
        #     monitor="val_loss", verbose=1, mode="auto", patience=10
        # )
        # callbacks.append(early_stopping)

        self._model.fit(
            X_train,
            y_train,
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=1,
            callbacks=callbacks
        )

    def to_categorical_tensor(self, data: [int], num_classes: int = None) -> np.array:
        """
        Converts the data to a categorical tensor

        :param data:
        :param num_classes: the number of classes
        :return:
        """
        return tf.keras.utils.to_categorical(y=data, num_classes=num_classes)

    def save(self):
        self._model.save(self.fine_tuned_model_location)

    def test(self, x: tf.Tensor, y: tf.Tensor):
        test_accuracy = tf.keras.metrics.Accuracy()

        logits = self._model(x)
        prediction = tf.argmax(logits, axis=1, output_type=tf.int32)

        y_integers = tf.argmax(y, axis=1, output_type=tf.int32)
        test_accuracy(prediction, y_integers)

        print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

        print("A")