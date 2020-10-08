import logging
import traceback
import tensorflow as tf
import tensorflow_hub as hub
import src.handlers.tokenization as tokenization
import numpy as np

logger = logging.getLogger(__name__)


class BertHandler:

    def __init__(self, model_path, model_name="bert", model_version="default", max_sequence_length=128, plot_path=None,
                 checkpoint_path=None):
        """
        Is the constuctor of the handler.

        :param model_path: the url to download the model from tensorflow hub
        :param model_name: the name of the model
        :param model_version: the version of the model
        :param max_sequence_length: the max sequence length accepted
        :param plot_path: True if to plot the model
        :param checkpoint_path: the path to the model
        """

        if model_path is None:
            raise Exception("model_path is None")

        self._model_path = model_path
        self.model_version = model_version
        self.model_name = model_name
        self._params = None
        self._bert_model = None
        self._tokenizer = None
        self._max_sequence_length = max_sequence_length
        self._plot_path = plot_path
        self._model = None
        self._checkpoint_path = checkpoint_path

        self._init_bert_layer(trainable=False)

    def _init_bert_layer(self, trainable=False):
        """
        Creates the bert layer

        :param trainable: (bool) true if you want to re-train the layer via transfer learning
        :return: None
        """
        try:
            self._bert_model = hub.KerasLayer(self._model_path, trainable=trainable, name="BERT")

            vocab_file = self._bert_model.resolved_object.vocab_file.asset_path.numpy()
            do_lower_case = self._bert_model.resolved_object.do_lower_case.numpy()
            self._tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

        except Exception as e:
            stacktrace = traceback.format_exc()

            logger.error("{}".format(stacktrace))
            raise e

    def _encode_sentence(self, s):
        """
        Encodes the sentence as required by BERT

        :param s: the sentence
        :return: the tokenized sentence
        """
        # Tokenizing the sentence
        tokens = list(self._tokenizer.tokenize(s))

        # Adding `[SEP]` at the end of the sentence as required by BERT
        tokens.append('[SEP]')

        # Converting tokens to ids
        tokens = self._tokenizer.convert_tokens_to_ids(tokens)

        # Padding the vector of ids with zeros up to a length=max_sequence_length - 1, where -1 because `[CLS]` will be
        # added later
        return tokens + [0] * (self._max_sequence_length - len(tokens) - 1)

    def get_feature_from_ids(self, data, num_classes):
        """
        Returns the features given the ids of the keywords
        :param data: the vector of ids
        :param num_classes: number of keywords
        :return:
        """

        feature = np.zeros(shape=(len(data), num_classes), dtype=np.bool)
        for idx, keyword_id in enumerate(data):
            if keyword_id is not None:
                feature[idx, keyword_id] = True

        return tf.constant(feature)

    def encode(self, data):
        """
        Encodes the vector of sentences as requred by BERT: ['CLS'] + token + ['SEP'], then converts the padded token
        vector to ids and computes the input mask and the input type id.

        :param data: the dataset
        :return:
        """
        encoded_data = tf.ragged.constant([self._encode_sentence(s) for s in np.array(data)])

        cls = [self._tokenizer.convert_tokens_to_ids(['[CLS]'])] * encoded_data.shape[0]
        input_word_ids = tf.concat([cls, encoded_data], axis=-1)

        input_mask = tf.ones_like(input_word_ids).to_tensor()
        input_word_ids = input_word_ids.to_tensor()

        type_cls = tf.ones_like(cls)
        type_data = tf.ones_like(encoded_data)
        input_type_ids = tf.concat([type_cls, type_data], axis=-1).to_tensor()

        inputs = {
            'input_word_ids': input_word_ids,
            'input_mask': input_mask,
            'input_type_ids': input_type_ids}

        return inputs

    def get_bert_layer(self):
        """
        Returns the bert layer

        :return: object
        """
        return self._bert_model

    def build_custom_model(self, num_keywords, output_classes):
        """
        Builds the custom model

        :param num_keywords: number of keywords
        :param output_classes: number of output classes
        :return: None
        """
        # Model definition
        input_word_ids = tf.keras.layers.Input(shape=(self._max_sequence_length,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(self._max_sequence_length,), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(self._max_sequence_length,), dtype=tf.int32, name="segment_ids")
        keywords_ids = tf.keras.layers.Input(shape=(num_keywords, ), name='keywords_ids')

        pooled_output, sequence_output = self._bert_model([input_word_ids, input_mask, segment_ids])
        bert_output = tf.keras.layers.Lambda(lambda x: x[:, 0, :], name='extract_CLS', output_shape=(None, 768))(
            sequence_output)  # extract representation of [CLS] token
        # bert_output = tf.keras.layers.Flatten(name="bert_pooled_output_flatten")(pooled_output)

        hidden = tf.concat([bert_output, keywords_ids], -1)
        hidden = tf.keras.layers.Dense(256, activation=tf.nn.relu, name='dense_1')(hidden)
        hidden = tf.keras.layers.BatchNormalization(name='batch_norm_1')(hidden)
        hidden = tf.keras.layers.Dropout(0.5)(hidden)
        hidden = tf.keras.layers.Dense(256, activation=tf.nn.relu, name='dense_2')(hidden)
        hidden = tf.keras.layers.BatchNormalization(name='batch_norm_2')(hidden)
        hidden = tf.keras.layers.Dropout(0.5)(hidden)
        output = tf.keras.layers.Dense(output_classes, activation=tf.nn.softmax, name='output')(hidden)

        self._model = tf.keras.Model(
            inputs=[input_word_ids, input_mask, segment_ids, keywords_ids],
            outputs=output,
            name=self.model_name
        )
        self._model.summary()
        logger.info("Model defined")

        self._model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.optimizers.Adam(lr=0.00001),
            metrics=['accuracy'])
        logger.info("Model compiled")

        if self._plot_path is not None:
            tf.keras.utils.plot_model(
                self._model, to_file=self._plot_path, show_shapes=True, show_layer_names=True,
                rankdir='TB', expand_nested=True, dpi=96
            )

    def train(self, X_train, y_train, X_val, y_val, epochs=200, load_checkpoint=False):
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
        if self._checkpoint_path is not None:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self._checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1,
                                                             save_freq=50)
            callbacks = [cp_callback]

        if self._checkpoint_path is not None and load_checkpoint:
            self._model.load_weights(self._checkpoint_path)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            verbose=1,
            mode='auto',
            patience=100
        )
        callbacks.append(early_stopping)

        self._model.fit(
            X_train,
            y_train,
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=1,
            callbacks=callbacks
        )

    def to_categorical_tensor(self, data, num_classes):
        """
        Converts the data to a categorical tensor

        :param data:
        :param num_classes: the number of classes
        :return:
        """
        return tf.keras.utils.to_categorical(y=data, num_classes=num_classes)