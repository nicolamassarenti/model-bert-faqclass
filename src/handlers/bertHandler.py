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
        TODO: aggiornare guida e mettere log
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
        tokens = list(self._tokenizer.tokenize(s))
        tokens.append('[SEP]')
        tokens = self._tokenizer.convert_tokens_to_ids(tokens)
        # TODO: spiegare che è padding e -1 per il [CLS] che metto dopo
        return tokens + [0] * (self._max_sequence_length - len(tokens) - 1)

    def get_feature_from_ids(self, data, num_classes):
        """TODO descrizione"""

        feature = np.zeros(shape=(len(data), num_classes), dtype=np.bool)
        for idx, id in enumerate(data):
            if id is not None:
                feature[idx, id] = True

        return tf.constant(feature)

    def encode(self, data):
        """
        TODO: descrizione
        :param data:
        :return:
        """
        encoded_data = tf.ragged.constant([self._encode_sentence(s) for s in np.array(data)])

        cls = [self._tokenizer.convert_tokens_to_ids(['[CLS]'])] * encoded_data.shape[0]
        input_word_ids = tf.concat([cls, encoded_data], axis=-1)

        input_mask = tf.ones_like(input_word_ids).to_tensor()
        input_word_ids = input_word_ids.to_tensor()

        type_cls = tf.zeros_like(cls)
        type_data = tf.zeros_like(encoded_data)
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
        TODO commenti e logs
        :return:
        """
        # Model definition
        input_word_ids = tf.keras.layers.Input(shape=(self._max_sequence_length,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(self._max_sequence_length,), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(self._max_sequence_length,), dtype=tf.int32, name="segment_ids")
        keywords_ids = tf.keras.layers.Input(shape=(num_keywords, ), name='keywords_ids')

        pooled_output, sequence_output = self._bert_model([input_word_ids, input_mask, segment_ids])
        bert_output = tf.keras.layers.Flatten(name="bert_pooled_output_flatten")(pooled_output)

        hidden = tf.concat([bert_output, keywords_ids], -1)
        hidden = tf.keras.layers.Dense(100, activation=tf.nn.relu, name='dense_1')(hidden)
        hidden = tf.keras.layers.BatchNormalization(name='batch_norm_1')(hidden)
        hidden = tf.keras.layers.Dropout(0.5)(hidden)
        hidden = tf.keras.layers.Dense(100, activation=tf.nn.relu, name='dense_2')(hidden)
        hidden = tf.keras.layers.BatchNormalization(name='batch_norm_2')(hidden)
        hidden = tf.keras.layers.Dropout(0.5)(hidden)
        hidden = tf.keras.layers.Dense(50, activation=tf.nn.relu, name='dense_3')(hidden)
        hidden = tf.keras.layers.BatchNormalization(name='batch_norm_3')(hidden)
        hidden = tf.keras.layers.Dropout(0.5)(hidden)
        output = tf.keras.layers.Dense(output_classes, activation=tf.nn.softmax, name='output')(hidden)

        self._model = tf.keras.Model(
            inputs=[input_word_ids, input_mask, segment_ids, keywords_ids],
            outputs=output,
            name=self.model_name
        )
        self._model.summary()
        self._model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.optimizers.Adam(lr=0.00001),
            metrics=['accuracy'])

        if self._plot_path is not None:
            tf.keras.utils.plot_model(
                self._model, to_file=self._plot_path, show_shapes=True, show_layer_names=True,
                rankdir='TB', expand_nested=True, dpi=96
            )

    def train(self, X_train, y_train, X_val, y_val, epochs=200, load_checkpoint=False):
        """
        TODO: descrizione
        :return:
        """
        callbacks = []
        if self._checkpoint_path is not None:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self._checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1,
                                                             save_freq=10)
            callbacks = [cp_callback]

        if self._checkpoint_path is not None and load_checkpoint:
            self._model.load_weights(self._checkpoint_path)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            verbose=1,
            mode='auto',
            patience=10
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
        TODO: commenti
        :param data:
        :param num_classes:
        :return:
        """
        return tf.keras.utils.to_categorical(y=data, num_classes=num_classes)