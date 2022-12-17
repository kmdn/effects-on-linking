import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # INFO messages are not printed

import random
import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, GRU, LSTM, Bidirectional, Activation, Dropout, Concatenate, BatchNormalization, Masking
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, DistilBertConfig, TFDistilBertModel

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)


class GCN_QA(object):
    _max_text_length = 512
    _nodes_vocab_size = 300 * 3
    _question_vocab_size = 300
    _nodes_vector_size = 150
    _question_vector_size = 150
    _bert_embedding_size = 768
    _hidden_layer1_size = 250
    _hidden_layer2_size = 250
    _output_size = 2

    _distil_bert = 'distilbert-base-uncased'
    _memory_dim = 100
    _mask_value = -10.0

    def __init__(self, config, dropout=1.0):
        tf.compat.v1.reset_default_graph()
        self._config = config

        # Logging
        self._logger = logging.getLogger(__name__) # own logger
        self._tf_logger = tf.get_logger() # TensorFlow logger
        self._tf_logger.handlers = [] # remove the original handler from the TensorFlow logger
        logging.basicConfig(level=self._config.log_level, format=self._config.log_format,
                handlers=[logging.FileHandler(self._config.log_path), logging.StreamHandler()])

        # Tokenizer
        self._tokenizer = DistilBertTokenizer.from_pretrained(self._distil_bert, do_lower_case=True,
                add_special_tokens=True, max_length=self._max_text_length, pad_to_max_length=True)

        # Part I: Question text sequence -> BERT
        config = DistilBertConfig(dropout=0.2, attention_dropout=0.2)
        config.output_hidden_states = False
        transformer_model = TFDistilBertModel.from_pretrained(self._distil_bert, config=config)

        input_question = Input(shape=(self._max_text_length,), dtype='int32', name='question')
        input_attention_mask = Input(shape=(self._max_text_length,), dtype='int32', name='attention_mask')
        input_sf_mask = Input(shape=(self._max_text_length, self._bert_embedding_size), dtype='float32', name='question_mask')

        embedding_layer = transformer_model.distilbert(input_question, attention_mask=input_attention_mask)[0]
        #cls_token = embedding_layer[:,0,:]
        sf_token = tf.math.reduce_mean(tf.math.multiply(embedding_layer, input_sf_mask), axis=1)
        question_outputs = BatchNormalization()(sf_token)
        question_outputs = Dense(self._question_vector_size)(question_outputs)
        question_outputs = Activation('relu')(question_outputs)

        # Part II: Entity graph node (as text) -> Bi-LSTM
        fw_lstm = LSTM(self._memory_dim)
        bw_lstm = LSTM(self._memory_dim, go_backwards=True)

        input_nodes = Input(shape=(None, self._nodes_vocab_size), name='node_vectors')
        masked_nodes = Masking(mask_value=self._mask_value)(input_nodes)
        nodes_outputs = Bidirectional(layer=fw_lstm, backward_layer=bw_lstm)(masked_nodes)
        nodes_outputs = Dense(self._nodes_vector_size)(nodes_outputs)
        nodes_outputs = Activation('relu')(nodes_outputs)

        # Part III: Comparator
        # concatenation size = _nodes_vector_size + _question_vector_size
        concatenated = Concatenate(axis=1)([question_outputs, nodes_outputs])
        mlp_outputs = Dense(self._hidden_layer2_size)(concatenated)
        mlp_outputs = Activation('relu')(mlp_outputs)
        mlp_outputs = Dropout(0.0)(mlp_outputs)
        mlp_outputs = Dense(self._output_size)(mlp_outputs) # 2-dim. output
        mlp_outputs = Activation('softmax')(mlp_outputs)

        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        self._model = tf.keras.models.Model(
                inputs=[input_question, input_attention_mask, input_sf_mask, input_nodes],
                outputs=mlp_outputs)
        self._model.get_layer('distilbert').trainable = False # make BERT layers untrainable
        self._model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        #self._model.summary()

    def __tokenize(self, sentences, tokenizer, max_length):
        input_ids, input_masks = [], []
        for sentence in sentences:
            inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=max_length,
                    padding='max_length', return_attention_mask=True)
            input_ids.append(inputs['input_ids'])
            input_masks.append(inputs['attention_mask'])
        return np.asarray(input_ids, dtype='int32'), np.asarray(input_masks, dtype='int32')

    def __generate_data(self, dataset, batch_size):
        dataset.pop('item_vector')
        dataset.pop('question_vectors')

        # https://stackoverflow.com/questions/46493419/use-a-generator-for-keras-model-fit-generator
        i = 0
        while True:
            # get a batch from the shuffled dataset, preprocess it, and give it to the model
            batch = {
                    'text': [], # required by BERT
                    'question': [],
                    'attention_mask': [],
                    'question_mask': [],
                    'node_vectors': [],
                    'y': []
                }

            # draw a (ordered) batch from the (shuffled) dataset
            for b in range(batch_size):
                if i == len(dataset['text']): # re-shuffle when processed whole dataset
                    i = 0
                    lists = list(zip(dataset['text'], dataset['node_vectors'], dataset['question_mask'], dataset['y']))
                    random.shuffle(lists)
                    dataset['text'], dataset['node_vectors'], dataset['question_mask'], dataset['y'] = zip(*lists)
                # add sample
                batch['text'].append(dataset['text'][i])
                batch['node_vectors'].append(dataset['node_vectors'][i])
                batch['question_mask'].append(dataset['question_mask'][i])
                batch['y'].append(dataset['y'][i])
                i += 1

            # preprocess batch (array, pad, tokenize)
            X = {}
            X['node_vectors'] = tf.keras.preprocessing.sequence.pad_sequences(
                    batch['node_vectors'], value=self._mask_value, padding='post', dtype='float64')
            X['question_mask'] = tf.keras.preprocessing.sequence.pad_sequences(
                    batch['question_mask'], maxlen=self._max_text_length, value=0.0)
            X['question'], X['attention_mask'] = self.__tokenize(
                    batch['text'], self._tokenizer, self._max_text_length)
            y = np.asarray(batch['y'])

            yield X, y

    def __train(self, datasets, saving_dir, epochs=1, batch_size=1):
        saving_path = saving_dir + "/cp-{epoch:04d}.ckpt"
        save_model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=saving_path,
                save_weights_only=False)
        logging_callback = tf.keras.callbacks.LambdaCallback(
                on_epoch_end = lambda epoch, logs: self._logger.info(f'\nEpoch {epoch + 1}: loss: {logs["loss"]} - accuracy: {logs["accuracy"]} - val_loss: {logs["val_loss"]} - val_accuracy: {logs["val_accuracy"]}')
        )

        # train dataset
        dataset_train = datasets[0]
        dataset_length_train = len(dataset_train['text'])
        steps_per_epoch = dataset_length_train // batch_size
        # validation dataset
        dataset_val = datasets[1]
        dataset_length_val = len(dataset_val['text'])
        validation_steps_per_epoch = dataset_length_val // batch_size

        self._logger.info('')
        self._logger.info('=== Training settings ===')
        self._logger.info(f'epochs={epochs}, batch_size={batch_size}, dataset_length_train={dataset_length_train}, dataset_length_val={dataset_length_val}, steps_per_epoch={steps_per_epoch}')

        history = self._model.fit(
                self.__generate_data(dataset_train, batch_size),
                epochs = epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=self.__generate_data(dataset_val, batch_size),
                validation_steps=validation_steps_per_epoch,
                callbacks=[save_model_callback, logging_callback]
        )
        return history

    def train(self, datasets, saving_dir, epochs=20, batch_size=32):
        self.__train(datasets, saving_dir, epochs, batch_size)

    def __predict(self, text, node_X, item_vector, question_vectors, question_mask):
        question, attention_mask = self.__tokenize(text, self._tokenizer, self._max_text_length)
        output = self._model.predict([question, attention_mask, question_mask, node_X])
        return output

    def __standardize_item(self, item):
        if item[0] < item[1]:
            return [0., 1.]
        return [1., 0.]

    def predict(self, text, node_X, item_vector, question_vectors, question_mask):
        # in contrast to train(), no generator method is required, as the dev set is small enough to fit into memory also with padding
        text = [text]
        node_X = np.expand_dims(node_X, axis=0)
        node_X = tf.keras.preprocessing.sequence.pad_sequences(node_X, maxlen=self._max_text_length, value=self._mask_value, padding='post', dtype='float64')
        question_vectors = np.expand_dims(question_vectors, axis=0)
        question_mask = np.expand_dims(question_mask, axis=0)
        question_mask = tf.keras.preprocessing.sequence.pad_sequences(question_mask, maxlen=self._max_text_length, value=0.0)

        output = self.__predict(text, node_X, item_vector, question_vectors, question_mask)
        return self.__standardize_item(output[0])

    # Loading and saving functions

    def save(self, filename):
        saver = self._model.save(filename)

    @classmethod
    def load(self, filename, config, dropout=1.0):
        model = GCN_QA(config, dropout)
        #model._model.load_weights(filename)
        model._model = tf.keras.models.load_model(filename)
        model._config = config
        return model
