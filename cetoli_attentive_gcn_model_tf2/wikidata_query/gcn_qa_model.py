import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # INFO messages are not printed

import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import StackedRNNCells
from tensorflow.keras.layers import GRUCell

tf.compat.v1.disable_eager_execution()

TINY = 1e-6
ONE = tf.constant(1.)
NAMESPACE = 'gcn_qa'
forbidden_weight = 1.
_weight_for_positive_matches = 1.
_rw = 1e-1


def compute_new_adjacency_matrix(embedding_size, attention_size, memory_dim, A, H, question_vector, name):
    Wa = tf.Variable(tf.random.uniform([embedding_size + memory_dim, attention_size], -_rw, _rw),
                     name='Wa_' + name)
    ba = tf.Variable(tf.random.uniform([attention_size], -_rw, _rw), name='b0_fw' + name)
    WHQ_projection = lambda x: tf.nn.relu(tf.matmul(tf.concat([x, question_vector], axis=1), Wa) + ba)
    WHQ = tf.map_fn(WHQ_projection, H)
    WHQ = tf.transpose(a=WHQ, perm=[1, 0, 2])
    WHQ_squared_projection = lambda x: tf.nn.softmax(tf.matmul(x, tf.transpose(a=x, perm=[1, 0])))
    WHQ_squared = tf.map_fn(WHQ_squared_projection, WHQ)
    new_A = tf.multiply(A, WHQ_squared)
    return new_A

def GCN_layer_fw(embedding_size, hidden_layer1_size, memory_dim, hidden, Atilde_fw, question_vector, name):
    new_A = compute_new_adjacency_matrix(embedding_size, 250, memory_dim, Atilde_fw, hidden, question_vector, name)
    W0_fw = tf.Variable(tf.random.uniform([embedding_size, hidden_layer1_size], -_rw, _rw),
                        name='W0_fw' + name)
    b0_fw = tf.Variable(tf.random.uniform([hidden_layer1_size], -_rw, _rw), name='b0_fw' + name)
    left_X1_projection_fw = lambda x: tf.matmul(x, W0_fw) + b0_fw
    left_X1_fw = tf.map_fn(left_X1_projection_fw, hidden)
    left_X1_fw = tf.transpose(a=left_X1_fw, perm=[1, 0, 2], name='left_X1_fw' + name)
    X1_fw = tf.nn.relu(tf.matmul(new_A, left_X1_fw))
    X1_fw = tf.transpose(a=X1_fw, perm=[1, 0, 2])
    return X1_fw


class GCN_QA(object):
    _nodes_vocab_size = 300
    _question_vocab_size = 300
    _question_vector_size = 150
    _types_size = 3
    _mask_size = 200
    _types_proj_size = 5
    _word_proj_size = 50
    _word_proj_size_for_rnn = 50
    _word_proj_size_for_item = 50
    _internal_proj_size = 250
    _hidden_layer1_size = 250
    _hidden_layer2_size = 250
    _output_size = 2

    _memory_dim = 100
    _stack_dimension = 2

    def __init__(self, dropout=1.0):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.variable_scope(NAMESPACE):
            config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(config=config)

            # Input variables
            self.node_X = tf.compat.v1.placeholder(tf.float32, shape=(None, None, self._nodes_vocab_size), name='node_X')
            self.types = tf.compat.v1.placeholder(tf.float32, shape=(None, None, self._types_size), name='types')
            self.Wt = tf.Variable(tf.random.uniform([self._types_size,
                                                     self._types_proj_size], -_rw, _rw))
            self.bt = tf.Variable(tf.random.uniform([self._types_proj_size], -_rw, _rw))
            self.types_projection = lambda x: tf.nn.relu(tf.matmul(x, self.Wt) + self.bt)
            self.types_internal = tf.map_fn(self.types_projection, self.types)
            self.question_vectors_fw = tf.compat.v1.placeholder(tf.float32, shape=(None, None, self._question_vocab_size),
                                                      name='question_vectors_inp_fw')
            self.question_vectors_bw = tf.compat.v1.placeholder(tf.float32, shape=(None, None, self._question_vocab_size),
                                                      name='question_vectors_inp_nw')
            self.question_mask = tf.compat.v1.placeholder(tf.float32, shape=(None, None, self._mask_size),
                                                name='question_mask')

            # The question is pre-processed by a bi-GRU
            self.Wq = tf.Variable(tf.random.uniform([self._question_vocab_size,
                                                     self._word_proj_size_for_rnn], -_rw, _rw))
            self.bq = tf.Variable(tf.random.uniform([self._word_proj_size_for_rnn], -_rw, _rw))
            self.internal_projection = lambda x: tf.nn.relu(tf.matmul(x, self.Wq) + self.bq)
            self.question_int_fw = tf.map_fn(self.internal_projection, self.question_vectors_fw)
            self.question_int_bw = tf.map_fn(self.internal_projection, self.question_vectors_bw)

            self.rnn_cell_fw = StackedRNNCells([GRUCell(self._memory_dim) for _ in range(self._stack_dimension)])
            self.rnn_cell_bw = StackedRNNCells([GRUCell(self._memory_dim) for _ in range(self._stack_dimension)])
            with tf.compat.v1.variable_scope('fw'):
                output_fw, state_fw = tf.compat.v1.nn.dynamic_rnn(self.rnn_cell_fw, self.question_int_fw, time_major=True,
                                                        dtype=tf.float32)
            with tf.compat.v1.variable_scope('bw'):
                output_bw, state_bw = tf.compat.v1.nn.dynamic_rnn(self.rnn_cell_bw, self.question_int_bw, time_major=True,
                                                        dtype=tf.float32)

            self.states = tf.concat(values=[output_fw, tf.reverse(output_bw, [0])], axis=2)
            self.question_vector_pre = tf.reduce_mean(input_tensor=tf.multiply(self.question_mask, self.states), axis=0)
            self.Wqa = tf.Variable(
                tf.random.uniform([2 * self._memory_dim, self._question_vector_size], -_rw, _rw),
                name='Wqa')
            self.bqa = tf.Variable(tf.random.uniform([self._question_vector_size], -_rw, _rw), name='bqa')
            self.question_vector = tf.nn.relu(tf.matmul(self.question_vector_pre, self.Wqa) + self.bqa)

            # Dense layer before gcn
            self.Wi = tf.Variable(tf.random.uniform([self._nodes_vocab_size,
                                                     self._word_proj_size], -_rw, _rw))
            self.bi = tf.Variable(tf.random.uniform([self._word_proj_size], -_rw, _rw))
            self.internal_projection2 = lambda x: tf.nn.relu(tf.matmul(x, self.Wi) + self.bi)
            self.word_embeddings = tf.map_fn(self.internal_projection2, self.node_X)

            self.inputs = tf.concat(values=[self.word_embeddings, self.types_internal], axis=2)
            self.Wp = tf.Variable(tf.random.uniform([self._word_proj_size + self._types_proj_size,
                                                     self._internal_proj_size], -_rw, _rw))
            self.bp = tf.Variable(tf.random.uniform([self._internal_proj_size], -_rw, _rw))
            self.enc_int_projection = lambda x: tf.nn.relu(tf.matmul(x, self.Wp) + self.bp)
            self.enc_int = tf.map_fn(self.enc_int_projection, self.inputs)

            # GCN part
            self.Atilde_fw = tf.nn.dropout(tf.compat.v1.placeholder(tf.float32, shape=(None, None, None), name="Atilde_fw"), 1 - (1.))

            self.X1_fw = GCN_layer_fw(self._internal_proj_size,
                                      self._hidden_layer1_size,
                                      self._question_vector_size,
                                      self.enc_int,
                                      self.Atilde_fw,
                                      self.question_vector,
                                      '_1')
            self.X1_fw_dropout = tf.nn.dropout(self.X1_fw, 1 - (dropout))

            self.X2_fw = GCN_layer_fw(self._hidden_layer1_size,
                                      self._hidden_layer1_size,
                                      self._question_vector_size,
                                      self.X1_fw_dropout,
                                      self.Atilde_fw,
                                      self.question_vector,
                                      '_2')
            self.X2_fw_dropout = tf.nn.dropout(self.X2_fw, 1 - (dropout))

            self.X3_fw = GCN_layer_fw(self._hidden_layer1_size,
                                      self._hidden_layer1_size,
                                      self._question_vector_size,
                                      self.X2_fw_dropout,
                                      self.Atilde_fw,
                                      self.question_vector,
                                      '_3')
            self.X3_fw_dropout = tf.nn.dropout(self.X3_fw, 1 - (dropout))

            self.X4_fw = GCN_layer_fw(self._hidden_layer1_size,
                                      self._hidden_layer1_size,
                                      self._question_vector_size,
                                      self.X3_fw_dropout,
                                      self.Atilde_fw,
                                      self.question_vector,
                                      '_4')
            self.X4_fw_dropout = tf.nn.dropout(self.X4_fw, 1 - (dropout))
            self.first_node = self.X4_fw_dropout[0]
            self.concatenated = tf.concat(values=[self.question_vector, self.first_node], axis=1)

            # Final feedforward layers
            self.Ws1 = tf.Variable(
                tf.random.uniform([self._question_vector_size
                                   + self._hidden_layer1_size,
                                   self._hidden_layer2_size], -_rw, _rw),
                name='Ws1')
            self.bs1 = tf.Variable(tf.random.uniform([self._hidden_layer2_size], -_rw, _rw), name='bs1')
            self.first_hidden = tf.nn.relu(tf.matmul(self.concatenated, self.Ws1) + self.bs1)
            self.first_hidden_dropout = tf.nn.dropout(self.first_hidden, 1 - (dropout))

            self.Wf = tf.Variable(
                tf.random.uniform([self._hidden_layer2_size, self._output_size], -_rw,
                                  _rw),
                name='Wf')
            self.bf = tf.Variable(tf.random.uniform([self._output_size], -_rw, _rw), name='bf')
            self.outputs = tf.nn.softmax(tf.matmul(self.first_hidden_dropout, self.Wf) + self.bf)

            # Loss function and training
            self.y_ = tf.compat.v1.placeholder(tf.float32, shape=(None, self._output_size), name='y_')
            self.outputs2 = tf.squeeze(self.outputs)
            self.y2_ = tf.squeeze(self.y_)
            self.one = tf.ones_like(self.outputs)
            self.tiny = self.one * TINY
            self.cross_entropy = (tf.reduce_mean(
                input_tensor=-tf.reduce_sum(input_tensor=self.y_ * tf.math.log(self.outputs + self.tiny) * _weight_for_positive_matches
                               + (self.one - self.y_) * tf.math.log(
                    self.one - self.outputs + self.tiny))
            ))

        # Clipping the gradient
        optimizer = tf.compat.v1.train.AdamOptimizer(1e-4)
        gvs = optimizer.compute_gradients(self.cross_entropy)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if var.name.find(NAMESPACE) != -1]
        self.train_step = optimizer.apply_gradients(capped_gvs)
        self.sess.run(tf.compat.v1.global_variables_initializer())

        # Adding the summaries
        tf.compat.v1.summary.scalar('cross_entropy', self.cross_entropy)
        self.merged = tf.compat.v1.summary.merge_all()
        self.train_writer = tf.compat.v1.summary.FileWriter('./train', self.sess.graph)

    def _add_identity(self, A):
        num_nodes = A.shape[0]
        identity = np.identity(num_nodes)
        return identity + A

    def __train(self, A_fw, node_X, types, item_vector, question_vectors, question_mask, y):
        item_vector = np.array(item_vector)
        Atilde_fw = np.array([self._add_identity(item) for item in A_fw])

        node_X = np.array(node_X)
        node_X = np.transpose(node_X, (1, 0, 2))

        types = np.array(types)
        types = np.transpose(types, (1, 0, 2))

        question_vectors = np.array(question_vectors)
        question_vectors_fw = np.transpose(question_vectors, (1, 0, 2))
        question_vectors_bw = question_vectors_fw[::-1, :, :]

        question_mask = np.array(question_mask)
        question_mask = np.transpose(question_mask, (1, 0, 2))

        y = np.array(y)

        feed_dict = {}
        feed_dict.update({self.node_X: node_X})
        feed_dict.update({self.types: types})
        feed_dict.update({self.question_vectors_fw: question_vectors_fw})
        feed_dict.update({self.question_vectors_bw: question_vectors_bw})
        feed_dict.update({self.question_mask: question_mask})
        feed_dict.update({self.Atilde_fw: Atilde_fw})
        feed_dict.update({self.y_: y})

        loss, _, summary, outputs2, y2 = self.sess.run(
            [self.cross_entropy, self.train_step, self.merged, self.outputs2, self.y2_], feed_dict)
        return loss, summary

    def train(self, data, epochs=20):
        for epoch in range(epochs):
            loss, _ = self.__train([data[i][0] for i in range(len(data))],
                                   [data[i][1] for i in range(len(data))],
                                   [data[i][2] for i in range(len(data))],
                                   [data[i][3] for i in range(len(data))],
                                   [data[i][4] for i in range(len(data))],
                                   [data[i][5] for i in range(len(data))],
                                   [data[i][6] for i in range(len(data))])
            return loss

    def __predict(self, A_fw, node_X, types, item_vector, question_vectors, question_mask):
        item_vector = np.array(item_vector)
        Atilde_fw = np.array([self._add_identity(item) for item in A_fw])

        node_X = np.array(node_X)
        node_X = np.transpose(node_X, (1, 0, 2))

        types = np.array(types)
        types = np.transpose(types, (1, 0, 2))

        question_vectors = np.array(question_vectors)
        question_vectors_fw = np.transpose(question_vectors, (1, 0, 2))
        question_vectors_bw = question_vectors_fw[::-1, :, :]

        question_mask = np.array(question_mask)
        question_mask = np.transpose(question_mask, (1, 0, 2))

        feed_dict = {}
        feed_dict.update({self.node_X: node_X})
        feed_dict.update({self.types: types})
        feed_dict.update({self.question_vectors_fw: question_vectors_fw})
        feed_dict.update({self.question_vectors_bw: question_vectors_bw})
        feed_dict.update({self.question_mask: question_mask})
        feed_dict.update({self.Atilde_fw: Atilde_fw})

        y_batch = self.sess.run([self.outputs2], feed_dict)
        return y_batch

    def __standardize_item(self, item):
        if item[0] < item[1]:
            return [0., 1.]
        return [1., 0.]

    def predict(self, A_fw, node_X, types, item_vector, question_vectors, question_mask):
        output = self.__predict([A_fw], [node_X], [types], [item_vector], [question_vectors], [question_mask])
        return self.__standardize_item(output[0])


    # Loading and saving functions

    def save(self, filename):
        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, filename)

    def load_tensorflow(self, filename):
        saver = tf.compat.v1.train.Saver([v for v in tf.compat.v1.global_variables() if NAMESPACE in v.name])
        saver.restore(self.sess, filename)

    @classmethod
    def load(self, filename, dropout=1.0):
        model = GCN_QA(dropout)
        model.load_tensorflow(filename)
        return model
