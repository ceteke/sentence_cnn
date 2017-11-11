import tensorflow as tf

class SentenceCNN:
    def __init__(self, sess, num_classes, learning_rate, batch_size, filter_sizes, num_filters, word_embeddings, seq_len,
                 dropout, padding_id=0):
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.word_embeddings = word_embeddings
        self.seq_len = seq_len
        self.embedding_size = word_embeddings.shape[1]
        self.vocab_size = word_embeddings.shape[0]
        self.kernel_initializer = tf.truncated_normal_initializer(stddev=5e-2)
        self.bias_initializer = tf.constant_initializer(0.1)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.sess = sess
        self.padding_id = padding_id
        self.dropout = dropout

        self.__form_variables()
        self.loss_op, self.train_op, self.train_acc_op, self.train_acc_update_op, self.test_acc_op, self.test_acc_update_op = self.__form_graph()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def __form_variables(self):
        raw_mask_array = [[1.]] * self.padding_id + [[0.]] + [[1.]] * (self.vocab_size - self.padding_id - 1)
        self.input_sentence = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.seq_len], name='input_sentence')
        self.labels = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name='labels')
        self.dropout_ph = tf.placeholder(dtype=tf.float32, shape=[], name='dropout')
        self.global_step = tf.Variable(0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        with tf.device('/cpu:0'):
            self.word_embeddings_static = tf.Variable(self.word_embeddings, name='static_embedding', trainable=False, dtype=tf.float32)
            self.word_embeddings_dynamic = tf.Variable(self.word_embeddings, name='dynamic_embedding', trainable=True, dtype=tf.float32)
            self.mask_padding = tf.get_variable("mask_padding",  initializer=raw_mask_array, trainable=False, dtype=tf.float32)

    def __form_graph(self):
        input_embedded_static = tf.nn.embedding_lookup(self.word_embeddings_static, self.input_sentence)
        input_embedded_dynamic = tf.nn.embedding_lookup(self.word_embeddings_dynamic, self.input_sentence)
        mask_input = tf.nn.embedding_lookup(self.mask_padding, self.input_sentence)

        input_embedded_static = tf.multiply(input_embedded_static, mask_input)
        input_embedded_dynamic = tf.multiply(input_embedded_dynamic, mask_input)

        input_embedded = tf.stack([input_embedded_static, input_embedded_dynamic], axis=3)

        conv_outputs = []

        for filter_size in self.filter_sizes:
            with tf.variable_scope('layer_{}'.format(filter_size)):
                conv = tf.layers.conv2d(input_embedded, self.num_filters, (filter_size, self.embedding_size),
                                        activation=tf.nn.relu, kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
                                        name='conv_layer')

                pooled = tf.layers.max_pooling2d(conv, (self.seq_len-filter_size+1,1), 1, name='max_pooling_layer')
                flattened = tf.squeeze(pooled, [1,2])
                conv_outputs.append(flattened)

        representation = tf.concat(conv_outputs, 1, name='features')
        representation_dropped = tf.nn.dropout(representation, keep_prob=self.dropout_ph, name='dropped_features')

        with tf.variable_scope('dense'):
            output = tf.layers.dense(representation_dropped, self.num_classes, kernel_initializer=self.kernel_initializer,
                                     bias_initializer=self.bias_initializer)
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=output),
                                  name='cross_entropy')

        with tf.variable_scope('optimization'):
            train_op = self.optimizer.minimize(loss, global_step=self.global_step)

        predictions = tf.argmax(output, 1)
        accuracy, acc_update = tf.metrics.accuracy(self.labels, predictions)

        accuracy_2, acc_update_2 = tf.metrics.accuracy(self.labels, predictions)

        return loss, train_op, accuracy, acc_update, accuracy_2, acc_update_2

    def train(self, input_sentence, labels):
        loss, _, acc, _ = self.sess.run([self.loss_op, self.train_op, self.train_acc_op, self.train_acc_update_op], feed_dict={self.input_sentence:input_sentence,
                                                                                                                        self.labels:labels,
                                                                                                                        self.dropout_ph:self.dropout})
        return loss, acc

    def test(self, input_sentence, labels):
        loss, acc, _ = self.sess.run([self.loss_op, self.test_acc_op, self.test_acc_update_op],
                                  feed_dict={self.input_sentence: input_sentence,
                                             self.labels: labels,
                                             self.dropout_ph: 1.0})
        return loss, acc

        #import pickle
        #embeddings = pickle.load(open('/home/ceteke/Desktop/langpy/glove-embedding-text8.pkl', 'rb'))
        #sess = tf.Session()
        #cnn_test = SentenceCNN(sess, 2, 0.001, 32, [3,5], 100, embeddings, 50)