import tensorflow as tf

class SentenceCNN:
    def __init__(self, sess, num_classes, learning_rate, batch_size, filter_sizes, num_filters, word_embeddings, seq_len):
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.word_embeddings = word_embeddings
        self.seq_len = seq_len
        self.embedding_size = word_embeddings.shape[1]
        self.kernel_initializer = tf.truncated_normal_initializer(stddev=1e-3)
        self.bias_initializer = tf.zeros_initializer()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.sess = sess

        self.__form_variables()
        self.__form_graph()
        self.sess.run(tf.global_variables_initializer())

    def __form_variables(self):
        self.input_sentence = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.seq_len], name='input_sentence')
        self.labels = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name='labels')
        self.global_step = tf.Variable(0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.word_embeddings_static = tf.Variable(self.word_embeddings, name='static_embedding', trainable=False)
        self.word_embeddings_dynamic = tf.Variable(self.word_embeddings, name='dynamic_embedding', trainable=True)

    def __form_graph(self):
        input_embedded_static = tf.nn.embedding_lookup(self.word_embeddings_static, self.input_sentence)
        input_embedded_dynamic = tf.nn.embedding_lookup(self.word_embeddings_static, self.input_sentence)

        input_embedded = tf.stack([input_embedded_static, input_embedded_dynamic], axis=3)

        conv_outputs = []

        for filter_size in self.filter_sizes:
            with tf.variable_scope('layer_{}'.format(filter_size)):
                conv = tf.layers.conv2d(input_embedded, self.num_filters, (filter_size, self.embedding_size),
                                        activation=tf.nn.relu, kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
                                        name='conv_layer')

                pooled = tf.layers.max_pooling2d(conv, (self.seq_len-filter_size+1,1), 1, name='max_pooling_layer')
                flattened = tf.reshape(pooled, [self.batch_size, self.num_filters])
                conv_outputs.append(flattened)

        representation = tf.concat(conv_outputs, 1, name='features')

        with tf.variable_scope('dense'):
            output = tf.layers.dense(representation, self.num_classes, kernel_initializer=self.kernel_initializer,
                                     bias_initializer=self.bias_initializer)
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.labels, self.num_classes, 1.0, 0.0),
                                                                          logits=output), name='cross_entropy')

        with tf.variable_scope('optimization'):
            train_op = self.optimizer.minimize(loss, global_step=self.global_step)

        predictions = tf.argmax(output, 1)
        accuracy = tf.metrics.accuracy(self.labels, predictions)

        return loss, train_op, accuracy

import pickle
embeddings = pickle.load(open('/home/ceteke/Desktop/langpy/glove-embedding-text8.pkl', 'rb'))
sess = tf.Session()
cnn_test = SentenceCNN(sess, 2, 0.001, 32, [3,5], 100, embeddings, 50)