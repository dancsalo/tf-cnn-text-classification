import tensorflow as tf
from tensorflow.contrib import layers


class OneLayerConv:
    def __init__(self, flags, Data):
        self.flags = flags
        self.vocab_size = Data.vocab_size
        self.seq_length = Data.seq_length
        self.input_x, self.input_y, self.dropout_keep_prob = self.inputs()
        self.scores = self.network()
        self.cost, self.predictions, self.accuracy = self.losses()
        self.train_outputs = list()

    def inputs(self):
        input_x = tf.placeholder(tf.int32, [None, self.seq_length], name="input_x")
        input_y = tf.placeholder(tf.float32, [None, self.flags['DATA']['NUM_CLASSES']], name="input_y")
        dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        return input_x, input_y, dropout_keep_prob

    def network(self):
        weights_regularizer = tf.contrib.layers.l2_regularizer(self.flags['TRAIN']['WEIGHT_DECAY'])

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([self.vocab_size, self.flags['NETWORK']['EMBED_SIZE']], -1.0, 1.0),
                name="W")
            embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = list()
        for i, filter_size in enumerate(self.flags['NETWORK']['FILTER_SIZES']):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                conv = layers.conv2d(inputs=embedded_chars_expanded, num_outputs=self.flags['NETWORK']['NUM_FILTERS'],
                                     kernel_size=[filter_size, self.flags['NETWORK']['EMBED_SIZE']], padding='VALID',
                                     weights_regularizer=weights_regularizer)
                pooled = tf.nn.max_pool(conv, ksize=[1, self.seq_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.flags['NETWORK']['NUM_FILTERS'] * len(self.flags['NETWORK']['FILTER_SIZES'])
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) mean and stddev
        with tf.name_scope("scores"):
            scores = layers.fully_connected(h_drop, num_outputs=self.flags['DATA']['NUM_CLASSES'], activation_fn=None,
                                            weights_regularizer=weights_regularizer)
        return scores

    def losses(self):
        predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y))

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        return cost, predictions, accuracy

    def feed_dict(self, batch, mode):
        if mode == 'train':
            x_batch, y_batch = zip(*batch)
            return {self.input_x: x_batch, self.input_y: y_batch, self.dropout_keep_prob: self.flags['TRAIN']['KEEP_PROB']}
        else:  # mode == 'test
            x_batch, y_batch = zip(*batch)
            return {self.input_x: x_batch, self.input_y: y_batch, self.dropout_keep_prob: 1.0}

    def outputs(self, list_to_add, mode):
        if mode == 'train':
            list_to_add.append(self.cost)
            self.train_outputs.append('Total Cost')
            list_to_add.append(self.accuracy)
            self.train_outputs.append('Accuracy')
            return list_to_add
        else:  # mode == 'test
            list_to_add.append(self.accuracy)
            self.train_outputs.append('Accuracy')
            return list_to_add

    def record_metrics(self, outputs, mode):
        """ Record training metrics """
        if mode == 'train':
            output_string = ""
            for string, value in zip(self.train_outputs, outputs[2:]):
                output_string += string + " = {:.6f}  ".format(value)
            return output_string

    def summaries(self):
        tf.summary.scalar("Total_Loss", self.cost)
        tf.summary.scalar("Accuracy", self.accuracy)
