
from unittest import TestCase
from tf_2.finegrained.wsdan.losses import WSDANLoss

import tensorflow as tf


class TestAttentionLoss(TestCase):
    # official code
    @staticmethod
    def calculate_pooling_center_loss(features, label, alfa, nrof_classes, weights, name, centers=None):
        features = tf.reshape(features, [features.shape[0], -1])
        label = tf.argmax(label, 1)

        nrof_features = features.get_shape()[1]
        if centers is None:
            centers = tf.compat.v1.get_variable(name, [nrof_classes, nrof_features], dtype=tf.float32,
                                                initializer=tf.constant_initializer(0), trainable=False)
        label = tf.reshape(label, [-1])
        centers_batch = tf.gather(centers, label)
        centers_batch = tf.nn.l2_normalize(centers_batch, axis=-1)

        diff = (1 - alfa) * (centers_batch - features)
        centers = tf.tensor_scatter_nd_sub(centers, tf.expand_dims(label, axis=-1), diff)

        with tf.control_dependencies([centers]):
            distance = tf.square(features - centers_batch)
            distance = tf.reduce_sum(distance, axis=-1)
            center_loss = tf.reduce_mean(distance)

        center_loss = tf.identity(center_loss * weights, name=name + '_loss')

        return center_loss, centers

    def test_compare_attention_loss_with_official_code(self):
        labels = tf.constant([1, 2, 3, 4, 5])
        one_hot_labels = tf.one_hot(labels-1, depth=5)

        features = tf.random.normal(shape=(5, 1, 1, 100))

        alfa = 0.95
        nrof_classes = 5
        weights = 1
        name = ''

        # official code test
        _, centers = self.calculate_pooling_center_loss(features=features,
                                                        label=one_hot_labels,
                                                        alfa=alfa,
                                                        nrof_classes=nrof_classes,
                                                        weights=weights,
                                                        name=name,
                                                        centers=None)
        official_loss, _ = self.calculate_pooling_center_loss(features=features,
                                                              label=one_hot_labels,
                                                              alfa=alfa,
                                                              nrof_classes=nrof_classes,
                                                              weights=weights,
                                                              name=name,
                                                              centers=centers)

        print(official_loss)

        # private code test
        wsdan_loss = WSDANLoss(num_class=nrof_classes)
        _ = wsdan_loss.calculate_attention_loss(labels=labels,
                                                embeddings=features,
                                                beta=0.05)

        private_loss = wsdan_loss.calculate_attention_loss(labels=labels,
                                                           embeddings=features,
                                                           beta=0.05)

        print(private_loss)

        self.assertEqual(official_loss, private_loss)






