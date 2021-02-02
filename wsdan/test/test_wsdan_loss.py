import unittest
from tf_2.finegrained.wsdan.losses import WSDANLoss
import tensorflow as tf


class TestWSDANLoss(unittest.TestCase):
    def test_calculate_classification_loss(self):
        y_pred = tf.constant([0, 0, 0])
        y_pred = tf.one_hot(y_pred, depth=3) * 100

        labels = [1, 1, 1]
        loss = WSDANLoss(3).calculate_classification_loss(labels, y_pred)
        self.assertGreater(0.001, loss)

        labels = [2, 2, 2]
        loss = WSDANLoss(3).calculate_classification_loss(labels, y_pred)
        self.assertLess(10, loss)

        labels = [0, 1, 2]
        self.assertIn(0, labels)
        # assert error test
        # loss = WSDANLoss().calculate_classification_loss(labels, y_pred)

    def test_calculate_attention_loss(self):
        labels = [2, 2, 2]
        embeddigns = tf.random.uniform(shape=(3, 1, 1, 32), minval=-100, maxval=100)
        loss = WSDANLoss(3).calculate_attention_loss(labels, embeddigns)
        self.assertEqual(tf.float32, type(loss.numpy()))

    def test_calculate_total_loss(self):
        loss_fn = WSDANLoss(num_class=5)

        y_true = [2, 2, 2, 1, 3]
        origin_pred = tf.one_hot(tf.zeros(5, dtype=tf.int32), depth=5)
        crop_pred = tf.one_hot(tf.zeros(5, dtype=tf.int32), depth=5)
        drop_pred = tf.one_hot(tf.zeros(5, dtype=tf.int32), depth=5)
        feature_matrix = tf.random.uniform(shape=(5, 1, 1, 10))
        beta = 0.95

        total_loss, classification_loss, attention_loss = loss_fn.calculate_total_loss(y_true,
                                                                                       origin_pred,
                                                                                       crop_pred,
                                                                                       drop_pred,
                                                                                       feature_matrix,
                                                                                       beta)

        self.assertEqual(tf.float32, type(total_loss.numpy()))
        self.assertEqual(tf.float32, type(classification_loss.numpy()))
        self.assertEqual(tf.float32, type(attention_loss.numpy()))
