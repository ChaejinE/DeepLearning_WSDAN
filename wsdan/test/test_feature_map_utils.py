import unittest
import numpy as np
import tensorflow as tf
from tf_2.finegrained.wsdan.utils.feature_map_utils import feature_map_random_sampling, point_wise_average_pooling


class TestFeatureMapUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 3
        self.height = 17
        self.width = 17
        self.channel = 10
        self.feature_maps = tf.random.uniform(shape=(self.batch_size, self.height, self.width, self.channel))

    def test_feature_map_random_sampling(self):
        feature_map = feature_map_random_sampling(self.feature_maps)
        self.assertEqual((self.batch_size, self.height, self.width, 1), np.shape(feature_map))

    def test_point_wise_average_pooling(self):
        feature_map = point_wise_average_pooling(self.feature_maps)
        self.assertTrue(np.equal((self.batch_size, self.height, self.width, 1), np.shape(feature_map)).all)

