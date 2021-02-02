import unittest
import tensorflow as tf
from tf_2.finegrained.wsdan.models.backbone import Backbone
import numpy as np


class TestBackbone(unittest.TestCase):

    def setUp(self) -> None:
        self.images = tf.random.uniform(shape=(32, 299, 299, 3))

    def test_inception_backbone(self):
        model = Backbone.inception_v3(output_layer=['mixed6'])
        feature_maps = model(self.images)

        self.assertEqual((32, 17, 17, 768), np.shape(feature_maps))

        model = Backbone.inception_v3(output_layer=['mixed6', 'mixed7'])
        feature_maps = model(self.images)

        self.assertEqual((32, 17, 17, 768), np.shape(feature_maps[0]))
        self.assertEqual((32, 17, 17, 768), np.shape(feature_maps[1]))

        model = Backbone.inception_v3(output_layer=None)
        feature_maps = model(self.images)

        self.assertEqual((32, 8, 8, 2048), np.shape(feature_maps))


