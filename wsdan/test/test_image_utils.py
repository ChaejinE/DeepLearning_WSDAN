from unittest import TestCase
from tf_2.finegrained.wsdan.utils.image_utils import min_max_normalize, visualize, create_heat_map
from tf_2.finegrained.wsdan.utils.image_utils import crop_resize_positive_area, drop_positive_area
from tf_2.finegrained.wsdan.utils.image_utils import attention_crop, attention_drop

import numpy as np
import tensorflow as tf


class TestImageUtils(TestCase):
    def setUp(self) -> None:
        self.padding = tf.constant([[0, 0],
                                    [6, 6],
                                    [6, 6],
                                    [0, 0]])

    def test_min_max_normalize(self):
        feature_map = tf.random.uniform(shape=(32, 17, 17, 1))
        normalized_feature = min_max_normalize(feature_map)
        self.assertEqual((32, 17, 17, 1), np.shape(normalized_feature))

        feature_map = tf.ones(shape=(32, 17, 17, 1)) * 255
        normalized_feature = min_max_normalize(feature_map)
        self.assertTrue(np.array_equal(np.zeros_like(feature_map), normalized_feature))

    def test_crop_resize_positive_area(self):
        images = tf.random.normal(shape=(32, 299, 299, 3))
        score_maps = tf.random.normal(shape=(32, 17, 17, 1))
        crop_images = crop_resize_positive_area(images, score_maps, threshold=0.5)

        self.assertEqual(np.shape(images), np.shape(crop_images))

        images = tf.ones(shape=(3, 10, 10, 3)) * 0.5
        images = tf.pad(images, self.padding, "CONSTANT")
        score_maps = tf.ones(shape=(3, 10, 10, 1))
        score_maps = tf.pad(score_maps, self.padding, "CONSTANT")
        crop_images = crop_resize_positive_area(images, score_maps, 0.3)

        expect_crop_images = tf.ones(shape=(3, 22, 22, 3))*0.5

        self.assertTrue(np.array_equal(expect_crop_images.numpy(), crop_images.numpy()))

    def test_drop_positive_area(self):
        images = tf.random.normal(shape=(32, 299, 299, 3))
        score_maps = tf.random.normal(shape=(32, 17, 17, 1))
        drop_images = drop_positive_area(images, score_maps, threshold=0.5)

        self.assertEqual(np.shape(images), np.shape(drop_images))

        images = tf.ones(shape=(3, 22, 22, 3))

        score_maps = tf.ones(shape=(3, 10, 10, 1))
        score_maps = tf.pad(score_maps, self.padding, "CONSTANT")

        drop_images = drop_positive_area(images, score_maps, threshold=0.5)

        expect_drop_images = tf.cast(tf.pad(tf.ones(shape=(3, 10, 10, 3)), self.padding, "CONSTANT"), tf.bool)
        expect_drop_images = tf.cast(tf.math.logical_not(expect_drop_images), tf.float32)

        self.assertTrue(np.array_equal(expect_drop_images.numpy(), drop_images))

    def test_visualize(self):
        image = tf.random.normal(shape=(30, 30, 3))
        crop_attention = tf.random.normal(shape=(30, 30))
        drop_attention = tf.random.normal(shape=(30, 30, 1))
        crop_image = tf.random.normal(shape=(30, 30, 3))
        drop_image = tf.random.normal(shape=(30, 30, 3))

        concated_heat_map = visualize(image, crop_attention, drop_attention, crop_image, drop_image)

        self.assertEqual((30, 150, 3), np.shape(concated_heat_map))

    def test_create_heat_map(self):
        feature_map = tf.ones(shape=(11, 11, 1))
        for i in range(45):
            feature_map = tf.pad(feature_map, tf.constant([[1, 1], [1, 1], [0, 0]])) + 1.

        image = tf.ones_like(feature_map)
        image = tf.tile(image, [1, 1, 3])
        image = image.numpy()

        # Test 1. create heat map 정상 동작
        heat_map = create_heat_map(image, feature_map)

        b, g, r = heat_map[50, 50, :]
        self.assertTrue(max(b, g) < r)

        b, g, r = heat_map[50, 3, :]
        self.assertTrue(max(g, r) < b)

        # Test 2. Invalid arg
        heat_map = create_heat_map(None, feature_map)
        self.assertEqual(None, heat_map)

        heat_map = create_heat_map(image, None)
        self.assertEqual(None, heat_map)

    def test_attention_crop(self):
        height = 94
        width = 94
        pad = 3
        ch = 10
        padding = tf.constant([[pad, pad], [pad, pad], [0, 0]])
        _attention_map = tf.ones(shape=(height, width, ch))
        attention_map = tf.pad(_attention_map, padding, "CONSTANT")

        # Test 1 : attention crop 정상 동작 테스트
        bbox, mask = attention_crop(attention_map, min_thr=0.4, max_thr=0.6)

        self.assertTrue(np.array_equal(np.array([3., 3., 96., 96.]), bbox * (attention_map.shape[0]-1)))
        self.assertEqual((4,), np.shape(bbox))

        # Test 2 : attention map is not valid
        bbox, mask = attention_crop(None)
        self.assertEqual((None, None), (bbox, mask))

        bbox, mask = attention_crop(3, 4)
        self.assertEqual((None, None), (bbox, mask))

        bbox, mask = attention_crop([1, 2])
        self.assertEqual((None, None), (bbox, mask))

        # Test 3 : thr is not valid
        bbox, mask = attention_crop(attention_map, min_thr='a', max_thr=0.6)
        self.assertEqual((None, None), (bbox, mask))

        bbox, mask = attention_crop(attention_map, min_thr=0.4, max_thr='b')
        self.assertEqual((None, None), (bbox, mask))

        bbox, mask = attention_crop(attention_map, min_thr=0.4, max_thr=None)
        self.assertEqual((None, None), (bbox, mask))

        # Test 4 : min_thr > max_thr
        bbox, mask = attention_crop(attention_map, min_thr=0.7, max_thr=0.6)
        self.assertTrue(np.array_equal(np.array([3., 3., 96., 96.]), bbox * (attention_map.shape[0]-1)))
        self.assertEqual((4,), np.shape(bbox))

        # Test 5 : Mask shape check
        bbox, mask = attention_crop(attention_map, min_thr=0.4, max_thr=0.6)
        self.assertEqual(attention_map.shape[:2], mask.shape)

    def test_attention_drop(self):
        height = 94
        width = 94
        pad = 3
        ch = 10
        padding = tf.constant([[pad, pad], [pad, pad], [0, 0]])
        _attention_map = tf.ones(shape=(height, width, ch))
        attention_map = tf.pad(_attention_map, padding, "CONSTANT")

        # Test 1 : attention drop 정상 동작
        mask, _mask = attention_drop(attention_map, min_thr=0.2, max_thr=0.5)

        false_area = mask[3:97, 3:97]
        self.assertTrue(np.array_equal(tf.zeros_like(false_area).numpy(), false_area))

        true_area = mask[0:3, :]
        self.assertTrue(np.array_equal(tf.ones_like(true_area).numpy(), true_area))

        true_area = mask[97:, :]
        self.assertTrue(np.array_equal(tf.ones_like(true_area).numpy(), true_area))

        # Test 2 : attention_map is not valid
        mask, _mask = attention_drop(attention_map=None, min_thr=0.2, max_thr=0.5)
        self.assertEqual((None, None), (mask, _mask))

        mask, _mask = attention_drop(attention_map=1, min_thr=0.2, max_thr=0.5)
        self.assertEqual((None, None), (mask, _mask))

        mask, _mask = attention_drop(attention_map=tf.random.normal(shape=(3, 100, 100, 10)), min_thr=0.2, max_thr=0.5)
        self.assertEqual((None, None), (mask, _mask))

        # Test 3 : threshold is not valid

        mask, _mask = attention_drop(attention_map, min_thr=0.2, max_thr='k')
        self.assertEqual((None, None), (mask, _mask))

        mask, _mask = attention_drop(attention_map, min_thr='h', max_thr=0.5)
        self.assertEqual((None, None), (mask, _mask))

        mask, _mask = attention_drop(attention_map, min_thr=None, max_thr=0.5)
        self.assertEqual((None, None), (mask, _mask))

        # Test 4 : min_thr > max thr

        mask, _mask = attention_drop(attention_map, min_thr=0.5, max_thr=0.2)

        false_area = mask[3:97, 3:97]
        self.assertTrue(np.array_equal(tf.zeros_like(false_area).numpy(), false_area))

        true_area = mask[0:3, :]
        self.assertTrue(np.array_equal(tf.ones_like(true_area).numpy(), true_area))

        true_area = mask[97:, :]
        self.assertTrue(np.array_equal(tf.ones_like(true_area).numpy(), true_area))
