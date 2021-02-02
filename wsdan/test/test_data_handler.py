import os
import shutil
import unittest

import cv2
import numpy as np
import tensorflow as tf

from tf_2.finegrained.wsdan.dataset.datahandler import DataHandler


class TestDataHandler(unittest.TestCase):
    file_dir = '/tmp/test'

    @classmethod
    def setUpClass(cls) -> None:
        os.makedirs(cls.file_dir, exist_ok=True)
        for i in range(7):
            image = np.random.uniform(size=(17, 17, 3))
            cv2.imwrite(cls.file_dir+'/{0:03}_{1:03}.jpg'.format(i, i+1), image)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.file_dir, ignore_errors=True)

    def test_create_batch_datasets(self):
        batch_size = 2
        resize_h_w = (299, 299)
        file_paths = [os.path.join(self.file_dir, file) for file in os.listdir(self.file_dir)]
        datasets = DataHandler(resize_h_w).create_batch_dataset(file_paths, batch_size, shuffle=True)

        # Test 1. batch size must be constant
        for imgs, labels in datasets:
            x = tf.shape(imgs)[0]
            self.assertEqual(batch_size, x)


