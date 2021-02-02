import unittest
import cv2
import numpy as np
import os
import shutil

from tf_2.finegrained.wsdan.dataset.datahandler import CubDataHandler


class TestCubDataHandler(unittest.TestCase):
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

    def test_get_label(self):
        resize_h_w = (299, 299)
        data_handler = CubDataHandler(resize_h_w)
        expected_label = {1: 'lotto_bird',
                          3: 'beauty_bird',
                          5: 'angry_bird'}

        file_folders = []
        file_names = []
        for _k, _v in expected_label.items():
            folder_name = os.path.join(self.file_dir, 'images', str(_k)+'.'+str(_v))
            os.makedirs(folder_name, exist_ok=True)
            file_folders.append(folder_name)
            file_names.append(str(_v+'_001.jpg'))

        for file_folder, file_name in zip(file_folders, file_names):
            cv2.imwrite(os.path.join(file_folder, file_name), np.random.normal(size=(3, 3, 3)))

        file_paths = [os.path.join(file_folder, file_name) for file_folder in file_folders
                      for file_name in os.listdir(file_folder)]

        for file_path in file_paths:
            label = data_handler.get_label(file_path)
            label = label.numpy()

            expected_name = expected_label[label]
            self.assertTrue(expected_name in file_path)

    def test_get_label_if_path_is_not_valid(self):
        resize_h_w = (299, 299)
        data_handler = CubDataHandler(resize_h_w)

        self.assertIsNone(data_handler.get_label(None))
        self.assertIsNone(data_handler.get_label(''))


