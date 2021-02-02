from unittest import TestCase
from tf_2.finegrained.wsdan.train import WSDANTrainer

import argparse
import yaml
import os
import shutil
import cv2
import numpy as np


class TestWSDANTrainer(TestCase):
    test_dir = '/tmp/test'

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.test_dir, exist_ok=True)
        os.makedirs(os.path.join(cls.test_dir, 'images'))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)

    def create_cub_data(self):
        labels = {1: 'lotto_bird',
                  3: 'beauty_bird',
                  5: 'angry_bird'}

        file_folders = []
        file_names = []
        for _k, _v in labels.items():
            folder_name = os.path.join(self.test_dir, 'images', str(_k)+'.'+str(_v))
            os.makedirs(folder_name, exist_ok=True)
            file_folders.append(folder_name)
            file_names.append(str(_v+'_001.jpg'))

        for file_folder, file_name in zip(file_folders, file_names):
            cv2.imwrite(os.path.join(file_folder, file_name), np.random.normal(size=(3, 3, 3)))
            for i in range(1, 5):
                cv2.imwrite(os.path.join(file_folder, ('{}'.format(i)+file_name)), np.random.normal(size=(3, 3, 3)))

    def test_create_data_paths_with_cub(self):
        self.create_cub_data()

        parser = argparse.ArgumentParser()
        parser.add_argument('--config-file-path', default='./tf_2/finegrained/wsdan/train.yml')
        args = parser.parse_args()

        with open(args.config_file_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        config['image_dir'] = os.path.join(self.test_dir, 'images')

        trainer = WSDANTrainer(config)
        train_paths, valid_paths = trainer.create_data_paths()

        expected_train_paths = ['/tmp/test/images/3.beauty_bird/1beauty_bird_001.jpg',
                                '/tmp/test/images/3.beauty_bird/2beauty_bird_001.jpg',
                                '/tmp/test/images/3.beauty_bird/3beauty_bird_001.jpg',
                                '/tmp/test/images/1.lotto_bird/1lotto_bird_001.jpg',
                                '/tmp/test/images/1.lotto_bird/2lotto_bird_001.jpg',
                                '/tmp/test/images/1.lotto_bird/3lotto_bird_001.jpg',
                                '/tmp/test/images/5.angry_bird/1angry_bird_001.jpg',
                                '/tmp/test/images/5.angry_bird/2angry_bird_001.jpg',
                                '/tmp/test/images/5.angry_bird/3angry_bird_001.jpg']
        expected_valid_paths = ['/tmp/test/images/3.beauty_bird/4beauty_bird_001.jpg',
                                '/tmp/test/images/1.lotto_bird/4lotto_bird_001.jpg',
                                '/tmp/test/images/5.angry_bird/4angry_bird_001.jpg']

        for path in expected_train_paths:
            self.assertTrue(path in train_paths)

        for path in expected_valid_paths:
            self.assertTrue(path in valid_paths)
