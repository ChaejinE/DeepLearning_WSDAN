from tf_2.finegrained.wsdan.models.wsdan import WSDAN
import tensorflow as tf
import unittest


class TestWSDAN(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 32
        self.images = tf.random.uniform(shape=(self.batch_size, 299, 299, 3))
        self.attention_num = 32

    def test_WSDAN(self):
        net = WSDAN(class_num=32, attention_num=32)
        output, attention_maps, embeddings = net(self.images)
        self.assertEqual(output.shape, (self.batch_size, 32))
        self.assertEqual(attention_maps.shape, (self.batch_size, 17, 17, self.attention_num))
        self.assertEqual(embeddings.shape, (self.batch_size, 1, 1, 32 * 768))
        net.summary()

