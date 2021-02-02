import tensorflow as tf


class DataHandler:
    def __init__(self, resize_h_w):
        self.resize_h_w = resize_h_w

    @staticmethod
    def get_label(path):
        path = tf.strings.split(path, sep='/', maxsplit=-1)[-1]
        label = tf.strings.split(path, sep='_', maxsplit=-1)[0]
        label = tf.strings.to_number(label, out_type=tf.dtypes.int32)

        return label

    def load_jpeg(self, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.gather(img, [2, 1, 0], axis=-1)
        img = tf.image.resize(img, size=self.resize_h_w)
        img = tf.image.convert_image_dtype(img, tf.float32)

        return img / 255.

    def map_fn(self, path):
        image = self.load_jpeg(path)
        label = self.get_label(path)

        return image, label

    def create_batch_dataset(self, dataset_paths, batch_size, shuffle=False):
        assert len(dataset_paths) > 1

        batch_dataset = tf.data.Dataset.from_tensor_slices(dataset_paths)
        if shuffle:
            batch_dataset = batch_dataset.shuffle(8000)
        batch_dataset = batch_dataset.map(self.map_fn).batch(batch_size, drop_remainder=True)

        return batch_dataset


class PangyoDataHandler(DataHandler):
    def __init__(self, resize_h_w):
        super().__init__(resize_h_w)


class CubDataHandler(DataHandler):
    def __init__(self, resize_h_w):
        super().__init__(resize_h_w)

    @staticmethod
    def get_label(path):
        try:
            path = tf.strings.split(path, sep='/', maxsplit=-1)[-2]
            label = tf.strings.split(path, sep='.', maxsplit=-1)[0]
            label = tf.strings.to_number(label, out_type=tf.dtypes.int32)

            return label
        except Exception as e:
            print(e)

            return None
