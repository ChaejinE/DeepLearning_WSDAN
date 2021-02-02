import tensorflow as tf


def feature_map_random_sampling(feature_maps):
    """
    :param feature_maps: Tensor of shape (Batch, h, w, ch)
    :return: Tensor of shape (Batch, h, w, 1)
    """
    batch = tf.shape(feature_maps)[0]
    indices = tf.random.shuffle(tf.range(batch))[0]
    feature_map = tf.gather(feature_maps, indices, axis=-1)
    feature_map = tf.expand_dims(feature_map, axis=-1)

    return feature_map


def point_wise_average_pooling(feature_maps):
    """
    :param feature_maps: Tensor of shape (Batch, h, w, ch)
    :return: Tensor of shape (Batch, h, w, 1)
    """
    feature_map = tf.reduce_mean(feature_maps, axis=-1)
    feature_map = tf.expand_dims(feature_map, axis=-1)

    return feature_map
