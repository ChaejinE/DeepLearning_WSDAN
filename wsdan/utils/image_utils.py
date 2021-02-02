import tensorflow as tf
import cv2
import numpy as np
import random


def min_max_normalize(feature_map):
    """min_max_normalize_per_feature_map
    :param feature_map: Tensor of shape (B, H, W, CH)
    :return: Tensor of shape (B, H, W, CH)
    """
    eps = 0.00001

    minimum = tf.reduce_min(feature_map, axis=[1, 2])[:, tf.newaxis, tf.newaxis, :]
    maximum = tf.reduce_max(feature_map, axis=[1, 2])[:, tf.newaxis, tf.newaxis, :]

    a = tf.math.subtract(feature_map, minimum)
    b = tf.math.subtract(maximum, minimum) + eps
    normalized_feature_map = tf.math.divide(a, b)

    return normalized_feature_map


# private
def crop_resize_positive_area(images, score_maps, threshold):
    """
    :param images: Original image for cropping and resizing (B, H, W, 3)
    :param score_maps: Tensor of shape (B, h, w, 1)
    :param threshold: Threshold for masking
    :return: crop_images (B, H, W, 3)
    """
    try:
        def get_positive_tlbr(mask):
            mask = tf.squeeze(mask, axis=-1)
            h, w = tf.shape(mask)
            indices = tf.where(mask)
            min_y, min_x = tf.cast(tf.reduce_min(indices, axis=0), dtype=tf.int32)
            max_y, max_x = tf.cast(tf.reduce_max(indices, axis=0), dtype=tf.int32)
            min_y, min_x = min_y / (h-1), min_x / (w-1)
            max_y, max_x = max_y / (h-1), max_x / (w-1)
            tlbr = tf.stack([min_y, min_x, max_y, max_x])
            tlbr = tf.cast(tlbr, dtype=tf.float32)

            return tlbr
        batch, height, width, _ = tf.shape(images)
        _max = tf.math.reduce_max(score_maps, axis=[1, 2])
        _max = _max[:, tf.newaxis, tf.newaxis, :]
        threshold = tf.random.uniform(shape=[], minval=0.4, maxval=0.6)
        crop_cond = tf.greater(score_maps, _max * threshold)
        crop_masks = tf.where(crop_cond, 1, 0)
        normalized_tlbr = tf.map_fn(get_positive_tlbr, crop_masks, dtype=tf.float32)
        box_idx = tf.range(batch, dtype=tf.int32)
        crop_images = tf.image.crop_and_resize(images, normalized_tlbr,
                                               box_idx, crop_size=[height, width])

        return crop_images

    except Exception as e:
        print("Error in crop_resize_positive_area as {}".format(e))

        raise


# private
def drop_positive_area(images, score_maps, threshold):
    """
    :param images: Original image for dropping (B, H, W, 3)
    :param score_maps: Tensor of shape (B, h, w, 1)
    :param threshold: Threshold for dropping
    :return: drop_images (B, H, W, 3)
    """
    try:
        _, height, width, _ = tf.shape(images)
        _max = tf.math.reduce_max(score_maps, axis=[1, 2])
        _max = _max[:, tf.newaxis, tf.newaxis, :]
        threshold = tf.random.uniform(shape=[], minval=0.2, maxval=0.5)
        drop_cond = tf.less(score_maps, _max * threshold)
        drop_masks = tf.where(drop_cond, 1, 0)
        drop_masks = tf.image.resize(drop_masks, size=(height, width))
        drop_masks = tf.cast(drop_masks, dtype=tf.float32)

        drop_images = tf.math.multiply(drop_masks, images)

        return drop_images

    except Exception as e:
        print('Error in "drop_positive_area" as {}'.format(e))

        raise


# official
def create_heat_map(image, feature_map):
    try:
        mask = np.mean(feature_map.numpy(), axis=-1, keepdims=True)  # (h, w, 1)
        mask = (mask / np.max(mask) * 255.0).astype(np.uint8)

        color_map = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)  # (h, w, 3)
        heat_map = cv2.addWeighted(image.astype(np.uint8), 0.5, color_map.astype(np.uint8), 0.5, 0)

        return heat_map
    except Exception as e:
        print('create_heat_map failed because error of {}'.format(e))

        return None


# official
def visualize(image, crop_attention, drop_attention, crop_image, drop_image):
    image = image.numpy() * 255.
    crop_heat_map = create_heat_map(image, crop_attention[:, :, tf.newaxis])
    drop_heat_map = create_heat_map(image, drop_attention)
    crop_image *= 255.
    drop_image *= 255.

    concated_heat_map = np.concatenate([image, crop_heat_map, crop_image, drop_heat_map, drop_image], axis=1)

    return concated_heat_map


# official
def attention_crop(attention_map, min_thr=0.4, max_thr=0.6):
    try:
        height, width, num_parts = attention_map.shape
        attention_map = attention_map.numpy()
        part_weights = attention_map.mean(axis=0).mean(axis=0)
        part_weights = np.sqrt(part_weights)
        part_weights = part_weights / np.sum(part_weights)
        selected_index = np.random.choice(np.arange(0, num_parts), 1, p=part_weights)[0]

        mask = attention_map[:, :, selected_index]

        threshold = random.uniform(min(min_thr, max_thr), max(min_thr, max_thr))
        itemindex = np.where(mask >= mask.max() * threshold)

        ymin = itemindex[0].min() / (height - 1.)
        ymax = itemindex[0].max() / (height - 1.)
        xmin = itemindex[1].min() / (width - 1.)
        xmax = itemindex[1].max() / (width - 1.)

        bbox = np.asarray([ymin, xmin, ymax, xmax], dtype=np.float32)

        return bbox, mask
    except Exception as e:
        print('attention crop failed because error of {}'.format(e))

        return None, None


# official
def attention_drop(attention_map, min_thr=0.2, max_thr=0.5):
    """

    :param attention_map: Tensor of shape (H, W, CH)
    :return:
    """
    try:
        height, width, num_parts = attention_map.shape
        attention_map = attention_map.numpy()
        part_weights = attention_map.mean(axis=0).mean(axis=0)
        part_weights = np.sqrt(part_weights)
        part_weights = part_weights / np.sum(part_weights)
        selected_index = np.random.choice(np.arange(0, num_parts), 1, p=part_weights)[0]

        _mask = attention_map[:, :, selected_index:selected_index+1]

        # soft mask
        threshold = random.uniform(min(min_thr, max_thr), max(min_thr, max_thr))
        mask = (_mask < threshold * _mask.max()).astype(np.float32)

        return mask, _mask
    except Exception as e:
        print('attention_drop failed because error of {}'.format(e))

        return None, None
