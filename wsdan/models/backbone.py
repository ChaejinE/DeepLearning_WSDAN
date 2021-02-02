import tensorflow as tf
from tensorflow.keras import Model


class Backbone:
    def __init__(self):
        pass

    @staticmethod
    def inception_v3(weights='imagenet', trainable=False, output_layer=None):
        """
        :param weights: None or 'imagenet' or weight_path
        :param trainable: True or False
        :param output_layer: None or 'mixed6' or list type
        :return: Model
        """
        inputs = tf.keras.layers.Input((None, None, 3))
        model = tf.keras.applications.InceptionV3(include_top=False, weights=weights, input_tensor=inputs)
        if trainable:
            model.trainable = trainable

        outputs = []

        if type(output_layer) is list:
            for i, _ in enumerate(output_layer):
                outputs.append(model.get_layer(output_layer[i]).output)
        else:
            outputs.append(model(inputs))

        return Model(inputs, outputs)
