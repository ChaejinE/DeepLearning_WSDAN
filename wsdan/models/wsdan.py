import tensorflow as tf
from tf_2.finegrained.wsdan.models.backbone import Backbone

import cv2
import time


class WSDAN(tf.keras.Model):
    def __init__(self, class_num, attention_num):
        super(WSDAN, self).__init__()
        self.class_num = class_num
        self.attention_num = attention_num

        # backbone model 생성
        self.backbone = Backbone().inception_v3(trainable=True, output_layer=['mixed6', 'mixed7'])

        # attention maps를 추출하려는 1x1 convolution layer
        self.conv_1x1 = tf.keras.layers.Conv2D(filters=attention_num, kernel_size=(1, 1),
                                               strides=1, padding='same',
                                               name='attention_maps', use_bias=False)

        # bilinear attention pooling에서 쓰이는 global average pooling layer
        self.global_average_pool = tf.keras.layers.GlobalAveragePooling2D()

        # classifier
        self.classification_conv = tf.keras.layers.Conv2D(filters=self.class_num, kernel_size=(1, 1),
                                                      strides=1, padding='same',
                                                      name='classification', use_bias=False)  # class_num or embedding dimension

    def call(self, inputs):
        feature_maps = self.backbone(inputs)
        attention_maps = self.conv_1x1(feature_maps[-1]) # (batch, height, width, attention_num)

        # bilinear attention pooling
        # (batch, height, width, attention_num, feature_num)
        attention_features = tf.expand_dims(attention_maps, axis=-1) * tf.expand_dims(feature_maps[0], axis=-2)
        # (batch, height, width, attention_num * feature_num)
        attention_features = tf.reshape(attention_features, shape=(feature_maps[0].shape[0], feature_maps[0].shape[1],
                                                                   feature_maps[0].shape[2],  -1))
        # (batch, attention_num * feature_num)
        parts_attention_features = self.global_average_pool(attention_features)

        # (batch, attention_num, feature_num)
        feature_matrix = tf.reshape(parts_attention_features, shape=(-1, self.attention_num,
                                                                     feature_maps[0].shape[-1]))

        # 논문 내용 적용
        feature_matrix = tf.multiply(tf.sign(feature_matrix), tf.sqrt(tf.abs(feature_matrix) + 1e-12))

        feature_matrix = tf.nn.l2_normalize(feature_matrix, axis=[1, 2])
        # official code에서 raw_feature라고 지칭하는 부분 : embeddings
        embeddings = tf.reshape(feature_matrix, [-1, 1, 1, attention_maps.shape[-1] * feature_maps[0].shape[-1]])
        # official code에서 pool_features라고 지칭하는 부분
        features = embeddings * 100  # logit

        logits = self.classification_conv(features)
        logits = tf.squeeze(logits, axis=[1, 2])

        return logits, attention_maps, embeddings
