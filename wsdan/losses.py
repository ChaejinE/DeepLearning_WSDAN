import tensorflow as tf


class WSDANLoss:
    def __init__(self, num_class):
        self.global_feature_centers = None
        self.num_class = num_class

    @staticmethod
    def calculate_classification_loss(y_true, y_pred):
        """
        :param y_true: Tensor of shape (batch, )
        :param y_pred: Tensor of shape (batch, class_num)
        :return:
                Tensor of shape ()
                type = float32
        """
        assert tf.constant(0) not in y_true, 'Check your labels <= 0'
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False,
                                                                      reduction=tf.keras.losses.Reduction.NONE,
                                                                      name='sparse_categorical_crossentropy')
        y_true = y_true - tf.constant(1)
        y_pred = tf.math.softmax(y_pred)
        loss = cross_entropy(y_true, y_pred)
        loss = tf.reduce_mean(loss)

        return loss

    def calculate_attention_loss(self, labels, embeddings, beta=0.05):
        """
        :param labels: Tensor shape (B, )
        :param embeddings: Tensor shape (B, 1, 1, attention_num * feature_num)
        :param beta: float
        :return: loss Tensor, float32
        """
        embeddings = tf.squeeze(embeddings, axis=[1, 2])
        embeddings = tf.cast(embeddings, dtype=tf.float32)
        batch, dims = embeddings.shape

        labels = labels - tf.constant(1)

        if self.global_feature_centers is None:
            self.global_feature_centers = tf.zeros(shape=(self.num_class, dims))
        self.global_feature_centers = tf.cast(self.global_feature_centers, dtype=tf.float32)

        batch_centers = tf.gather(self.global_feature_centers, labels)
        batch_centers = tf.math.l2_normalize(batch_centers, axis=-1)

        diff = beta * (batch_centers - embeddings)
        labels = tf.expand_dims(labels, axis=-1)
        self.global_feature_centers = tf.tensor_scatter_nd_sub(self.global_feature_centers, labels, diff)
        distance = tf.math.square(embeddings - batch_centers)
        distance = tf.math.reduce_sum(distance, axis=-1)
        loss = tf.reduce_mean(distance)

        return loss

    def calculate_total_loss(self, y_true, origin_pred, crop_pred, drop_pred, embeddings, beta):
        loss_of_origin = self.calculate_classification_loss(y_true, origin_pred)
        loss_of_crop = self.calculate_classification_loss(y_true, crop_pred)
        loss_of_drop = self.calculate_classification_loss(y_true, drop_pred)

        classification_loss = (loss_of_origin + loss_of_crop + loss_of_drop) / 3
        attention_loss = self.calculate_attention_loss(y_true, embeddings, beta)

        loss = classification_loss + attention_loss

        return loss, classification_loss, attention_loss
