import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models


class SiamModel(tf.keras.layers.Layer):
    def __init__(self, input_shape=16, dense1_shape=32, dense2_shape=64, name='IntenalSiamModel', **kwargs):
        super(SiamModel, self).__init__(name=name, **kwargs)
        self.inputs = layers.Input(shape=input_shape)
        self.fc1 = layers.Dense(dense1_shape, activation='relu')
        self.batchnorm = layers.BatchNormalization()
        self.fc2 = layers.Dense(dense2_shape, activation='sigmoid')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.batchnorm(x)
        output = self.fc2(x)
        return output


class SiameseNetwork(tf.keras.Model):
    def __init__(self, input_shape=16, dense1_shape=32, dense2_shape=64, name='SiameseNetwork', **kwargs):
        super(SiameseNetwork, self).__init__(name=name, **kwargs)
        self.internal_model = SiamModel(input_shape=input_shape, dense1_shape=dense1_shape, dense2_shape=dense2_shape)
        self.distance_layer = layers.Lambda(lambda features: tf.math.abs(features[0] - features[1]))
        self.classifier = layers.Dense(1, activation='sigmoid')
    
    def call(self, inputs):
        sample1_vector, sample2_vector = inputs

        sample1_features = self.internal_model(sample1_vector)
        sample2_features = self.internal_model(sample2_vector)

        # distance layer
        distance_vector = tf.math.abs(sample1_features - sample2_features)
        
        # classification head - only for training
        output = self.classifier(distance_vector)

        return output, distance_vector

    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            y_pred, distance = self(x, training=True)
            # Compute the loss function (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # compute the gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # update the model's weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # update metrics
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metrics names to current value
        return {m.name: m.result() for m in self.metrics}