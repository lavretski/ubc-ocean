from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


def make_model(input_shape: tuple[int, int], num_classes: int) \
    -> tf.keras.Model:
        inputs = keras.Input(shape=input_shape)

        x = layers.Conv2D(128, 3, strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x

        for size in [256, 512, 728]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            residual = layers.Conv2D(size, 1, strides=2, padding="same")(
                previous_block_activation)
            x = layers.add([x, residual])
            previous_block_activation = x

        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        return keras.Model(inputs, outputs)