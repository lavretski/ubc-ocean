from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import keras_cv


def make_scratch_model(image_size: tuple[int, int], num_classes: int) \
    -> tf.keras.Model:
    # https://keras.io/examples/vision/image_classification_from_scratch/
        inputs = keras.Input(shape=image_size)

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


def make_aritra_model() -> tf.keras.Model:
    # https://www.kaggle.com/code/aritrag/kerascv-train-and-infer-on-thumbnails
    resnet_backbone = keras_cv.models.ResNetV2Backbone.from_preset("resnet152_v2")
    resnet_backbone.trainable = False

    image_inputs = resnet_backbone.input
    image_embeddings = resnet_backbone(image_inputs)
    image_embeddings = keras.layers.GlobalAveragePooling2D()(image_embeddings)

    x = keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1)(image_embeddings)
    x = keras.layers.Dense(units=1024, activation="relu")(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(units=512, activation="relu")(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Dense(units=256, activation="relu")(x)
    outputs = keras.layers.Dense(units=5, activation="softmax")(x)

    return keras.Model(inputs=image_inputs, outputs=outputs)
