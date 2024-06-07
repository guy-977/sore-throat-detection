import tensorflow as tf


def build_model(img_width, img_height):
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=(img_width, img_height, 3),
        include_top=False)  # Do not include the ImageNet classifier at the top.
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(img_width, img_height, 3))
    x = base_model(inputs, training=False)
    
    x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2)(x)
    model = tf.keras.Model(inputs, outputs)
    

    return model