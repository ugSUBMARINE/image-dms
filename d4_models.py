import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

# from tensorflow.keras import layers
from keras import layers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def simple_model_imp(wt_seq, channel_num, model_name="simple_model_imp", reduce=-1):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(16, 3, padding="same", activation="leaky_relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="leaky_relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="leaky_relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1, activation="leaky_relu")(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def res_net(wt_seq, channel_num, model_name="res_net", reduce=-1):
    # function for creating an identity or projection residual module
    def residual_module(layer_in, n_filters):
        """https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional
        -neural-networks/"""
        merge_input = layer_in
        # check if the number of filters needs to be increase, assumes channels last format
        if layer_in.shape[-1] != n_filters:
            merge_input = layers.Conv2D(
                n_filters,
                (1, 1),
                padding="same",
                activation="leaky_relu",
                kernel_initializer="he_normal",
            )(layer_in)
        # conv1
        conv1 = layers.Conv2D(
            n_filters,
            (3, 3),
            padding="same",
            activation="leaky_relu",
            kernel_initializer="he_normal",
        )(layer_in)
        # conv2
        conv2 = layers.Conv2D(
            n_filters,
            (3, 3),
            padding="same",
            activation="linear",
            kernel_initializer="he_normal",
        )(conv1)
        # add filters, assumes filters/channels last
        layer_out = layers.add([conv2, merge_input])
        # activation function
        layer_out = layers.Activation("leaky_relu")(layer_out)
        return layer_out

    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(16, 3, padding="same", activation="leaky_relu")(inputs)
    x = residual_module(x, 32)
    x = layers.MaxPooling2D()(x)
    x = residual_module(x, 32)
    x = layers.MaxPooling2D()(x)
    x = residual_module(x, 64)
    x = layers.MaxPooling2D()(x)
    x = residual_module(x, 64)
    x = layers.MaxPooling2D()(x)
    x = residual_module(x, 128)
    x = layers.MaxPooling2D()(x)
    x = residual_module(x, 128)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def vgg(wt_seq, channel_num, model_name="vgg", reduce=-1):
    """https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional
    -neural-networks/"""

    def vgg_block(layer_in, n_filters, n_conv):
        for _ in range(n_conv):
            layer_in = layers.Conv2D(
                n_filters, (3, 3), padding="same", activation="leaky_relu"
            )(layer_in)
        layer_in = layers.MaxPooling2D((2, 2), strides=(2, 2))(layer_in)
        return layer_in

    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = vgg_block(inputs, 32, 3)
    x = vgg_block(x, 32, 4)
    x = vgg_block(x, 64, 4)
    x = vgg_block(x, 64, 4)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def activation_block(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def conv_stem(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_block(x)


def conv_mixer_block(x, filters: int, kernel_size: int):
    # Depth wise convolution.
    x0 = x
    x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
    x = layers.Add()([activation_block(x), x0])  # Residual.

    # Point wise convolution.
    x = layers.Conv2D(filters, kernel_size=1)(x)
    x = activation_block(x)

    return x


def get_conv_mixer_256_8(
    wt_seq, channel_num, filters=128, depth=8, kernel_size=5, patch_size=2, reduce=-1
):  # 128 8 5 2
    """modified after https://keras.io/examples/vision/convmixer/"""
    inputs = keras.Input((len(wt_seq), len(wt_seq), channel_num))

    # Extract patch embeddings.
    x = conv_stem(inputs, filters, patch_size)

    # ConvMixer blocks.
    for _ in range(depth):
        x = conv_mixer_block(x, filters, kernel_size)

    # Classification block.
    x = layers.GlobalAvgPool2D()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1, activation="leaky_relu")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model


def dense_net2(
    wt_seq,
    channel_num,
    filter_num=12,
    block_num=4,
    block_depth=4,
    reduce=False,
    bn=False,
    classif_l=2,
    filter_size=3,
    l_pool="avg",
    e_pool="avg",
    model_name="dense_net_tune",
):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    if filter_num == 0:
        filter_num = 1

    def d_layer(p_in):
        if bn:
            p_in = layers.BatchNormalization(momentum=0.9)(p_in)
        dl = layers.Conv2D(128, 1, padding="same", activation="leaky_relu")(p_in)
        dl = layers.Conv2D(
            filter_num, filter_size, padding="same", activation="leaky_relu"
        )(dl)
        return dl

    def d_block(prev_in):
        temp = prev_in
        for i in range(block_depth):
            x = d_layer(temp)
            temp = layers.concatenate([x, temp])
        return temp

    if intro_layer:
        x = layers.Conv2D(128, 3, 2, padding="same")(inputs)
        x = layers.MaxPooling2D(3, 2)(x)
        x = d_block(x)
    else:
        x = d_block(inputs)
    for i in range(block_num):
        if bn:
            x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.Conv2D(128, 1, padding="same")(x)
        if l_pool == "avg":
            x = layers.AveragePooling2D()(x)
        else:
            x = layers.MaxPooling2D()(x)
        x = d_block(x)

    if e_pool == "avg":
        x = layers.GlobalAveragePooling2D()(x)
    else:
        x = layers.GlobalMaxPool2D()(x)

    for i in range(classif_l):
        x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1, dtype=tf.float32)(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def sep_conv_mix(wt_seq, channel_num, model_name="sep_conv_mix", reduce=False):
    def block(prev_in):
        bx = layers.SeparableConv2D(32, 3, padding="same", activation="leaky_relu")(
            prev_in
        )
        bxx = layers.SeparableConv2D(32, 3, padding="same", activation="leaky_relu")(bx)
        bo = layers.add([prev_in, bx, bxx])
        return bo

    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    if reduce:
        kernel_size = 9
        strides_ = 9
    else:
        kernel_size = 3
        strides_ = 1
    x = layers.SeparableConv2D(
        32, kernel_size, strides=strides_, padding="same", activation="leaky_relu"
    )(inputs)
    for i in range(9):
        x = block(x)

    x = layers.GlobalMaxPool2D()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


if __name__ == "__main__":
    mod = sep_conv_res(np.arange(237), 7)
