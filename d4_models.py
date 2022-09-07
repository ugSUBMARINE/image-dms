import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

# from tensorflow.keras import layers
from keras import layers

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def simple_model(wt_seq, channel_num, model_name="simple_model"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(16, 3, padding="same", activation="leaky_relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="leaky_relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="leaky_relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def simple_model_norm(wt_seq, channel_num, model_name="simple_model_norm"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(16, 3, padding="same", activation="leaky_relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="leaky_relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="leaky_relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def simple_model_imp(wt_seq, channel_num, model_name="simple_model_imp"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(16, 3, padding="same", activation="leaky_relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="leaky_relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="leaky_relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    # x = layers.GlobalAvgPool2D()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1, activation="leaky_relu")(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def simple_model_128(wt_seq, channel_num, model_name="simple_model_imp"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(128, 3, padding="same", activation="leaky_relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="leaky_relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="leaky_relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalMaxPool2D()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1, activation="leaky_relu")(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def simple_model_longer(wt_seq, channel_num, model_name="simple_model_longer"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)

    def block(prev_in, filter_s=32):
        # x0 = layers.Conv2D(32, 2, strides=2, padding='same', activation='leaky_relu')(prev_in)
        filter_s = prev_in.shape[-1]
        x0 = layers.AveragePooling2D()(prev_in)
        x1 = layers.Conv2D(filter_s, 3, padding="same", activation="leaky_relu")(x0)
        x2 = layers.Conv2D(filter_s, 3, padding="same", activation="leaky_relu")(x1)
        x3 = layers.Conv2D(filter_s, 3, padding="same", activation="leaky_relu")(x2)
        x4 = layers.Conv2D(filter_s, 3, padding="same", activation="leaky_relu")(x3)
        # x5 = layers.Conv2D(filter_s, 3, padding='same', activation='leaky_relu')(x4)
        # x6 = layers.Conv2D(filter_s, 3, padding='same', activation='leaky_relu')(x5)
        # x7 = layers.Conv2D(filter_s, 3, padding='same', activation='leaky_relu')(x6)

        # out = layers.add([x1, x2, x3, x4, x5, x6, x7])
        out = layers.concatenate([x1, x4])
        return out

    x = block(inputs)
    for i in range(4):
        x = block(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1, activation="leaky_relu")(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def create_simple_model(wt_seq, channel_num, model_name="create_simple_model"):
    inputs = keras.Input(shape=(235, 40, 1), name=model_name)
    x = layers.Conv2D(
        filters=128,
        kernel_size=[3, inputs.shape[2]],
        strides=[1, 1],
        padding="valid",
        activation="leaky_relu",
        use_bias=True,
    )(inputs)
    x = layers.Conv2D(
        filters=128,
        kernel_size=[3, x.shape[2]],
        strides=[1, 1],
        padding="valid",
        activation="leaky_relu",
        use_bias=True,
    )(x)
    x = layers.Conv2D(
        filters=128,
        kernel_size=[3, x.shape[2]],
        strides=[1, 1],
        padding="valid",
        activation="leaky_relu",
        use_bias=True,
    )(x)
    x = layers.Conv2D(
        filters=128,
        kernel_size=[3, x.shape[2]],
        strides=[1, 1],
        padding="valid",
        activation="leaky_relu",
        use_bias=True,
    )(x)
    x = layers.Conv2D(
        filters=128,
        kernel_size=[3, x.shape[2]],
        strides=[1, 1],
        padding="valid",
        activation="leaky_relu",
        use_bias=True,
    )(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation="leaky_relu", use_bias=True)(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, use_bias=True)(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def simple_model_gap(wt_seq, channel_num, model_name="simple_model_gap"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(16, 3, padding="same", activation="leaky_relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="leaky_relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="leaky_relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="leaky_relu")(x)
    x = layers.GlobalMaxPool2D()(x)
    x = layers.Dense(512, activation="leaky_relu")(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    return model


def simple_stride_model_test(
    wt_seq, channel_num, model_name="simple_stride_model_test"
):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(
        filters=32, kernel_size=1, strides=1, padding="same", activation="leaky_relu"
    )(inputs)
    x = layers.Conv2D(
        filters=64, kernel_size=2, strides=2, padding="same", activation="leaky_relu"
    )(x)
    x = layers.Conv2D(
        filters=128, kernel_size=2, strides=2, padding="same", activation="leaky_relu"
    )(x)
    x = layers.Conv2D(
        filters=64, kernel_size=2, strides=2, padding="same", activation="leaky_relu"
    )(x)
    x = layers.Conv2D(
        filters=64, kernel_size=2, strides=2, padding="same", activation="leaky_relu"
    )(x)
    # x = layers.Conv2D(filters=256, kernel_size=2, strides=2, padding='same', activation='leaky_relu')(x)
    # x = layers.Conv2D(filters=64, kernel_size=2, strides=2, padding='same', activation='leaky_relu')(x)
    # x = layers.Conv2D(filters=64, kernel_size=3, strides=3, padding='same', activation='leaky_relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1, activation="leaky_relu")(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def shrinking_res(wt_seq, channel_num, blocks=6, model_name="shrinking_res"):
    def block(prev_output, count, num_convs=4, block_depth=2, reduce_size=True):
        kernel_list = [3, 3, 5, 7]
        filter_list = [32, 32, 32, 32]
        missing_filter = abs(len(filter_list) - blocks)
        if missing_filter > 0:
            for m in range(missing_filter):
                filter_list += [filter_list[-1]]
        filter_size = filter_list[count - 1]

        b2 = layers.Conv2D(
            filters=filter_size,
            kernel_size=2,
            strides=2,
            padding="same",
            activation="leaky_relu",
        )(prev_output)

        diff_kl_convs = len(kernel_list) - num_convs
        if diff_kl_convs != 0:
            for j in range(abs(diff_kl_convs)):
                kernel_list.insert(0, kernel_list[0])

        layer_list = [b2]
        for k in range(num_convs):
            li = layers.Conv2D(
                filters=filter_size,
                kernel_size=kernel_list[k],
                strides=1,
                padding="same",
                activation="leaky_relu",
            )(b2)
            if block_depth > 1:
                for f in range(block_depth - 1):
                    li = layers.Conv2D(
                        filters=filter_size,
                        kernel_size=kernel_list[k],
                        strides=1,
                        padding="same",
                        activation="leaky_relu",
                    )(li)
                layer_list += [li]
            else:
                layer_list += [li]

        if reduce_size:
            layer_count = len(layer_list)
            if count == 0:
                pass
            elif layer_count - count <= 2:
                layer_list = layer_list[:2]
            else:
                layer_list = layer_list[: layer_count - count]
        b_out = layers.add(layer_list)

        return b_out

    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    # x = layers.Conv2D(filters=256, kernel_size=1, strides=1, padding="same", activation="leaky_relu")(inputs)
    x = block(inputs, count=0)
    for i in range(blocks):
        x = block(x, count=i + 1)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()

    return model


def inception_res(wt_seq, channel_num, model_name="inception_res"):
    def block(prev_out):
        b2 = layers.Conv2D(
            filters=32,
            kernel_size=2,
            strides=2,
            padding="same",
            activation="leaky_relu",
        )(prev_out)
        c1 = layers.Conv2D(
            filters=16,
            kernel_size=5,
            strides=1,
            padding="same",
            activation="leaky_relu",
        )(b2)
        c11 = layers.Conv2D(
            filters=16,
            kernel_size=5,
            strides=1,
            padding="same",
            activation="leaky_relu",
        )(c1)
        c2 = layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="leaky_relu",
        )(b2)
        c22 = layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="leaky_relu",
        )(c2)
        c3 = layers.Conv2D(
            filters=32,
            kernel_size=1,
            strides=1,
            padding="same",
            activation="leaky_relu",
        )(b2)
        c33 = layers.Conv2D(
            filters=32,
            kernel_size=1,
            strides=1,
            padding="same",
            activation="leaky_relu",
        )(c3)
        b_out = layers.concatenate([c11, c22, c33])
        return b_out

    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(
        filters=16, kernel_size=3, strides=1, padding="same", activation="leaky_relu"
    )(inputs)
    x = layers.Conv2D(
        filters=16, kernel_size=3, strides=1, padding="same", activation="leaky_relu"
    )(x)
    x = layers.Conv2D(
        filters=16, kernel_size=3, strides=1, padding="same", activation="leaky_relu"
    )(x)

    # x = block(inputs)
    for i in range(5):
        x = block(x)

    # x = layers.Conv2D(filters=32, kernel_size=1, strides=1, padding="same", activation="leaky_relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def deeper_res(wt_seq, channel_num, model_name="deeper_res"):
    def block(prev_out):
        b2 = layers.Conv2D(
            filters=32,
            kernel_size=2,
            strides=2,
            padding="same",
            activation="leaky_relu",
        )(prev_out)
        i1 = layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="leaky_relu",
        )(b2)
        i2 = layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="leaky_relu",
        )(b2)
        i3 = layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="leaky_relu",
        )(b2)
        i12 = layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="leaky_relu",
        )(i1)
        i22 = layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="leaky_relu",
        )(i2)
        i32 = layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            padding="same",
            activation="leaky_relu",
        )(i3)
        return layers.add([b2, i12, i22, i32])

    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(
        filters=128, kernel_size=1, strides=1, padding="same", activation="leaky_relu"
    )(inputs)
    x = block(x)
    x = block(x)
    x = block(x)
    x = block(x)
    x = block(x)
    x = block(x)
    x = block(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def res_net(wt_seq, channel_num, model_name="res_net"):
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


def vgg(wt_seq, channel_num, model_name="vgg"):
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
    # x = vgg_block(x, 64, 4)
    # x = vgg_block(x, 64, 4)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def simple_longer(wt_seq, channel_num, model_name="simple_longer"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(16, 3, padding="same", activation="leaky_relu")(inputs)
    x = layers.Conv2D(16, 3, padding="same", activation="leaky_relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="leaky_relu")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="leaky_relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="leaky_relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="leaky_relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def simple_stride_model(wt_seq, channel_num, model_name="simple_stride_model"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(
        filters=32, kernel_size=1, strides=1, padding="same", activation="leaky_relu"
    )(inputs)
    x = layers.Conv2D(
        filters=64, kernel_size=2, strides=2, padding="same", activation="leaky_relu"
    )(x)
    x = layers.Conv2D(
        filters=128, kernel_size=2, strides=2, padding="same", activation="leaky_relu"
    )(x)
    x = layers.Conv2D(
        filters=64, kernel_size=2, strides=2, padding="same", activation="leaky_relu"
    )(x)
    x = layers.Conv2D(
        filters=64, kernel_size=3, strides=3, padding="same", activation="leaky_relu"
    )(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def autoencoder(wt_seq, channel_num, model_name="autoencoder"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    # Encoder
    x = layers.Conv2D(16, 3, activation="leaky_relu", padding="same")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, activation="leaky_relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="leaky_relu", padding="same")(x)
    # x = layers.MaxPooling2D()(x)
    xl = layers.MaxPooling2D()(x)
    xf = layers.Flatten()(xl)
    x = layers.Dense(256, activation="leaky_relu")(xf)
    # x = layers.Dense(128, activation='leaky_relu')(x)

    # Decoder
    # x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(xf.shape[1], activation="leaky_relu")(x)
    x = layers.Reshape((xl.shape[1], xl.shape[2], xl.shape[3]))(x)
    x = layers.Conv2DTranspose(
        64, 3, strides=2, activation="leaky_relu", padding="same"
    )(x)
    x = layers.Conv2DTranspose(
        32, 3, strides=2, activation="leaky_relu", padding="same"
    )(x)
    x = layers.Conv2DTranspose(
        16, 3, strides=2, activation="leaky_relu", padding="same"
    )(x)
    x = layers.Resizing(len(wt_seq), len(wt_seq))(x)
    output = layers.Conv2D(6, 3, activation="tanh", padding="same")(x)

    # Autoencoder
    ae = keras.Model(inputs, output)
    ae.summary()
    return ae


def ae_conv(wt_seq, channel_num, model_name="ae_conv"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(128, 3, padding="same", activation="leaky_relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="leaky_relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="leaky_relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    # x = layers.GlobalAvgPool2D()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)  # 128
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1, activation="leaky_relu")(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def sep_conv(wt_seq, channel_num, model_name="sep_conv"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.SeparableConv2D(16, 3, padding="same", activation="leaky_relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.SeparableConv2D(32, 3, padding="same", activation="leaky_relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.SeparableConv2D(64, 3, padding="same", activation="leaky_relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.SeparableConv2D(128, 3, padding="same", activation="leaky_relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.SeparableConv2D(256, 3, padding="same", activation="leaky_relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.SeparableConv2D(512, 3, padding="same", activation="leaky_relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.SeparableConv2D(1024, 3, padding="same", activation="leaky_relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)  # 128
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1, activation="leaky_relu")(x)
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
    wt_seq, channel_num, filters=128, depth=8, kernel_size=5, patch_size=2
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


def depth_conv(
    wt_seq,
    channel_num,
    filters=128,
    blocks=5,
    kernel_size=5,
    patch_size=2,
    model_name="depth_conv",
):
    def res(prev_in):
        y0 = layers.DepthwiseConv2D(
            kernel_size=kernel_size, padding="same", activation="leaky_relu"
        )(prev_in)
        y0 = layers.Add()([y0, prev_in])
        y0 = layers.BatchNormalization()(y0)
        yp = layers.Conv2D(filters, kernel_size=1, activation="leaky_relu")(y0)
        y = layers.BatchNormalization()(yp)
        return y

    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(
        filters, kernel_size=patch_size, strides=patch_size, activation="leaky_relu"
    )(inputs)

    for i in range(blocks):
        x = res(x)

    x = layers.GlobalAvgPool2D()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1, activation="leaky_relu")(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def sep_stride(wt_seq, channel_num, model_name="sep_stride"):
    def re_block(prev_in):
        y0 = layers.SeparableConv2D(
            128, 3, strides=1, padding="same", activation="leaky_relu"
        )(prev_in)
        y = layers.SeparableConv2D(
            128, 3, strides=1, padding="same", activation="leaky_relu"
        )(y0)
        y = layers.Add()([y0, y])
        y = layers.BatchNormalization()(y)
        return y

    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = re_block(inputs)
    for i in range(5):
        x = re_block(x)
    x = layers.GlobalAvgPool2D()(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1, activation="leaky_relu")(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


# sequence


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def seq_transformer(vocab_size=21, maxlen=75, embed_dim=32, num_heads=6, ff_dim=64):
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    # x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation="leaky_relu")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model


def simple_depth(wt_seq, channel_num, model_name="simple_separable"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x0 = layers.Conv2D(
        filters=256, strides=2, kernel_size=2, padding="same", activation="leaky_relu"
    )(inputs)
    x = layers.BatchNormalization()(x0)
    x = layers.DepthwiseConv2D(kernel_size=7, padding="same", activation="leaky_relu")(
        x
    )
    x = layers.BatchNormalization()(x)
    x = layers.DepthwiseConv2D(kernel_size=7, padding="same", activation="leaky_relu")(
        x
    )
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x0, x])
    x = layers.GlobalAvgPool2D()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def simple_model_evo(wt_seq, channel_num, model_name="simple_model_evo"):
    seq_len = len(wt_seq)
    evo_layer_size = int(seq_len * 1.2)

    evo_inputs = keras.Input(shape=(seq_len,), name="evo")
    e = layers.Dense(evo_layer_size, activation="leaky_relu")(evo_inputs)
    # e = layers.Dense(evo_layer_size, activation='leaky_relu')(e)
    e = layers.Dense(evo_layer_size, activation="leaky_relu")(e)
    # e = layers.Dense(8, activation='leaky_relu')(e)

    inputs = keras.Input(shape=(seq_len, seq_len, channel_num), name="structure")
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

    x = layers.concatenate([x, e])

    outputs = layers.Dense(1, activation="leaky_relu")(x)
    model = keras.Model([inputs, evo_inputs], outputs, name=model_name + "_")
    model.summary()
    return model


def seq_model(wt_seq, model_name="seq_model"):
    inputs = keras.Input(shape=(len(wt_seq), 20, 1), name=model_name)
    x = layers.Conv2D(16, 3, padding="same", activation="leaky_relu")(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="leaky_relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="leaky_relu")(x)
    oh_output = layers.Conv2D(1, 3, padding="same", activation="sigmoid")(x)
    f = layers.Flatten()(oh_output)
    s = layers.Dense(128, activation="leaky_relu")(f)
    s = layers.Dense(128, activation="leaky_relu")(s)
    s = layers.Dense(64, activation="leaky_relu")(s)
    s = layers.Dense(1)(s)
    model = keras.Model(inputs, [oh_output, s], name=model_name + "_")
    model.summary()
    return model


def simple_model_row(wt_seq, channel_num, model_name="simple_model_row"):
    wt_len = len(wt_seq)
    inputs = keras.Input(shape=(wt_len, wt_len, channel_num), name=model_name)
    x = layers.Conv2D(16, [10, 1], activation="leaky_relu", padding="same")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, [10, 1], activation="leaky_relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, [10, 1], activation="leaky_relu", padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    # x = layers.GlobalAvgPool2D()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(256, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1, activation="leaky_relu")(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def simple_model_test(wt_seq, channel_num, model_name="simple_model_imp"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)

    def block(prev_in):
        x1 = layers.Conv2D(128, 3, padding="same", activation="leaky_relu")(prev_in)
        xx1 = layers.Conv2D(128, 3, padding="same", activation="leaky_relu")(x1)
        xx1 = layers.Conv2D(128, 3, padding="same", activation="leaky_relu")(xx1)
        x1 = layers.Add()([x1, xx1])
        return x1

    x = block(inputs)
    for i in range(5):
        x = layers.MaxPooling2D()(x)
        x = block(x)
    # x = layers.GlobalAvgPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    # x = layers.Dense(256, activation='leaky_relu')(x)
    # x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1, activation="leaky_relu")(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def dense_net(wt_seq, channel_num, model_name="dense_net"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)

    def dense_block(prev_in):
        d0 = layers.Conv2D(12, 3, padding="same", activation="leaky_relu")(prev_in)
        d1 = layers.concatenate([prev_in, d0])
        d1 = layers.Conv2D(128, 1, padding="same", activation="leaky_relu")(d1)  #
        d1 = layers.Conv2D(12, 3, padding="same", activation="leaky_relu")(d1)
        d2 = layers.concatenate([prev_in, d0, d1])
        d2 = layers.Conv2D(128, 1, padding="same", activation="leaky_relu")(d2)  #
        d2 = layers.Conv2D(12, 3, padding="same", activation="leaky_relu")(d2)
        d3 = layers.concatenate([prev_in, d0, d1, d2])
        d3 = layers.Conv2D(128, 1, padding="same", activation="leaky_relu")(d3)  #
        d3 = layers.Conv2D(12, 3, padding="same", activation="leaky_relu")(d3)
        d3 = layers.concatenate([prev_in, d0, d1, d2, d3])
        return d3

    x = dense_block(inputs)
    for i in range(2):
        # x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, 1, padding="same", activation="leaky_relu")(x)
        x = layers.AvgPool2D()(x)
        x = dense_block(x)
    x = layers.Conv2D(256, 1, padding="same", activation="leaky_relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def dense_net2(wt_seq, channel_num, model_name="dense_net"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)

    def d_layer(p_in):
        # dl = layers.BatchNormalization()(p_in)
        dl = layers.Conv2D(128, 1, padding="same", activation="leaky_relu")(p_in)
        dl = layers.Conv2D(12, 3, padding="same", activation="leaky_relu")(dl)
        return dl

    def d_block(prev_in):
        temp = prev_in
        for i in range(4):
            x = d_layer(temp)
            temp = layers.concatenate([x, temp])
        return temp

    x = d_block(inputs)
    for i in range(4):
        # x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, 1, padding="same")(x)
        x = layers.AveragePooling2D()(x)
        x = d_block(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(128, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def dense_net2(
    wt_seq,
    channel_num,
    filter_num=12,
    block_num=4,
    block_depth=4,
    intro_layer=False,
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
        x = layers.Conv2D(128, 7, 2, padding="same")(inputs)
        x = layers.MaxPooling2D(3, 2)(x)
        x = d_block(x)

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
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def simple_model_imp_tune(
    wt_seq,
    channel_num,
    model_name="simple_model_imp_tune",
    times=2,
    num_blocks=3,
    filter_s=3,
    l_pool="max",
    layer_base_size=16,
    num_dense=4,
    dense_size=128,
    f_b="flat",
):

    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)

    def block(prev_in, depth, fs, pool_type):
        bx = layers.Conv2D(depth, fs, padding="same", activation="leaky_relu")(prev_in)
        if pool_type == "max":
            bx = layers.MaxPooling2D()(bx)
        else:
            bx = layers.AvgPool2D()(bx)
        return bx

    x = layers.Conv2D(
        layer_base_size, filter_s, padding="same", activation="leaky_relu"
    )(inputs)
    if l_pool == "max":
        x = layers.MaxPooling2D()(x)
    else:
        x = layers.AvgPool2D()(x)

    for i in range(1, num_blocks + 1):
        x = block(
            prev_in=x, depth=i * times * layer_base_size, fs=filter_s, pool_type=l_pool
        )
    if f_b == "flat":
        x = layers.Flatten()(x)
    else:
        x = layers.GlobalAvgPool2D()(x)

    for i in range(num_dense):
        x = layers.Dense(dense_size, activation="leaky_relu")(x)
    x = layers.Dense(64, activation="leaky_relu")(x)
    outputs = layers.Dense(1, activation="leaky_relu")(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


if __name__ == "__main__":
    mod = dense_test(np.arange(237), 7)
