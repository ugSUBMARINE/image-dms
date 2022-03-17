import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def simple_model(wt_seq, channel_num, model_name="simple_model"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(16, 3, padding='same', activation='leaky_relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='leaky_relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='leaky_relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='leaky_relu')(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def simple_model_norm(wt_seq, channel_num, model_name="simple_model_norm"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(16, 3, padding='same', activation='leaky_relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='leaky_relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='leaky_relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='leaky_relu')(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(64, activation='leaky_relu')(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def simple_model_imp(wt_seq, channel_num, model_name="simple_model_imp"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(16, 3, padding='same', activation='leaky_relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='leaky_relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='leaky_relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    # x = layers.GlobalAvgPool2D()(x)
    x = layers.Dense(128, activation='leaky_relu')(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(64, activation='leaky_relu')(x)
    outputs = layers.Dense(1, activation='leaky_relu')(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def simple_model_128(wt_seq, channel_num, model_name="simple_model_imp"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(128, 3, padding='same', activation='leaky_relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='leaky_relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='leaky_relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalMaxPool2D()(x)
    x = layers.Dense(128, activation='leaky_relu')(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(64, activation='leaky_relu')(x)
    outputs = layers.Dense(1, activation='leaky_relu')(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def create_simple_model(wt_seq, channel_num, model_name="create_simple_model"):
    inputs = keras.Input(shape=(235, 40, 1), name=model_name)
    x = layers.Conv2D(filters=128, kernel_size=[3, inputs.shape[2]], strides=[1, 1], padding='valid',
                      activation='leaky_relu', use_bias=True)(inputs)
    x = layers.Conv2D(filters=128, kernel_size=[3, x.shape[2]], strides=[1, 1], padding='valid',
                      activation='leaky_relu', use_bias=True)(x)
    x = layers.Conv2D(filters=128, kernel_size=[3, x.shape[2]], strides=[1, 1], padding='valid',
                      activation='leaky_relu', use_bias=True)(x)
    x = layers.Conv2D(filters=128, kernel_size=[3, x.shape[2]], strides=[1, 1], padding='valid',
                      activation='leaky_relu', use_bias=True)(x)
    x = layers.Conv2D(filters=128, kernel_size=[3, x.shape[2]], strides=[1, 1], padding='valid',
                      activation='leaky_relu', use_bias=True)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100, activation='leaky_relu', use_bias=True)(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, use_bias=True)(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def simple_model_gap(wt_seq, channel_num, model_name="simple_model_gap"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(16, 3, padding='same', activation='leaky_relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='leaky_relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='leaky_relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='leaky_relu')(x)
    x = layers.GlobalMaxPool2D()(x)
    x = layers.Dense(512, activation='leaky_relu')(x)
    x = layers.Dense(128, activation='leaky_relu')(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    return model


def simple_stride_model_test(wt_seq, channel_num, model_name="simple_stride_model_test"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='leaky_relu')(inputs)
    x = layers.Conv2D(filters=64, kernel_size=2, strides=2, padding='same', activation='leaky_relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=2, strides=2, padding='same', activation='leaky_relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=2, strides=2, padding='same', activation='leaky_relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=2, strides=2, padding='same', activation='leaky_relu')(x)
    # x = layers.Conv2D(filters=256, kernel_size=2, strides=2, padding='same', activation='leaky_relu')(x)
    # x = layers.Conv2D(filters=64, kernel_size=2, strides=2, padding='same', activation='leaky_relu')(x)
    # x = layers.Conv2D(filters=64, kernel_size=3, strides=3, padding='same', activation='leaky_relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='leaky_relu')(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(64, activation='leaky_relu')(x)
    outputs = layers.Dense(1, activation='leaky_relu')(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def shrinking_res(wt_seq, channel_num, blocks=4, model_name="shrinking_res"):
    def block(prev_output, count, num_convs=4, block_depth=2, reduce_size=True):
        kernel_list = [3, 3, 5, 7]
        filter_list = [32, 32, 32, 32]
        missing_filter = abs(len(filter_list) - blocks)
        if missing_filter > 0:
            for m in range(missing_filter):
                filter_list += [filter_list[-1]]
        filter_size = filter_list[count - 1]

        b2 = layers.Conv2D(filters=filter_size, kernel_size=2, strides=2, padding="same", activation="leaky_relu") \
            (prev_output)

        diff_kl_convs = len(kernel_list) - num_convs
        if diff_kl_convs != 0:
            for j in range(abs(diff_kl_convs)):
                kernel_list.insert(0, kernel_list[0])

        layer_list = [b2]
        for k in range(num_convs):
            li = layers.Conv2D(filters=filter_size, kernel_size=kernel_list[k], strides=1, padding="same",
                               activation="leaky_relu")(b2)
            if block_depth > 1:
                for f in range(block_depth - 1):
                    li = layers.Conv2D(filters=filter_size, kernel_size=kernel_list[k], strides=1, padding="same",
                                       activation="leaky_relu")(li)
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
                layer_list = layer_list[:layer_count - count]
        b_out = layers.add(layer_list)
        return b_out

    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    # x = layers.Conv2D(filters=256, kernel_size=1, strides=1, padding="same", activation="leaky_relu")(inputs)
    x = block(inputs, count=0)
    for i in range(blocks):
        x = block(x, count=i + 1)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='leaky_relu')(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(64, activation='leaky_relu')(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()

    return model


def inception_res(wt_seq, channel_num, model_name="inception_res"):
    def block(prev_out):
        b2 = layers.Conv2D(filters=32, kernel_size=2, strides=2, padding="same", activation="leaky_relu")(prev_out)
        c1 = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding="same", activation="leaky_relu")(b2)
        c11 = layers.Conv2D(filters=16, kernel_size=5, strides=1, padding="same", activation="leaky_relu")(c1)
        c2 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="leaky_relu")(b2)
        c22 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same", activation="leaky_relu")(c2)
        c3 = layers.Conv2D(filters=32, kernel_size=1, strides=1, padding="same", activation="leaky_relu")(b2)
        c33 = layers.Conv2D(filters=32, kernel_size=1, strides=1, padding="same", activation="leaky_relu")(c3)
        b_out = layers.concatenate([c11, c22, c33])
        return b_out

    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding="same", activation="leaky_relu")(inputs)
    x = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding="same", activation="leaky_relu")(x)
    x = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding="same", activation="leaky_relu")(x)

    # x = block(inputs)
    for i in range(5):
        x = block(x)

    # x = layers.Conv2D(filters=32, kernel_size=1, strides=1, padding="same", activation="leaky_relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='leaky_relu')(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(64, activation='leaky_relu')(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def deeper_res(wt_seq, channel_num, model_name="deeper_res"):
    def block(prev_out):
        b2 = layers.Conv2D(filters=32, kernel_size=2, strides=2, padding="same", activation="leaky_relu")(prev_out)
        i1 = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation="leaky_relu")(b2)
        i2 = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation="leaky_relu")(b2)
        i3 = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation="leaky_relu")(b2)
        i12 = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation="leaky_relu")(i1)
        i22 = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation="leaky_relu")(i2)
        i32 = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation="leaky_relu")(i3)
        return layers.add([b2, i12, i22, i32])

    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(filters=128, kernel_size=1, strides=1, padding="same", activation="leaky_relu")(inputs)
    x = block(x)
    x = block(x)
    x = block(x)
    x = block(x)
    x = block(x)
    x = block(x)
    x = block(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='leaky_relu')(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(64, activation='leaky_relu')(x)
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
            merge_input = layers.Conv2D(n_filters, (1, 1), padding='same', activation='leaky_relu',
                                        kernel_initializer='he_normal')(layer_in)
        # conv1
        conv1 = layers.Conv2D(n_filters, (3, 3), padding='same', activation='leaky_relu',
                              kernel_initializer='he_normal')(layer_in)
        # conv2
        conv2 = layers.Conv2D(n_filters, (3, 3), padding='same', activation='linear',
                              kernel_initializer='he_normal')(conv1)
        # add filters, assumes filters/channels last
        layer_out = layers.add([conv2, merge_input])
        # activation function
        layer_out = layers.Activation('leaky_relu')(layer_out)
        return layer_out

    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(16, 3, padding='same', activation='leaky_relu')(inputs)
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
    x = layers.Dense(128, activation='leaky_relu')(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(64, activation='leaky_relu')(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def vgg(wt_seq, channel_num, model_name="vgg"):
    def vgg_block(layer_in, n_filters, n_conv):
        for _ in range(n_conv):
            layer_in = layers.Conv2D(n_filters, (3, 3), padding='same', activation='leaky_relu')(layer_in)
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
    x = layers.Dense(128, activation='leaky_relu')(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(64, activation='leaky_relu')(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def simple_longer(wt_seq, channel_num, model_name="simple_longer"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(16, 3, padding='same', activation='leaky_relu')(inputs)
    x = layers.Conv2D(16, 3, padding='same', activation='leaky_relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='leaky_relu')(x)
    x = layers.Conv2D(32, 3, padding='same', activation='leaky_relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='leaky_relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='leaky_relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='leaky_relu')(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(64, activation='leaky_relu')(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


def simple_stride_model(wt_seq, channel_num, model_name="simple_stride_model"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='leaky_relu')(inputs)
    x = layers.Conv2D(filters=64, kernel_size=2, strides=2, padding='same', activation='leaky_relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=2, strides=2, padding='same', activation='leaky_relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=2, strides=2, padding='same', activation='leaky_relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=3, padding='same', activation='leaky_relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='leaky_relu')(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(64, activation='leaky_relu')(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name=model_name + "_")
    model.summary()
    return model


if __name__ == "__main__":
    simple_model_128(np.arange(235), 6, )
