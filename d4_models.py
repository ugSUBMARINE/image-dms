import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow_addons as tfa


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
    return model


def simple_model_imp(wt_seq, channel_num, model_name="simple_model"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(16, 3, padding='same', activation='leaky_relu', use_bias=True, bias_initializer="HeNormal")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='leaky_relu', use_bias=True, bias_initializer="HeNormal")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='leaky_relu', use_bias=True, bias_initializer="HeNormal")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='leaky_relu', use_bias=True, bias_initializer="HeNormal")(x)
    x = layers.Dense(256, activation='leaky_relu', use_bias=True, bias_initializer="HeNormal")(x)
    x = layers.Dense(256, activation='leaky_relu', use_bias=True, bias_initializer="HeNormal")(x)
    x = layers.Dense(64, activation='leaky_relu', use_bias=True, bias_initializer="HeNormal")(x)
    outputs = layers.Dense(1, activation='leaky_relu', use_bias=True, bias_initializer="HeNormal")(x)
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


def simple_model_gap(wt_seq, channel_num, model_name="simple_model"):
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
    x = layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same', activation='leaky_relu')(inputs)
    x = layers.Conv2D(filters=256, kernel_size=2, strides=2, padding='same', activation='leaky_relu')(x)
    x = layers.Conv2D(filters=256, kernel_size=2, strides=2, padding='same', activation='leaky_relu')(x)
    x = layers.Conv2D(filters=256, kernel_size=2, strides=2, padding='same', activation='leaky_relu')(x)
    x = layers.Conv2D(filters=256, kernel_size=2, strides=2, padding='same', activation='leaky_relu')(x)
    x = layers.Conv2D(filters=256, kernel_size=2, strides=2, padding='same', activation='leaky_relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=2, strides=2, padding='same', activation='leaky_relu')(x)
    # x = layers.Conv2D(filters=64, kernel_size=3, strides=3, padding='same', activation='leaky_relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='leaky_relu')(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dense(64, activation='leaky_relu')(x)
    outputs = layers.Dense(1)(x)
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
        filter_size = filter_list[count-1]

        b2 = layers.Conv2D(filters=filter_size, kernel_size=2, strides=2, padding="same", activation="leaky_relu",
                           name="SizeReduction{}".format(str(count)))(prev_output)

        diff_kl_convs = len(kernel_list) - num_convs
        if diff_kl_convs != 0:
            for j in range(abs(diff_kl_convs)):
                kernel_list.insert(0, kernel_list[0])

        layer_list = [b2]
        for k in range(num_convs):
            li = layers.Conv2D(filters=filter_size, kernel_size=kernel_list[k], strides=1, padding="same",
                               activation="leaky_relu", name="B{}K{}NC{}D0".format(str(count), str(kernel_list[k]),
                                                                                   str(k)))(b2)
            if block_depth > 1:
                for f in range(block_depth - 1):
                    li = layers.Conv2D(filters=filter_size, kernel_size=kernel_list[k], strides=1, padding="same",
                                       activation="leaky_relu", name="B{}K{}NC{}D{}".format(str(count),
                                                                                            str(kernel_list[k]),
                                                                                            str(k), str(f + 1)))(li)
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


def mlp(x, hidden_units, dropout_rate, activation_function=tf.nn.gelu):
    for units in hidden_units:
        x = layers.Dense(units, activation=activation_function)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def vit_classifier(num_classes, input_shape, patch_size, num_patches, projection_dim, num_heads,
                   transformer_units, transformer_layers, mlp_head_units):
    inputs = layers.Input(shape=input_shape)
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5, activation_function="leaky_relu")
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


# --------------------------VIT TEST-------------------------------------------------------------------
"""
num_classes_ex = 1
input_shape_ex = (len(wt_seq), len(wt_seq), channel_num)
image_size_ex = len(wt_seq)  # We'll resize input images to this size
patch_size_ex = 6  # Size of the patches to be extract from the input images
num_patches_ex = (image_size_ex // patch_size_ex) ** 2
projection_dim_ex = 64  # 64
num_heads_ex = 2  # 4
transformer_units_ex = [projection_dim_ex * 2, projection_dim_ex]  # Size of the transformer layers
transformer_layers_ex = 8  # 8
mlp_head_units_ex = [2048, 1024]  # Size of the dense layers of the final classifier
model = model_to_use(num_classes_ex, input_shape_ex, patch_size_ex, num_patches_ex, projection_dim_ex, num_heads_ex,
                     transformer_units_ex, transformer_layers_ex, mlp_head_units_ex)
model.compile(optimizer, loss="mean_absolute_error", metrics=["mae"])
"""
# -----------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    """
    num_classes_ex = 1
    # input_shape_ex = (len(wt_seq), len(wt_seq), 5)
    input_shape_ex = (75, 75, 5)

    learning_rate_ex = 0.001
    weight_decay_ex = 0.0001
    batch_size_ex = 64  # 256
    # image_size_ex = len(wt_seq)  # We'll resize input images to this size
    image_size_ex = 75  # We'll resize input images to this size
    patch_size_ex = 6  # Size of the patches to be extract from the input images
    num_patches_ex = (image_size_ex // patch_size_ex) ** 2
    projection_dim_ex = 64
    num_heads_ex = 4
    transformer_units_ex = [projection_dim_ex * 2, projection_dim_ex]  # Size of the transformer layers
    transformer_layers_ex = 8
    mlp_head_units_ex = [2048, 1024]  # Size of the dense layers of the final classifier
    model = vit_classifier(num_classes_ex, input_shape_ex, patch_size_ex, num_patches_ex, projection_dim_ex, num_heads_ex,
                   transformer_units_ex, transformer_layers_ex, mlp_head_units_ex)
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate_ex, weight_decay=weight_decay_ex)
    model.compile(optimizer=optimizer, loss="mean_absolute_error", metrics=["mae"])
    # tf.keras.utils.plot_model(model)
    """
    simple_model_imp(np.arange(235), 6, )


