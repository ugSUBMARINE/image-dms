import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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


def create_simple_model(wt_seq, channel_num, model_name="simple_model"):
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


def simple_stride_model_test(wt_seq, channel_num, model_name="simple_model_test"):
    inputs = keras.Input(shape=(len(wt_seq), len(wt_seq), channel_num), name=model_name)
    x = layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation='leaky_relu')(inputs)
    x = layers.Conv2D(filters=64, kernel_size=2, strides=2, padding='same', activation='leaky_relu')(x)
    x = layers.Conv2D(filters=128, kernel_size=2, strides=2, padding='same', activation='leaky_relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=2, strides=2, padding='same', activation='leaky_relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=3, padding='same', activation='leaky_relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='leaky_relu')(x)
    x = layers.Dropout(0.05)(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dropout(0.05)(x)
    x = layers.Dense(256, activation='leaky_relu')(x)
    x = layers.Dropout(0.05)(x)
    x = layers.Dense(64, activation='leaky_relu')(x)
    x = layers.Dropout(0.05)(x)
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
    simple_stride_model_test(np.arange(250), 5, "simple")
    print(str(simple_stride_model).split(" ")[1])
