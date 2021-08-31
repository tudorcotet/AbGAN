import tensorflow as tf
from tensorflow_addons.layers import GroupNormalization



DIM_EMBED = 128
#Need to implement image-to-h and y-to-h mapping

class LabelEmbedding(tf.keras.Model): #y to h mapping
    def __init__(self, dim_embed=DIM_EMBED):
        super(LabelEmbedding, self).__init__()

        self.model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=dim_embed),
        GroupNormalization(groups=8),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Dense(units=dim_embed),
        GroupNormalization(groups=8),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Dense(units=dim_embed),
        GroupNormalization(groups=8),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Dense(units=dim_embed),
        GroupNormalization(groups=8),
        tf.keras.layers.ReLU(),

        tf.keras.layers.Dense(units=dim_embed, activation='relu')
        ])


    def call(self, y):
        return self.model(y)
