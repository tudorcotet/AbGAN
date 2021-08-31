import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow_addons.layers import SpectralNormalization
from layers import DiscriminatorBlock



DISC_SIZE = 64

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.relu = layers.ReLU()


        self.discblock1 = tf.keras.Sequential([
        DiscriminatorBlock(DISC_SIZE, stride=2, first_block=True),
        DiscriminatorBlock(DISC_SIZE*2, stride=2),
        DiscriminatorBlock(DISC_SIZE*4, stride=2),
        ])

        self.discblock2 = tf.keras.Sequential([
        DiscriminatorBlock(DISC_SIZE*8, stride=2),
        ])

        self.discblock3 = tf.keras.Sequential([
        DiscriminatorBlock(DISC_SIZE*16, stride=1),
        self.relu
        ])


        self.linear1 = SpectralNormalization(layers.Dense(units=1))
        self.linear2 = SpectralNormalization(layers.Dense(units=DISC_SIZE*16*4*4, use_bias=False))


    def call(self, x, y):
        out = self.discblock1(x)
        out = self.discblock2(out)
        out = self.discblock3(out)

        out = tf.reshape(out, [-1,DISC_SIZE*16*4*4])
        out_y = tf.reduce_sum(out*self.linear2(y), 1, keepdims=True)
        out = self.linear1(out) + out_y

        out = tf.reshape(out, [-1,1])

        return out
