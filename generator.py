import tensorflow as tf
from layers import ConditionalBatchNorm2D, GeneratorBlock
import tensorflow.keras.layers as layers


GEN_SIZE = 64


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()

        self.dense = layers.Dense(units=4*4*GEN_SIZE*16, activation=None)
        self._final = layers.Conv2D(filters=1, kernel_size=(3,3), padding='same', activation='sigmoid')
        self.batch_norm = layers.BatchNormalization()
        self.relu = layers.ReLU()

        self.genblock0 = GeneratorBlock(in_channels=GEN_SIZE*16, out_channels=GEN_SIZE*8)
        self.genblock1 = GeneratorBlock(in_channels=GEN_SIZE*8, out_channels=GEN_SIZE*4)
        self.genblock2 = GeneratorBlock(in_channels=GEN_SIZE*4, out_channels=GEN_SIZE*2)
        self.genblock3 = GeneratorBlock(in_channels=GEN_SIZE*2, out_channels=GEN_SIZE)

        self.final = tf.keras.Sequential([
        self.batch_norm, self.relu,
        self._final
        ])


    def call(self, inputs):
        z, y = inputs
        out = self.dense(z)
        out = tf.reshape(out, [-1, 4, 4, GEN_SIZE*16])
        out = self.genblock0((out,y))
        out = self.genblock1((out,y))
        out = self.genblock2((out,y))
        out = self.genblock3((out,y))
        out = self.final(out)

        return out
