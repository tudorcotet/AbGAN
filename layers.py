import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow_addons.layers import SpectralNormalization



class ConditionalBatchNorm2D(layers.Layer):
    def __init__(self, out_features):
        super(ConditionalBatchNorm2D, self).__init__(name='ConditionalBatchNorm')
        self.out_features = out_features
        self.batch_norm = layers.BatchNormalization(center=False, scale=False)

        self.embed_gamma = layers.Dense(units = out_features, activation=None, use_bias=False)
        self.embed_beta = layers.Dense(units = out_features, activation=None, use_bias=False)


    def call(self, inputs):
        x, y = inputs
        out = self.batch_norm(x)

        gamma = self.embed_gamma(y)
        gamma = tf.reshape(gamma, [-1, 1, 1, self.out_features])
        beta = self.embed_beta(y)
        beta = tf.reshape(beta, [-1, 1, 1, self.out_features])

        out = out + out*gamma + beta

        return out



class GeneratorBlock(layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(GeneratorBlock, self).__init__()

        self.conv1 = layers.Conv2D(filters=out_channels, kernel_size=(3,3), activation=None, padding='same')
        self.conv2 = layers.Conv2D(filters=out_channels, kernel_size=(3,3), activation=None, padding='same')
        self.relu = layers.ReLU()
        self.batch_norm = layers.BatchNormalization()
        self.cond_norm_1 = ConditionalBatchNorm2D(in_channels)
        self.cond_norm_2 = ConditionalBatchNorm2D(out_channels)
        self.upsample = layers.UpSampling2D(size=2)


        self.bypass_conv = layers.Conv2D(filters=out_channels, kernel_size=(1,1), activation=None)

        self.bypass = tf.keras.Sequential([
            self.upsample, self.bypass_conv
            ])


    def call(self, inputs):
        x, y = inputs

        out = self.cond_norm_1((x, y))
        out = self.relu(out)
        out = self.upsample(out)
        out = self.conv1(out)
        out = self.cond_norm_2((out, y))
        out = self.relu(out)
        out = self.conv2(out)
        out = out + self.bypass(x)


        return out



class DiscriminatorBlock(layers.Layer):
    def __init__(self, out_channels, stride=1, first_block = False, residual=True):
        super(DiscriminatorBlock, self).__init__()
        self.residual = residual

        self.conv1 = SpectralNormalization(layers.Conv2D(filters=out_channels, kernel_size=(3,3), activation=None, padding='same'))
        self.conv2 = SpectralNormalization(layers.Conv2D(filters=out_channels, kernel_size=(3,3), activation=None, padding='same'))
        self.bypass_conv = SpectralNormalization(layers.Conv2D(filters=out_channels, kernel_size=(1,1), activation=None))
        self.relu = layers.ReLU()

        if first_block:
            self.model = tf.keras.Sequential([
            self.conv1,
            self.relu,
            self.conv2,
            layers.AvgPool2D(pool_size=(2,2), strides=stride)
            ])

        else:
            if stride == 1:
                self.model = tf.keras.Sequential([
                self.relu,
                self.conv1,
                self.relu,
                self.conv2,
                ])
            else:
                self.model = tf.keras.Sequential([
                self.relu,
                self.conv1,
                self.relu,
                layers.AvgPool2D(pool_size=(2,2), strides=stride)
                ])

        if residual:
            if first_block:
                self.bypass = tf.keras.Sequential([
                layers.AvgPool2D(pool_size=(2,2), strides=stride),
                self.bypass_conv
                ])
            else:
                if stride != 1:
                    self.bypass = tf.keras.Sequential([
                    self.bypass_conv,
                    layers.AvgPool2D(pool_size=(2,2), strides=stride)
                    ])
                else:
                    self.bypass = tf.keras.Sequential([
                    self.bypass_conv
                    ])

    def call(self, x):
        if self.residual:
            return self.model(x) + self.bypass(x)
        else:
            return self.model(x)
