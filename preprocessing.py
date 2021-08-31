import tensorflow as tf
import numpy as np


aa_list = ["A", "C", "D", "E",
           "F", "G", "H", "I",
           "K", "L", "M", "N",
           "P", "Q", "R", "S",
           "T", "W", "Y", "V"]


def aa_to_tensor(sequence, aa_list = aa_list):
    indices = [aa_list.index(aa) for aa in sequence]
    return tf.one_hot(indices, depth = len(aa_list))


def tensor_to_aa(tensor, aa_list = aa_list, as_list=False):
    indices = tf.argmax(tensor, axis=1).numpy()
    sequence = [aa_list[i] for i in indices]
    if as_list:
        return sequence
    else:
        return ''.join(sequence)

def pad_tensor_downwards(tensor, input_shape):
    assert tf.shape(tensor)[1] == input_shape[1], "Padding only on sequence length, not for different AAs"
    paddings = [[0, input_shape[0]-tf.shape(hot)[0]], [0,0]]
    out = tf.pad(hot, paddings, 'CONSTANT', constant_values=0)
    return out

def pad_tensor_variable(tensor, image_size=[64,64]):
    left_right = int((image_size[1] - tf.shape(tensor)[1].numpy())/2)
    paddings = [[0, image_size[0] - tf.shape(tensor)[0].numpy()], [left_right, left_right]]
    return tf.pad(tensor, paddings)

def minmax_label_normalize(labels):
    return (labels - np.min(labels)) / (np.max(labels) - np.min(labels))

def z_score_label_normalize(labels):
    return (labels - np.mean(labels)) / np.std(labels)
