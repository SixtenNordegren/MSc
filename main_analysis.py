import numpy
import tensorflow as tf
from loss_functions.d7 import d7

theory = d7()
file = numpy.load(
    "/home/sixten/Projects/GDS_saves/test_file.txt.npz", allow_pickle=True
)["arr_0"]


def is_different(tensor_1, tensor_2, tol=1e-7):
    delta = tf.norm(tensor_1 - tensor_2)
    if delta > tol:
        print("is the sasame")
    else:
        print("is different")
