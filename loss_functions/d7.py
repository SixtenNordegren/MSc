import numpy as np
import itertools
import tensorflow as tf


def LeviCivitaR5(dim):
    # Initialize a numpy array filled with zeros
    levi_civita = np.zeros((dim, dim, dim, dim, dim), dtype=int)

    # Iterate over all permutations of a given dimension
    for perm in itertools.permutations(range(dim)):
        # Determine the sign of the permutation
        inversions = sum(
            i < j
            for i, index_i in enumerate(perm)
            for j, index_j in enumerate(perm)
            if index_i > index_j
        )
        sign = (-1) ** inversions

        # Assign the sign to the corresponding position in the tensor
        levi_civita[perm] = sign

    return tf.constant(levi_civita, dtype=tf.float32)


LC = LeviCivitaR5(5)


class d7:
    def __init__(self):
        pass

    @tf.function
    def Y(self, y):
        values = [
            y[0],
            y[1],
            y[2],
            y[3],
            y[4],
            y[1],
            y[6],
            y[7],
            y[8],
            y[9],
            y[2],
            y[7],
            y[12],
            y[13],
            y[14],
            y[3],
            y[8],
            y[13],
            y[18],
            y[19],
            y[4],
            y[9],
            y[14],
            y[19],
            y[24],
        ]
        indices = [
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
        ]

        output = tf.sparse.SparseTensor(
            indices=indices, values=values, dense_shape=[5, 5]
        )
        output = tf.sparse.to_dense(output)
        return output

    @tf.function
    def Z(self, z):
        values = [
            z[0],
            z[1],
            z[2],
            z[3],
            z[4],
            z[5],
            z[6],
            z[7],
            z[8],
            z[9],
            z[10],
            z[11],
            z[12],
            z[13],
            z[14],
            z[15],
            z[16],
            z[17],
            z[18],
            z[19],
            -z[0],
            -z[1],
            -z[2],
            -z[3],
            -z[4],
            -z[2] + z[6],
            z[21],
            z[22],
            z[23],
            z[24],
            -z[3] + z[11],
            z[26],
            z[27],
            z[28],
            z[29],
            -z[4] + z[16],
            z[31],
            z[32],
            z[33],
            z[34],
            -z[5],
            -z[6],
            -z[7],
            -z[8],
            -z[9],
            z[2] - z[6],
            -z[21],
            -z[22],
            -z[23],
            -z[24],
            -z[8] + z[12],
            -z[23] + z[27],
            z[37],
            z[38],
            z[39],
            -z[9] + z[17],
            -z[24] + z[32],
            z[42],
            z[43],
            z[44],
            -z[10],
            -z[11],
            -z[12],
            -z[13],
            -z[14],
            z[3] - z[11],
            -z[26],
            -z[27],
            -z[28],
            -z[29],
            z[8] - z[12],
            z[23] - z[27],
            -z[37],
            -z[38],
            -z[39],
            -z[14] + z[18],
            -z[29] + z[33],
            -z[39] + z[43],
            z[3],
            z[4],
            -z[15],
            -z[16],
            -z[17],
            -z[18],
            -z[19],
            z[4] - z[16],
            -z[31],
            -z[32],
            -z[33],
            -z[34],
            z[9] - z[17],
            z[24] - z[32],
            -z[42],
            -z[43],
            -z[44],
            z[14] - z[18],
            z[29] - z[33],
            z[39] - z[43],
            -z[3],
            -z[4],
        ]
        indices = [
            (0, 1, 0),
            (0, 1, 1),
            (0, 1, 2),
            (0, 1, 3),
            (0, 1, 4),
            (0, 2, 0),
            (0, 2, 1),
            (0, 2, 2),
            (0, 2, 3),
            (0, 2, 4),
            (0, 3, 0),
            (0, 3, 1),
            (0, 3, 2),
            (0, 3, 3),
            (0, 3, 4),
            (0, 4, 0),
            (0, 4, 1),
            (0, 4, 2),
            (0, 4, 3),
            (0, 4, 4),
            (1, 0, 0),
            (1, 0, 1),
            (1, 0, 2),
            (1, 0, 3),
            (1, 0, 4),
            (1, 2, 0),
            (1, 2, 1),
            (1, 2, 2),
            (1, 2, 3),
            (1, 2, 4),
            (1, 3, 0),
            (1, 3, 1),
            (1, 3, 2),
            (1, 3, 3),
            (1, 3, 4),
            (1, 4, 0),
            (1, 4, 1),
            (1, 4, 2),
            (1, 4, 3),
            (1, 4, 4),
            (2, 0, 0),
            (2, 0, 1),
            (2, 0, 2),
            (2, 0, 3),
            (2, 0, 4),
            (2, 1, 0),
            (2, 1, 1),
            (2, 1, 2),
            (2, 1, 3),
            (2, 1, 4),
            (2, 3, 0),
            (2, 3, 1),
            (2, 3, 2),
            (2, 3, 3),
            (2, 3, 4),
            (2, 4, 0),
            (2, 4, 1),
            (2, 4, 2),
            (2, 4, 3),
            (2, 4, 4),
            (3, 0, 0),
            (3, 0, 1),
            (3, 0, 2),
            (3, 0, 3),
            (3, 0, 4),
            (3, 1, 0),
            (3, 1, 1),
            (3, 1, 2),
            (3, 1, 3),
            (3, 1, 4),
            (3, 2, 0),
            (3, 2, 1),
            (3, 2, 2),
            (3, 2, 3),
            (3, 2, 4),
            (3, 4, 0),
            (3, 4, 1),
            (3, 4, 2),
            (3, 4, 3),
            (3, 4, 4),
            (4, 0, 0),
            (4, 0, 1),
            (4, 0, 2),
            (4, 0, 3),
            (4, 0, 4),
            (4, 1, 0),
            (4, 1, 1),
            (4, 1, 2),
            (4, 1, 3),
            (4, 1, 4),
            (4, 2, 0),
            (4, 2, 1),
            (4, 2, 2),
            (4, 2, 3),
            (4, 2, 4),
            (4, 3, 0),
            (4, 3, 1),
            (4, 3, 2),
            (4, 3, 3),
            (4, 3, 4),
        ]

        output = tf.sparse.SparseTensor(
            indices=indices, values=values, dense_shape=[5, 5, 5]
        )
        output = tf.sparse.to_dense(output)
        return output

    @tf.function
    def X(self, y, z):
        dim = 5
        delta = tf.eye(dim, dtype=tf.float32)
        x = 1 / 2 * (
            tf.einsum("QM,NP->QMNP", delta, self.Y(y))
            - tf.einsum("QN,MP->QMNP", delta, self.Y(y))
        ) - 2 * tf.einsum("MNPRS,RSQ->MNPQ", LC, self.Z(z))
        return x

    @tf.function
    def QC(self, y, z):
        """
        Returns the quadratic constraint
        """
        qc = tf.einsum("MQ,QNP->MNP", self.Y(y), self.Z(z)) + 2 * tf.einsum(
            "MRSTU,RSN,TUP->MNP", LC, self.Z(z), self.Z(z)
        )
        return qc

    @tf.function
    def loss_func(self, y, z):
        loss = tf.einsum("MNP,MNP->", self.QC(y, z), self.QC(y, z))
        return loss

    def inputs(self):
        return (25, 50)

    def tensors(self):
        return (self.Y, self.Z)


if __name__ == "__main__":
    print(tf.einsum("ABCDE,ABCDE->", LC, LC))
