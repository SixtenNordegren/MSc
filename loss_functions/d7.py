import numpy as np
import itertools
import tensorflow as tf


def generators():
    """Creates the generators for SL(5)/SU(5), specifically the positive roots and the cartans."""
    dim = 5
    positive_roots = []
    for i in range(dim):
        for j in range(dim):
            if i > j:
                gen = np.zeros((dim, dim), dtype=np.float32)
                gen[i, j] = 1.0
                gen = gen + np.transpose(gen)
                positive_roots.append(gen)

    cartans = [
        np.diag(np.roll((-4.0, 1.0, 1.0, 1.0, 1.0), index)) for index in range(dim)
    ][:4]
    generators = positive_roots + cartans
    generators = tf.constant(np.stack(generators, axis=0), dtype=tf.float32)

    return generators


def LeviCivitaR5(dim):
    # Initialize a numpy array filled with zeros
    levi_civita = np.zeros((dim, dim, dim, dim, dim), dtype=np.float32)

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
t = generators()


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
        output = tf.cast(tf.sparse.to_dense(output), dtype=tf.float32)
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
        output = tf.cast(tf.sparse.to_dense(output), dtype=tf.float32)
        return output

    # @tf.function
    # def X(self, y, z):
    #     dim = 5
    #     delta = tf.eye(dim, dtype=tf.float32)
    #     x = 1 / 2 * (
    #         tf.einsum("QM,NP->QMNP", delta, self.Y(y))
    #         - tf.einsum("QN,MP->QMNP", delta, self.Y(y))
    #     ) - 2 * tf.einsum("MNPRS,RSQ->MNPQ", LC, self.Z(z))
    #     return x

    @tf.function
    def QC(self, y, z):
        """
        Returns the quadratic constraint
        """
        return tf.einsum("MQ,QNP->MNP", self.Y(y), self.Z(z)) + 2 * tf.einsum(
            "MRSTU,RSN,TUP->MNP", LC, self.Z(z), self.Z(z)
        )

    @tf.function
    def hessian(self, y, z):
        # Be careful with i and j indices here, make sure that they
        # have the 14 index free
        tt = tf.einsum("jMA,iAQ->jMiQ", t, t)
        tt_sym = 0.5 * (tt + tf.einsum("iMjQ->jMiQ", tt))

        A = 0.125 * (
            2 * tf.einsum("iMjN,NP,PM->ij", tt_sym, self.Y(y), self.Y(y))
            + 2 * tf.einsum("iMN,NP,jPQ,QM->ij", t, self.Y(y), t, self.Y(y))
            - tf.einsum("iMjN,MN,PP->ij", tt_sym, self.Y(y), self.Y(y))
            - tf.einsum("iMN,MN,jPQ,PQ->ij", t, self.Y(y), t, self.Y(y))
        )

        B = 8 * (
            (
                tf.einsum("MRP,QRP,iMjQ->ij", self.Z(z), self.Z(z), tt_sym)
                + tf.einsum("MNP,QRP,iMQ,jNR->ij", self.Z(z), self.Z(z), t, t)
                + tf.einsum("MRP,QRS,iMQ,jPS->ij", self.Z(z), self.Z(z), t, t)
                + tf.einsum("RMP,RQP,iMjQ->ij", self.Z(z), self.Z(z), tt_sym)
                + tf.einsum("NMP,RQP,iMQ,jNR->ij", self.Z(z), self.Z(z), t, t)
                + tf.einsum("RMP,RQS,iMQ,jPS->ij", self.Z(z), self.Z(z), t, t)
                + tf.einsum("PRM,PRQ,iMjQ->ij", self.Z(z), self.Z(z), tt_sym)
                + tf.einsum("PNM,PRQ,iMQ,jNR->ij", self.Z(z), self.Z(z), t, t)
                + tf.einsum("PRM,SRQ,iMQ,jPS->ij", self.Z(z), self.Z(z), t, t)
            )
            - (
                tf.einsum("MRR,QPP,iMjQ->ij", self.Z(z), self.Z(z), tt_sym)
                + tf.einsum("MNR,QPP,iMQ,jNR->ij", self.Z(z), self.Z(z), t, t)
                + tf.einsum("MRR,QPS,iMQ,jPS->ij", self.Z(z), self.Z(z), t, t)
                + tf.einsum("RMQ,RPP,iMjQ->ij", self.Z(z), self.Z(z), tt_sym)
                + tf.einsum("NMQ,RPP,iMQ,jNR->ij", self.Z(z), self.Z(z), t, t)
                + tf.einsum("RMQ,RPS,iMQ,jPS->ij", self.Z(z), self.Z(z), t, t)
                + tf.einsum("PRR,PMQ,iMjQ->ij", self.Z(z), self.Z(z), tt_sym)
                + tf.einsum("PNR,PMQ,iMQ,jNR->ij", self.Z(z), self.Z(z), t, t)
                + tf.einsum("PRR,SMQ,iMQ,jPS->ij", self.Z(z), self.Z(z), t, t)
            )
        )
        return A + B

    @tf.function
    def grad_V(self, y, z):
        grad_V = 0.0625 * (
            2 * tf.einsum("iMN,NP,PM->i", t, self.Y(y), self.Y(y))
            - tf.einsum("iMN,NM,PP->i", t, self.Y(y), self.Y(y))
        ) + 4.0 * (
            tf.einsum("iMQ,MRP,QRP->i", t, self.Z(z), self.Z(z))
            + tf.einsum("iMQ,RMP,RQP->i", t, self.Z(z), self.Z(z))
            + tf.einsum("iMQ,PRM,PRQ->i", t, self.Z(z), self.Z(z))
            - (
                tf.einsum("iMQ,MRR,QPP->i", t, self.Z(z), self.Z(z))
                + tf.einsum("iMQ,RMQ,RPP->i", t, self.Z(z), self.Z(z))
                + tf.einsum("iMQ,PRR,PMQ->i", t, self.Z(z), self.Z(z))
            )
        )
        return grad_V

    # @tf.function
    def V(self, y, z):
        """
        TensorFlow version of the potential V function.
        Ensure that Y and Z functions are compatible with TensorFlow for this to work.
        """
        M = tf.eye(5, dtype=tf.float32)

        # Potential V calculation (adapted for TensorFlow)
        V = (1 / 64) * (
            2 * tf.einsum("MN,NP,PQ,QM->", M, self.Y(y), M, self.Y(y))
            - tf.einsum("MN,MN->", M, self.Y(y)) ** 2
            + tf.einsum("MNP,QRS,MQ,NR,PS->", self.Z(z), self.Z(z), M, M, M)
            - tf.einsum("MNP,QRS,MQ,NP,RS->", self.Z(z), self.Z(z), M, M, M)
        )
        return V

    @tf.function
    def loss_func(self, y, z):
        # Quadratic constraint squared
        qc2 = tf.einsum("MNP,MNP->", self.QC(y, z), self.QC(y, z))
        grad_V2 = tf.einsum("i,i->", self.grad_V(y, z), self.grad_V(y, z))
        # norm_X = (tf.einsum("MNPQ,MNPX", self.X(y, z), self.X(y, z))) ** 2
        # loss = tf.asinh(qc2)  # squashed loss
        loss = tf.asinh(qc2 + grad_V2)
        return loss

    def inputs(self):
        return (25, 50)

    def tensors(self):
        return (self.Y, self.Z)


if __name__ == "__main__":
    print(tf.einsum("ABCDE,ABCDE", LC, LC))
