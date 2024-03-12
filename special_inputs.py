import tensorflow as tf
from func import *
from loss_functions.d7 import generators, LeviCivitaR5

LC = LeviCivitaR5(5)
t = generators()

Y = tf.eye(5, dtype=tf.float32)
Z = tf.zeros((5, 5, 5), dtype=tf.float32)


def V():
    """
    TensorFlow version of the potential V function.
    Ensure that Y and Z functions are compatible with TensorFlow for this to work.
    """
    M = tf.eye(5, dtype=tf.float32)

    # Potential V calculation (adapted for TensorFlow)
    V = (
        (1.0 / 64.0)
        * (
            2.0 * tf.einsum("MN,NP,PQ,QM->", M, Y, M, Y)
            - tf.einsum("MN,NM->", M, Y) ** 2
        )
        + tf.einsum("MNP,QRS,MQ,NR,PS->", Z, Z, M, M, M)
        - tf.einsum("MNP,QRS,MQ,NP,RS->", Z, Z, M, M, M)
    )

    return V


def hessian():
    # Be careful with i and j indices here, make sure that they
    # have the 14 index free
    tt = tf.einsum("iMA,jAQ->iMjQ", t, t)
    tt_sym = 0.5 * (tt + tf.einsum("iMjQ->jMiQ", tt))
    A = 0.125 * (
        2 * tf.einsum("iMjN,NP,PM->ij", tt_sym, Y, Y)
        + 2 * tf.einsum("iMN,NP,jPQ,QM->ij", t, Y, t, Y)
        - tf.einsum("iMjN,MN,PP->ij", tt_sym, Y, Y)
        - tf.einsum("iMN,MN,jPQ,PQ->ij", t, Y, t, Y)
    )
    print(t.shape, tt.shape, tt_sym.shape, A.shape)
    # A = 0.125 * (
    #     tf.einsum("iMjN,NP,PM->ij", tt, Y, Y)
    #     + tf.einsum("jMiN,NP,PM->ij", tt, Y, Y)
    #     - 0.5 * tf.einsum("iMjN,MN,PP->ji", tt, Y, Y)
    #     - 0.5 * tf.einsum("jMiN,MN,PP->ji", tt, Y, Y)
    #     + 2 * tf.einsum("iMN,NP,jPQ,QM->ji", t, Y, t, Y)
    #     - tf.einsum("iMN,MN,jPQ,PQ->ij", t, Y, t, Y)
    # )
    ## B is probably zero, although I am not sure
    # B = 4 * (
    #     tf.einsum("MRP,QRP,jMiQ->ij", Z, Z, tt_sym)
    #     + tf.einsum("MRP,QRP,jMiQ->ji", Z, Z, tt_sym)
    #     + 2 * tf.einsum("MNP,QRP,iMQ,jNR->ij", Z, Z, t, t)
    #     + 2 * tf.einsum("MRP,QRS,iMQ,jPS->ij", Z, Z, t, t)
    #     + 0.5 * tf.einsum("PRM,PRQ,jMiQ->ij", Z, Z, tt_sym)
    #     + 0.5 * tf.einsum("PRM,PRQ,jMiQ->ji", Z, Z, tt_sym)
    #     + tf.einsum("PNM,PRQ,iMQ,jNR->ij", Z, Z, t, t)
    #     + tf.einsum("PRM,SRQ,iMQ,jPS->ij", Z, Z, t, t)
    #     - (
    #         tf.einsum("MRR,QPP,jMiQ->ij", Z, Z, tt_sym)
    #         + tf.einsum("MRR,QPP,jMiQ->ji", Z, Z, tt_sym)
    #         + 2 * tf.einsum("MNR,QPP,iMQ,jNR->ij", Z, Z, t, t)
    #         + 2 * tf.einsum("MRR,QPS,iMQ,jPS->ij", Z, Z, t, t)
    #         + 0.5 * tf.einsum("PRR,PMQ,jMiQ->ij", Z, Z, tt_sym)
    #         + 0.5 * tf.einsum("PRR,PMQ,jMiQ->ji", Z, Z, tt_sym)
    #         + tf.einsum("PNR,PMQ,iMQ,jNR->ij", Z, Z, t, t)
    #         + tf.einsum("PRR,SMQ,iMQ,jPS->ij", Z, Z, t, t)
    #     )
    # )
    return A  # + B


if __name__ == "__main__":
    inv_killing = tf.linalg.inv(tf.einsum("iAB,jBA->ij", t, t))
    output = -15.0 * (
        tf.math.real(tf.linalg.eigvals(tf.einsum("AB,BC->AC", inv_killing, hessian())))
        / tf.math.abs(V())
    )
    print(tf.sort(output))
    # print(killing)

    # mass_matrix = tf.math.real(tf.linalg.eigvals(hessian() / V()))
    # pretty_string = print_Fridrik_understands(mass_matrix, 1e-5, 1e-9)
    # print(pretty_string)
