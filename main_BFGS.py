import tensorflow as tf
import numpy
import scipy.optimize
from func import print_Fridrik_understands
from loss_functions.d7 import *
from gds.optimizers import *

theory = d7()
processor = Adam()


f_opt = theory.loss_func
x0 = tf.random.uniform((75,), dtype=tf.float32)
tol = 1e-4
ttol = 1e-9


save_location = "/home/sixten/Projects/GDS_saves/"
file_name = "test_2024-03-01_2"


def auto_grad(bar):
    bar = tf.constant(bar)
    with tf.GradientTape() as tape:
        tape.watch(bar)
        loss_func = loss_wrapper(bar)
        gradient = tape.gradient(loss_func, bar)
    return gradient.numpy()


def loss_wrapper(bar):
    return tf.asinh(f_opt(bar[:25], bar[25:]))


def scanner(x0):
    opt = scipy.optimize.fmin_bfgs(
        loss_wrapper,
        x0.numpy(),
        fprime=auto_grad,
        gtol=tol,
        maxiter=10 ** 4,
        disp=0,
    )
    return tf.constant(opt)


def masses_computation(sol):
    inv_killing = tf.linalg.inv(tf.einsum("iAB,jBA->ij", t, t))
    masses = -15.0 * (
        tf.math.real(
            tf.linalg.eigvals(
                tf.einsum("AB,BC->AC", inv_killing, theory.hessian(sol[:25], sol[25:]))
            )
        )
        / tf.math.abs(theory.V(sol[:25], sol[25:]))
    )
    return masses


if __name__ == "__main__":
    number_of_solutions = 100
    solutions = []
    for _ in range(number_of_solutions):
        x0 = tf.random.uniform((75,), dtype=tf.float32)
        sol = scanner(x0)

        solutions.append(sol.numpy())
        print_str = print_Fridrik_understands(
            masses_computation(sol), theory.V(sol[:25], sol[25:]), 1, 1
        )
        tf.print(print_str)
    np.savez(
        save_location + file_name + ".npz",
        **{f"array_{i}": array for i, array in enumerate(solutions)},
    )
