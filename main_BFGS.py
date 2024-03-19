import tensorflow as tf
import numpy as np
import scipy.optimize
from func import print_Fridrik_understands
import loss_functions.d7
import gds.optimizers

x0 = tf.random.uniform((75,), dtype=tf.float32)
tol = 1e-4
ttol = 1e-9

theory = loss_functions.d7.d7()
f_opt = theory.loss_func
processor = gds.optimizers.Adam(ttol, f_opt, theory.inputs)

t = loss_functions.d7.generators()

save_location = "/home/sixten/Projects/GDS_saves/"
file_name = "test_2024-03-01_2"


def search(maxiter=1e4):
    iter = 0
    while True:
        gradients = processor.minimizer(input_point)
        tf.print(tf.linalg.norm(gradients))
        if tf.linalg.norm(gradients) < tol:
            yield input_point
        elif iter > maxiter:
            print("Ran out of range; will stop.")
            iter += 1


# There is a peculiar interaction with tensorflow and the numpy
# interface where the tensorflow interface requires the Variable to be
# kept outside the scope of the function, for reasons contained in the
# structure of the scipy implementation of the BFGS function, it
# demands the passing of the variable being minimized within the scope
# of the function. This means that we have to define two different
# minimization functions. One for scipy and one for tensorflow.


def auto_grad(bar):
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
        x0,
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
    number_of_solutions = 1
    solutions = []
    for _ in range(number_of_solutions):
        x0 = tf.random.uniform((75,), dtype=tf.float32)
        input_point = tf.Variable(scanner(x0))

        print_str = print_Fridrik_understands(
            masses_computation(sol), theory.V(sol[:25], sol[25:]), 1, 1
        )
        tf.print(print_str)
        precision_scanner = search()
        sol = next(precision_scanner)

        solutions.append(sol.numpy())
        print_str = print_Fridrik_understands(
            masses_computation(sol), theory.V(sol[:25], sol[25:]), 1, 1
        )
        tf.print(print_str)
    np.savez(
        save_location + file_name + ".npz",
        **{f"array_{i}": array for i, array in enumerate(solutions)},
    )
