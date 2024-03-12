import tensorflow as tf
import numpy as np
import fractions
import os
from datetime import datetime
from itertools import permutations
import itertools
from scipy.sparse import coo_matrix

# Disable the obligatory tensorflow warnings.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
tf.get_logger().setLevel("ERROR")


def eigen_formater(tensor):
    string = "("
    for eigenval in tf.linalg.eigvals(tensor):
        if eigenval.numpy() > 0.0:
            string += "+, "
        elif eigenval.numpy() < 0.0:
            string += "-, "
        else:
            string += "NaN, "

    string = string[:-2] + ")"
    return string


def LeviCivitaR4(dimension):
    # Initialize a numpy array filled with zeros
    levi_civita_np = np.zeros((dimension, dimension, dimension, dimension), dtype=float)

    # Generate all permutations of [0, 1, 2, 3] and their signs
    for perm in permutations(range(dimension)):
        # The sign of the permutation is +1 if the permutation is even, -1 if odd
        sign = (-1) ** sum(
            i < j
            for i in range(dimension)
            for j in range(i + 1, dimension)
            if perm[i] > perm[j]
        )
        levi_civita_np[perm] = sign

    # Convert the numpy array to a TensorFlow tensor
    levi_civita_tensor = tf.constant(levi_civita_np, dtype=tf.float32)

    return levi_civita_tensor


def save_usefull(y, z, V_val, directory, file_name="testfile_delete"):
    """
    Saves the minima details to a file within the specified directory. The file name includes the current date and time.

    Args:
        y (tf.Tensor): The y values at the minima.
        z (tf.Tensor): The z values at the minima.
        V_val (float): The potential value at the minima.
        directory (str): Directory where minima details will be saved.
    """

    file_name = f"minimas_{file_name}.txt"
    file_path = os.path.join(directory, file_name)

    # Ensure the values are in a numpy format for easier writing
    y_val = y.numpy()
    z_val = z.numpy()
    V_val = V_val.numpy()

    with open(file_path, "a") as file:  # 'a' mode appends to the file if it exists
        file.write(
            f"{[z_val,y_val,V_val]}",
        )


def save(y, z, V_val, directory, time, start=False):
    """
    Saves the minima details to a file within the specified directory. The file name includes the current date and time.

    Args:
        y (tf.Tensor): The y values at the minima.
        z (tf.Tensor): The z values at the minima.
        V_val (float): The potential value at the minima.
        directory (str): Directory where minima details will be saved.
    """

    # Generate the timestamp
    # current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_time = datetime.now().strftime("%Y-%m-%d_%H")
    current_time_precise = datetime.now().strftime("%M-%S")
    file_name = f"minimas_{current_time}.txt"
    file_path = os.path.join(directory, file_name)

    if start == True:
        with open(file_path, "a") as file:
            file.write(f"Started search at: {current_time_precise}")

    # Ensure the values are in a numpy format for easier writing
    y_val = y.numpy()
    z_val = z.numpy()
    V_val = V_val.numpy()

    with open(file_path, "a") as file:  # 'a' mode appends to the file if it exists
        file.write(f"Minima located at:\n")
        file.write(f"Y: {y_val}\nZ: {z_val}\nV: {V_val}\n\n")
        file.write(f"This minima was located in {time}s\n")


def Build_Y(input_list, dim):
    """
    Builds the dim-dimmensional Y tensor

    Args:
        input_list - list of independent variables to be used as inputs

    """
    # Set the dimension of the tensor (5, 5 by default)
    y = tf.zeros((dim, dim), dtype=tf.float64)
    counter = 0
    for i in range(dim):
        for j in range(dim):
            if i == j:
                y = tf.tensor_scatter_nd_update(y, [[i, j]], [input_list[counter]])
                counter += 1
            elif i < j:
                y = tf.tensor_scatter_nd_update(y, [[i, j]], [2 * input_list[counter]])
                counter += 1
    # Symmetrize
    y = 0.5 * (y + tf.einsum("ij->ji", y))
    return y


def print_Fridrik_understands(
    mass_matrix, potential, parameter_accuracy, loss_function_accuracy, n_found=1
):
    # print_matrix = [
    #     str(fractions.Fraction(str(element.numpy())).limit_denominator())
    #     for element in mass_matrix
    # ]
    pretty_string = f"Mass matrix:\n {tf.sort(mass_matrix)}\nPotential: {potential}\n Parameter accuracy: {parameter_accuracy}\n Loss function accuracy: {loss_function_accuracy}\n Times mass was found: {n_found}\n"
    # pretty_string = pretty_string + f"rounded Mass matrix:\n {print_matrix}"
    return pretty_string


# Example usage
if __name__ == "__main__":
    init_y = tf.random.uniform((15,), maxval=1, dtype=tf.float32)
    init_z = tf.random.uniform((40,), maxval=1, dtype=tf.float32)

    y = Build_Y(init_y, 5)
    print(spare_matrix_csr.numpy())

    print(f"y -matrix: {y}")
    print(f"z -matrix: {z}")
