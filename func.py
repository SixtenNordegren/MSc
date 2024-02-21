import tensorflow as tf
import numpy as np
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


def stationarity(gradients, tol=1e-5):
    """
    computes the norm of the gradients with tensorflow tools.

    args:
            gradients - list of gradients.
            tol  -- tolerance/ epsilon


    returns:
            true or false - true corresponds to having found a minimum
    """

    gradient_norm = tf.norm(gradients[0]) + tf.norm(gradients[1])
    if gradient_norm < tol:
        return true

        # def parser(text, dim_i=10, dim_j=5):
        #     """
        #     Translates linear combinations of Mathematica subscripts e.g.
        #     Subscript[i,j]->x[k]. Where k is related to i and j through an
        #     internal dictionary. It does return a string, use pythons eval()
        #     function to execute the code.

        #     Example:
        #         -Subscript[z, 1, 3] + Subscript[z, 2, 2] -> -z[2] + z[6]
        #     Args:
        #         text - string containing the linear combination to be
        #         tranlsated
        #         dim_i - dimension of first slot
        #         dim_j - dimension of second slot
        #     Returns:
        #         string - translated string
        #     """
        #     dictionary = {}
        #     counter = 0
        #     counter_i = 1
        #     counter_j = 1
        #     for i in range(dim_i):
        #         for j in range(dim_j):
        #             dictionary[(i + 1, j + 1)] = counter
        #             # print(f"{(i,j)}->{counter}")
        #             counter += 1
        #             counter_j += 1
        #         counter_i += 1

        #     try:
        #         index = int(text.index("S"))
        #     except ValueError:
        #         return text
        #     row_number = int(text[index + 13])
        #     col_number = int(text[index + 16])
        #     text = (
        #         text[:index] +
        #         f"x[{dictionary[row_number, col_number]}]" + text[index + 18:]
        #     )
        #     text = parser(text)
        #     return text


def LeviCivitaR5(dimension):
    # Initialize a numpy array filled with zeros
    levi_civita = np.zeros(
        (dimension, dimension, dimension, dimension, dimension), dtype=int
    )

    # Iterate over all permutations of a given dimension
    for perm in itertools.permutations(range(dimension)):
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


# Example usage
if __name__ == "__main__":
    init_y = tf.random.uniform((15,), maxval=1, dtype=tf.float32)
    init_z = tf.random.uniform((40,), maxval=1, dtype=tf.float32)

    y = Build_Y(init_y, 5)
    print(spare_matrix_csr.numpy())

    print(f"y -matrix: {y}")
    print(f"z -matrix: {z}")
