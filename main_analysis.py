import numpy
import math
import tensorflow as tf
from func import print_Fridrik_understands
from loss_functions.d7 import d8

data = numpy.load("/home/sixten/Projects/GDS_saves/test_2024-03-01_2.npz")
theory = d8()


# Iterate through each item in the loaded data and print
# for key in data:
#     print(f"{key}: {data[key]}")


def is_same(new_val, mass_dict, tol=1e-3):
    if len(mass_dict) == 0:
        return False
    elif len(mass_dict) > 0:
        for mass in mass_dict:
            if tf.linalg.norm(new_val - mass.deref()) > tol:
                return False
            else:
                print("Degenerate minima detected")
                mass_dict[mass.ref()] += 1
                return True


mass_dict = {}
pretty_string = ""
for key in data:
    mass_matrix = theory.hessian(data[key][:25], data[key][25:]) / theory.V(
        data[key][:25], data[key][25:]
    )
    mass_matrix = tf.math.real(tf.linalg.eigvals(mass_matrix))
    if not is_same(mass_matrix, mass_dict) or not tf.math.is_nan(mass_matrix):
        mass_dict[mass_matrix.ref()] = 1
        pretty_string += print_Fridrik_understands(
            mass_matrix,
            1e-4,
            "e^{}".format(
                math.floor(
                    math.log(theory.loss_func(data[key][:25], data[key][25:]), 10)
                )
            ),
            n_found=mass_dict[mass_matrix.ref()],
        )
print(pretty_string, mass_dict)
