import numpy as np
from gds.optimizers import *
from loss_functions.d7 import *


# Options
file_path = "/home/sixten/Projects/GDS_saves/"
search_length = 3  # Number of minimas to look for

# Search paramters
tol = 1e-9
initial_learning_rate = 2e-1
decay_steps = 1250
decay_rate = 0.6

# Operationl paramters
good_minima = False
max_steps = 3000
theory = d7()
# processor = Adam(initial_learning_rate, theory.loss_func, *theory.inputs())
processor = GD_op(
    initial_learning_rate,
    theory.loss_func,
    *theory.inputs(),
    decay_steps=decay_steps,
    decay_rate=decay_rate,
)

step = None
norm = None
minima = []
for _ in range(search_length):
    for step in range(max_steps):
        gradients = processor.gds()
        if step % 10 == 0:
            norm = processor.norm(gradients)
            if norm < tol:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("Minimum found!")
                print(f"Steps: {step}")
                print(f"Total norm: {norm}")
                print("######################################")
                file_name = "test_file.txt"
                minima.append((step, *processor.optimizer.variables()))
                break
        if step % 250 == 0:
            norm = processor.norm(gradients)
            print(f"Steps: {step}")
            print(f"Total norm:{norm}")
            print("######################################")
    if good_minima == False:
        print("Solution abandoned")
        print(f"Steps: {step}")
        print(f"Total norm: {norm}")
        print("######################################")
        file_name = "test_file.txt"
        minima.append((step, *processor.optimizer.variables()))

    good_minima = False
    processor.reset()
np.savez(file_path + file_name, minima)
