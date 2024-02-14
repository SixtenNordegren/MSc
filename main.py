# Packages
import tensorflow as tf

# Modules
from gds.opt_adam import gradient_descent_step
from loss_functions.d7_loss import loss_function

# Search paramters
tol = 1e-9
learning_rate = 1e-1
search_length = 1 # Number of minimas to look for

# Operationl paramters
bad_minima = False
stationary = False
counter = 0

optimizer = tf.optimizers.Adam(learning_rate=learning_rate, epsilon=tol)
y = tf.Variable(tf.random.uniform((25,)), dtype=tf.float32)
z = tf.Variable(tf.random.uniform((50,)), dtype=tf.float32)

def stationarity(gradients, tol=1e-5):
	"""
	Computes the norm of the gradients with tensorflow tools.

	Args:
		gradients - list of gradients.
		tol = tolerance lever.

	Returns:
		True or False - True corresponds to having found a minimum
	"""

	# Cheat norm
	# gradient_norm = min([tf.norm(grad) for grad in gradients if grad is not None])

	# Not cheat norm
	gradient_norm = tf.norm(gradients[0]) + tf.norm(gradients[1])
	if gradient_norm < tol:
		return True
	else:
		return False

for i in range(search_length):
	y.assign(tf.random.uniform((25,), dtype=tf.float32))
	z.assign(tf.random.uniform((50,), dtype=tf.float32))

	while stationary == False:
		# Perform a gradient descent step
		gradients, loss_val = gradient_descent_step(loss_function, y, z)
		counter += 1
		if counter % 10 == 0:
			stationary = stationarity(gradients, tol=tol)
			if counter % 2000 == 0:
				print(f"Solution abandoned")
				# print(f"Time spent {ti-tf_2000}")
				# print(f"Updated y: {Y(y).numpy()}")
				# print(f"Updated z: {Z(z).numpy()}")
				# print(f"Loss : {loss_val.numpy()}")
				# print(f"Grad^2 y:{tf.norm(gradients[0]).numpy()}")
				# print(f"Grad^2 z:{tf.norm(gradients[1]).numpy()}")
				# print(
				# f"Total gradient^2 :{tf.norm(gradients[0])+tf.norm(gradients[1])}"
				# )
				# x = np.einsum("QMNP,QMNP->", X(y, z), X(y, z))
				# print(f"Updated x: {x}")
				print("######################################")
				# bad_minima = True
				break

	if bad_minima == False:
		print("Minimum found!")
		print(f"norm grad y:{tf.norm(gradients[0]).numpy()}")
		print(f"norm grad z:{tf.norm(gradients[1]).numpy()}")
		print(f"norm y:{tf.norm(Y(y)).numpy()}")
		print(f"norm z:{tf.norm(Z(z)).numpy()}")
		print(f"Updated y: {Y(y).numpy()}")
		print(f"Signature of Y: {eigen_formater(Y(y))}")
		# print(f"Updated z: {Z(z).numpy()}")
		print(f"Loss: {loss_val.numpy()}")
		x = np.einsum("QMNP,QMNP->", X(y, z), X(y, z))
		print(f"Updated x: {x}")
		print(f"Steps : {counter}")
		print("######################################")
	bad_minima = False
