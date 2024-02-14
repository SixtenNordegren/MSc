import tensorflow as tf

optimizer = tf.optimizers.Adam(learning_rate = 1e-1, epsilon=1e-9)

@tf.function
def gradient_descent_step(loss_func, *args):
	"""
	Performs a single gradient descent step.

	Args:
		loss_func - The loss function to be minimized.
		*args - Tensor(s) the loss function is a function off. 

	Returns:
		gradients - Tensorflow computed gradient. 
		loss_func - Returns loss. (Might be removed at a later point)
	"""
	with tf.GradientTape() as tape:
		tape.watch([*args])
		loss = loss_func(*args)
	gradients = tape.gradient(loss, [*args])
	optimizer.apply_gradients(zip(gradients, [*args]))
	return gradients, loss
