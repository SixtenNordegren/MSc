import tensorflow as tf

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
        loss_func = loss_func
    gradients = tape.gradient(loss_func, [*args])
    optimizer.apply_gradients(zip(gradients, [*args]))
    return gradients, loss_func
