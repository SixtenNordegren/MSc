import tensorflow as tf


class GD_op:
    def __init__(
        self, learning_rate, loss_func, *inputs, decay_steps=1000, decay_rate=0.5
    ):

        self.step_decay_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps,
            decay_rate,
        )
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=self.step_decay_schedule, clipnorm=1.0
        )
        self.input_dims = inputs
        self.inputs = [
            tf.Variable(tf.random.uniform((dim,)), dtype=tf.float32) for dim in inputs
        ]
        self.loss_func = tf.function(loss_func)

    def reset(self):
        self.step_decay_schedule.step = 0
        for var in self.optimizer.variables():
            var.assign(tf.zeros_like(var))
        for input in self.inputs:
            input.assign(tf.random.uniform((input.shape[0],), dtype=tf.float32))

    @tf.function
    def gds(self):
        """
        Performs a single gradient descent step.

        Args:
                loss_func - The loss function to be minimized.
                *args - Tensor(s) the loss function is a function off.

        Returns:
                gradients - Numerical gradient.
                loss - Returns value of loss function, important that this is a tensor
                        and not a function, otherwise tensorflow will not
                        compile the function.
        """
        with tf.GradientTape() as tape:
            tape.watch(list(self.inputs))
            loss = self.loss_func(*self.inputs)
        gradients = tape.gradient(loss, list(self.inputs))
        self.optimizer.apply_gradients(zip(gradients, list(self.inputs)))
        return gradients

    @tf.function
    def norm(self, gradients):
        norm = tf.linalg.norm(tf.concat([*gradients], axis=0))
        return norm


class Adam:
    def __init__(self, learning_rate, loss_func, *inputs):
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.input_dims = inputs
        self.inputs = [
            tf.Variable(tf.random.uniform((dim,)), dtype=tf.float32) for dim in inputs
        ]
        self.loss_func = tf.function(loss_func)

    def reset(self):
        for input in self.inputs:
            input.assign(tf.random.uniform((input.shape[0],), dtype=tf.float32))

    @tf.function
    def gds(self):
        """
        Performs a single gradient descent step.

        Args:
                loss_func - The loss function to be minimized.
                *args - Tensor(s) the loss function is a function off.

        Returns:
                gradients - Numerical gradient.
                loss - Returns value of loss function, important that this is a tensor
                        and not a function, otherwise tensorflow will not
                        compile the function.
        """
        with tf.GradientTape() as tape:
            tape.watch(list(self.inputs))
            loss = self.loss_func(*self.inputs)
        gradients = tape.gradient(loss, list(self.inputs))
        self.optimizer.apply_gradients(zip(gradients, list(self.inputs)))
        return gradients

    @tf.function
    def norm(self, gradients):
        norm = tf.linalg.norm(tf.concat([*gradients], axis=0))
        return norm


class GD:
    def __init__(self, learning_rate, tolerance, loss_func, *inputs):

        self.learning_rate = learning_rate
        self.input_dims = inputs
        self.inputs = [tf.random.uniform((dim,), dtype=tf.float32) for dim in inputs]
        self.loss_func = tf.function(loss_func)

    def reset(self):
        for index in range(len(self.inputs)):
            self.inputs[index] = tf.random.uniform(
                (self.inputs[index].shape[0],), dtype=tf.float32
            )

    @tf.function
    def gds(self):
        """
        Performs a single gradient descent step.

        Args:
                loss_func - The loss function to be minimized.
                *args - Tensor(s) the loss function is a function off.

        Returns:
                gradients - Numerical gradient.
                loss - Returns value of loss function, important that this is a tensor
                        and not a function, otherwise tensorflow will not
                        compile the function.
        """
        with tf.GradientTape() as tape:
            tape.watch(list(self.inputs))
            loss = self.loss_func(*self.inputs)
        gradients = tape.gradient(loss, list(self.inputs))
        for index in range(len(self.inputs)):
            self.inputs[index] = self.inputs[index] - self.learning_rate * tf.asinh(
                gradients[index]
            )
        return gradients

    @tf.function
    def norm(self, gradients):
        norm = tf.linalg.norm(tf.concat([*gradients], axis=0))
        return norm
