import tensorflow as tf
import scipy.optimize
import numpy


class GD_op:
    def __init__(
        self,
        learning_rate,
        loss_func,
        *inputs,
        decay_steps=1000,
        decay_rate=0.5,
        squash_func=tf.asinh
    ):

        # self.step_decay_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        #     [10, 30, 500, 750],
        #     [1.0, 1e-1, 1e-2, 2e-3, 1.9e-3],
        #     # [1.0, 1e-1, 1e-2, 4e-3, 2e-3, 8e-4],
        # )
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
        # for input in self.inputs:
        #     input.assign(tf.random.uniform((input.shape[0],), dtype=tf.float32))

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
    def __init__(self, learning_rate, loss_func, input_dims, tol=1e-9):
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.input_dims = input_dims
        self.tol = tol
        self.loss_func = loss_func

    @tf.function
    def loss_wrapper(self, bar):
        return self.loss_func(bar[:25], bar[25:])

    def norm(self, gradients):
        return tf.linalg.norm(tf.concat(list(*gradients), axis=0))

    @tf.function
    def minimizer(self, inputs):
        tf.print("Opened minimizer")
        with tf.GradientTape(persistent=False) as tape:
            loss = self.loss_wrapper(inputs)
        gradients = tape.gradient(loss, list(inputs))
        self.optimizer.apply_gradients(zip(loss, list(list(inputs))))
        print(gradients)
        return gradients


class BFGS:
    def __init__(self, learning_rate, loss_func, *input_dims):

        self.dims = sum(input_dims)
        self.learning_rate = learning_rate
        self.inputs_ = tf.random.uniform((self.dims,), dtype=tf.float32)
        self.pure_loss_func = loss_func

    @property
    def inputs(self):
        return self.inputs_

    @inputs.setter
    def inputs(self, bar):
        self.inputs_ = bar

    def reset(self):
        self.inputs = tf.random.uniform((self.dims,), dtype=tf.float32)

    @tf.function
    def loss_wrapper(self, bar):
        return self.loss_func(bar[:25], bar[25:])

    def loss_func(self, *bar):
        return tf.asinh(self.pure_loss_func(*bar))

    def auto_gradient(self, foo, bar):
        with tf.GradientTape() as tape:
            tape.watch(bar)
            loss = foo(bar)
        return tape.gradient(loss, bar)

    def gds(self):
        fprime = self.auto_gradient(self.loss_wrapper, self.inputs)
        opt = scipy.optimize.fmin_bfgs(
            tf.function(self.loss_wrapper),
            self.inputs,
            fprime=fprime.numpy(),
            gtol=1e-4,
            maxiter=10 ** 4,
            disp=0,
        )
        print(opt)
        print(self.loss_wrapper(opt))
        self.inputs = tf.constant(opt)
        # self.inputs = tf.random.uniform((self.dims,), dtype=tf.float32)
        return fprime

    def norm(self, gradients):
        return tf.linalg.norm(gradients)


class GD:
    def __init__(self, learning_rate, loss_func, *inputs):

        self.learning_rate = learning_rate
        self.input_dims = [inputs]
        self.inputs = [
            tf.Variable(tf.random.uniform((dim,), dtype=tf.float32)) for dim in inputs
        ]

        self.loss_func = tf.function(loss_func)

    def reset(self):
        for index in range(len(self.input_dims)):
            self.inputs[index].assign(
                tf.random.uniform((self.inputs[index].shape[0],), dtype=tf.float32)
            )

    @tf.function
    def gds(self):

        with tf.GradientTape() as tape:
            tape.watch(list(self.inputs))
            loss = self.loss_func(*self.inputs)
            # loss = tf.asinh(self.loss_func(*self.inputs))
        gradients = tape.gradient(loss, list(self.inputs))
        for index in range(len(self.input_dims)):
            self.inputs[index].assign(
                self.inputs[index] - self.learning_rate * gradients[index]
            )
        return gradients

    @tf.function
    def norm(self, gradients):
        norm = tf.linalg.norm(tf.concat([*gradients], axis=0))
        return norm
