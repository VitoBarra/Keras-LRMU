
import tensorflow as tf


@staticmethod
def _cont2discrete_zoh(A, B):
    """
    Function to discretize A and B matrices using Zero Order Hold method.

    Functionally equivalent to
    ``scipy.signal.cont2discrete((A.T, B.T, _, _), method="zoh", dt=1.0)``
    (but implemented in TensorFlow so that it is differentiable).

    Note that this accepts and returns matrices that are transposed from the
    standard linear system implementation (as that makes it easier to use in
    `.call`).
    """

    # combine A/B and pad to make square matrix
    em_upper = tf.concat([A, B], axis=0)  # pylint: disable=no-value-for-parameter
    em = tf.pad(em_upper, [(0, 0), (0, B.shape[0])])

    # compute matrix exponential
    ms = tf.linalg.expm(em)

    # slice A/B back out of combined matrix
    discrt_A = ms[: A.shape[0], : A.shape[1]]
    discrt_B = ms[A.shape[0] :, : A.shape[1]]

    return discrt_A, discrt_B