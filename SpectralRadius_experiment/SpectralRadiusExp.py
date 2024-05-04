import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
import tensorflow as tf
import random
from matplotlib import cm


# https://danielrapp.github.io/rnn-spectral-radius/
def positive_matrix_normalized_gaussian(size):

    mean = 0
    randomMatrix = tf.random.normal(
        shape=[size, size],
        mean=mean,
        stddev=1.0/ np.sqrt(size),
        dtype=tf.dtypes.float64,
        seed=123456,
        name=None
    )

    return tf.math.abs(randomMatrix)/16


def positive_matrix_uniform(size):
    randomMatrix = tf.random.uniform(
        shape=[size, size],
        minval=0,
        maxval=1,
        dtype=tf.dtypes.float64,
        seed=123456,
        name=None
    )
    return randomMatrix


# https://stackoverflow.com/questions/64056164/how-to-randomly-change-the-signs-of-some-of-the-elements-in-a-numpy-array
def random_switch_signs(matrix, rate=0.5):
    mewMatrix = [each * (1 if random.uniform(0, 1) < rate else -1) for each in matrix]
    return mewMatrix


# https://scicomp.stackexchange.com/questions/34117/implementing-gelfand-s-formula-for-the-spectral-radius-in-python-lack-of-conve
def gelfands_kth_term(matrix, k=20):
    """Gelfand's formula"""
    matrix = la.matrix_power(matrix, k)
    f_norm = la.norm(matrix, 'fro')
    term = f_norm ** (1.0 / k)

    return term


def convergence_data_gelfands_formula(matrix, step=50):
    data = np.array([])
    for k in range(1, step):
        data = np.append(data, [k, gelfands_kth_term(matrix, k)])
    return data


def generate_data(generate_random_matrix_fun, size, rate):
    matrix = generate_random_matrix_fun(size)
    dataPoint = gelfands_kth_term(matrix, 80)

    matrix = random_switch_signs(matrix, rate)
    dataPointSwitch = gelfands_kth_term(matrix, 80)
    return np.array([dataPoint]), np.array([dataPointSwitch])


def run_mesh_test(generate_random_matrix_fun, sizes, rates):
    results = np.array([])
    resultsSwitch = np.array([])
    for s in sizes:
        for r in rates:
            (data, switchData) = generate_data(generate_random_matrix_fun, s, r)
            results = np.append(results, data[-1])
            resultsSwitch = np.append(resultsSwitch, switchData[-1])

    return results.reshape(sizes.size, rates.size), resultsSwitch.reshape(sizes.size, rates.size)


def test_gaussian_matrix_plot(filename, generate_random_matrix_fun=positive_matrix_normalized_gaussian):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), subplot_kw=dict(projection='3d'))

    ax1.set_xlabel('Flip Rate')
    ax1.set_ylabel('Matrix Size')
    ax1.set_zlabel('Spectral Radius')

    ax2.set_xlabel('Flip Rate')
    ax2.set_ylabel('Matrix Size')
    ax2.set_zlabel('Spectral Radius')

    matrix_sizes = np.linspace(50, 500, 10).astype(int)
    flip_rates = np.linspace(0, 1, 25)

    Z, Z_switch = run_mesh_test(generate_random_matrix_fun, matrix_sizes, flip_rates)

    X, Y = np.meshgrid(flip_rates, matrix_sizes)

    ax1.plot_surface(X, Y, Z, cmap=cm.plasma)
    ax2.plot_surface(X, Y, Z_switch, cmap=cm.plasma)

    fig.suptitle("Spectral Radius of Random Matrices with " + filename + " Distribution")
    fig.savefig("SRE_Plot/" + filename + ".png")
    plt.close(fig)


if __name__ == "__main__":
    test_gaussian_matrix_plot("gaussian", positive_matrix_normalized_gaussian)
    #test_gaussian_matrix_plot("uniform", positive_matrix_uniform)
