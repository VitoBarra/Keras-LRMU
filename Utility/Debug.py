import tensorflow as tf


def PrintMatrix(matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            print(matrix[i][j], end=" ")
        print()


def PrintAvailableGPU():
    print("\n")
    for i in tf.config.list_physical_devices('GPU'):
        print(i)
    print("\n")
