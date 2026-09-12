# matrix.py - makes the matrices and does the actual math with tensorflow

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide tensorflow info spam

import numpy as np
import tensorflow as tf

MATRIX_SIZE = 100


def create_matrix(rows, cols):
    return np.random.uniform(0, 10, size=(rows, cols)).astype(np.float32)


def multiply_rows(A, B, start_row, end_row):
    # the tensorflow part, every thread calls this for its own rows of A
    part_of_a = tf.constant(A[start_row:end_row])
    result = tf.matmul(part_of_a, tf.constant(B))
    return result.numpy()


def check_result(A, B, our_result):
    # compare our threaded answer with tensorflow doing everything in one go
    normal = tf.matmul(tf.constant(A), tf.constant(B)).numpy()
    return bool(np.allclose(our_result, normal, atol=0.1))
