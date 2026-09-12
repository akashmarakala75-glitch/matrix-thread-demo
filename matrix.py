import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

MATRIX_SIZE = 100


def create_matrix(rows: int, columns: int) -> np.ndarray:
    """Create a matrix of random values between 0 and 10."""
    return np.random.uniform(0, 10, size=(rows, columns)).astype(np.float32)


def multiply_rows(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    start_row: int,
    end_row: int,
) -> np.ndarray:
    """Multiply one range of rows from matrix A by matrix B."""
    row_block = tf.constant(matrix_a[start_row:end_row])
    result = tf.matmul(row_block, tf.constant(matrix_b))
    return result.numpy()


def check_result(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    threaded_result: np.ndarray,
) -> bool:
    """Compare the threaded result with a single TensorFlow multiplication."""
    expected_result = tf.matmul(tf.constant(matrix_a), tf.constant(matrix_b)).numpy()
    return bool(np.allclose(threaded_result, expected_result, atol=0.1))
