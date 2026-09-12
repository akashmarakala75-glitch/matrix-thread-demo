"""Matrix multiplication using TensorFlow, a task queue, and worker threads."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import queue
import threading
import time

import numpy as np
import tensorflow as tf


MINIMUM_SIZE = 100
NUMBER_OF_THREADS = 4


def read_size(message):
    """Read one matrix dimension and make sure it is at least 100."""
    while True:
        try:
            value = int(input(message))
            if value < MINIMUM_SIZE:
                print("Error: the value must be at least 100.")
                continue
            return value
        except ValueError:
            print("Error: please enter a whole number.")


def worker(thread_number, tasks, matrix_a, matrix_b, matrix_c, print_lock):
    """Take C[i][j] tasks from the queue until all tasks are finished."""
    while True:
        try:
            row, column = tasks.get_nowait()
        except queue.Empty:
            return

        with print_lock:
            print(f"Thread {thread_number} -> calculating C[{row}][{column}]")

        # C[i][j] is the dot product of row i of A and column j of B.
        products = tf.multiply(matrix_a[row, :], matrix_b[:, column])
        matrix_c[row, column] = tf.reduce_sum(products).numpy()
        tasks.task_done()


def multiply_with_threads(matrix_a, matrix_b):
    """Create one task for every C[i][j] and run four workers."""
    result_rows = matrix_a.shape[0]
    result_columns = matrix_b.shape[1]
    matrix_c = np.zeros((result_rows, result_columns), dtype=np.float32)
    tasks = queue.Queue()

    for row in range(result_rows):
        for column in range(result_columns):
            tasks.put((row, column))

    print(f"\nScheduler created {tasks.qsize()} tasks.\n")

    print_lock = threading.Lock()
    threads = []
    for thread_number in range(NUMBER_OF_THREADS):
        thread = threading.Thread(
            target=worker,
            args=(thread_number, tasks, matrix_a, matrix_b, matrix_c, print_lock),
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return matrix_c


def main():
    """Read dimensions, create matrices, schedule the work, and show the result."""
    rows = read_size("Enter number of rows: ")
    columns = read_size("Enter number of columns: ")

    print("\nCreating matrices using TensorFlow...")
    # A is rows x columns, B is columns x columns, so C is rows x columns.
    matrix_a = tf.random.uniform(
        (rows, columns), minval=1, maxval=10, dtype=tf.int32
    )
    matrix_b = tf.random.uniform(
        (columns, columns), minval=1, maxval=10, dtype=tf.int32
    )
    matrix_a = tf.cast(matrix_a, tf.float32)
    matrix_b = tf.cast(matrix_b, tf.float32)

    start_time = time.perf_counter()
    matrix_c = multiply_with_threads(matrix_a, matrix_b)
    execution_time = time.perf_counter() - start_time

    # Compare against TensorFlow's complete multiplication as a correctness check.
    expected = tf.matmul(matrix_a, matrix_b).numpy()
    is_correct = np.allclose(matrix_c, expected)

    print("\nMatrix multiplication completed.")
    print(f"Execution time: {execution_time:.3f} seconds")
    print(f"Result verified with tf.matmul: {is_correct}")
    print("\nTop-left 3 x 3 portion of Matrix C:")
    print(matrix_c[:3, :3])


if __name__ == "__main__":
    main()
