"""Threaded matrix multiplication with TensorFlow and Matplotlib animation."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.animation import FuncAnimation, PillowWriter


MINIMUM_SIZE = 100
STEPS_PER_FRAME = 25


class ExecutionRecorder:
    """Safely record the order in which cells finish."""

    def __init__(self):
        self.completed_cells = deque()
        self.lock = threading.Lock()

    def record(self, row, column):
        with self.lock:
            self.completed_cells.append((row, column))


class CellComputer:
    """Calculate one result cell using TensorFlow."""

    def __init__(self, matrix_a, matrix_b, matrix_c, recorder):
        self.matrix_a = matrix_a
        self.matrix_b = matrix_b
        self.matrix_c = matrix_c
        self.recorder = recorder

    def calculate(self, row, column):
        row_from_a = self.matrix_a[row, :]
        column_from_b = self.matrix_b[:, column]
        value = tf.tensordot(row_from_a, column_from_b, axes=1).numpy()
        self.matrix_c[row, column] = value
        self.recorder.record(row, column)
        return row, column


def read_size(message):
    """Read a whole number that is at least 100."""
    while True:
        try:
            value = int(input(message))
            if value < MINIMUM_SIZE:
                print("Error: the value must be at least 100.")
            else:
                return value
        except ValueError:
            print("Error: please enter a whole number.")


def multiply_with_threads(matrix_a, matrix_b):
    """Submit every C[i][j] calculation to a thread pool."""
    result_rows = matrix_a.shape[0]
    result_columns = matrix_b.shape[1]
    matrix_c = np.zeros((result_rows, result_columns), dtype=np.float32)
    recorder = ExecutionRecorder()
    computer = CellComputer(matrix_a, matrix_b, matrix_c, recorder)
    positions = [
        (row, column)
        for row in range(result_rows)
        for column in range(result_columns)
    ]
    worker_count = os.cpu_count() or 4

    print(f"Scheduler created {len(positions)} cell tasks.")
    print(f"Thread pool is using {worker_count} worker threads.")

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(computer.calculate, row, column)
            for row, column in positions
        ]

        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 1000 == 0 or completed == len(positions):
                print(f"Completed {completed} / {len(positions)} cells")

    return matrix_c, list(recorder.completed_cells)


def create_animation(matrix_a, matrix_b, matrix_c, execution_order, gif_name=None):
    """Animate completed cells in their recorded threaded execution order."""
    rows, columns = matrix_c.shape
    visible_c = np.full((rows, columns), np.nan, dtype=np.float32)

    figure, axes = plt.subplots(1, 3, figsize=(11, 4))
    figure.suptitle("Threaded Matrix Multiplication using TensorFlow")

    axes[0].imshow(matrix_a.numpy(), cmap="Blues", aspect="auto")
    axes[0].set_title("Matrix A")
    axes[1].imshow(matrix_b.numpy(), cmap="Greens", aspect="auto")
    axes[1].set_title("Matrix B")
    result_image = axes[2].imshow(
        visible_c,
        cmap="YlOrRd",
        aspect="auto",
        vmin=float(np.min(matrix_c)),
        vmax=float(np.max(matrix_c)),
    )
    axes[2].set_title("Matrix C")

    for axis in axes:
        axis.set_xlabel("Column")
        axis.set_ylabel("Row")

    row_marker = axes[0].axhline(-1, color="red", linewidth=2)
    column_marker = axes[1].axvline(-1, color="red", linewidth=2)
    progress_text = figure.text(0.5, 0.02, "Completed 0 cells", ha="center")
    total_frames = (len(execution_order) + STEPS_PER_FRAME - 1) // STEPS_PER_FRAME

    def update(frame_number):
        start = frame_number * STEPS_PER_FRAME
        end = min(start + STEPS_PER_FRAME, len(execution_order))

        for row, column in execution_order[start:end]:
            visible_c[row, column] = matrix_c[row, column]

        current_row, current_column = execution_order[end - 1]
        row_marker.set_ydata([current_row, current_row])
        column_marker.set_xdata([current_column, current_column])
        result_image.set_data(visible_c)
        progress_text.set_text(
            f"Completed {end} / {len(execution_order)} cells  |  "
            f"Displaying C[{current_row}][{current_column}]"
        )
        return result_image, row_marker, column_marker, progress_text

    animation = FuncAnimation(
        figure,
        update,
        frames=total_frames,
        interval=80,
        repeat=False,
        blit=False,
    )
    figure.tight_layout(rect=[0, 0.08, 1, 0.92])

    if gif_name:
        print(f"Saving animation as {gif_name}...")
        animation.save(gif_name, writer=PillowWriter(fps=12), dpi=70)
        print("GIF saved.")
    else:
        plt.show()

    plt.close(figure)


def main():
    rows = read_size("Enter number of rows: ")
    columns = read_size("Enter number of columns: ")

    print("\nCreating random matrices using TensorFlow...")
    matrix_a = tf.random.uniform((rows, columns), minval=1, maxval=10)
    matrix_b = tf.random.uniform((columns, columns), minval=1, maxval=10)

    started_at = time.perf_counter()
    matrix_c, execution_order = multiply_with_threads(matrix_a, matrix_b)
    elapsed_time = time.perf_counter() - started_at

    expected = tf.matmul(matrix_a, matrix_b).numpy()
    print("\nMatrix multiplication completed.")
    print(f"Execution time: {elapsed_time:.3f} seconds")
    print(f"Result verified with tf.matmul: {np.allclose(matrix_c, expected)}")
    print("\nTop-left 3 x 3 portion of Matrix C:")
    print(matrix_c[:3, :3])

    save_answer = input("\nSave the animation as Matrix_multiplication.gif? (y/n): ")
    gif_name = "Matrix_multiplication.gif" if save_answer.lower() == "y" else None
    create_animation(matrix_a, matrix_b, matrix_c, execution_order, gif_name)


if __name__ == "__main__":
    main()
