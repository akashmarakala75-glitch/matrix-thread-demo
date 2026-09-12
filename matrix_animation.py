import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

MIN_SIZE = 100
STEPS_PER_FRAME = 25


class ExecutionRecorder:
    def __init__(self):
        self.order = deque()
        self.lock = threading.Lock()

    def record(self, i, j):
        with self.lock:
            self.order.append((i, j))


class CellComputer:
    def __init__(self, A, B, C, recorder):
        self.rows = tf.unstack(A, axis=0)
        self.cols = tf.unstack(B, axis=1)
        self.C = C
        self.recorder = recorder
        # run once here so tf.function is compiled before threads start
        self.dot(self.rows[0], self.cols[0])

    @staticmethod
    @tf.function
    def dot(row, col):
        return tf.tensordot(row, col, axes=1)

    def compute(self, i, j):
        self.C[i][j] = float(self.dot(self.rows[i], self.cols[j]))
        self.recorder.record(i, j)


def ask_size(msg):
    while True:
        try:
            n = int(input(msg))
        except ValueError:
            print("Please enter a number")
            continue
        if n < MIN_SIZE:
            print("Minimum allowed is", MIN_SIZE)
            continue
        return n


def multiply(A, B):
    rows = A.shape[0]
    cols = B.shape[1]
    C = np.zeros((rows, cols), dtype=np.float32)

    recorder = ExecutionRecorder()
    computer = CellComputer(A, B, C, recorder)

    tasks = [(i, j) for i in range(rows) for j in range(cols)]
    workers = os.cpu_count() or 4
    print("Total tasks:", len(tasks))
    print("Worker threads:", workers)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(computer.compute, i, j) for i, j in tasks]
        done = 0
        for f in as_completed(futures):
            f.result()
            done += 1
            if done % 1000 == 0:
                print("done", done, "/", len(tasks))

    return C, list(recorder.order)


def animate(A, B, C, order, gif=None):
    rows, cols = C.shape
    shown = np.full((rows, cols), np.nan)   # nan = not filled yet

    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    fig.suptitle("Matrix multiplication with threads (TensorFlow)")

    ax[0].imshow(A.numpy(), cmap="Blues", aspect="auto")
    ax[0].set_title("A")
    ax[1].imshow(B.numpy(), cmap="Greens", aspect="auto")
    ax[1].set_title("B")
    img = ax[2].imshow(shown, cmap="YlOrRd", aspect="auto",
                       vmin=C.min(), vmax=C.max())
    ax[2].set_title("C = A x B")

    row_line = ax[0].axhline(-1, color="red", lw=2)
    col_line = ax[1].axvline(-1, color="red", lw=2)
    label = fig.text(0.5, 0.02, "", ha="center")

    total = len(order)
    frames = (total + STEPS_PER_FRAME - 1) // STEPS_PER_FRAME

    def update(k):
        start = k * STEPS_PER_FRAME
        end = min(start + STEPS_PER_FRAME, total)
        for i, j in order[start:end]:
            shown[i][j] = C[i][j]
        i, j = order[end - 1]
        row_line.set_ydata([i, i])
        col_line.set_xdata([j, j])
        img.set_data(shown)
        label.set_text(f"{end}/{total} cells done   current: C[{i}][{j}]")
        return img, row_line, col_line, label

    anim = FuncAnimation(fig, update, frames=frames, interval=80,
                         repeat=False, blit=True)
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])

    if gif:
        print("saving", gif, "...")
        anim.save(gif, writer=PillowWriter(fps=12), dpi=70)
        print("saved")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    rows = ask_size("Rows: ")
    cols = ask_size("Columns: ")

    print("Generating matrices with TensorFlow...")
    A = tf.random.uniform((rows, cols), 1, 10, dtype=tf.int32)
    B = tf.random.uniform((cols, cols), 1, 10, dtype=tf.int32)

    t0 = time.perf_counter()
    C, order = multiply(A, B)
    t1 = time.perf_counter()

    print("\nFinished in %.3f seconds" % (t1 - t0))
    ok = np.array_equal(C, tf.matmul(A, B).numpy())
    print("Matches tf.matmul:", ok)
    print("Top-left 3x3 of C:")
    print(C[:3, :3])

    ans = input("\nSave as Matrix_multiplication.gif instead of showing? (y/n): ")
    animate(A, B, C, order, "Matrix_multiplication.gif" if ans.lower() == "y" else None)
