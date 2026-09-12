import queue
import threading
import time

import numpy as np

import matrix

NUM_THREADS = 4
ROWS_PER_TASK = 5
DEMO_DELAY = 0.5


class Scheduler:
    """Split matrix multiplication into row blocks shared by worker threads."""

    def __init__(self, matrix_a: np.ndarray, matrix_b: np.ndarray):
        self.matrix_a = matrix_a
        self.matrix_b = matrix_b
        self.result = np.zeros(
            (matrix_a.shape[0], matrix_b.shape[1]), dtype=np.float32
        )

        self.tasks = []
        self.task_queue = queue.Queue()
        row_ranges = range(0, matrix_a.shape[0], ROWS_PER_TASK)
        for task_id, start_row in enumerate(row_ranges):
            end_row = min(start_row + ROWS_PER_TASK, matrix_a.shape[0])
            task = {
                "id": task_id,
                "start_row": start_row,
                "end_row": end_row,
                "status": "pending",
                "thread": None,
            }
            self.tasks.append(task)
            self.task_queue.put(task)

        self.progress_lock = threading.Lock()
        self.state = "idle"
        self.rows_done = 0
        self.total_time = 0
        self.compute_time = 0
        print(
            f"Scheduler: created {len(self.tasks)} tasks for {NUM_THREADS} threads"
        )

    def worker(self, thread_name: str) -> None:
        """Process queued row blocks until no tasks remain."""
        while True:
            try:
                task = self.task_queue.get_nowait()
            except queue.Empty:
                return

            with self.progress_lock:
                task["status"] = "working"
                task["thread"] = thread_name
            print(
                f"  {thread_name} is processing rows "
                f"{task['start_row']} to {task['end_row'] - 1}"
            )

            compute_started_at = time.perf_counter()
            row_block = matrix.multiply_rows(
                self.matrix_a,
                self.matrix_b,
                task["start_row"],
                task["end_row"],
            )
            compute_duration = time.perf_counter() - compute_started_at

            self.result[task["start_row"]:task["end_row"]] = row_block

            time.sleep(DEMO_DELAY)

            with self.progress_lock:
                task["status"] = "done"
                self.rows_done += task["end_row"] - task["start_row"]
                self.compute_time += compute_duration

    def run(self) -> None:
        """Run all workers and wait for the complete result."""
        self.state = "running"
        started_at = time.perf_counter()

        threads = []
        for thread_number in range(1, NUM_THREADS + 1):
            worker_thread = threading.Thread(
                target=self.worker,
                args=(f"Thread-{thread_number}",),
            )
            threads.append(worker_thread)
            worker_thread.start()

        for worker_thread in threads:
            worker_thread.join()

        self.total_time = time.perf_counter() - started_at
        self.state = "done"
        print(
            f"Scheduler: finished in {self.total_time:.2f} s "
            f"(TensorFlow took {self.compute_time * 1000:.1f} ms)"
        )

    def get_progress(self) -> dict:
        """Return a consistent snapshot for the web interface."""
        with self.progress_lock:
            return {
                "state": self.state,
                "num_threads": NUM_THREADS,
                "rows_done": self.rows_done,
                "total_rows": self.matrix_a.shape[0],
                "tasks": [task.copy() for task in self.tasks],
                "total_time": round(self.total_time, 3),
                "compute_time": round(self.compute_time * 1000, 2),
            }
