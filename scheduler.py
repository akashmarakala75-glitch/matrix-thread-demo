# scheduler.py - the scheduler and the worker threads
# the scheduler cuts matrix A into row tasks and the threads take them
# from a queue one by one until nothing is left

import queue
import threading
import time

import numpy as np

import matrix

NUM_THREADS = 4
ROWS_PER_TASK = 5  # 100 rows / 5 = 20 tasks
DEMO_DELAY = 0.5   # pause after each task so the animation is visible (0 = full speed)


class Scheduler:

    def __init__(self, A, B):
        self.A = A
        self.B = B
        self.result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)

        # make the task list, every task = a range of rows of A
        self.tasks = []
        self.task_queue = queue.Queue()
        task_id = 0
        for start in range(0, A.shape[0], ROWS_PER_TASK):
            end = min(start + ROWS_PER_TASK, A.shape[0])
            task = {"id": task_id, "start_row": start, "end_row": end,
                    "status": "pending", "thread": None}
            self.tasks.append(task)
            self.task_queue.put(task)
            task_id += 1

        self.lock = threading.Lock()
        self.state = "idle"
        self.rows_done = 0
        self.total_time = 0
        self.compute_time = 0
        print("Scheduler: made", len(self.tasks), "tasks for", NUM_THREADS, "threads")

    def worker(self, name):
        # this runs inside every thread, keeps taking tasks until none are left
        while True:
            try:
                task = self.task_queue.get_nowait()
            except queue.Empty:
                break  # no more tasks, this thread is done

            task["status"] = "working"
            task["thread"] = name
            print(" ", name, "doing rows", task["start_row"], "to", task["end_row"] - 1)

            # the actual multiplication of these rows (tensorflow)
            t0 = time.time()
            block = matrix.multiply_rows(self.A, self.B, task["start_row"], task["end_row"])
            t1 = time.time()

            # put the rows into the result (every task has different rows so this is safe)
            self.result[task["start_row"]:task["end_row"]] = block

            time.sleep(DEMO_DELAY)  # just so humans can watch the animation

            with self.lock:  # lock because all threads update these counters
                task["status"] = "done"
                self.rows_done += task["end_row"] - task["start_row"]
                self.compute_time += t1 - t0

    def run(self):
        self.state = "running"
        start = time.time()

        threads = []
        for i in range(NUM_THREADS):
            t = threading.Thread(target=self.worker, args=("Thread-" + str(i + 1),))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        self.total_time = time.time() - start
        self.state = "done"
        print("Scheduler: finished in %.2f s (tensorflow itself took %.1f ms)"
              % (self.total_time, self.compute_time * 1000))

    def get_progress(self):
        # everything the web page needs to draw the animation
        return {
            "state": self.state,
            "num_threads": NUM_THREADS,
            "rows_done": self.rows_done,
            "total_rows": self.A.shape[0],
            "tasks": self.tasks,
            "total_time": round(self.total_time, 3),
            "compute_time": round(self.compute_time * 1000, 2),  # ms
        }
