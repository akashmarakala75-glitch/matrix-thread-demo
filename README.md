# 100 x 100 Matrix Multiplication with Threads + TensorFlow

My project for demonstrating thread based task scheduling:
two 100x100 matrices get multiplied, but the work is split into small
tasks by a scheduler and done by 4 worker threads with TensorFlow.
A web page shows the whole thing as an animation.

Flow: Matrix A + B -> scheduler makes tasks -> threads take tasks ->
TensorFlow multiplies the rows -> result matrix -> checked and animated.

## Files

```
matrix-thread-demo/
|-- main.py           entry point, small web server + a console test mode
|-- scheduler.py      the scheduler and the worker threads
|-- matrix.py         creates the matrices + the TensorFlow multiplication
|-- frontend/
|   |-- index.html    page layout
|   |-- style.css     styling and css animations
|   |-- script.js     asks the server for progress and animates it
|-- requirements.txt
```

## How to run

Needs Python 3.9 - 3.12 (TensorFlow does not support every version) and pip.

```
cd matrix-thread-demo
py -3.12 -m venv .venv          (Linux/Mac: python3 -m venv .venv)
.venv\Scripts\activate          (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

python main.py test    runs everything once in the console and checks the answer
python main.py         starts the server, then open http://localhost:8000
```

## How the scheduler works (scheduler.py)

1. In `Scheduler.__init__` the 100 rows of matrix A get divided into tasks of
   5 rows each, so 20 tasks. A task is just a small dictionary like
   `{"id": 3, "start_row": 15, "end_row": 20, "status": "pending", "thread": None}`.
2. All tasks go into a `queue.Queue`, that is the scheduler's to-do list.
3. A free thread takes the next task from the queue (`get_nowait()` in
   `worker()`). So a fast thread automatically gets more tasks and no thread
   sits around doing nothing. This is dynamic scheduling / load balancing.
4. When the queue is empty the threads just stop.

## How the threads work (scheduler.py)

- The threads are created in `Scheduler.run()` with
  `threading.Thread(target=self.worker, ...)`, 4 of them.
- Every thread runs `worker()`: take a task, multiply those rows, write them
  into the shared result matrix, repeat.
- Writing the result needs no lock because every task has different rows.
  A `threading.Lock` is only used for the shared counters (rows_done etc).
- `t.join()` makes the program wait for all threads, and `time.time()` around
  the whole run measures the execution time.

Note about the GIL: Python threads share one interpreter, so normal Python
code does not run on multiple cores at the same time. TensorFlow releases the
GIL during the heavy math so the threads do overlap, but the main point of
this project is showing how tasks are scheduled and distributed, not raw speed.

## How TensorFlow is used (matrix.py)

- `multiply_rows(A, B, start_row, end_row)` takes only the given rows of A and
  the whole B and calls `tf.matmul` on them. Every thread calls this for its
  own row range, so the TensorFlow work is really split between the tasks.
- `check_result()` is the built in test: it multiplies the full matrices with
  one normal `tf.matmul(A, B)` and compares with `np.allclose`. The demo
  prints/shows if the answer is correct.

## Execution time

Two numbers are shown:

- total time = wall clock time of the whole run. It contains the artificial
  `DEMO_DELAY` pause (0.5 s per task) which only exists so the animation is
  watchable. Set it to 0 in scheduler.py for the real speed.
- pure TensorFlow time = only the time spent inside `tf.matmul`, added up
  over all tasks. For 100x100 it is just a few milliseconds.

## How the animation works (frontend/)

- main.py runs a tiny web server (Python's built in http.server, no framework)
  with two JSON endpoints: `/api/start` starts the multiplication in a
  background thread, `/api/progress` returns the status of every task.
- script.js asks `/api/progress` every 200 ms and draws the answer:
  - Matrix A and the result are 20 strips (1 strip = 1 task = 5 real rows).
    The backend still calculates the full 100x100, the page is a simple view.
  - The scheduler box shows the task queue as small chips (0-4, 5-9, ...).
  - When a thread takes a task, a chip flies from the scheduler to that
    thread's card (css transform transition).
  - Working strips blink, finished strips get the color of the thread that
    did them, so you can see who calculated what.
  - At the end it shows the times, the correctness check and a corner of the
    result matrix.

## Where is what

| What                     | File               | Where                                  |
| ------------------------ | ------------------ | -------------------------------------- |
| threads are created      | scheduler.py       | `run()`, the `threading.Thread` loop   |
| scheduler splits work    | scheduler.py       | `__init__`, task list + queue          |
| threads get their work   | scheduler.py       | `worker()`, `task_queue.get_nowait()`  |
| matrix multiplication    | matrix.py          | `multiply_rows()`                      |
| TensorFlow               | matrix.py          | `tf.matmul(...)`                       |
| time measurement         | scheduler.py       | `run()`, `time.time()`                 |
| 100x100 test             | main.py, matrix.py | `console_test()` + `check_result()`    |
| animation                | frontend/script.js | polling + strips/chips + `flyChip()`   |

## For the presentation

1. I multiply two 100x100 matrices, but instead of one big operation I split
   the work so I can show scheduling.
2. The scheduler cuts matrix A into 20 tasks of 5 rows and puts them in a queue.
3. I create 4 worker threads. A free thread takes the next task from the
   queue, which balances the load automatically.
4. Each thread multiplies its rows with matrix B using TensorFlow's tf.matmul
   and writes the rows into the shared result.
5. The main program joins all threads and measures the time.
6. As a test I compare my threaded result with one full tf.matmul, they match.
7. The web page polls the server 5 times a second and animates the scheduler
   giving tasks to the threads and the result filling up.

Questions they might ask:

- Why is the total time seconds but the compute time only milliseconds?
  Because of the 0.5 s pause per task that makes the animation watchable.
- What if one thread is slow? The queue gives the remaining tasks to the
  other threads, so it balances itself.
- Does it work with bigger matrices? Yes, change MATRIX_SIZE in matrix.py,
  nothing is hard coded to 100.
- Is the answer correct? Yes, every run is checked with np.allclose against
  a normal full TensorFlow multiplication.

## Settings you can change

- MATRIX_SIZE in matrix.py (default 100)
- NUM_THREADS in scheduler.py (default 4)
- ROWS_PER_TASK in scheduler.py (default 5, gives 20 tasks)
- DEMO_DELAY in scheduler.py (0.5 s pause per task, 0 = full speed)

