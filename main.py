# main.py - run this file to start the project
#   python main.py        -> web server for the animation, open http://localhost:8000
#   python main.py test   -> run the multiplication once in the console and check it

print("Starting... (importing tensorflow takes a few seconds)")

import json
import os
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import matrix
from scheduler import Scheduler

PORT = 8000
FRONTEND = os.path.join(os.path.dirname(__file__), "frontend")

# shared between the server and the multiplication run
current = None    # the Scheduler of the current run
verified = None   # True/False after the answer was checked
sample = None     # top left corner of the result for the web page


def run_demo():
    global current, verified, sample

    A = matrix.create_matrix(matrix.MATRIX_SIZE, matrix.MATRIX_SIZE)
    B = matrix.create_matrix(matrix.MATRIX_SIZE, matrix.MATRIX_SIZE)
    print("\nCreated matrix A and B, size:", A.shape)
    print("A top left 3x3:\n", np.round(A[:3, :3], 1))
    print("B top left 3x3:\n", np.round(B[:3, :3], 1))

    current = Scheduler(A, B)
    current.run()  # waits here until all threads are finished

    # check our threaded answer against tensorflow doing it in one go
    sample = [[round(float(v), 1) for v in row] for row in current.result[:4, :4]]
    verified = matrix.check_result(A, B, current.result)
    print("Answer correct:", verified)


class MyHandler(SimpleHTTPRequestHandler):
    # serves the frontend folder + 2 small json endpoints

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND, **kwargs)

    def do_GET(self):
        global current, verified, sample
        if self.path == "/api/start":
            if current is None or current.state != "running":
                current = None  # forget the old run
                verified = None
                sample = None
                # background thread so the server can still answer /api/progress
                threading.Thread(target=run_demo, daemon=True).start()
            self.send_json({"ok": True})
        elif self.path == "/api/progress":
            if current is None:
                self.send_json({"state": "idle"})
            else:
                data = current.get_progress()
                data["verified"] = verified
                data["sample"] = sample
                self.send_json(data)
        else:
            super().do_GET()

    def send_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # don't print a line for every request


def console_test():
    run_demo()
    print("\n--- test with 100 x 100 matrices ---")
    print("rows done:", current.rows_done, "/", matrix.MATRIX_SIZE)
    print("total time: %.2f s (includes the demo pause per task)" % current.total_time)
    print("pure tensorflow time: %.1f ms" % (current.compute_time * 1000))
    print("result top left 4x4:")
    for row in sample:
        print("  ", row)
    print("TEST PASSED" if verified else "TEST FAILED")


if __name__ == "__main__":
    # first tensorflow call is always slow, warm it up with a tiny multiplication
    matrix.multiply_rows(matrix.create_matrix(2, 2), matrix.create_matrix(2, 2), 0, 2)

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        console_test()
    else:
        print("Server running on http://localhost:%d (Ctrl+C to stop)" % PORT)
        ThreadingHTTPServer(("localhost", PORT), MyHandler).serve_forever()
