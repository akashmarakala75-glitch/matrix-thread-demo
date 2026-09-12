"""Run the matrix multiplication demo in a browser or the console."""

print("Starting... TensorFlow may take a few seconds to import.")

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

active_scheduler = None
verification_result = None
result_sample = None
demo_is_starting = False
demo_lock = threading.Lock()


def run_demo() -> None:
    """Create two matrices, multiply them, and verify the result."""
    global active_scheduler, verification_result, result_sample, demo_is_starting

    matrix_a = matrix.create_matrix(matrix.MATRIX_SIZE, matrix.MATRIX_SIZE)
    matrix_b = matrix.create_matrix(matrix.MATRIX_SIZE, matrix.MATRIX_SIZE)
    print(f"\nCreated matrices A and B with shape {matrix_a.shape}")
    print("Top-left 3x3 of A:\n", np.round(matrix_a[:3, :3], 1))
    print("Top-left 3x3 of B:\n", np.round(matrix_b[:3, :3], 1))

    scheduler = Scheduler(matrix_a, matrix_b)
    with demo_lock:
        active_scheduler = scheduler
        demo_is_starting = False
    scheduler.run()

    sample = [
        [round(float(value), 1) for value in row]
        for row in scheduler.result[:4, :4]
    ]
    verified = matrix.check_result(matrix_a, matrix_b, scheduler.result)
    with demo_lock:
        result_sample = sample
        verification_result = verified
    print(f"Result verified: {verified}")


def start_demo() -> bool:
    """Start a run unless one is already being prepared or processed."""
    global active_scheduler, verification_result, result_sample, demo_is_starting

    with demo_lock:
        already_running = demo_is_starting or (
            active_scheduler is not None and active_scheduler.state == "running"
        )
        if already_running:
            return False

        active_scheduler = None
        verification_result = None
        result_sample = None
        demo_is_starting = True

    threading.Thread(target=run_demo, daemon=True).start()
    return True


class DemoRequestHandler(SimpleHTTPRequestHandler):
    """Serve the frontend and the demo's two JSON endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=FRONTEND, **kwargs)

    def do_GET(self):
        if self.path == "/api/start":
            self.send_json({"ok": start_demo()})
        elif self.path == "/api/progress":
            with demo_lock:
                scheduler = active_scheduler
                verified = verification_result
                sample = result_sample

            if scheduler is None:
                self.send_json({"state": "idle"})
            else:
                data = scheduler.get_progress()
                data["verified"] = verified
                data["sample"] = sample
                self.send_json(data)
        else:
            super().do_GET()

    def send_json(self, data: dict) -> None:
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def console_test() -> None:
    run_demo()
    scheduler = active_scheduler
    print("\n--- test with 100 x 100 matrices ---")
    print(f"Rows completed: {scheduler.rows_done} / {matrix.MATRIX_SIZE}")
    print(f"Total time: {scheduler.total_time:.2f} s (includes demo pauses)")
    print(f"TensorFlow time: {scheduler.compute_time * 1000:.1f} ms")
    print("Top-left 4x4 of the result:")
    for row in result_sample:
        print(f"  {row}")
    print("TEST PASSED" if verification_result else "TEST FAILED")


if __name__ == "__main__":
    matrix.multiply_rows(matrix.create_matrix(2, 2), matrix.create_matrix(2, 2), 0, 2)

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        console_test()
    else:
        print(f"Server running at http://localhost:{PORT} (Ctrl+C to stop)")
        ThreadingHTTPServer(("localhost", PORT), DemoRequestHandler).serve_forever()
