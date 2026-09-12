// script.js - asks the python server for progress every 200 ms and animates it.
// Matrix A and the result are drawn as 20 strips (1 strip = 1 task = 5 rows).

const COLORS = ["#2196f3", "#4caf50", "#ff9800", "#9c27b0"]; // one color per thread

function $(id) { return document.getElementById(id); }

let pollTimer = null;
let built = false;     // did we build the strips/cards for this run yet
let lastStatus = {};   // task id -> status we saw in the previous poll
let threadColor = {};  // thread name -> color
let startedAt = 0;

// matrix B never changes so just draw a 10x10 block picture for it,
// and grey placeholder strips for A and the result
for (let i = 0; i < 100; i++) {
  $("matrixB").innerHTML += '<div class="cell"></div>';
}
makeStrips($("matrixA"), 20);
makeStrips($("matrixR"), 20);

$("startBtn").onclick = async function () {
  $("startBtn").disabled = true;
  $("finalInfo").classList.add("hidden");
  built = false;
  lastStatus = {};
  startedAt = Date.now();
  $("status").textContent = "running...";
  await fetch("/api/start");
  pollTimer = setInterval(poll, 200);
};

async function poll() {
  const res = await fetch("/api/progress");
  const data = await res.json();
  if (data.state === "idle") return; // python is still creating the matrices

  if (!built) buildEverything(data);

  // react to tasks that changed since the last poll
  for (const task of data.tasks) {
    const before = lastStatus[task.id] || "pending";
    if (task.status !== before) {
      if (task.status === "working") taskStarted(task);
      if (task.status === "done") {
        if (before === "pending") taskStarted(task); // we missed the working step
        taskDone(task);
      }
      lastStatus[task.id] = task.status;
    }
  }

  // status text on every thread card
  for (const name in threadColor) {
    const now = data.tasks.find(t => t.thread === name && t.status === "working");
    if (now) {
      $("status-" + name).textContent = "working on rows " + now.start_row + "-" + (now.end_row - 1);
    } else {
      $("status-" + name).textContent = data.state === "done" ? "finished" : "waiting...";
    }
  }

  $("progressBar").style.width = (100 * data.rows_done / data.total_rows) + "%";
  $("progressText").textContent = data.rows_done + " / " + data.total_rows + " rows";
  const secs = data.state === "done" ? data.total_time : (Date.now() - startedAt) / 1000;
  $("timer").textContent = secs.toFixed(1) + " s";

  // finished? stop polling and show the summary
  if (data.state === "done" && data.verified !== null) {
    clearInterval(pollTimer);
    showSummary(data);
  }
}

function buildEverything(data) {
  $("matrixA").innerHTML = "";
  $("matrixR").innerHTML = "";
  $("taskQueue").innerHTML = "";
  $("threads").innerHTML = "";
  threadColor = {};

  // a strip in A, a strip in the result and a queue chip for every task
  for (const task of data.tasks) {
    const label = task.start_row + "-" + (task.end_row - 1);
    $("matrixA").innerHTML += '<div class="strip" id="a-' + task.id + '"></div>';
    $("matrixR").innerHTML += '<div class="strip" id="r-' + task.id + '"></div>';
    $("taskQueue").innerHTML += '<span class="chip" id="chip-' + task.id + '">' + label + "</span>";
  }

  // a card for every thread
  for (let i = 0; i < data.num_threads; i++) {
    const name = "Thread-" + (i + 1);
    threadColor[name] = COLORS[i % COLORS.length];
    $("threads").innerHTML +=
      '<div class="thread-card" id="card-' + name + '" style="border-left-color:' + threadColor[name] + '">' +
      '<div class="thread-name"><span class="dot" style="background:' + threadColor[name] + '"></span>' + name + "</div>" +
      '<div class="thread-status" id="status-' + name + '">waiting...</div>' +
      '<div class="thread-done" id="done-' + name + '"></div></div>';
  }
  built = true;
}

// a thread took a task -> color the strip in A and fly a chip to the thread
function taskStarted(task) {
  const color = threadColor[task.thread];
  const strip = $("a-" + task.id);
  strip.style.background = color;
  strip.classList.add("working");
  $("chip-" + task.id).classList.add("taken");
  flyChip($("chip-" + task.id), $("card-" + task.thread), color);
}

// a task is finished -> fill the result strip with the thread's color
function taskDone(task) {
  const color = threadColor[task.thread];
  const stripA = $("a-" + task.id);
  stripA.classList.remove("working");
  stripA.classList.add("done");
  $("r-" + task.id).style.background = color;
  $("chip-" + task.id).classList.add("done");
  $("done-" + task.thread).innerHTML += '<span class="mini-chip" style="background:' + color + '"></span>';
}

// a little chip flies from the scheduler box to the thread card
function flyChip(fromEl, toEl, color) {
  const a = fromEl.getBoundingClientRect();
  const b = toEl.getBoundingClientRect();
  const fly = document.createElement("span");
  fly.className = "fly-chip";
  fly.textContent = fromEl.textContent;
  fly.style.background = color;
  fly.style.left = a.left + "px";
  fly.style.top = a.top + "px";
  document.body.appendChild(fly);
  setTimeout(function () { // small delay so the browser animates the move
    fly.style.transform = "translate(" + (b.left - a.left + 10) + "px," + (b.top - a.top + 15) + "px)";
    fly.style.opacity = "0.2";
  }, 20);
  setTimeout(function () { fly.remove(); }, 600);
}

function showSummary(data) {
  $("startBtn").disabled = false;
  $("startBtn").textContent = "Run again";
  $("status").textContent = data.verified ? "done - result verified" : "done - WRONG RESULT";

  let rows = "";
  for (const r of data.sample) {
    rows += "<tr>";
    for (const v of r) rows += "<td>" + v + "</td>";
    rows += "<td>...</td></tr>";
  }
  rows += "<tr><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr>";

  $("finalInfo").innerHTML =
    '<div class="badge ' + (data.verified ? "ok" : "bad") + '">' +
    (data.verified ? "Correct: same answer as one big tf.matmul" : "Verification failed") + "</div>" +
    "<p>Total time: <b>" + data.total_time.toFixed(2) + " s</b> (includes the demo delays)<br>" +
    "Pure TensorFlow time: <b>" + data.compute_time + " ms</b></p>" +
    "<p>Top left corner of the result:</p><table>" + rows + "</table>";
  $("finalInfo").classList.remove("hidden");
}

function makeStrips(container, n) {
  for (let i = 0; i < n; i++) {
    container.innerHTML += '<div class="strip"></div>';
  }
}
