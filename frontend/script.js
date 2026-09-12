const THREAD_COLORS = ["#2196f3", "#4caf50", "#ff9800", "#9c27b0"];
const POLL_INTERVAL_MS = 200;

function getElement(id) {
  return document.getElementById(id);
}

let pollTimer = null;
let isInterfaceBuilt = false;
let previousTaskStatuses = {};
let threadColors = {};
let startedAt = 0;

for (let cell = 0; cell < 100; cell++) {
  getElement("matrixB").insertAdjacentHTML("beforeend", '<div class="cell"></div>');
}
makeStrips(getElement("matrixA"), 20);
makeStrips(getElement("matrixR"), 20);

getElement("startBtn").addEventListener("click", startDemo);

async function startDemo() {
  const startButton = getElement("startBtn");
  startButton.disabled = true;
  getElement("finalInfo").classList.add("hidden");
  isInterfaceBuilt = false;
  previousTaskStatuses = {};
  startedAt = Date.now();
  getElement("status").textContent = "Starting...";

  try {
    const response = await fetch("/api/start");
    if (!response.ok) throw new Error(`Server returned ${response.status}`);
    clearInterval(pollTimer);
    pollTimer = setInterval(updateProgress, POLL_INTERVAL_MS);
  } catch (error) {
    showRequestError(error);
  }
}

async function updateProgress() {
  try {
    const response = await fetch("/api/progress");
    if (!response.ok) throw new Error(`Server returned ${response.status}`);
    const progress = await response.json();
    if (progress.state === "idle") return;

    if (!isInterfaceBuilt) buildInterface(progress);

    for (const task of progress.tasks) {
      const previousStatus = previousTaskStatuses[task.id] || "pending";
      if (task.status !== previousStatus) {
        if (task.status === "working") markTaskStarted(task);
        if (task.status === "done") {
          if (previousStatus === "pending") markTaskStarted(task);
          markTaskDone(task);
        }
        previousTaskStatuses[task.id] = task.status;
      }
    }

    updateThreadStatuses(progress);
    updateOverallProgress(progress);

    if (progress.state === "done" && progress.verified !== null) {
      clearInterval(pollTimer);
      showSummary(progress);
    }
  } catch (error) {
    showRequestError(error);
  }
}

function updateThreadStatuses(progress) {
  for (const threadName in threadColors) {
    const activeTask = progress.tasks.find(
      task => task.thread === threadName && task.status === "working"
    );
    const statusElement = getElement(`status-${threadName}`);

    statusElement.textContent = activeTask
      ? `Working on rows ${activeTask.start_row}-${activeTask.end_row - 1}`
      : progress.state === "done" ? "Finished" : "Waiting...";
  }
}

function updateOverallProgress(progress) {
  const percentage = 100 * progress.rows_done / progress.total_rows;
  const elapsedSeconds = progress.state === "done"
    ? progress.total_time
    : (Date.now() - startedAt) / 1000;

  getElement("progressBar").style.width = `${percentage}%`;
  getElement("progressText").textContent = `${progress.rows_done} / ${progress.total_rows} rows`;
  getElement("timer").textContent = `${elapsedSeconds.toFixed(1)} s`;
}

function buildInterface(progress) {
  getElement("matrixA").innerHTML = "";
  getElement("matrixR").innerHTML = "";
  getElement("taskQueue").innerHTML = "";
  getElement("threads").innerHTML = "";
  threadColors = {};

  for (const task of progress.tasks) {
    const rowLabel = `${task.start_row}-${task.end_row - 1}`;
    getElement("matrixA").insertAdjacentHTML(
      "beforeend", `<div class="strip" id="a-${task.id}"></div>`
    );
    getElement("matrixR").insertAdjacentHTML(
      "beforeend", `<div class="strip" id="r-${task.id}"></div>`
    );
    getElement("taskQueue").insertAdjacentHTML(
      "beforeend", `<span class="chip" id="chip-${task.id}">${rowLabel}</span>`
    );
  }

  for (let threadNumber = 1; threadNumber <= progress.num_threads; threadNumber++) {
    const threadName = `Thread-${threadNumber}`;
    const color = THREAD_COLORS[(threadNumber - 1) % THREAD_COLORS.length];
    threadColors[threadName] = color;
    getElement("threads").insertAdjacentHTML("beforeend", `
      <div class="thread-card" id="card-${threadName}" style="border-left-color: ${color}">
        <div class="thread-name">
          <span class="dot" style="background: ${color}"></span>${threadName}
        </div>
        <div class="thread-status" id="status-${threadName}">Waiting...</div>
        <div class="thread-done" id="done-${threadName}"></div>
      </div>
    `);
  }
  isInterfaceBuilt = true;
}

function markTaskStarted(task) {
  const color = threadColors[task.thread];
  const strip = getElement(`a-${task.id}`);
  strip.style.background = color;
  strip.classList.add("working");
  getElement(`chip-${task.id}`).classList.add("taken");
  animateTaskAssignment(
    getElement(`chip-${task.id}`),
    getElement(`card-${task.thread}`),
    color
  );
}

function markTaskDone(task) {
  const color = threadColors[task.thread];
  const inputStrip = getElement(`a-${task.id}`);
  inputStrip.classList.remove("working");
  inputStrip.classList.add("done");
  getElement(`r-${task.id}`).style.background = color;
  getElement(`chip-${task.id}`).classList.add("done");
  getElement(`done-${task.thread}`).insertAdjacentHTML(
    "beforeend", `<span class="mini-chip" style="background: ${color}"></span>`
  );
}

function animateTaskAssignment(source, destination, color) {
  const sourcePosition = source.getBoundingClientRect();
  const destinationPosition = destination.getBoundingClientRect();
  const movingChip = document.createElement("span");
  movingChip.className = "fly-chip";
  movingChip.textContent = source.textContent;
  movingChip.style.background = color;
  movingChip.style.left = `${sourcePosition.left}px`;
  movingChip.style.top = `${sourcePosition.top}px`;
  document.body.appendChild(movingChip);

  setTimeout(() => {
    const horizontalDistance = destinationPosition.left - sourcePosition.left + 10;
    const verticalDistance = destinationPosition.top - sourcePosition.top + 15;
    movingChip.style.transform = `translate(${horizontalDistance}px, ${verticalDistance}px)`;
    movingChip.style.opacity = "0.2";
  }, 20);
  setTimeout(() => movingChip.remove(), 600);
}

function showSummary(data) {
  const startButton = getElement("startBtn");
  startButton.disabled = false;
  startButton.textContent = "Run again";
  getElement("status").textContent = data.verified
    ? "Done - result verified"
    : "Done - verification failed";

  let tableRows = "";
  for (const row of data.sample) {
    tableRows += "<tr>";
    for (const value of row) tableRows += `<td>${value}</td>`;
    tableRows += "<td>...</td></tr>";
  }
  tableRows += "<tr><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr>";

  const summary = getElement("finalInfo");
  summary.innerHTML = `
    <div class="badge ${data.verified ? "ok" : "bad"}">
      ${data.verified ? "Correct: matches a single tf.matmul" : "Verification failed"}
    </div>
    <p>Total time: <b>${data.total_time.toFixed(2)} s</b> (includes demo delays)<br>
    TensorFlow time: <b>${data.compute_time} ms</b></p>
    <p>Top-left corner of the result:</p>
    <table>${tableRows}</table>
  `;
  summary.classList.remove("hidden");
}

function showRequestError(error) {
  clearInterval(pollTimer);
  getElement("status").textContent = "Could not reach the server";
  getElement("startBtn").disabled = false;
  console.error(error);
}

function makeStrips(container, count) {
  for (let strip = 0; strip < count; strip++) {
    container.insertAdjacentHTML("beforeend", '<div class="strip"></div>');
  }
}
