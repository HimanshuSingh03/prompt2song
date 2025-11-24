const form = document.getElementById("promptForm");
const promptInput = document.getElementById("promptInput");
const topKInput = document.getElementById("topKInput");
const startBtn = document.getElementById("startBtn");
const statusBadge = document.getElementById("statusBadge");
const logOutput = document.getElementById("logOutput");
const questionSection = document.getElementById("questionSection");
const questionCounter = document.getElementById("questionCounter");
const songATitle = document.getElementById("songATitle");
const songAArtist = document.getElementById("songAArtist");
const songAEmbed = document.getElementById("songAEmbed");
const songBTitle = document.getElementById("songBTitle");
const songBArtist = document.getElementById("songBArtist");
const songBEmbed = document.getElementById("songBEmbed");
const chooseABtn = document.getElementById("chooseABtn");
const chooseBBtn = document.getElementById("chooseBBtn");
const skipBtn = document.getElementById("skipBtn");
const resultsSection = document.getElementById("resultsSection");
const resultsTable = document.getElementById("resultsTable");
const csvPathEl = document.getElementById("csvPath");

let sessionId = null;

function setStatus(text, variant = "default") {
  statusBadge.textContent = text;
  statusBadge.classList.toggle("success", variant === "success");
}

function renderLogs(logs = []) {
  if (!Array.isArray(logs) || logs.length === 0) {
    logOutput.textContent = "Waiting for output…";
    return;
  }
  logOutput.textContent = logs.join("\n");
  logOutput.scrollTop = logOutput.scrollHeight;
}

function embedContent(container, info) {
  const embedUrl = info?.spotify_embed;
  if (embedUrl) {
    container.innerHTML = `<iframe src="${embedUrl}" allow="encrypted-media; clipboard-write"></iframe>`;
  } else {
    container.textContent = "No Spotify snippet available for this track.";
  }
}

function renderQuestion(question) {
  if (!question) {
    questionSection.hidden = true;
    return;
  }
  questionSection.hidden = false;
  questionCounter.textContent = `Question ${question.index} / ${question.total}`;
  songATitle.textContent = question.a.name || "Unknown title";
  songAArtist.textContent = question.a.artists || "Unknown artist";
  embedContent(songAEmbed, question.a);

  songBTitle.textContent = question.b.name || "Unknown title";
  songBArtist.textContent = question.b.artists || "Unknown artist";
  embedContent(songBEmbed, question.b);
}

function renderResults(finalCsv) {
  if (!finalCsv || !finalCsv.rows || finalCsv.rows.length === 0) {
    resultsSection.hidden = true;
    return;
  }
  resultsSection.hidden = false;
  csvPathEl.textContent = finalCsv.path || finalCsv.filename || "recs_rlhf.csv";

  const rows = finalCsv.rows;
  const headers = Object.keys(rows[0]).filter((h) => h !== "lyrics");
  if (headers.length === 0) {
    resultsSection.hidden = true;
    return;
  }
  const thead = resultsTable.querySelector("thead");
  const tbody = resultsTable.querySelector("tbody");
  thead.innerHTML = `<tr>${headers.map((h) => `<th>${h}</th>`).join("")}</tr>`;
  tbody.innerHTML = rows
    .map((row) => `<tr>${headers.map((h) => `<td>${row[h] ?? ""}</td>`).join("")}</tr>`)
    .join("");
}

async function postJSON(url, payload) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(text || "Request failed");
  }
  return resp.json();
}

function applyPayload(data) {
  if (!data) {
    return;
  }
  if (data.sessionId) {
    sessionId = data.sessionId;
  }
  renderLogs(data.logs);
  renderQuestion(data.question);
  if (data.completed) {
    setStatus("Complete", "success");
    renderResults(data.finalCsv);
  } else if (data.question) {
    setStatus("Awaiting RLHF feedback");
  } else {
    setStatus("Working…");
  }
}

async function startRun(event) {
  event.preventDefault();
  const prompt = promptInput.value.trim();
  const topKRaw = topKInput.value.trim();
  if (!prompt) return;

  resultsSection.hidden = true;
  sessionId = null;
  setStatus("Starting…");
  startBtn.disabled = true;
  chooseABtn.disabled = true;
  chooseBBtn.disabled = true;
  skipBtn.disabled = true;

  const payload = { prompt };
  if (topKRaw) {
    payload.top_k = Number(topKRaw);
  }

  try {
    const data = await postJSON("/api/start", payload);
    applyPayload(data);
    chooseABtn.disabled = !data.question;
    chooseBBtn.disabled = !data.question;
    skipBtn.disabled = !data.question;
  } catch (err) {
    setStatus("Error");
    logOutput.textContent = `Error: ${err.message}`;
  } finally {
    startBtn.disabled = false;
  }
}

async function answer(choice) {
  if (!sessionId) return;
  chooseABtn.disabled = true;
  chooseBBtn.disabled = true;
  skipBtn.disabled = true;
  setStatus("Submitting choice…");
  try {
    const data = await postJSON("/api/answer", { sessionId, choice });
    applyPayload(data);
    chooseABtn.disabled = data.completed || !data.question;
    chooseBBtn.disabled = data.completed || !data.question;
    skipBtn.disabled = data.completed || !data.question;
  } catch (err) {
    logOutput.textContent += `\nError: ${err.message}`;
    setStatus("Error");
  }
}

form.addEventListener("submit", startRun);
chooseABtn.addEventListener("click", () => answer("a"));
chooseBBtn.addEventListener("click", () => answer("b"));
skipBtn.addEventListener("click", () => answer("skip"));
