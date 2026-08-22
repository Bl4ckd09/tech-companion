const elements = {
  form: document.querySelector("#help-form"),
  helpText: document.querySelector("#help-text"),
  startButton: document.querySelector("#start-button"),
  exampleButton: document.querySelector("#example-button"),
  message: document.querySelector("#message"),
  guidance: document.querySelector("#guidance"),
  understoodAs: document.querySelector("#understood-as"),
  instruction: document.querySelector("#instruction"),
  stepNumber: document.querySelector("#step-number"),
  actionRow: document.querySelector("#action-row"),
  assistantButton: document.querySelector("#assistant-button"),
  refreshButton: document.querySelector("#refresh-button"),
  manualHint: document.querySelector("#manual-hint"),
  resetButton: document.querySelector("#reset-button"),
  diagnosticList: document.querySelector("#diagnostic-list"),
  deviceChip: document.querySelector("#device-chip"),
  deviceLabel: document.querySelector("#device-label"),
  phonePanel: document.querySelector(".phone-panel"),
  phoneScreen: document.querySelector("#phone-screen"),
  screenEmpty: document.querySelector("#screen-empty"),
  screenWrap: document.querySelector("#screen-wrap"),
  tapRing: document.querySelector("#tap-ring"),
};

const state = {
  session: null,
  busy: false,
};

async function requestJson(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = payload?.detail || `The request failed with status ${response.status}.`;
    throw new Error(detail);
  }
  return payload;
}

function setBusy(busy) {
  state.busy = busy;
  elements.startButton.disabled = busy;
  elements.exampleButton.disabled = busy;
  elements.assistantButton.disabled = busy;
  elements.refreshButton.disabled = busy;
  elements.resetButton.disabled = busy;
  elements.startButton.textContent = busy ? "Working..." : "Show me how";
}

function showError(error) {
  elements.message.textContent = error instanceof Error ? error.message : String(error);
  elements.message.hidden = false;
}

function clearError() {
  elements.message.hidden = true;
  elements.message.textContent = "";
}

function formatLatency(value) {
  return typeof value === "number" ? `${value} ms` : null;
}

function renderDiagnostics(session) {
  const rows = {
    "Intent router": session.providers.intent_router,
    "Visual guide": session.providers.visual_agent,
    Android: session.providers.android,
    "Intent model": session.diagnostics.intent_model,
    "Intent score": session.diagnostics.intent_score,
    "Intent latency": formatLatency(session.diagnostics.intent_latency_ms),
    "Visual latency": formatLatency(session.diagnostics.visual_latency_ms),
    "Visual steps": session.diagnostics.visual_steps,
    "Fallback used": session.fallback_used ? "yes" : "no",
  };
  elements.diagnosticList.replaceChildren();
  for (const [label, value] of Object.entries(rows)) {
    if (value === null || value === undefined || value === "") {
      continue;
    }
    const term = document.createElement("dt");
    const detail = document.createElement("dd");
    term.textContent = label;
    detail.textContent = String(value);
    elements.diagnosticList.append(term, detail);
  }
}

function renderSession(session) {
  state.session = session;
  elements.guidance.hidden = false;
  elements.guidance.classList.toggle("blocked", session.status === "blocked");
  elements.guidance.classList.toggle("complete", session.status === "complete");
  elements.understoodAs.textContent = session.understood_as;
  elements.instruction.textContent = session.instruction;
  const visualSteps = Number(session.diagnostics.visual_steps || 0);
  elements.stepNumber.textContent = session.status === "complete" ? "OK" : String(Math.max(1, visualSteps));

  const actionable = session.status === "guidance" && session.action && session.action.type !== "done";
  elements.actionRow.hidden = !actionable;
  elements.manualHint.hidden = !actionable;
  elements.phonePanel.classList.toggle("interactive", Boolean(actionable));

  if (session.status === "blocked") {
    elements.stepNumber.textContent = "!";
  }
  if (session.status === "unsupported" || session.status === "error") {
    elements.stepNumber.textContent = "i";
  }

  if (session.screenshot_url) {
    elements.phoneScreen.src = session.screenshot_url;
    elements.phoneScreen.hidden = false;
    elements.screenEmpty.hidden = true;
  } else {
    elements.phoneScreen.hidden = true;
    elements.phoneScreen.removeAttribute("src");
    elements.screenEmpty.hidden = false;
  }
  renderDiagnostics(session);
}

async function startSession(text) {
  clearError();
  setBusy(true);
  try {
    const session = await requestJson("/api/session", {
      method: "POST",
      body: JSON.stringify({ text }),
    });
    renderSession(session);
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
  }
}

async function advanceSession(mode, coordinates = {}) {
  if (!state.session || state.busy) {
    return;
  }
  clearError();
  setBusy(true);
  try {
    const session = await requestJson(`/api/session/${state.session.session_id}/execute`, {
      method: "POST",
      body: JSON.stringify({ mode, ...coordinates }),
    });
    renderSession(session);
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
  }
}

async function refreshSession() {
  if (!state.session || state.busy) {
    return;
  }
  clearError();
  setBusy(true);
  try {
    const session = await requestJson(`/api/session/${state.session.session_id}/refresh`, {
      method: "POST",
    });
    renderSession(session);
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
  }
}

async function resetDemo() {
  clearError();
  setBusy(true);
  try {
    await requestJson("/api/demo/reset", { method: "POST" });
    state.session = null;
    elements.guidance.hidden = true;
    elements.phoneScreen.hidden = true;
    elements.phoneScreen.removeAttribute("src");
    elements.screenEmpty.hidden = false;
    elements.phonePanel.classList.remove("interactive");
    elements.helpText.value = "";
    elements.helpText.focus();
    await loadDeviceStatus();
  } catch (error) {
    showError(error);
  } finally {
    setBusy(false);
  }
}

async function loadDeviceStatus() {
  try {
    const device = await requestJson("/api/device");
    elements.deviceChip.classList.toggle("connected", device.connected);
    elements.deviceChip.classList.toggle("disconnected", !device.connected);
    elements.deviceLabel.textContent = device.connected
      ? `${device.model || "Android phone"} connected`
      : device.detail;
  } catch (error) {
    elements.deviceChip.classList.add("disconnected");
    elements.deviceLabel.textContent = "Phone unavailable";
  }
}

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = elements.helpText.value.trim();
  if (text) {
    startSession(text);
  }
});

elements.exampleButton.addEventListener("click", () => {
  elements.helpText.value = "The words on my phone are too small";
  elements.helpText.focus();
});

elements.assistantButton.addEventListener("click", () => advanceSession("assistant"));
elements.refreshButton.addEventListener("click", refreshSession);
elements.resetButton.addEventListener("click", resetDemo);

elements.screenWrap.addEventListener("click", (event) => {
  const actionable = state.session?.status === "guidance" && state.session.action?.type !== "done";
  if (!actionable || state.busy || elements.phoneScreen.hidden) {
    return;
  }
  const rect = elements.phoneScreen.getBoundingClientRect();
  const x = Math.round(((event.clientX - rect.left) / rect.width) * 1000);
  const y = Math.round(((event.clientY - rect.top) / rect.height) * 1000);
  if (x < 0 || x > 1000 || y < 0 || y > 1000) {
    return;
  }
  elements.tapRing.style.left = `${event.clientX - rect.left}px`;
  elements.tapRing.style.top = `${event.clientY - rect.top}px`;
  elements.tapRing.hidden = false;
  advanceSession("manual", { x, y });
});

loadDeviceStatus();
