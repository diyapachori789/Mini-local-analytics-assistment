(function () {
  "use strict";

  // Browser storage holds display preferences only. Query history is owned by
  // the backend so it survives refreshes, restarts, and cleared site data.
  const STORAGE_KEYS = Object.freeze({
    theme: "mini-local-analytics.theme",
    view: "mini-local-analytics.view",
  });
  const LEGACY_HISTORY_KEY = "mini-local-analytics.history";
  const HISTORY_LIMIT = 50;
  const CHART_TYPES = new Set(["auto", "bar", "line", "pie", "scatter"]);
  const VIEWS = new Set(["ask", "history", "charts", "about"]);

  let isSubmitting = false;
  let savedHistory = [];
  let ui;

  function initialize() {
    ui = {
      answerText: document.getElementById("answer-text"),
      answerWarning: document.getElementById("answer-warning"),
      askedBlock: document.getElementById("asked-block"),
      askedQuestion: document.getElementById("asked-question"),
      askButton: document.getElementById("ask-button"),
      chartEmpty: document.getElementById("chart-empty"),
      chartFigure: document.getElementById("chart-figure"),
      chartImage: document.getElementById("chart-image"),
      chartMessage: document.getElementById("chart-message"),
      chartNote: document.getElementById("chart-note"),
      chartRequested: document.getElementById("chart-requested"),
      chartType: document.getElementById("chart-type"),
      chartTypeBadge: document.getElementById("chart-type-badge"),
      chartsEmpty: document.getElementById("charts-empty"),
      chartGallery: document.getElementById("chart-gallery"),
      clearButtons: Array.from(document.querySelectorAll("[data-clear-session]")),
      clearHistoryButtons: Array.from(document.querySelectorAll("[data-clear-history]")),
      exampleButtons: Array.from(document.querySelectorAll("[data-example-question]")),
      form: document.getElementById("query-form"),
      historyEmpty: document.getElementById("history-empty"),
      historyList: document.getElementById("history-list"),
      liveRegion: document.getElementById("live-region"),
      navButtons: Array.from(document.querySelectorAll("[data-view-target]")),
      navToggle: document.getElementById("nav-toggle"),
      questionInput: document.getElementById("question-input"),
      questionMessage: document.getElementById("question-message"),
      resultStatus: document.getElementById("result-status"),
      resultsCard: document.getElementById("results-card"),
      resultTableBody: document.getElementById("result-table-body"),
      resultTableCaption: document.getElementById("result-table-caption"),
      resultTableHead: document.getElementById("result-table-head"),
      rowMeta: document.getElementById("row-meta"),
      sidebar: document.getElementById("sidebar"),
      sidebarBackdrop: document.getElementById("sidebar-backdrop"),
      statusDot: document.getElementById("status-dot"),
      statusList: document.getElementById("system-status-list"),
      tableWrap: document.querySelector(".table-wrap"),
      themeIcon: document.querySelector(".theme-toggle-icon"),
      themeLabel: document.querySelector(".theme-toggle-label"),
      themeToggle: document.getElementById("theme-toggle"),
      truncationNote: document.getElementById("truncation-note"),
      emptyResult: document.getElementById("empty-result"),
      views: Array.from(document.querySelectorAll("[data-view]")),
    };

    applyTheme(readStoredTheme());
    bindEvents();
    // Any history left by an earlier build is display data only and is no
    // longer authoritative, so it is discarded rather than merged.
    removeStoredValue(LEGACY_HISTORY_KEY);
    renderHistory([]);
    renderChartGallery();
    activateView(readStoredView(), false);
    loadStatus();
    loadHistory();
  }

  function bindEvents() {
    ui.form.addEventListener("submit", submitQuestion);
    ui.questionInput.addEventListener("keydown", submitOnEnter);
    ui.chartRequested.addEventListener("change", updateChartControls);
    ui.themeToggle.addEventListener("click", toggleTheme);
    ui.navToggle.addEventListener("click", toggleNavigation);
    ui.sidebarBackdrop.addEventListener("click", closeNavigation);
    ui.chartImage.addEventListener("error", showChartImageError);

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && document.body.classList.contains("nav-open")) {
        closeNavigation();
        ui.navToggle.focus();
      }
    });

    for (const button of ui.navButtons) {
      button.addEventListener("click", function () {
        activateView(button.dataset.viewTarget, true);
      });
    }

    for (const button of ui.exampleButtons) {
      button.addEventListener("click", function () {
        const question = button.dataset.exampleQuestion;
        if (typeof question !== "string") {
          return;
        }
        ui.questionInput.value = question;
        setQuestionMessage("Example added. Review it, then submit when ready.", "success");
        activateView("ask", false);
        ui.questionInput.focus();
      });
    }

    for (const button of ui.clearButtons) {
      button.addEventListener("click", clearSession);
    }

    for (const button of ui.clearHistoryButtons) {
      button.addEventListener("click", clearSavedHistory);
    }

    window.addEventListener("resize", closeNavigation);
  }

  function submitOnEnter(event) {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      ui.form.requestSubmit();
    }
  }

  async function submitQuestion(event) {
    event.preventDefault();
    if (isSubmitting) {
      return;
    }

    const question = ui.questionInput.value;
    if (question.trim().length === 0) {
      setQuestionMessage("Enter a question before asking for an analysis.", "error");
      announce("A question is required before submitting.");
      ui.questionInput.focus();
      return;
    }

    const requestBody = {
      question,
      chart_requested: ui.chartRequested.checked,
      chart_type: selectedChartType(),
    };

    setSubmitting(true);
    setQuestionMessage("Analyzing your question…", "");
    announce("Your question is being analyzed.");

    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(requestBody),
      });
      const payload = await readJson(response);

      if (!response.ok || !isRecord(payload) || payload.ok !== true) {
        throw new RequestFailure(getSafeErrorMessage(payload));
      }

      renderResult(payload, question);
      // The backend saved this request already. Reloading is the single source
      // of truth and avoids inserting a duplicate entry locally.
      loadHistory();
      setQuestionMessage("Analysis complete.", "success");
      announce("Analysis complete. The result has been updated.");
    } catch (error) {
      clearCurrentResult();
      resetChart();
      const message = error instanceof RequestFailure ? error.message : "The analysis could not be completed. Please try again later.";
      setQuestionMessage(message, "error");
      announce("The analysis could not be completed.");
    } finally {
      setSubmitting(false);
    }
  }

  async function loadStatus() {
    try {
      const response = await fetch("/api/status", { headers: { Accept: "application/json" } });
      const payload = await readJson(response);
      if (!response.ok || !isRecord(payload) || payload.ok === false) {
        throw new Error("Status unavailable");
      }
      renderStatus(payload);
    } catch (_error) {
      renderUnavailableStatus();
    }
  }

  async function readJson(response) {
    try {
      return await response.json();
    } catch (_error) {
      return null;
    }
  }

  function renderResult(payload, typedQuestion) {
    const columns = Array.isArray(payload.columns) ? payload.columns : [];
    const rows = Array.isArray(payload.rows) ? payload.rows : [];
    const answer = typeof payload.answer === "string" ? payload.answer : "";
    const question = typeof payload.question === "string" ? payload.question : typedQuestion;
    const metadata = isRecord(payload.meta) ? payload.meta : {};
    const refused = payload.refused === true || metadata.refused === true;

    ui.resultsCard.hidden = false;
    // A refusal is a normal answer, not a failure: neutral label, no error styling.
    ui.resultStatus.textContent = refused ? "ASSISTANT RESPONSE" : "RESULT";
    ui.resultStatus.classList.remove("is-error");

    // Shows which question this answer belongs to. Untrusted text, so it is set
    // as text content and never parsed as markup.
    const askedText = typeof question === "string" ? question.trim() : "";
    ui.askedQuestion.textContent = askedText;
    ui.askedBlock.hidden = askedText.length === 0;

    ui.answerText.textContent = answer;
    ui.answerWarning.hidden = metadata.answer_fallback_used !== true;
    ui.answerWarning.textContent = metadata.answer_fallback_used === true
      ? "Answer generation was unavailable, so the returned result rows are shown below."
      : "";

    if (refused) {
      // Nothing was computed, so there is no table, no row count and no chart
      // to show. Clearing them prevents stale output from a previous question.
      ui.rowMeta.textContent = "";
      ui.truncationNote.textContent = "";
      ui.truncationNote.hidden = true;
      renderResultTable([], []);
      ui.tableWrap.hidden = true;
      ui.emptyResult.hidden = true;
      resetChart();
      return;
    }

    ui.tableWrap.hidden = false;
    renderRowMetadata(payload, rows);
    renderResultTable(columns, rows);
    renderChart(payload.chart, question);
  }

  function renderRowMetadata(payload, rows) {
    const rowCount = asNonNegativeInteger(payload.row_count);
    const isTruncated = payload.truncated === true;

    if (rowCount === null) {
      ui.rowMeta.textContent = "Results returned.";
    } else {
      ui.rowMeta.textContent = `${formatCount(rowCount)} ${rowCount === 1 ? "row" : "rows"} returned.`;
    }

    ui.truncationNote.hidden = !isTruncated;
    if (isTruncated) {
      const maxRows = asNonNegativeInteger(payload.max_rows);
      if (maxRows !== null) {
        ui.truncationNote.textContent = `Showing the first ${formatCount(maxRows)} rows. Additional rows were not fetched because of the configured result limit.`;
      } else {
        ui.truncationNote.textContent = "Results were truncated by the configured result limit.";
      }
    } else {
      ui.truncationNote.textContent = "";
    }

    ui.emptyResult.hidden = rows.length !== 0;
  }

  function renderResultTable(columns, rows) {
    ui.resultTableHead.replaceChildren();
    ui.resultTableBody.replaceChildren();
    ui.tableWrap.hidden = columns.length === 0 || rows.length === 0;
    ui.resultTableCaption.textContent = "Query result rows";

    if (columns.length === 0 || rows.length === 0) {
      return;
    }

    const headerRow = document.createElement("tr");
    for (const column of columns) {
      const heading = document.createElement("th");
      heading.scope = "col";
      heading.textContent = displayValue(column);
      headerRow.append(heading);
    }
    ui.resultTableHead.append(headerRow);

    for (const sourceRow of rows) {
      const tableRow = document.createElement("tr");
      const values = Array.isArray(sourceRow) ? sourceRow : [];
      for (let index = 0; index < columns.length; index += 1) {
        const cell = document.createElement("td");
        cell.textContent = displayValue(values[index]);
        tableRow.append(cell);
      }
      ui.resultTableBody.append(tableRow);
    }
  }

  function renderChart(sourceChart, question) {
    const chart = isRecord(sourceChart) ? sourceChart : {};
    const chartUrl = safeChartUrl(chart.url);
    const chartType = safeChartType(chart.type);
    const requested = chart.requested === true;

    ui.chartNote.hidden = true;
    ui.chartNote.textContent = "";
    ui.chartMessage.hidden = true;
    ui.chartMessage.textContent = "";
    ui.chartTypeBadge.hidden = chartType === null;
    ui.chartTypeBadge.textContent = chartType === null ? "" : `${chartType} chart`;

    if (chartUrl !== null) {
      ui.chartEmpty.hidden = true;
      ui.chartFigure.hidden = false;
      ui.chartImage.alt = chartAlt(chartType, question);
      ui.chartImage.src = chartUrl;

      if (typeof chart.note === "string" && chart.note.trim().length > 0) {
        ui.chartNote.textContent = chart.note;
        ui.chartNote.hidden = false;
      }
      return;
    }

    ui.chartFigure.hidden = true;
    ui.chartImage.removeAttribute("src");
    ui.chartEmpty.hidden = false;
    setChartEmptyCopy(
      requested ? "Chart unavailable." : "No chart requested yet.",
      requested
        ? "The analysis completed, but a chart image was not available."
        : "Enable “Generate chart” when you ask a question to see a local visualization here.",
    );

    if (typeof chart.note === "string" && chart.note.trim().length > 0) {
      ui.chartMessage.textContent = chart.note;
      ui.chartMessage.hidden = false;
    }
  }

  function showChartImageError() {
    ui.chartImage.removeAttribute("src");
    ui.chartFigure.hidden = true;
    ui.chartEmpty.hidden = false;
    setChartEmptyCopy("Chart unavailable.", "The generated chart image could not be displayed.");
    ui.chartMessage.textContent = "The chart file is unavailable. Submit a new question if you need another chart.";
    ui.chartMessage.hidden = false;
  }

  function resetChart() {
    ui.chartImage.removeAttribute("src");
    ui.chartImage.alt = "";
    ui.chartFigure.hidden = true;
    ui.chartEmpty.hidden = false;
    ui.chartTypeBadge.hidden = true;
    ui.chartTypeBadge.textContent = "";
    ui.chartNote.hidden = true;
    ui.chartNote.textContent = "";
    ui.chartMessage.hidden = true;
    ui.chartMessage.textContent = "";
    setChartEmptyCopy(
      "No chart requested yet.",
      "Enable “Generate chart” when you ask a question to see a local visualization here.",
    );
  }

  function setChartEmptyCopy(title, description) {
    const titleElement = ui.chartEmpty.querySelector("p");
    const descriptionElement = ui.chartEmpty.querySelector("span:last-child");
    titleElement.textContent = title;
    descriptionElement.textContent = description;
  }

  async function loadHistory() {
    try {
      const response = await fetch("/api/history", { headers: { Accept: "application/json" } });
      const payload = await readJson(response);
      if (!response.ok || !isRecord(payload) || payload.ok !== true) {
        throw new Error("History unavailable");
      }
      savedHistory = normalizeHistory(payload.history);
    } catch (_error) {
      // A history outage must never obscure the current analysis.
      savedHistory = [];
    }
    renderHistory(savedHistory);
    renderChartGallery();
  }

  function renderHistory(entries) {
    ui.historyList.replaceChildren();
    ui.historyEmpty.hidden = entries.length !== 0;
    for (const entry of entries) {
      ui.historyList.append(createHistoryItem(entry));
    }
  }

  function renderChartGallery() {
    // The gallery is rebuilt from saved history, so charts survive a refresh
    // instead of living only in page memory.
    const chartEntries = savedHistory.filter(function (entry) {
      return entry.chartRequested;
    });
    ui.chartGallery.replaceChildren();
    ui.chartsEmpty.hidden = chartEntries.length !== 0;
    for (const entry of chartEntries) {
      ui.chartGallery.append(createGalleryItem(entry));
    }
  }

  function normalizeHistory(entries) {
    if (!Array.isArray(entries)) {
      return [];
    }
    const normalized = [];
    for (const entry of entries) {
      if (!isRecord(entry) || typeof entry.question !== "string" || entry.question.trim().length === 0) {
        continue;
      }
      const chart = isRecord(entry.chart) ? entry.chart : {};
      const chartUrl = chart.available === true ? safeChartUrl(chart.url) : null;
      const answer = typeof entry.answer === "string" ? entry.answer : "";
      normalized.push({
        id: typeof entry.id === "string" ? entry.id : "",
        answerPreview: answer.slice(0, 180),
        chartRequested: chart.requested === true,
        chartAvailable: chartUrl !== null,
        chartType: safeChartType(chart.type),
        chartUrl,
        question: entry.question.slice(0, 2000),
        rowCount: asNonNegativeInteger(entry.row_count),
        timestamp: safeTimestamp(entry.created_at),
      });
      if (normalized.length === HISTORY_LIMIT) {
        break;
      }
    }
    return normalized;
  }

  function createHistoryItem(entry) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    const copy = document.createElement("span");
    const question = document.createElement("span");
    const preview = document.createElement("span");
    const metadata = document.createElement("span");
    const timestamp = document.createElement("span");
    const rowCount = document.createElement("span");

    button.type = "button";
    button.className = "history-item";
    button.addEventListener("click", function () {
      ui.questionInput.value = entry.question;
      setQuestionMessage("Question added from history. Submit it when you are ready.", "success");
      activateView("ask", true);
      ui.questionInput.focus();
    });

    copy.className = "history-copy";
    question.className = "history-question";
    question.textContent = entry.question;
    preview.className = "history-preview";
    preview.textContent = entry.answerPreview || "Completed analysis";
    copy.append(question, preview);

    metadata.className = "history-meta";
    timestamp.textContent = formatTimestamp(entry.timestamp);
    rowCount.textContent = entry.rowCount === null ? "Results returned" : `${formatCount(entry.rowCount)} rows`;
    metadata.append(timestamp, rowCount);
    if (entry.chartRequested) {
      const chartState = document.createElement("span");
      chartState.className = "history-chart";
      chartState.textContent = entry.chartAvailable ? "Chart generated" : "Chart unavailable";
      metadata.append(chartState);
    }

    button.append(copy, metadata);
    item.append(button);
    return item;
  }

  function createGalleryItem(entry) {
    // A chart file can be removed from disk while its history row remains, so
    // an unavailable chart renders as a plain, non-linking card.
    const card = document.createElement(entry.chartAvailable ? "a" : "div");
    const content = document.createElement("span");
    const question = document.createElement("span");
    const metadata = document.createElement("span");

    card.className = entry.chartAvailable ? "gallery-card" : "gallery-card is-unavailable";
    if (entry.chartAvailable) {
      const image = document.createElement("img");
      card.href = entry.chartUrl;
      card.target = "_blank";
      card.rel = "noopener";
      image.src = entry.chartUrl;
      image.alt = chartAlt(entry.chartType, entry.question);
      card.append(image);
    } else {
      const placeholder = document.createElement("span");
      placeholder.className = "gallery-card-missing";
      placeholder.textContent = "Chart unavailable";
      card.append(placeholder);
    }
    content.className = "gallery-card-content";
    question.className = "gallery-card-question";
    question.textContent = entry.question;
    metadata.className = "gallery-card-meta";
    metadata.textContent = `${formatTimestamp(entry.timestamp)}${entry.chartType === null ? "" : ` · ${entry.chartType} chart`}`;
    content.append(question, metadata);
    card.append(content);
    return card;
  }

  function clearSession() {
    if (isSubmitting) {
      return;
    }
    // Display state only. Saved history stays on the backend until it is
    // explicitly deleted through "Clear saved history".
    removeStoredValue(STORAGE_KEYS.view);
    clearCurrentResult();
    resetChart();
    ui.questionInput.value = "";
    setQuestionMessage("The displayed result was cleared. Saved history was kept.", "success");
    activateView("ask", false);
    announce("The displayed result was cleared. Saved history was kept.");
  }

  async function clearSavedHistory() {
    if (isSubmitting) {
      return;
    }
    const confirmed = window.confirm(
      "Delete all saved query history? This cannot be undone. Charts, data, and logs are not affected.",
    );
    if (!confirmed) {
      return;
    }

    try {
      const response = await fetch("/api/history", {
        method: "DELETE",
        headers: { Accept: "application/json" },
      });
      const payload = await readJson(response);
      if (!response.ok || !isRecord(payload) || payload.ok !== true) {
        throw new Error("History could not be cleared");
      }
      savedHistory = [];
      renderHistory(savedHistory);
      renderChartGallery();
      setQuestionMessage("Saved history was deleted.", "success");
      announce("Saved history was deleted.");
    } catch (_error) {
      setQuestionMessage("Saved history could not be deleted. Please try again.", "error");
      announce("Saved history could not be deleted.");
    }
  }

  function clearCurrentResult() {
    ui.resultsCard.hidden = true;
    ui.askedQuestion.textContent = "";
    ui.askedBlock.hidden = true;
    ui.answerText.textContent = "";
    ui.answerWarning.textContent = "";
    ui.answerWarning.hidden = true;
    ui.rowMeta.textContent = "";
    ui.truncationNote.textContent = "";
    ui.truncationNote.hidden = true;
    ui.emptyResult.hidden = true;
    ui.resultTableHead.replaceChildren();
    ui.resultTableBody.replaceChildren();
    ui.tableWrap.hidden = false;
    ui.resultTableCaption.textContent = "Query result rows";
  }

  function setSubmitting(submitting) {
    isSubmitting = submitting;
    ui.askButton.disabled = submitting;
    ui.askButton.classList.toggle("is-loading", submitting);
    ui.questionInput.disabled = submitting;
    ui.chartRequested.disabled = submitting;
    ui.chartType.disabled = submitting || !ui.chartRequested.checked;
    for (const button of ui.exampleButtons) {
      button.disabled = submitting;
    }
    for (const button of ui.clearButtons) {
      button.disabled = submitting;
    }
  }

  function updateChartControls() {
    ui.chartType.disabled = isSubmitting || !ui.chartRequested.checked;
  }

  function selectedChartType() {
    return CHART_TYPES.has(ui.chartType.value) ? ui.chartType.value : "auto";
  }

  function toggleTheme() {
    const nextTheme = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    applyTheme(nextTheme);
    writeStoredValue(STORAGE_KEYS.theme, nextTheme);
  }

  function applyTheme(theme) {
    const selectedTheme = theme === "light" ? "light" : "dark";
    const lightThemeActive = selectedTheme === "light";
    document.documentElement.dataset.theme = selectedTheme;
    ui.themeToggle.setAttribute("aria-pressed", String(lightThemeActive));
    ui.themeIcon.textContent = lightThemeActive ? "◐" : "☼";
    ui.themeLabel.textContent = lightThemeActive ? "Dark theme" : "Light theme";
    ui.themeToggle.setAttribute("aria-label", lightThemeActive ? "Use dark theme" : "Use light theme");
  }

  function activateView(requestedView, shouldFocus) {
    const viewName = VIEWS.has(requestedView) ? requestedView : "ask";
    for (const view of ui.views) {
      const active = view.dataset.view === viewName;
      view.hidden = !active;
      view.classList.toggle("is-active", active);
      view.setAttribute("aria-hidden", String(!active));
      if (active && shouldFocus) {
        view.focus({ preventScroll: true });
      }
    }

    for (const button of ui.navButtons) {
      const active = button.dataset.viewTarget === viewName;
      button.classList.toggle("is-active", active);
      if (active) {
        button.setAttribute("aria-current", "page");
      } else {
        button.removeAttribute("aria-current");
      }
    }

    writeStoredValue(STORAGE_KEYS.view, viewName);
    closeNavigation();
  }

  function toggleNavigation() {
    if (document.body.classList.contains("nav-open")) {
      closeNavigation();
      return;
    }
    document.body.classList.add("nav-open");
    ui.sidebar.inert = false;
    ui.sidebar.removeAttribute("aria-hidden");
    ui.navToggle.setAttribute("aria-expanded", "true");
    ui.sidebarBackdrop.hidden = false;
  }

  function closeNavigation() {
    document.body.classList.remove("nav-open");
    ui.navToggle.setAttribute("aria-expanded", "false");
    ui.sidebarBackdrop.hidden = true;
    if (isMobileViewport()) {
      ui.sidebar.inert = true;
      ui.sidebar.setAttribute("aria-hidden", "true");
    } else {
      ui.sidebar.inert = false;
      ui.sidebar.removeAttribute("aria-hidden");
    }
  }

  function renderStatus(payload) {
    const source = isRecord(payload.status) ? payload.status : payload;
    const database = pickStatusValue(source, ["database", "database_status"]);
    const api = pickStatusValue(source, ["api", "api_status"]);
    const state = determineStatusState(database, api);
    if (state === "error") {
      renderUnavailableStatus();
      return;
    }

    const statusEntries = [
      ["Database", database],
      ["Table", pickStatusValue(source, ["table", "table_name"])],
      ["Analytics engine", pickStatusValue(source, ["analytics_engine", "engine"])],
      ["Model", pickStatusValue(source, ["model", "model_name"])],
      ["Web mode", pickStatusValue(source, ["web_mode", "mode"])],
      ["API", api],
    ].filter(function (entry) {
      return entry[1] !== null;
    });

    ui.statusList.replaceChildren();
    for (const [label, value] of statusEntries) {
      const row = document.createElement("div");
      const term = document.createElement("dt");
      const detail = document.createElement("dd");
      row.className = "status-row";
      term.textContent = label;
      detail.textContent = value;
      row.append(term, detail);
      ui.statusList.append(row);
    }
    setStatusDotState(state);
  }

  function renderUnavailableStatus() {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    const detail = document.createElement("dd");
    row.className = "status-row";
    term.textContent = "Service";
    detail.textContent = "Unavailable";
    row.append(term, detail);
    ui.statusList.replaceChildren(row);
    setStatusDotState("error");
  }

  function determineStatusState(database, api) {
    const validDatabase = database === "Ready" || database === "Not initialized";
    const validApi = api === "Configured" || api === "Missing";
    if (!validDatabase || !validApi) {
      return "error";
    }
    if (database === "Ready" && api === "Configured") {
      return "ready";
    }
    return "warning";
  }

  function setStatusDotState(state) {
    ui.statusDot.classList.remove("is-ready", "is-warning", "is-error");
    ui.statusDot.classList.add(`is-${state}`);
  }

  function pickStatusValue(source, keys) {
    for (const key of keys) {
      const value = safeStatusValue(source[key]);
      if (value !== null) {
        return value;
      }
    }
    return null;
  }

  function safeStatusValue(value) {
    if (typeof value === "boolean") {
      return value ? "Yes" : "No";
    }
    if (typeof value === "number" && Number.isFinite(value)) {
      return String(value);
    }
    if (typeof value !== "string") {
      return null;
    }
    const trimmed = value.trim();
    if (trimmed.length === 0 || trimmed.length > 100 || /[\\/]/.test(trimmed) || trimmed.includes("..")) {
      return null;
    }
    return trimmed;
  }

  function setQuestionMessage(message, tone) {
    ui.questionMessage.textContent = message;
    ui.questionMessage.classList.toggle("is-error", tone === "error");
    ui.questionMessage.classList.toggle("is-success", tone === "success");
  }

  function announce(message) {
    ui.liveRegion.textContent = "";
    window.setTimeout(function () {
      ui.liveRegion.textContent = message;
    }, 0);
  }

  function getSafeErrorMessage(payload) {
    if (isRecord(payload) && isRecord(payload.error) && typeof payload.error.message === "string") {
      const message = payload.error.message.trim();
      if (message.length > 0 && message.length <= 500) {
        return message;
      }
    }
    return "The analysis could not be completed. Please review your question and try again.";
  }

  function readStoredTheme() {
    const savedTheme = readStoredValue(STORAGE_KEYS.theme);
    return savedTheme === "light" || savedTheme === "dark" ? savedTheme : "dark";
  }

  function readStoredView() {
    const savedView = readStoredValue(STORAGE_KEYS.view);
    return VIEWS.has(savedView) ? savedView : "ask";
  }

  function safeTimestamp(value) {
    if (typeof value !== "string") {
      return new Date().toISOString();
    }
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? new Date().toISOString() : parsed.toISOString();
  }

  function readStoredValue(key) {
    try {
      return window.localStorage.getItem(key);
    } catch (_error) {
      return null;
    }
  }

  function writeStoredValue(key, value) {
    try {
      window.localStorage.setItem(key, value);
    } catch (_error) {
      // The dashboard remains usable when browser storage is unavailable.
    }
  }

  function removeStoredValue(key) {
    try {
      window.localStorage.removeItem(key);
    } catch (_error) {
      // The displayed session state has already been cleared in memory.
    }
  }

  function safeChartUrl(value) {
    if (typeof value !== "string") {
      return null;
    }
    try {
      const parsed = new URL(value, window.location.origin);
      const chartPath = parsed.pathname;
      if (parsed.origin !== window.location.origin || !chartPath.startsWith("/charts/") || !chartPath.endsWith(".png")) {
        return null;
      }
      return chartPath;
    } catch (_error) {
      return null;
    }
  }

  function safeChartType(value) {
    return typeof value === "string" && CHART_TYPES.has(value) && value !== "auto" ? value : null;
  }

  function chartAlt(chartType, question) {
    const conciseQuestion = typeof question === "string" && question.trim().length > 0
      ? question.trim().slice(0, 180)
      : "the submitted question";
    return chartType === null
      ? `Generated chart for: ${conciseQuestion}`
      : `Generated ${chartType} chart for: ${conciseQuestion}`;
  }

  function isMobileViewport() {
    return typeof window.matchMedia === "function"
      ? window.matchMedia("(max-width: 820px)").matches
      : window.innerWidth <= 820;
  }

  function asNonNegativeInteger(value) {
    return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : null;
  }

  function formatCount(value) {
    return new Intl.NumberFormat().format(value);
  }

  function formatTimestamp(value) {
    const timestamp = new Date(value);
    return Number.isNaN(timestamp.getTime()) ? "Recent" : timestamp.toLocaleString();
  }

  function displayValue(value) {
    if (value === null || value === undefined) {
      return "—";
    }
    if (typeof value === "boolean") {
      return value ? "true" : "false";
    }
    return String(value);
  }

  function isRecord(value) {
    return value !== null && typeof value === "object" && !Array.isArray(value);
  }

  class RequestFailure extends Error {}

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize, { once: true });
  } else {
    initialize();
  }
}());
