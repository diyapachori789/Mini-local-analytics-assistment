(function () {
  "use strict";

  // Browser storage intentionally holds display preferences only. Saved chats
  // are owned by the local backend, not copied into the browser.
  const STORAGE_KEYS = Object.freeze({
    theme: "mini-local-analytics.theme",
    view: "mini-local-analytics.view",
  });
  const LEGACY_HISTORY_KEY = "mini-local-analytics.history";
  const CHART_TYPES = new Set(["auto", "bar", "line", "pie", "scatter"]);
  const VIEWS = new Set(["chat", "about"]);
  const CONVERSATION_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/;
  const MAX_CONVERSATIONS = 100;
  const MAX_MESSAGE_CHARS = 12000;
  const MAX_TITLE_CHARS = 180;

  let activeConversationId = null;
  let conversations = [];
  let dataPanelSequence = 0;
  let initialConversationLoadPending = true;
  let initialConversationRestorePending = true;
  let isConversationAction = false;
  let isSubmitting = false;
  let attachedImageName = null;
  let attachedImageUrl = null;
  let ui;

  function initialize() {
    ui = {
      activeConversationTitle: document.getElementById("conversation-heading"),
      askButton: document.getElementById("ask-button"),
      attachmentChip: document.getElementById("attachment-chip"),
      attachmentName: document.getElementById("attachment-name"),
      attachmentRemove: document.getElementById("attachment-remove"),
      attachmentThumb: document.getElementById("attachment-thumb"),
      imageInput: document.getElementById("image-input"),
      chartShortcut: document.querySelector("[data-chart-shortcut]"),
      conversationList: document.getElementById("conversation-list"),
      conversationListEmpty: document.getElementById("conversation-list-empty"),
      conversationMeta: document.getElementById("conversation-meta"),
      conversationStream: document.getElementById("conversation-stream"),
      deleteAllButton: document.querySelector("[data-delete-all-conversations]"),
      form: document.getElementById("query-form"),
      liveRegion: document.getElementById("live-region"),
      navButtons: Array.from(document.querySelectorAll("[data-view-target]")),
      navToggle: document.getElementById("nav-toggle"),
      newChatButtons: Array.from(document.querySelectorAll("[data-new-chat]")),
      questionInput: document.getElementById("question-input"),
      questionMessage: document.getElementById("question-message"),
      sidebar: document.getElementById("sidebar"),
      sidebarBackdrop: document.getElementById("sidebar-backdrop"),
      themeIcon: document.querySelector(".theme-toggle-icon"),
      themeLabel: document.querySelector(".theme-toggle-label"),
      themeToggle: document.getElementById("theme-toggle"),
      views: Array.from(document.querySelectorAll("[data-view]")),
    };

    applyTheme(readStoredTheme());
    bindEvents();
    // Do not merge display-only data written by an earlier browser build into
    // server-owned conversations.
    removeStoredValue(LEGACY_HISTORY_KEY);
    renderNewConversation();
    renderConversationList([]);
    activateView(readStoredView(), false);
    closeNavigation();
    loadConversations();
  }

  function bindEvents() {
    ui.form.addEventListener("submit", submitQuestion);
    ui.questionInput.addEventListener("keydown", submitOnEnter);
    ui.conversationStream.addEventListener("click", selectSuggestedQuestion);
    ui.chartShortcut.addEventListener("click", showChartsInConversation);
    ui.themeToggle.addEventListener("click", toggleTheme);
    ui.navToggle.addEventListener("click", toggleNavigation);
    ui.sidebarBackdrop.addEventListener("click", closeNavigation);
    document.addEventListener("click", closeConversationMenus);
    ui.deleteAllButton.addEventListener("click", deleteAllConversations);
    ui.imageInput.addEventListener("change", attachImage);
    ui.attachmentRemove.addEventListener("click", clearAttachment);

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape") {
        closeConversationMenus();
      }
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

    for (const button of ui.newChatButtons) {
      button.addEventListener("click", startNewConversation);
    }

    window.addEventListener("resize", closeNavigation);
  }


  // --- Image attachment -----------------------------------------------------
  //
  // Frontend only. There is no backend image understanding, so nothing is
  // uploaded: the file stays in the browser and the composer refuses to send
  // while one is attached. The alternative - accepting the file and quietly
  // analysing only the text - would look like support that does not exist.

  const ALLOWED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp"];

  function attachImage(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) {
      clearAttachment();
      return;
    }
    if (!ALLOWED_IMAGE_TYPES.includes(file.type)) {
      clearAttachment();
      setQuestionMessage("Choose a PNG, JPEG or WebP image.", "error");
      announce("That file type is not supported.");
      return;
    }

    attachedImageName = file.name;
    // Set as text, never as markup: the filename is user-controlled.
    ui.attachmentName.textContent = file.name;

    if (attachedImageUrl !== null) {
      URL.revokeObjectURL(attachedImageUrl);
    }
    attachedImageUrl = URL.createObjectURL(file);
    ui.attachmentThumb.src = attachedImageUrl;
    ui.attachmentChip.hidden = false;

    setQuestionMessage("Image analysis is not supported yet.", "");
    announce("Image attached: " + file.name + ". Image analysis is not supported yet.");
  }

  function clearAttachment() {
    attachedImageName = null;
    if (attachedImageUrl !== null) {
      URL.revokeObjectURL(attachedImageUrl);
      attachedImageUrl = null;
    }
    ui.imageInput.value = "";
    ui.attachmentThumb.removeAttribute("src");
    ui.attachmentName.textContent = "";
    ui.attachmentChip.hidden = true;
    setQuestionMessage("", "");
    ui.questionInput.focus();
  }


  function selectSuggestedQuestion(event) {
    const button = event.target.closest("[data-example-question]");
    if (!(button instanceof HTMLButtonElement) || !ui.conversationStream.contains(button)) {
      return;
    }
    const question = button.dataset.exampleQuestion;
    if (typeof question !== "string") {
      return;
    }
    ui.questionInput.value = question;
    setQuestionMessage("Suggestion added. Review it, then send when ready.", "success");
    activateView("chat", false);
    ui.questionInput.focus();
  }

  function showChartsInConversation() {
    activateView("chat", false);
    const chart = ui.conversationStream.querySelector(".message-chart");
    if (chart === null) {
      setQuestionMessage("Charts will appear inside assistant messages when they add value.", "");
      announce("This chat does not contain a chart yet.");
      return;
    }
    chart.scrollIntoView({ behavior: "smooth", block: "center" });
    chart.focus({ preventScroll: true });
    announce("Moved to the first chart in this chat.");
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
    if (initialConversationLoadPending) {
      setQuestionMessage("Loading your saved chats before sending a message...", "");
      announce("Saved chats are still loading.");
      return;
    }

    if (attachedImageName !== null) {
      // The backend accepts a question, chart preferences and a conversation id
      // and rejects anything else, so an attachment cannot be sent anywhere.
      // Saying so beats dropping it silently.
      setQuestionMessage(
        "Image analysis is not supported yet. Remove the image to send your message.",
        "error"
      );
      announce("Image analysis is not supported yet.");
      ui.attachmentRemove.focus();
      return;
    }

    const question = ui.questionInput.value;
    if (question.trim().length === 0) {
      setQuestionMessage("Enter a question before sending it.", "error");
      announce("A question is required before sending.");
      ui.questionInput.focus();
      return;
    }

    const requestBody = {
      question,
      conversation_id: null,
    };
    let pendingMessage = null;
    setSubmitting(true);
    setQuestionMessage("Thinking...", "");
    announce("Analytics Assistant is thinking.");

    try {
      const conversationId = await ensureActiveConversation();
      if (conversationId === null) {
        return;
      }

      requestBody.conversation_id = conversationId;
      pendingMessage = appendPendingTurn(question);
      ui.questionInput.value = "";

      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(requestBody),
      });
      const payload = await readJson(response);
      if (!response.ok || !isRecord(payload) || payload.ok !== true) {
        throw new RequestFailure(getSafeErrorMessage(payload));
      }

      const returnedConversationId = safeConversationId(payload.conversation_id);
      if (returnedConversationId !== null) {
        activeConversationId = returnedConversationId;
      }
      replacePendingMessage(pendingMessage, createAssistantMessage(currentResponseMessage(payload, question)));
      pendingMessage = null;
      updateConversationHeaderForActive();
      loadConversations();

      if (payload.conversation_saved === false) {
        setQuestionMessage("Response received, but this chat could not be saved.", "error");
        announce("The response was received, but the chat could not be saved.");
      } else {
        setQuestionMessage("", "");
        announce("The conversation was updated.");
      }
    } catch (error) {
      const message = error instanceof RequestFailure
        ? error.message
        : "The response could not be completed. Please try again later.";
      if (pendingMessage !== null) {
        replacePendingMessage(pendingMessage, createFailedAssistantMessage(message));
      }
      setQuestionMessage(message, "error");
      announce("The response could not be completed.");
    } finally {
      setSubmitting(false);
    }
  }

  async function ensureActiveConversation() {
    const current = safeConversationId(activeConversationId);
    if (current !== null) {
      return current;
    }
    const conversation = await createConversation();
    if (conversation === null) {
      return null;
    }
    activeConversationId = conversation.id;
    updateConversationHeader(conversation);
    return conversation.id;
  }

  async function startNewConversation() {
    if (isSubmitting || isConversationAction) {
      return;
    }
    const conversation = await createConversation();
    if (conversation === null) {
      return;
    }
    activeConversationId = conversation.id;
    renderNewConversation(conversation);
    activateView("chat", false);
    ui.questionInput.value = "";
    setQuestionMessage("New chat ready.", "success");
    announce("A new chat is ready.");
    ui.questionInput.focus();
    loadConversations();
  }

  async function createConversation() {
    if (isConversationAction) {
      return null;
    }
    setConversationActionBusy(true);
    try {
      const response = await fetch("/api/conversations", {
        method: "POST",
        headers: { Accept: "application/json" },
      });
      const payload = await readJson(response);
      if (!response.ok || !isRecord(payload) || payload.ok !== true) {
        throw new RequestFailure(getSafeErrorMessage(payload));
      }
      const conversation = normalizeConversationSummary(payload.conversation);
      if (conversation === null) {
        throw new RequestFailure("The new chat could not be created safely.");
      }
      return conversation;
    } catch (error) {
      const message = error instanceof RequestFailure
        ? error.message
        : "A new chat could not be created. Please try again later.";
      setQuestionMessage(message, "error");
      announce("A new chat could not be created.");
      return null;
    } finally {
      setConversationActionBusy(false);
    }
  }

  async function loadConversations() {
    try {
      const response = await fetch("/api/conversations", { headers: { Accept: "application/json" } });
      const payload = await readJson(response);
      if (!response.ok || !isRecord(payload) || payload.ok !== true) {
        throw new Error("Conversations unavailable");
      }
      conversations = normalizeConversationList(payload.conversations);
      renderConversationList(conversations);
      const active = conversations.find(function (conversation) {
        return conversation.id === activeConversationId;
      });
      if (active) {
        updateConversationHeader(active);
      }
      if (initialConversationRestorePending) {
        initialConversationRestorePending = false;
        // Conversation identity is backend-authoritative. On a fresh browser
        // load, restore the most recently updated saved chat instead of making
        // a different one until the user explicitly chooses New chat.
        if (activeConversationId === null && conversations.length > 0) {
          loadConversation(conversations[0].id, true);
          return;
        }
      }
      initialConversationLoadPending = false;
    } catch (_error) {
      initialConversationRestorePending = false;
      initialConversationLoadPending = false;
      if (conversations.length === 0) {
        ui.conversationList.replaceChildren();
        ui.conversationListEmpty.hidden = false;
        ui.conversationListEmpty.textContent = "Saved chats are unavailable right now.";
      }
    }
  }

  async function loadConversation(conversationId, restoringInitialConversation = false) {
    const safeId = safeConversationId(conversationId);
    if (safeId === null || isSubmitting || isConversationAction) {
      if (restoringInitialConversation) {
        initialConversationLoadPending = false;
      }
      return;
    }

    setConversationActionBusy(true);
    try {
      const response = await fetch(`/api/conversations/${encodeURIComponent(safeId)}`, {
        headers: { Accept: "application/json" },
      });
      const payload = await readJson(response);
      if (!response.ok || !isRecord(payload) || payload.ok !== true) {
        throw new RequestFailure(getSafeErrorMessage(payload));
      }
      const conversation = normalizeConversationDetail(payload.conversation);
      if (conversation === null) {
        throw new RequestFailure("This saved chat could not be loaded safely.");
      }

      activeConversationId = conversation.id;
      renderConversationDetail(conversation);
      renderConversationList(conversations);
      activateView("chat", false);
      // Announced for screen readers only; no visible toast for a
      // routine action that the user just asked for.
      setQuestionMessage("", "");
      announce("Chat opened.");
    } catch (error) {
      const message = error instanceof RequestFailure
        ? error.message
        : "This saved chat could not be loaded. Please try again later.";
      setQuestionMessage(message, "error");
      announce("Saved chat could not be loaded.");
    } finally {
      setConversationActionBusy(false);
      if (restoringInitialConversation) {
        initialConversationLoadPending = false;
      }
    }
  }

  async function deleteConversation(conversationId) {
    const safeId = safeConversationId(conversationId);
    if (safeId === null || isSubmitting || isConversationAction) {
      return;
    }
    const confirmed = window.confirm(
      "Delete this saved chat? This cannot be undone. Charts, data, and logs are not affected.",
    );
    if (!confirmed) {
      return;
    }

    setConversationActionBusy(true);
    try {
      const response = await fetch(`/api/conversations/${encodeURIComponent(safeId)}`, {
        method: "DELETE",
        headers: { Accept: "application/json" },
      });
      const payload = await readJson(response);
      if (!response.ok || !isRecord(payload) || payload.ok !== true) {
        throw new RequestFailure(getSafeErrorMessage(payload));
      }
      if (activeConversationId === safeId) {
        renderNewConversation();
        ui.questionInput.value = "";
      }
      conversations = conversations.filter(function (conversation) {
        return conversation.id !== safeId;
      });
      renderConversationList(conversations);
      setQuestionMessage("Saved chat deleted.", "success");
      announce("Saved chat deleted.");
      loadConversations();
    } catch (error) {
      const message = error instanceof RequestFailure
        ? error.message
        : "This saved chat could not be deleted. Please try again later.";
      setQuestionMessage(message, "error");
      announce("Saved chat could not be deleted.");
    } finally {
      setConversationActionBusy(false);
    }
  }

  async function deleteAllConversations() {
    if (isSubmitting || isConversationAction) {
      return;
    }
    const confirmed = window.confirm(
      "Delete all saved chats? This cannot be undone. Charts, data, and logs are not affected.",
    );
    if (!confirmed) {
      return;
    }

    setConversationActionBusy(true);
    try {
      const response = await fetch("/api/conversations", {
        method: "DELETE",
        headers: { Accept: "application/json" },
      });
      const payload = await readJson(response);
      if (!response.ok || !isRecord(payload) || payload.ok !== true) {
        throw new RequestFailure(getSafeErrorMessage(payload));
      }
      conversations = [];
      renderConversationList(conversations);
      renderNewConversation();
      ui.questionInput.value = "";
      setQuestionMessage("All saved chats were deleted.", "success");
      announce("All saved chats were deleted.");
    } catch (error) {
      const message = error instanceof RequestFailure
        ? error.message
        : "Saved chats could not be deleted. Please try again later.";
      setQuestionMessage(message, "error");
      announce("Saved chats could not be deleted.");
    } finally {
      setConversationActionBusy(false);
    }
  }

  function renderNewConversation(conversation) {
    activeConversationId = conversation ? safeConversationId(conversation.id) : null;
    ui.activeConversationTitle.textContent = conversation ? conversation.title : "New chat";
    ui.conversationMeta.textContent = conversation
      ? "This chat is ready for your first question."
      : "Ask a question to begin.";
    ui.conversationStream.replaceChildren(createConversationEmpty());
    ui.conversationStream.setAttribute("aria-busy", "false");
    renderConversationList(conversations);
  }

  function renderConversationDetail(conversation) {
    ui.activeConversationTitle.textContent = conversation.title;
    const count = conversation.messages.length;
    ui.conversationMeta.textContent = count === 0
      ? "This saved chat has no messages yet."
      : `${formatCount(count)} ${count === 1 ? "message" : "messages"} saved locally.`;
    ui.conversationStream.replaceChildren();
    ui.conversationStream.setAttribute("aria-busy", "false");

    if (conversation.messages.length === 0) {
      ui.conversationStream.append(createConversationEmpty());
      return;
    }

    let latestQuestion = "";
    for (const message of conversation.messages) {
      if (message.role === "user") {
        latestQuestion = message.content;
      }
      ui.conversationStream.append(createConversationMessage(message, latestQuestion));
    }
    if (conversation.messagesTruncated) {
      const notice = document.createElement("p");
      notice.className = "message-history-notice";
      notice.textContent = "Only the most recent saved messages are shown in this view.";
      ui.conversationStream.append(notice);
    }
  }

  function renderConversationList(entries) {
    ui.conversationList.replaceChildren();
    ui.conversationListEmpty.hidden = entries.length !== 0;
    if (entries.length === 0) {
      if (ui.conversationListEmpty.textContent !== "Saved chats are unavailable right now.") {
        ui.conversationListEmpty.textContent = "Your saved chats will appear here.";
      }
      syncConversationControls();
      return;
    }

    const groups = new Map([
      ["Today", []],
      ["Yesterday", []],
      ["Previous 7 Days", []],
      ["Older", []],
    ]);
    for (const entry of entries) {
      groups.get(conversationGroupLabel(entry.updatedAt)).push(entry);
    }

    for (const [label, group] of groups) {
      if (group.length === 0) {
        continue;
      }
      const section = document.createElement("section");
      const heading = document.createElement("h3");
      const list = document.createElement("ol");
      section.className = "conversation-group";
      heading.className = "conversation-group-title";
      heading.textContent = label;
      list.className = "conversation-group-list";
      for (const entry of group) {
        list.append(createConversationListItem(entry));
      }
      section.append(heading, list);
      ui.conversationList.append(section);
    }
    syncConversationControls();
  }

  function closeConversationMenus() {
    for (const popover of document.querySelectorAll(".conversation-menu-popover")) {
      popover.hidden = true;
    }
    for (const trigger of document.querySelectorAll(".conversation-menu-trigger")) {
      trigger.setAttribute("aria-expanded", "false");
    }
  }

  function createConversationListItem(entry) {
    const item = document.createElement("li");
    const selectButton = document.createElement("button");
    const copy = document.createElement("span");
    const title = document.createElement("span");
    const metadata = document.createElement("span");
    const deleteButton = document.createElement("button");

    item.className = "conversation-list-item";
    selectButton.type = "button";
    selectButton.className = "conversation-select";
    selectButton.classList.toggle("is-active", entry.id === activeConversationId);
    if (entry.id === activeConversationId) {
      selectButton.setAttribute("aria-current", "page");
    }
    selectButton.addEventListener("click", function () {
      loadConversation(entry.id);
    });

    copy.className = "conversation-list-copy";
    title.className = "conversation-list-title";
    title.textContent = entry.title;
    metadata.className = "conversation-list-meta";
    metadata.textContent = formatTimestamp(entry.updatedAt);
    copy.append(title, metadata);
    selectButton.append(copy);

    // Overflow menu rather than a permanent Delete on every row.
    const menu = document.createElement("div");
    menu.className = "conversation-menu";

    const trigger = document.createElement("button");
    trigger.type = "button";
    trigger.className = "conversation-menu-trigger";
    trigger.setAttribute("aria-haspopup", "true");
    trigger.setAttribute("aria-expanded", "false");
    trigger.setAttribute("aria-label", "Chat actions");
    const dots = document.createElement("span");
    dots.setAttribute("aria-hidden", "true");
    dots.textContent = "⋯";
    trigger.append(dots);

    const popover = document.createElement("div");
    popover.className = "conversation-menu-popover";
    popover.hidden = true;

    deleteButton.type = "button";
    deleteButton.className = "conversation-delete";
    deleteButton.dataset.deleteConversation = entry.id;
    deleteButton.textContent = "Delete chat";
    deleteButton.addEventListener("click", function () {
      closeConversationMenus();
      deleteConversation(entry.id);
    });
    popover.append(deleteButton);

    trigger.addEventListener("click", function (event) {
      event.stopPropagation();
      const willOpen = popover.hidden;
      closeConversationMenus();
      popover.hidden = !willOpen;
      trigger.setAttribute("aria-expanded", willOpen ? "true" : "false");
      if (willOpen) {
        deleteButton.focus();
      }
    });

    menu.append(trigger, popover);
    item.append(selectButton, menu);
    return item;
  }

  function appendPendingTurn(question) {
    removeConversationEmpty();
    ui.conversationStream.append(createUserMessage(question));
    const pending = createPendingAssistantMessage();
    ui.conversationStream.append(pending);
    scrollConversationToEnd();
    return pending;
  }

  function replacePendingMessage(pending, replacement) {
    if (pending && pending.parentElement === ui.conversationStream) {
      pending.replaceWith(replacement);
      scrollConversationToEnd();
    }
  }

  function createConversationMessage(message, latestQuestion) {
    return message.role === "user"
      ? createUserMessage(message.content, message.createdAt)
      : createAssistantMessage(message, latestQuestion);
  }

  function createUserMessage(content, createdAt) {
    const article = document.createElement("article");
    const header = document.createElement("header");
    const label = document.createElement("span");
    const timestamp = document.createElement("time");
    const body = document.createElement("p");
    article.className = "chat-message chat-message-user";
    header.className = "chat-message-header";
    label.className = "chat-message-role";
    label.textContent = "You";
    timestamp.className = "chat-message-time";
    timestamp.textContent = formatTimestamp(createdAt);
    body.className = "chat-message-content";
    body.textContent = content;
    header.append(label, timestamp);
    article.append(header, body);
    return article;
  }

  function createPendingAssistantMessage() {
    const article = document.createElement("article");
    const header = document.createElement("header");
    const label = document.createElement("span");
    const body = document.createElement("p");
    article.className = "chat-message chat-message-assistant is-pending";
    article.setAttribute("aria-live", "polite");
    header.className = "chat-message-header";
    label.className = "chat-message-role";
    label.textContent = "Analytics Assistant";
    body.className = "chat-message-content pending-copy";
    body.textContent = "Thinking...";
    header.append(label);
    article.append(header, body);
    return article;
  }

  function createFailedAssistantMessage(message) {
    const article = document.createElement("article");
    const header = document.createElement("header");
    const label = document.createElement("span");
    const body = document.createElement("p");
    article.className = "chat-message chat-message-assistant chat-message-failed";
    header.className = "chat-message-header";
    label.className = "chat-message-role";
    label.textContent = "Analytics Assistant";
    body.className = "chat-message-error";
    body.textContent = message;
    header.append(label);
    article.append(header, body);
    return article;
  }

  function createAssistantMessage(message, latestQuestion) {
    const article = document.createElement("article");
    const header = document.createElement("header");
    const label = document.createElement("span");
    const timestamp = document.createElement("time");
    const body = document.createElement("p");

    article.className = "chat-message chat-message-assistant";
    header.className = "chat-message-header";
    label.className = "chat-message-role";
    label.textContent = message.meta.refused ? "Assistant response" : "Analytics Assistant";
    timestamp.className = "chat-message-time";
    timestamp.textContent = formatTimestamp(message.createdAt);
    body.className = "chat-message-content";
    body.textContent = message.content;
    header.append(label, timestamp);
    article.append(header, body);

    if (message.meta.answerFallbackUsed && message.meta.hasResult) {
      const warning = document.createElement("p");
      warning.className = "message-warning";
      warning.setAttribute("role", "status");
      warning.textContent = "Answer generation was unavailable, so the returned result data is shown below.";
      article.append(warning);
    }

    appendDataBlock(article, message);
    appendChartBlock(article, message.chart, message.question || latestQuestion);
    return article;
  }

  function appendDataBlock(article, message) {
    if (message.meta.refused || !message.meta.hasResult) {
      return;
    }
    const hasCurrentRows = Array.isArray(message.columns) && Array.isArray(message.rows);
    if (hasCurrentRows) {
      if (message.columns.length === 0 || message.rows.length === 0) {
        const empty = document.createElement("p");
        empty.className = "message-data-empty";
        empty.textContent = "No rows returned.";
        article.append(empty);
        return;
      }
      article.append(createDataToggle(message));
      return;
    }

    if (message.historical) {
      const unavailable = document.createElement("p");
      unavailable.className = "message-data-unavailable";
      if (message.rowCount === 0) {
        unavailable.textContent = "No rows were returned for this saved message.";
      } else if (message.rowCount !== null) {
        unavailable.textContent = `${formatCount(message.rowCount)} ${message.rowCount === 1 ? "row was" : "rows were"} returned for this saved message. Detailed rows were not retained for this saved chat.`;
      } else {
        unavailable.textContent = "The returned data rows were not retained for this saved chat.";
      }
      article.append(unavailable);
    }
  }

  function createDataToggle(message) {
    const wrapper = document.createElement("section");
    const button = document.createElement("button");
    const panel = document.createElement("div");
    const panelId = `message-data-${dataPanelSequence += 1}`;
    const rowCount = message.rowCount === null ? message.rows.length : message.rowCount;

    wrapper.className = "message-data";
    button.type = "button";
    button.className = "data-toggle";
    button.setAttribute("aria-expanded", "false");
    button.setAttribute("aria-controls", panelId);
    button.textContent = `View data (${formatCount(rowCount)} ${rowCount === 1 ? "row" : "rows"})`;
    panel.className = "message-data-panel";
    panel.id = panelId;
    panel.hidden = true;
    panel.append(createResultTable(message.columns, message.rows));

    if (message.truncated) {
      const truncation = document.createElement("p");
      truncation.className = "message-truncation";
      truncation.textContent = message.maxRows === null
        ? "Results were truncated by the configured result limit."
        : `Showing the first ${formatCount(message.maxRows)} rows. Additional rows were not fetched because of the configured result limit.`;
      panel.append(truncation);
    }

    button.addEventListener("click", function () {
      const expanded = button.getAttribute("aria-expanded") === "true";
      button.setAttribute("aria-expanded", String(!expanded));
      button.textContent = expanded
        ? `View data (${formatCount(rowCount)} ${rowCount === 1 ? "row" : "rows"})`
        : "Hide data";
      panel.hidden = expanded;
    });

    wrapper.append(button, panel);
    return wrapper;
  }

  function createResultTable(columns, rows) {
    const wrap = document.createElement("div");
    const table = document.createElement("table");
    const caption = document.createElement("caption");
    const head = document.createElement("thead");
    const body = document.createElement("tbody");
    const headerRow = document.createElement("tr");

    wrap.className = "table-wrap message-table-wrap";
    wrap.tabIndex = 0;
    wrap.setAttribute("aria-label", "Returned result table");
    caption.className = "sr-only";
    caption.textContent = "Returned result rows";
    for (const column of columns) {
      const heading = document.createElement("th");
      heading.scope = "col";
      heading.textContent = displayValue(column);
      headerRow.append(heading);
    }
    head.append(headerRow);

    for (const sourceRow of rows) {
      const row = document.createElement("tr");
      const values = Array.isArray(sourceRow) ? sourceRow : [];
      for (let index = 0; index < columns.length; index += 1) {
        const cell = document.createElement("td");
        cell.textContent = displayValue(values[index]);
        row.append(cell);
      }
      body.append(row);
    }

    table.append(caption, head, body);
    wrap.append(table);
    return wrap;
  }

  function appendChartBlock(article, chart, question) {
    if (!chart.requested) {
      return;
    }
    const section = document.createElement("section");
    const heading = document.createElement("div");
    const label = document.createElement("p");
    const type = document.createElement("span");
    section.className = "message-chart";
    section.tabIndex = -1;
    heading.className = "message-chart-heading";
    label.className = "message-chart-label";
    label.textContent = "Chart";
    type.className = "message-chart-type";
    type.hidden = chart.type === null;
    type.textContent = chart.type === null ? "" : `${chart.type} chart`;
    heading.append(label, type);
    section.append(heading);

    if (chart.available && chart.url !== null) {
      const figure = document.createElement("figure");
      const image = document.createElement("img");
      figure.className = "message-chart-figure";
      image.src = chart.url;
      image.alt = chartAlt(chart.type, question);
      image.decoding = "async";
      image.addEventListener("error", function () {
        renderMissingChart(section, chart.note, chart.historical, chart.generated);
      });
      figure.append(image);
      section.append(figure);
      if (chart.note !== "") {
        const note = document.createElement("p");
        note.className = "message-chart-note";
        note.textContent = chart.note;
        section.append(note);
      }
    } else {
      renderMissingChart(section, chart.note, chart.historical, chart.generated);
    }
    article.append(section);
  }

  function renderMissingChart(section, note, historical, generated) {
    const heading = section.querySelector(".message-chart-heading");
    const missing = document.createElement("p");
    missing.className = "message-chart-missing";
    missing.textContent = historical && generated
      ? "This saved chart is unavailable because its PNG file is no longer on disk."
      : (note || "A meaningful chart could not be created from the returned data.");
    section.replaceChildren(heading, missing);
    if (historical && generated && note !== "") {
      const noteElement = document.createElement("p");
      noteElement.className = "message-chart-note";
      noteElement.textContent = note;
      section.append(noteElement);
    }
  }

  function currentResponseMessage(payload, question) {
    const metadata = isRecord(payload.meta) ? payload.meta : {};
    const columns = normalizeColumns(payload.columns);
    const rows = normalizeRows(payload.rows, columns.length);
    const rowCount = asNonNegativeInteger(payload.row_count);
    return {
      chart: normalizeCurrentChart(payload.chart),
      columns,
      content: safeMessageContent(payload.answer, "The answer could not be generated. The returned data is shown below."),
      createdAt: new Date().toISOString(),
      historical: false,
      meta: {
        answerFallbackUsed: metadata.answer_fallback_used === true,
        hasResult: metadata.has_result === true,
        refused: payload.refused === true || metadata.refused === true,
        success: metadata.success !== false,
      },
      question,
      rowCount: rowCount === null ? rows.length : rowCount,
      rows,
      truncated: payload.truncated === true,
      maxRows: asNonNegativeInteger(payload.max_rows),
    };
  }

  function normalizeConversationList(value) {
    if (!Array.isArray(value)) {
      return [];
    }
    const normalized = [];
    for (const entry of value) {
      const conversation = normalizeConversationSummary(entry);
      if (conversation === null || normalized.some(function (existing) { return existing.id === conversation.id; })) {
        continue;
      }
      normalized.push(conversation);
      if (normalized.length === MAX_CONVERSATIONS) {
        break;
      }
    }
    return normalized;
  }

  function normalizeConversationSummary(value) {
    if (!isRecord(value)) {
      return null;
    }
    const id = safeConversationId(value.conversation_id);
    if (id === null) {
      return null;
    }
    return {
      createdAt: safeTimestamp(value.created_at),
      id,
      messageCount: asNonNegativeInteger(value.message_count) || 0,
      title: safeTitle(value.title),
      updatedAt: safeTimestamp(value.updated_at),
    };
  }

  function normalizeConversationDetail(value) {
    if (!isRecord(value)) {
      return null;
    }
    const summary = normalizeConversationSummary(value);
    if (summary === null || !Array.isArray(value.messages)) {
      return null;
    }
    const messages = [];
    for (const source of value.messages) {
      const message = normalizeStoredMessage(source);
      if (message !== null) {
        messages.push(message);
      }
    }
    return { ...summary, messages, messagesTruncated: value.messages_truncated === true };
  }

  function normalizeStoredMessage(value) {
    if (!isRecord(value) || (value.role !== "user" && value.role !== "assistant")) {
      return null;
    }
    const metadata = isRecord(value.meta) ? value.meta : {};
    return {
      chart: normalizeStoredChart(value.chart),
      content: safeMessageContent(value.content, value.role === "assistant" ? "No response was saved." : ""),
      createdAt: safeTimestamp(value.created_at),
      historical: true,
      meta: {
        answerFallbackUsed: metadata.answer_fallback_used === true,
        hasResult: metadata.has_result === true,
        refused: metadata.refused === true,
        success: metadata.success !== false,
      },
      question: "",
      role: value.role,
      rowCount: asNonNegativeInteger(value.row_count),
      rows: null,
      columns: null,
      truncated: value.truncated === true,
      maxRows: null,
    };
  }

  function normalizeCurrentChart(value) {
    const source = isRecord(value) ? value : {};
    const url = safeChartUrl(source.url);
    return {
      available: url !== null,
      generated: url !== null,
      historical: false,
      note: safeOptionalText(source.note, 500),
      requested: source.requested === true,
      type: safeChartType(source.type),
      url,
    };
  }

  function normalizeStoredChart(value) {
    const source = isRecord(value) ? value : {};
    const url = source.available === true ? safeChartUrl(source.url) : null;
    const type = safeChartType(source.type);
    return {
      available: url !== null,
      generated: type !== null,
      historical: true,
      note: safeOptionalText(source.note, 500),
      requested: source.requested === true,
      type,
      url,
    };
  }

  function normalizeColumns(value) {
    if (!Array.isArray(value)) {
      return [];
    }
    return value.slice(0, 100).map(function (column) {
      return typeof column === "string" ? column.slice(0, 250) : displayValue(column).slice(0, 250);
    });
  }

  function normalizeRows(value, columnCount) {
    if (!Array.isArray(value) || columnCount === 0) {
      return [];
    }
    return value.slice(0, 1000).map(function (row) {
      return Array.isArray(row) ? row.slice(0, columnCount) : [];
    });
  }

  function createConversationEmpty() {
    const empty = document.createElement("div");
    const icon = document.createElement("span");
    const brand = document.createElement("p");
    const heading = document.createElement("h3");
    const copy = document.createElement("span");
    const suggestions = document.createElement("div");
    empty.className = "conversation-empty";
    empty.id = "conversation-empty";
    icon.className = "conversation-empty-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = "✦";
    brand.className = "welcome-brand";
    brand.textContent = "Analytics Assistant";
    heading.className = "welcome-heading";
    heading.textContent = "What can I help you analyze?";
    copy.className = "welcome-copy";
    copy.textContent = "Choose a starting point or ask your own question below.";
    suggestions.className = "suggestion-grid";
    suggestions.setAttribute("aria-label", "Suggested questions");
    suggestions.append(
      createSuggestionCard(
        "Pipeline Analysis",
        "Understand your current pipeline",
        "Analyze our current pipeline by stage.",
      ),
      createSuggestionCard(
        "Performance Insights",
        "Compare regions, owners, or accounts",
        "Compare performance across regions.",
      ),
      createSuggestionCard(
        "Opportunity Trends",
        "Explore how opportunities change over time",
        "Show opportunity creation trends over time.",
      ),
    );
    empty.append(icon, brand, heading, copy, suggestions);
    return empty;
  }

  function createSuggestionCard(title, description, question) {
    const button = document.createElement("button");
    const heading = document.createElement("strong");
    const copy = document.createElement("span");
    button.type = "button";
    button.className = "suggestion-card";
    button.dataset.exampleQuestion = question;
    heading.textContent = title;
    copy.textContent = description;
    button.append(heading, copy);
    return button;
  }

  function removeConversationEmpty() {
    const empty = document.getElementById("conversation-empty");
    if (empty && empty.parentElement === ui.conversationStream) {
      empty.remove();
    }
  }

  function updateConversationHeaderForActive() {
    const active = conversations.find(function (conversation) {
      return conversation.id === activeConversationId;
    });
    if (active) {
      updateConversationHeader(active);
      renderConversationList(conversations);
      return;
    }
    ui.conversationMeta.textContent = "This chat is being saved locally.";
  }

  function updateConversationHeader(conversation) {
    ui.activeConversationTitle.textContent = conversation.title;
    ui.conversationMeta.textContent = conversation.messageCount === 0
      ? "This chat is ready for your first question."
      : `${formatCount(conversation.messageCount)} ${conversation.messageCount === 1 ? "message" : "messages"} saved locally.`;
  }

  function scrollConversationToEnd() {
    window.setTimeout(function () {
      ui.conversationStream.scrollTop = ui.conversationStream.scrollHeight;
    }, 0);
  }

  async function readJson(response) {
    try {
      return await response.json();
    } catch (_error) {
      return null;
    }
  }

  function setSubmitting(submitting) {
    isSubmitting = submitting;
    ui.askButton.disabled = submitting;
    ui.askButton.classList.toggle("is-loading", submitting);
    ui.questionInput.disabled = submitting;
    ui.conversationStream.setAttribute("aria-busy", String(submitting));
    syncConversationControls();
  }

  function setConversationActionBusy(busy) {
    isConversationAction = busy;
    syncConversationControls();
  }

  function syncConversationControls() {
    const disabled = isSubmitting || isConversationAction;
    for (const button of ui.newChatButtons) {
      button.disabled = disabled;
    }
    ui.deleteAllButton.disabled = disabled;
    for (const button of document.querySelectorAll("[data-delete-conversation]")) {
      button.disabled = disabled;
    }
  }

  function toggleTheme() {
    applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
  }

  function applyTheme(theme) {
    const selectedTheme = theme === "light" ? "light" : "dark";
    const lightThemeActive = selectedTheme === "light";
    document.documentElement.dataset.theme = selectedTheme;
    ui.themeToggle.setAttribute("aria-pressed", String(lightThemeActive));
    ui.themeIcon.textContent = lightThemeActive ? "◐" : "☼";
    ui.themeLabel.textContent = lightThemeActive ? "Dark theme" : "Light theme";
    ui.themeToggle.setAttribute("aria-label", lightThemeActive ? "Use dark theme" : "Use light theme");
    writeStoredValue(STORAGE_KEYS.theme, selectedTheme);
  }

  function activateView(requestedView, shouldFocus) {
    const view = requestedView === "ask" ? "chat" : requestedView;
    const activeView = VIEWS.has(view) ? view : "chat";
    for (const section of ui.views) {
      const isActive = section.dataset.view === activeView;
      section.hidden = !isActive;
      section.classList.toggle("is-active", isActive);
    }
    for (const button of ui.navButtons) {
      const isActive = button.dataset.viewTarget === activeView;
      button.classList.toggle("is-active", isActive);
      if (isActive) {
        button.setAttribute("aria-current", "page");
      } else {
        button.removeAttribute("aria-current");
      }
    }
    writeStoredValue(STORAGE_KEYS.view, activeView);
    closeNavigation();
    if (shouldFocus) {
      const target = document.getElementById(`view-${activeView}`);
      if (target) {
        target.focus();
      }
    }
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
    return "The request could not be completed. Please review it and try again.";
  }

  function conversationGroupLabel(value) {
    if (value === null) {
      return "Older";
    }
    const timestamp = new Date(value);
    if (Number.isNaN(timestamp.getTime())) {
      return "Older";
    }
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    if (timestamp >= today) {
      return "Today";
    }
    if (timestamp >= yesterday) {
      return "Yesterday";
    }
    const previousSevenDays = new Date(today);
    previousSevenDays.setDate(previousSevenDays.getDate() - 7);
    if (timestamp >= previousSevenDays) {
      return "Previous 7 Days";
    }
    return "Older";
  }

  function readStoredTheme() {
    const savedTheme = readStoredValue(STORAGE_KEYS.theme);
    return savedTheme === "light" || savedTheme === "dark" ? savedTheme : "dark";
  }

  function readStoredView() {
    const savedView = readStoredValue(STORAGE_KEYS.view);
    if (savedView === "ask") {
      return "chat";
    }
    return VIEWS.has(savedView) ? savedView : "chat";
  }

  function safeConversationId(value) {
    if (typeof value !== "string") {
      return null;
    }
    const candidate = value.trim();
    return CONVERSATION_ID_PATTERN.test(candidate) ? candidate : null;
  }

  function safeTitle(value) {
    if (typeof value !== "string") {
      return "Untitled chat";
    }
    const title = value.trim().slice(0, MAX_TITLE_CHARS);
    return title.length === 0 ? "Untitled chat" : title;
  }

  function safeMessageContent(value, fallback) {
    if (typeof value !== "string") {
      return fallback;
    }
    const content = value.slice(0, MAX_MESSAGE_CHARS);
    return content.trim().length === 0 ? fallback : content;
  }

  function safeOptionalText(value, maxLength) {
    if (typeof value !== "string") {
      return "";
    }
    return value.trim().slice(0, maxLength);
  }

  function safeTimestamp(value) {
    if (typeof value !== "string") {
      return null;
    }
    const timestamp = new Date(value);
    return Number.isNaN(timestamp.getTime()) ? null : timestamp.toISOString();
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
      // The displayed UI remains usable when browser storage is unavailable.
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
      : "the saved question";
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
    if (value === null) {
      return "Recent";
    }
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
