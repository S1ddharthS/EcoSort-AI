(function () {
  "use strict";

  const grid = document.getElementById("history-grid");
  if (!grid) return;

  const totalEl = document.getElementById("history-total");
  const statusEl = document.getElementById("history-status");
  const clearButton = document.getElementById("clear-history");
  const errorBox = document.getElementById("history-error");
  const errorMessage = document.getElementById("history-error-message");
  const retryButton = document.getElementById("history-retry");
  const empty = document.getElementById("history-empty");
  const emptyTitle = document.getElementById("history-empty-title");
  const emptyText = document.getElementById("history-empty-text");
  const emptyAction = document.getElementById("history-empty-action");
  const showAllButton = document.getElementById("show-all-button");
  const filterButtons = Array.from(document.querySelectorAll(".filter-pill"));
  const esc =
    (window.EcoSort && window.EcoSort.escapeHtml) ||
    function (value) {
      return String(value == null ? "" : value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    };

  const state = {
    filter: "",
    items: [],
    total: 0,
    controller: null,
    loading: false
  };

  const categoryIcons = {
    plastic: "bottle",
    paper: "file",
    cardboard: "box",
    glass: "glass",
    metal: "can",
    organic: "apple"
  };

  function icon(name, cls) {
    return (
      '<svg class="' +
      (cls || "icon") +
      '" aria-hidden="true" focusable="false"><use href="/static/img/icons.svg#i-' +
      name +
      '"></use></svg>'
    );
  }

  function catIcon(category, cls) {
    return icon(categoryIcons[category] || "recycle", cls);
  }

  function formatDate(value) {
    if (!value) return "Unknown date";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    try {
      return new Intl.DateTimeFormat(undefined, {
        day: "numeric",
        month: "short",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit"
      }).format(date);
    } catch (error) {
      return date.toLocaleString();
    }
  }

  function filterLabel() {
    const active = filterButtons.find(function (button) {
      return button.dataset.category === state.filter;
    });
    return active ? active.textContent.trim() : "items";
  }

  function updatePills() {
    filterButtons.forEach(function (button) {
      const active = button.dataset.category === state.filter;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });
  }

  function skeletonMarkup() {
    let markup = "";
    for (let i = 0; i < 6; i += 1) {
      markup +=
        '<div class="history-card skeleton-card" aria-hidden="true">' +
        '<div class="skeleton skeleton-image"></div>' +
        '<div class="skeleton-body"><div class="skeleton skeleton-line skeleton-line-lg"></div>' +
        '<div class="skeleton skeleton-line"></div><div class="skeleton skeleton-line skeleton-line-sm"></div></div></div>';
    }
    return markup;
  }

  function setLoading() {
    state.loading = true;
    errorBox.hidden = true;
    empty.hidden = true;
    grid.hidden = false;
    grid.innerHTML = skeletonMarkup();
    grid.setAttribute("aria-busy", "true");
    statusEl.textContent = "Loading your history...";
    clearButton.disabled = true;
  }

  function showError(message) {
    grid.innerHTML = "";
    grid.hidden = true;
    grid.setAttribute("aria-busy", "false");
    empty.hidden = true;
    errorBox.hidden = false;
    errorMessage.textContent =
      message ||
      "Please check that the server is running and try again.";
    statusEl.textContent = "History couldn't be loaded.";
    clearButton.disabled = true;
  }

  function renderEmpty() {
    grid.innerHTML = "";
    grid.hidden = true;
    empty.hidden = false;
    if (state.filter) {
      const label = filterLabel().toLowerCase();
      emptyTitle.textContent = "No " + label + " items yet";
      emptyText.textContent =
        "There are other items in your history, but none in this category. Try another filter or classify something new.";
      if (emptyAction) emptyAction.hidden = true;
      showAllButton.hidden = false;
    } else {
      emptyTitle.textContent = "No classifications yet";
      emptyText.textContent =
        "Take your first photo and it will appear here automatically, with its category and confidence.";
      if (emptyAction) emptyAction.hidden = false;
      showAllButton.hidden = true;
    }
  }

  function cardMarkup(item, index) {
    const dateText = formatDate(item.created_at);
    const percent =
      item.confidence_percent != null
        ? item.confidence_percent
        : Math.round(Number(item.confidence) * 100);
    const confidenceClass = item.uncertain ? "caution" : "positive";
    const imageMarkup = item.image_available
      ? '<img src="' + esc(item.image_url) + '" alt="Saved photo of ' + esc(item.display_name) + '" loading="lazy">'
      : "";
    return (
      '<article class="history-card" data-id="' +
      Number(item.id) +
      '" style="--card-index:' +
      index +
      '">' +
      '<div class="history-thumb cat-bg-soft-' +
      esc(item.category) +
      '">' +
      '<span class="history-thumb-fallback" aria-hidden="true">' +
      catIcon(item.category, "icon icon-xl") +
      "</span>" +
      imageMarkup +
      '<span class="history-cat-badge cat-' +
      esc(item.category) +
      '">' +
      catIcon(item.category, "icon icon-sm") +
      " " +
      esc(item.display_name) +
      "</span>" +
      '<span class="history-confidence">' +
      percent +
      "%</span>" +
      "</div>" +
      '<div class="history-body">' +
      '<div class="history-card-top"><div><h2>' +
      esc(item.display_name) +
      '</h2><p class="history-date">' +
      icon("clock", "icon icon-sm") +
      " " +
      esc(dateText) +
      "</p></div>" +
      '<button class="icon-btn delete-button" type="button" data-id="' +
      Number(item.id) +
      '" aria-label="Delete ' +
      esc(item.display_name) +
      " classification from " +
      esc(dateText) +
      '">' +
      icon("trash") +
      "</button></div>" +
      '<div class="history-meta"><span class="badge badge-' +
      confidenceClass +
      '">' +
      esc(item.confidence_label) +
      "</span>" +
      '<span class="history-note">' +
      (item.uncertain ? "No bin advice given" : "Recycling advice shown") +
      "</span></div>" +
      "</div></article>"
    );
  }

  function bindCardEvents() {
    grid.querySelectorAll(".delete-button").forEach(function (button) {
      button.addEventListener("click", function () {
        deleteItem(Number(button.dataset.id));
      });
    });
    grid.querySelectorAll(".history-thumb img").forEach(function (image) {
      image.addEventListener("error", function () {
        image.hidden = true;
      });
    });
  }

  function render() {
    grid.setAttribute("aria-busy", "false");
    totalEl.textContent = String(state.total);
    if (state.total === 0) {
      // Never imply that other filtered items exist when the whole history is
      // empty (for example after deleting the last filtered result).
      state.filter = "";
    }
    updatePills();
    if (!state.items.length) {
      statusEl.textContent = state.total
        ? "No items in this view. " + state.total + " saved in total."
        : "Newest first";
      clearButton.disabled = state.total === 0;
      renderEmpty();
      return;
    }
    clearButton.disabled = false;
    empty.hidden = true;
    grid.hidden = false;
    grid.innerHTML = state.items.map(cardMarkup).join("");
    bindCardEvents();
    statusEl.textContent = state.filter
      ? "Showing " + state.items.length + " of " + state.total + " items · Newest first"
      : state.total + (state.total === 1 ? " item" : " items") + " · Newest first";
  }

  async function load() {
    if (state.controller) state.controller.abort();
    const controller = new AbortController();
    state.controller = controller;
    setLoading();
    const query = state.filter
      ? "?category=" + encodeURIComponent(state.filter)
      : "";
    try {
      const [historyResponse, statsResponse] = await Promise.all([
        fetch("/api/history" + query, {
          headers: { Accept: "application/json" },
          signal: controller.signal
        }),
        fetch("/api/stats", {
          headers: { Accept: "application/json" },
          signal: controller.signal
        })
      ]);
      if (!historyResponse.ok) {
        let message = "The history endpoint returned an error.";
        try {
          const payload = await historyResponse.json();
          message = payload.message || payload.error || message;
        } catch (error) {
          /* Keep the default message. */
        }
        throw new Error(message);
      }
      if (!statsResponse.ok) {
        throw new Error("The statistics endpoint returned an error.");
      }
      const data = await historyResponse.json();
      const stats = await statsResponse.json();
      if (state.controller !== controller) return;
      state.items = Array.isArray(data.items) ? data.items : [];
      state.total = Number(stats.total) || 0;
      state.loading = false;
      render();
    } catch (error) {
      if (error.name === "AbortError") return;
      if (state.controller !== controller) return;
      state.loading = false;
      state.items = [];
      showError(
        error && error.message
          ? error.message
          : "We couldn't load your history. Please try again."
      );
    }
  }

  async function deleteItem(id) {
    if (!id) return;
    const accepted = await window.EcoSort.confirmAction({
      title: "Delete this result?",
      message:
        "This removes the classification and its saved preview image from this device. This can't be undone.",
      confirmLabel: "Delete item"
    });
    if (!accepted) return;
    try {
      const response = await fetch("/api/history/" + id, {
        method: "DELETE",
        headers: { Accept: "application/json" }
      });
      const payload = await response.json().catch(function () {
        return {};
      });
      if (!response.ok) {
        throw new Error(
          payload.message || "The item couldn't be deleted. Please try again."
        );
      }
      window.EcoSort.toast("Item deleted.", "success");
      load();
    } catch (error) {
      window.EcoSort.toast(
        error.message || "The item couldn't be deleted. Please try again.",
        "error"
      );
    }
  }

  async function clearAll() {
    if (state.total === 0) return;
    const accepted = await window.EcoSort.confirmAction({
      title: "Clear all history?",
      message:
        "This deletes all " +
        state.total +
        " saved results and their preview images. This can't be undone.",
      confirmLabel: "Clear everything"
    });
    if (!accepted) return;
    try {
      const response = await fetch("/api/history", {
        method: "DELETE",
        headers: { Accept: "application/json" }
      });
      const payload = await response.json().catch(function () {
        return {};
      });
      if (!response.ok) {
        throw new Error(
          payload.message || "History couldn't be cleared. Please try again."
        );
      }
      state.filter = "";
      window.EcoSort.toast("History cleared.", "success");
      load();
    } catch (error) {
      window.EcoSort.toast(
        error.message || "History couldn't be cleared. Please try again.",
        "error"
      );
    }
  }

  filterButtons.forEach(function (button) {
    button.addEventListener("click", function () {
      const nextFilter = button.dataset.category || "";
      if (nextFilter === state.filter) return;
      state.filter = nextFilter;
      load();
    });
  });
  clearButton.addEventListener("click", clearAll);
  retryButton.addEventListener("click", load);
  showAllButton.addEventListener("click", function () {
    state.filter = "";
    load();
  });

  load();
})();
