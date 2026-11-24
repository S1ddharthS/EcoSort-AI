(function () {
  "use strict";

  const toastRegion = document.getElementById("toast-region");
  const dialog = document.getElementById("confirm-dialog");
  const confirmTitle = document.getElementById("confirm-title");
  const confirmMessage = document.getElementById("confirm-message");
  const confirmAccept = document.getElementById("confirm-accept");

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function toast(message, type) {
    if (!toastRegion) return;
    const item = document.createElement("div");
    item.className = "toast toast-" + (type || "info");
    item.setAttribute("role", "status");
    item.innerHTML =
      '<svg class="icon" aria-hidden="true" focusable="false"><use href="/static/img/icons.svg#i-' +
      (type === "error" ? "alert" : "check-circle") +
      '"></use></svg><span>' +
      escapeHtml(message) +
      "</span>";
    toastRegion.appendChild(item);
    window.setTimeout(function () {
      item.classList.add("is-leaving");
      window.setTimeout(function () {
        item.remove();
      }, 220);
    }, 3800);
  }

  function confirmAction(options) {
    const settings = options || {};
    if (!dialog || typeof dialog.showModal !== "function") {
      const accepted = window.confirm(settings.message || "Are you sure?");
      return Promise.resolve(accepted);
    }
    confirmTitle.textContent = settings.title || "Are you sure?";
    confirmMessage.textContent = settings.message || "";
    confirmAccept.textContent = settings.confirmLabel || "Delete";
    confirmAccept.className =
      "btn " + (settings.neutral ? "btn-primary" : "btn-danger");
    dialog.returnValue = "";
    dialog.showModal();

    return new Promise(function (resolve) {
      function onClose() {
        dialog.removeEventListener("close", onClose);
        resolve(dialog.returnValue === "confirm");
      }
      dialog.addEventListener("close", onClose);
    });
  }

  if (dialog) {
    dialog.addEventListener("click", function (event) {
      if (event.target === dialog) {
        dialog.close("cancel");
      }
    });
  }

  // Theme toggle -----------------------------------------------------------
  const themeToggle = document.getElementById("theme-toggle");
  const themeMeta = document.getElementById("theme-color-meta");
  function applyTheme(nextTheme, persist) {
    document.documentElement.setAttribute("data-theme", nextTheme);
    if (themeMeta) {
      themeMeta.setAttribute("content", nextTheme === "dark" ? "#0C1714" : "#F7F5EF");
    }
    if (themeToggle) {
      const isDark = nextTheme === "dark";
      themeToggle.setAttribute("aria-pressed", String(isDark));
      themeToggle.setAttribute(
        "aria-label",
        isDark ? "Switch to light mode" : "Switch to dark mode"
      );
    }
    if (persist) {
      try {
        localStorage.setItem("ecosort-theme", nextTheme);
      } catch (error) {
        /* Storage can be disabled; the visual toggle still works. */
      }
    }
  }

  applyTheme(document.documentElement.getAttribute("data-theme") || "light", false);
  if (themeToggle) {
    themeToggle.addEventListener("click", function () {
      const current = document.documentElement.getAttribute("data-theme");
      applyTheme(current === "dark" ? "light" : "dark", true);
    });
  }

  if (window.matchMedia) {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = function (event) {
      let hasStored = false;
      try {
        hasStored = Boolean(localStorage.getItem("ecosort-theme"));
      } catch (error) {
        hasStored = false;
      }
      if (!hasStored) applyTheme(event.matches ? "dark" : "light", false);
    };
    if (typeof media.addEventListener === "function") {
      media.addEventListener("change", onChange);
    }
  }

  window.EcoSort = {
    escapeHtml: escapeHtml,
    toast: toast,
    confirmAction: confirmAction
  };
})();
