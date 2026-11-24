(function () {
  "use strict";

  const input = document.getElementById("guide-search");
  if (!input) return;

  const list = document.getElementById("learn-list");
  const cards = Array.from(list.querySelectorAll(".learn-card"));
  const count = document.getElementById("search-count");
  const empty = document.getElementById("learn-empty");
  const clearButton = document.getElementById("clear-guide-search");

  function applySearch() {
    const query = input.value.trim().toLowerCase();
    let matches = 0;
    cards.forEach(function (card) {
      const isMatch = !query || card.textContent.toLowerCase().includes(query);
      card.hidden = !isMatch;
      if (isMatch) {
        matches += 1;
        if (query) card.open = true;
      }
    });
    if (query) {
      count.textContent =
        matches === 0
          ? "No guides match “" + input.value.trim() + "”"
          : matches + (matches === 1 ? " guide" : " guides") + " found";
    } else {
      count.textContent = "Showing all " + cards.length + " guides";
    }
    empty.hidden = matches !== 0;
    list.hidden = matches === 0;
  }

  input.addEventListener("input", applySearch);

  clearButton.addEventListener("click", function () {
    input.value = "";
    applySearch();
    input.focus();
  });

  function openHashTarget() {
    if (!window.location.hash) return;
    let target = null;
    try {
      target = document.querySelector(window.location.hash);
    } catch (error) {
      target = null;
    }
    if (target && target.classList.contains("learn-card")) {
      target.open = true;
      window.setTimeout(function () {
        target.scrollIntoView({
          block: "start",
          behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
            ? "auto"
            : "smooth"
        });
      }, 80);
    }
  }

  window.addEventListener("hashchange", openHashTarget);
  openHashTarget();
})();
