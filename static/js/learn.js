(function () {
  "use strict";

  const input = document.getElementById("guide-search");
  if (!input) return;

  const list = document.getElementById("learn-list");
  const cards = Array.from(list.querySelectorAll(".learn-card"));
  const count = document.getElementById("search-count");
  const empty = document.getElementById("learn-empty");
  const clearButton = document.getElementById("clear-guide-search");
  let savedOpenState = null;

  function applySearch() {
    const rawQuery = input.value.trim();
    const query = rawQuery.replace(/\s+/g, " ").toLowerCase();
    if (query && !savedOpenState) {
      savedOpenState = cards.map(function (card) {
        return card.open;
      });
    }
    let matches = 0;
    cards.forEach(function (card, index) {
      const text = card.textContent.replace(/\s+/g, " ").toLowerCase();
      const isMatch = !query || text.includes(query);
      card.hidden = !isMatch;
      if (isMatch) {
        matches += 1;
        if (query) {
          card.open = true;
        } else if (savedOpenState) {
          card.open = savedOpenState[index];
        }
      }
    });
    if (!query) {
      savedOpenState = null;
    }
    if (query) {
      count.textContent =
        matches === 0
          ? "No guides match “" + rawQuery + "”"
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
      savedOpenState = null;
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
