(function () {
  "use strict";

  var stored = null;
  try {
    stored = localStorage.getItem("ecosort-theme");
  } catch (error) {
    stored = null;
  }
  var prefersDark = window.matchMedia
    ? window.matchMedia("(prefers-color-scheme: dark)").matches
    : false;
  var theme = stored || (prefersDark ? "dark" : "light");
  document.documentElement.setAttribute("data-theme", theme);

  var meta = document.getElementById("theme-color-meta");
  if (meta) {
    meta.setAttribute("content", theme === "dark" ? "#0C1714" : "#F7F5EF");
  }
})();
