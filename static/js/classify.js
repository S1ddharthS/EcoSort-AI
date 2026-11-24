(function () {
  "use strict";

  const classifyBtn = document.getElementById("classify-btn");
  if (!classifyBtn) return;

  const esc =
    (window.EcoSort && window.EcoSort.escapeHtml) ||
    function (value) {
      return String(value == null ? "" : value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    };

  const tabs = [
    {
      button: document.getElementById("tab-upload"),
      panel: document.getElementById("panel-upload"),
      mode: "upload"
    },
    {
      button: document.getElementById("tab-camera"),
      panel: document.getElementById("panel-camera"),
      mode: "camera"
    }
  ];
  const fileInput = document.getElementById("file-input");
  const dropzone = document.getElementById("dropzone");
  const uploadError = document.getElementById("upload-error");
  const previewWrap = document.getElementById("selected-preview");
  const previewImage = document.getElementById("preview-image");
  const previewName = document.getElementById("preview-name");
  const removeImageBtn = document.getElementById("remove-image");
  const helpText = document.getElementById("classification-help");
  const analysis = document.getElementById("analysis-state");
  const errorBox = document.getElementById("classify-error");
  const errorTitle = document.getElementById("classify-error-title");
  const errorMessage = document.getElementById("classify-error-message");
  const errorHint = document.getElementById("classify-error-hint");
  const retryButton = document.getElementById("retry-classify");
  const resultSection = document.getElementById("result-section");

  const startCameraBtn = document.getElementById("start-camera");
  const cameraStart = document.getElementById("camera-start");
  const cameraLive = document.getElementById("camera-live");
  const cameraReview = document.getElementById("camera-review");
  const cameraPermission = document.getElementById("camera-permission");
  const cameraSelected = document.getElementById("camera-selected");
  const cameraVideo = document.getElementById("camera-video");
  const capturedImage = document.getElementById("captured-image");
  const captureBtn = document.getElementById("capture-btn");
  const retakeBtn = document.getElementById("retake-btn");
  const useCaptureBtn = document.getElementById("use-capture-btn");
  const retakeSelectedBtn = document.getElementById("retake-selected");
  const cameraErrorTitle = document.getElementById("camera-error-title");
  const cameraErrorText = document.getElementById("camera-error-text");
  const cameraFallback = document.getElementById("camera-fallback");
  const cameraHint = document.getElementById("camera-hint");

  const state = {
    mode: "upload",
    selectedFile: null,
    cameraDataUrl: null,
    capturedDataUrl: null,
    objectUrl: null,
    stream: null,
    busy: false,
    hasResult: false,
    modelReady: document.body.dataset.modelReady === "true"
  };

  const scanButtonMarkup =
    '<svg class="icon" aria-hidden="true" focusable="false"><use href="/static/img/icons.svg#i-scan"></use></svg><span>Classify item</span>';
  const loadingButtonMarkup =
    '<span class="btn-spinner" aria-hidden="true"></span><span>Analysing...</span>';
  const maxBytes = 5 * 1024 * 1024;

  function escape(value) {
    return esc(value);
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  }

  function icon(name, cls) {
    return (
      '<svg class="' +
      (cls || "icon") +
      '" aria-hidden="true" focusable="false"><use href="/static/img/icons.svg#i-' +
      name +
      '"></use></svg>'
    );
  }

  const categoryIcons = {
    plastic: "bottle",
    paper: "file",
    cardboard: "box",
    glass: "glass",
    metal: "can",
    organic: "apple"
  };

  function catIcon(category, cls) {
    return icon(categoryIcons[category] || "recycle", cls);
  }

  function showUploadError(message) {
    uploadError.textContent = message;
    uploadError.hidden = false;
  }

  function clearUploadError() {
    uploadError.hidden = true;
    uploadError.textContent = "";
  }

  function hasImage() {
    return Boolean(state.selectedFile || state.cameraDataUrl);
  }

  function updateActionState() {
    let disabled = !hasImage() || state.hasResult || state.busy;
    let message = "Add a photo to start.";
    if (hasImage() && !state.hasResult) message = "Ready when you are.";
    if (state.hasResult) {
      message = "Result saved. Choose “Classify another item” to start fresh.";
    }
    if (!state.modelReady) {
      disabled = true;
      message =
        "Classification is unavailable because the AI model isn't installed.";
    }
    classifyBtn.disabled = disabled;
    helpText.textContent = message;
    previewWrap.hidden = !hasImage();
  }

  function setBusy(isBusy) {
    state.busy = isBusy;
    tabs.forEach(function (tab) {
      tab.button.disabled = isBusy;
    });
    dropzone.setAttribute("aria-disabled", String(isBusy));
    dropzone.tabIndex = isBusy ? -1 : 0;
    fileInput.disabled = isBusy;
    removeImageBtn.disabled = isBusy;
    startCameraBtn.disabled = isBusy;
    captureBtn.disabled = isBusy;
    retakeBtn.disabled = isBusy;
    useCaptureBtn.disabled = isBusy;
    retakeSelectedBtn.disabled = isBusy;
    cameraFallback.disabled = isBusy;
    if (isBusy) {
      classifyBtn.innerHTML = loadingButtonMarkup;
      classifyBtn.disabled = true;
      classifyBtn.setAttribute("aria-busy", "true");
    } else {
      classifyBtn.innerHTML = scanButtonMarkup;
      classifyBtn.removeAttribute("aria-busy");
      updateActionState();
    }
  }

  function resetResult() {
    state.hasResult = false;
    resultSection.hidden = true;
    resultSection.innerHTML = "";
    updateActionState();
  }

  function revokeObjectUrl() {
    if (state.objectUrl) {
      URL.revokeObjectURL(state.objectUrl);
      state.objectUrl = null;
    }
  }

  function showPreview(src, label) {
    previewImage.src = src;
    previewImage.alt = "Selected photo: " + label;
    previewName.textContent = label;
    previewWrap.hidden = false;
  }

  function resetCameraView() {
    stopStream();
    state.capturedDataUrl = null;
    cameraStart.hidden = false;
    cameraLive.hidden = true;
    cameraReview.hidden = true;
    cameraPermission.hidden = true;
    cameraSelected.hidden = true;
  }

  function hideCameraStages() {
    cameraStart.hidden = true;
    cameraLive.hidden = true;
    cameraReview.hidden = true;
    cameraPermission.hidden = true;
    cameraSelected.hidden = true;
  }

  function stopStream() {
    if (state.stream) {
      state.stream.getTracks().forEach(function (track) {
        track.stop();
      });
      state.stream = null;
    }
    if (cameraVideo) cameraVideo.srcObject = null;
  }

  function activateTab(mode, focusTab) {
    if (state.busy) return;
    if (mode === state.mode && focusTab) return;
    if (mode !== "camera") {
      stopStream();
      if (!cameraLive.hidden) {
        hideCameraStages();
        cameraStart.hidden = false;
      }
    }
    state.mode = mode;
    tabs.forEach(function (tab) {
      const active = tab.mode === mode;
      tab.button.classList.toggle("is-active", active);
      tab.button.setAttribute("aria-selected", String(active));
      tab.button.tabIndex = active ? 0 : -1;
      tab.panel.hidden = !active;
    });
    if (mode === "camera") {
      if (state.cameraDataUrl) {
        hideCameraStages();
        cameraSelected.hidden = false;
      } else if (state.capturedDataUrl) {
        hideCameraStages();
        cameraReview.hidden = false;
      }
    } else {
      clearUploadError();
    }
    if (focusTab) {
      const activeTab = tabs.find(function (tab) {
        return tab.mode === mode;
      });
      activeTab.button.focus();
    }
  }

  function setSelectionFromFile(file) {
    clearUploadError();
    if (!file) return;
    if (file.size === 0) {
      showUploadError("That file is empty. Please choose another JPG, PNG or WEBP image.");
      return;
    }
    if (file.size > maxBytes) {
      showUploadError(
        "That image is larger than 5 MB. Please choose a smaller file."
      );
      return;
    }
    const allowedTypes = ["image/jpeg", "image/png", "image/webp"];
    const extension = (file.name.split(".").pop() || "").toLowerCase();
    const allowedExtensions = ["jpg", "jpeg", "png", "webp"];
    if (
      !allowedTypes.includes(file.type) &&
      !allowedExtensions.includes(extension)
    ) {
      showUploadError(
        "That file isn't an image. Please upload a JPG, PNG or WEBP. HEIC photos need to be exported as JPG first."
      );
      return;
    }

    revokeObjectUrl();
    const url = URL.createObjectURL(file);
    const probe = new Image();
    probe.onload = function () {
      stopStream();
      state.selectedFile = file;
      state.cameraDataUrl = null;
      state.objectUrl = url;
      state.mode = "upload";
      resetResult();
      showPreview(url, file.name + " · " + formatBytes(file.size));
      hideCameraStages();
      cameraStart.hidden = false;
      clearUploadError();
      updateActionState();
      uploadError.hidden = true;
    };
    probe.onerror = function () {
      URL.revokeObjectURL(url);
      showUploadError(
        "We couldn't open that image. Please try a different JPG, PNG or WEBP file."
      );
    };
    probe.src = url;
  }

  function setSelectionFromCamera(dataUrl) {
    state.selectedFile = null;
    state.cameraDataUrl = dataUrl;
    resetResult();
    showPreview(dataUrl, "Camera photo · ready to classify");
    hideCameraStages();
    cameraSelected.hidden = false;
    updateActionState();
  }

  function clearSelection(options) {
    const opts = options || {};
    stopStream();
    revokeObjectUrl();
    state.selectedFile = null;
    state.cameraDataUrl = null;
    state.capturedDataUrl = null;
    fileInput.value = "";
    previewImage.removeAttribute("src");
    previewWrap.hidden = true;
    hideCameraStages();
    cameraStart.hidden = false;
    resetResult();
    errorBox.hidden = true;
    analysis.hidden = true;
    clearUploadError();
    if (!opts.keepTab) activateTab("upload", false);
    updateActionState();
  }

  function showAnalysis() {
    errorBox.hidden = true;
    resultSection.hidden = true;
    analysis.hidden = false;
    analysis.scrollIntoView({
      block: "nearest",
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "auto"
        : "smooth"
    });
  }

  function showError(title, message, hint) {
    analysis.hidden = true;
    resultSection.hidden = true;
    state.hasResult = false;
    errorTitle.textContent = title || "We couldn't classify that";
    errorMessage.textContent = message;
    if (hint) {
      errorHint.textContent = hint;
      errorHint.hidden = false;
    } else {
      errorHint.hidden = true;
    }
    errorBox.hidden = false;
    errorBox.scrollIntoView({
      block: "nearest",
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "auto"
        : "smooth"
    });
  }

  function listItems(items, tone) {
    return (items || [])
      .map(function (item) {
        return (
          '<li>' +
          icon(tone === "do" ? "check" : "x") +
          "<span>" +
          escape(item) +
          "</span></li>"
        );
      })
      .join("");
  }

  function renderTopPredictions(topPredictions) {
    return topPredictions
      .map(function (prediction, index) {
        const percent = Math.round(Number(prediction.confidence) * 100);
        return (
          '<div class="top-prediction" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="' +
          percent +
          '" aria-label="' +
          escape(prediction.display_name) +
          ": " +
          percent +
          ' percent">' +
          '<div class="top-prediction-head"><span>' +
          catIcon(prediction.category, "icon icon-sm") +
          " " +
          escape(prediction.display_name) +
          '</span><span class="top-prediction-value">' +
          percent +
          "%</span></div>" +
          '<div class="prediction-track"><span style="width: ' +
          percent +
          '%"></span></div>' +
          "</div>"
        );
      })
      .join("");
  }

  function renderConfidentResult(payload) {
    const prediction = payload.prediction;
    const rec = payload.recommendation;
    if (!rec) {
      renderUncertainResult(payload);
      return;
    }
    const percent = Math.round(Number(prediction.confidence_percent));
    const tone =
      rec.recyclable === "Yes"
        ? "positive"
        : rec.recyclable === "No"
        ? "negative"
        : "caution";
    const steps = (rec.how_to_prepare || [])
      .map(function (step, index) {
        return (
          '<li><span class="prep-number">' +
          (index + 1) +
          "</span><span>" +
          escape(step) +
          "</span></li>"
        );
      })
      .join("");
    const doItems = listItems(rec.do_list, "do");
    const dontItems = listItems(rec.dont_list, "dont");
    const gradcamMarkup = payload.gradcam
      ? '<div class="gradcam-block">' +
        '<button class="btn btn-ghost btn-sm gradcam-toggle" id="gradcam-toggle" type="button" aria-expanded="false" aria-controls="gradcam-panel">' +
        icon("sparkles") +
        "<span>Show what the AI looked at</span>" +
        '<span class="gradcam-chevron">' +
        icon("chevron") +
        "</span></button>" +
        '<div class="gradcam-panel" id="gradcam-panel" hidden>' +
        '<img src="' +
        escape(payload.gradcam) +
        '" alt="Grad-CAM heatmap. Warmer colours show the areas that most influenced the classification." loading="lazy">' +
        "<p>Warmer colours had more influence on the decision. This is a guide to the model's focus, not a guarantee that it is right.</p>" +
        "</div></div>"
      : "";

    resultSection.innerHTML =
      '<article class="result-card" aria-labelledby="result-title">' +
      '<div class="result-hero">' +
      '<div class="result-photo"><img src="' +
      escape(prediction.image_url) +
      '" alt="Your classified item: ' +
      escape(prediction.display_name) +
      '" width="900" height="675"></div>' +
      '<div class="result-headline">' +
      '<p class="eyebrow">' +
      icon("check-circle") +
      " Classification complete</p>" +
      '<div class="result-title-row"><span class="cat-icon cat-' +
      escape(prediction.category) +
      ' result-cat-icon">' +
      catIcon(prediction.category) +
      '</span><div><h2 id="result-title">' +
      escape(prediction.display_name) +
      '</h2><p class="result-summary">This looks like ' +
      escape(prediction.display_name.toLowerCase()) +
      ". Here's how to handle it.</p></div></div>" +
      '<div class="confidence-wrap"><div class="confidence-ring" role="img" aria-label="' +
      percent +
      ' percent confidence" style="--pct: ' +
      percent +
      '%;--ring-color: var(--success)"><span>' +
      percent +
      "%</span></div><div><p class=\"confidence-label\">" +
      escape(prediction.confidence_label) +
      '</p><p class="confidence-copy">Based on the six waste types EcoSort knows.</p></div></div>' +
      "</div></div>" +
      '<div class="result-grid">' +
      '<div class="result-main">' +
      '<div class="result-badges">' +
      '<div class="status-panel status-panel-' +
      tone +
      '"><span class="status-panel-label">Recyclable</span><strong>' +
      escape(rec.recyclable) +
      "</strong><p>" +
      (tone === "positive"
        ? "This item is normally accepted in recycling."
        : tone === "negative"
        ? "Keep this out of your recycling bin."
        : "Accepted items and bin colours depend on your local service.") +
      "</p></div>" +
      '<div class="status-panel status-panel-bin">' +
      '<span class="status-panel-label">' +
      icon("map-pin") +
      " Typical bin</span><strong>" +
      escape(rec.bin_colour) +
      "</strong><p>Always check your local council's rules before binning.</p></div>" +
      "</div>" +
      '<section class="result-block" aria-labelledby="prepare-title"><div class="result-block-head"><h2 id="prepare-title">How to prepare it</h2><span class="muted-pill">' +
      (rec.how_to_prepare || []).length +
      " steps</span></div><ol class=\"prep-steps\">" +
      steps +
      "</ol></section>" +
      '<div class="dos-grid">' +
      '<section class="dos-card dos-yes" aria-labelledby="dos-title"><h2 id="dos-title">' +
      icon("check-circle") +
      " Do</h2><ul class=\"check-list\">" +
      doItems +
      "</ul></section>" +
      '<section class="dos-card dos-no" aria-labelledby="donts-title"><h2 id="donts-title">' +
      icon("x") +
      " Don't</h2><ul class=\"cross-list\">" +
      dontItems +
      "</ul></section>" +
      "</div>" +
      '<div class="fact-panel">' +
      icon("leaf") +
      "<div><h2>Good to know</h2><p>" +
      escape(rec.fun_fact) +
      "</p></div></div>" +
      '<p class="impact-copy"><strong>Why it matters:</strong> ' +
      escape(rec.environmental_impact) +
      "</p>" +
      gradcamMarkup +
      "</div>" +
      '<aside class="result-side" aria-label="Prediction details">' +
      '<h2>Top 3 predictions</h2>' +
      renderTopPredictions(payload.top_predictions) +
      '<p class="result-side-note">The model compares every photo against all six classes. A higher bar means a stronger match.</p>' +
      '<div class="saved-note">' +
      icon("check-circle") +
      "<p>Saved to History with its image and confidence.</p></div>" +
      "</aside></div>" +
      '<div class="result-actions"><button class="btn btn-primary" id="new-classification" type="button">' +
      icon("scan") +
      ' Classify another item</button><a class="btn btn-ghost" href="/history">' +
      icon("history") +
      " View history</a></div>" +
      "</article>";

    bindResultActions();
  }

  function renderUncertainResult(payload) {
    const prediction = payload.prediction;
    const tips = (payload.tips || [])
      .map(function (tip) {
        return "<li>" + icon("check") + "<span>" + escape(tip) + "</span></li>";
      })
      .join("");
    resultSection.innerHTML =
      '<article class="result-card uncertain-card" aria-labelledby="uncertain-title">' +
      '<div class="uncertain-head">' +
      '<span class="uncertain-icon">' +
      icon("question") +
      "</span>" +
      '<p class="eyebrow">Low confidence</p>' +
      '<h2 id="uncertain-title" tabindex="-1">We\'re not sure about this one</h2>' +
      '<p class="uncertain-lead">' +
      escape(
        payload.message ||
          "The model's best match was below 60% confident, so we're not recommending a bin. Guessing could put the wrong item in the wrong bin."
      ) +
      "</p>" +
      "</div>" +
      '<div class="uncertain-body">' +
      '<div class="uncertain-photo"><img src="' +
      escape(prediction.image_url) +
      '" alt="The photo that could not be classified confidently"></div>' +
      '<div class="uncertain-tips"><h2>Try again with</h2><ul class="check-list">' +
      tips +
      "</ul></div>" +
      "</div>" +
      '<p class="saved-note saved-note-light">' +
      icon("check-circle") +
      " This attempt was saved in History. No recycling advice was given because the model was not confident enough.</p>" +
      '<div class="result-actions"><button class="btn btn-primary" id="new-classification" type="button">' +
      icon("camera") +
      ' Try again</button><a class="btn btn-ghost" href="/history">' +
      icon("history") +
      " View history</a></div>" +
      "</article>";
    bindResultActions();
  }

  function bindResultActions() {
    const newButton = document.getElementById("new-classification");
    if (newButton) {
      newButton.addEventListener("click", function () {
        clearSelection();
        const dropzoneEl = document.getElementById("dropzone");
        dropzoneEl.focus({ preventScroll: true });
        dropzoneEl.scrollIntoView({
          block: "center",
          behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
            ? "auto"
            : "smooth"
        });
      });
    }
    const gradcamButton = document.getElementById("gradcam-toggle");
    const gradcamPanel = document.getElementById("gradcam-panel");
    if (gradcamButton && gradcamPanel) {
      gradcamButton.addEventListener("click", function () {
        const expanded = gradcamButton.getAttribute("aria-expanded") === "true";
        gradcamButton.setAttribute("aria-expanded", String(!expanded));
        gradcamPanel.hidden = expanded;
        const label = gradcamButton.querySelector("span:not(.gradcam-chevron)");
        if (label) {
          label.textContent = expanded
            ? "Show what the AI looked at"
            : "Hide AI focus map";
        }
      });
    }
    const resultImage = resultSection.querySelector(".result-photo img, .uncertain-photo img");
    if (resultImage) {
      resultImage.addEventListener("error", function () {
        resultImage.alt = "The saved preview image could not be loaded.";
        resultImage.classList.add("is-missing");
      });
    }
    const uncertainTitle = document.getElementById("uncertain-title");
    if (uncertainTitle) {
      window.requestAnimationFrame(function () {
        uncertainTitle.focus({ preventScroll: true });
      });
    }
  }

  async function classify() {
    if (state.busy || !hasImage() || state.hasResult) return;
    if (!state.modelReady) {
      showError(
        "The AI model isn't installed",
        "Classification can't run until the model file is in place.",
        "Add ml/model/ecosort_model.keras and restart the app. See README.md."
      );
      return;
    }

    setBusy(true);
    showAnalysis();
    try {
      let response;
      if (state.selectedFile) {
        const form = new FormData();
        form.append("image", state.selectedFile, state.selectedFile.name);
        response = await fetch("/api/predict", {
          method: "POST",
          body: form,
          headers: { Accept: "application/json" }
        });
      } else {
        response = await fetch("/api/predict", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json"
          },
          body: JSON.stringify({ image: state.cameraDataUrl })
        });
      }

      let payload = null;
      try {
        payload = await response.json();
      } catch (parseError) {
        throw new Error(
          "The server sent back an unexpected response. Please try again."
        );
      }
      if (!response.ok || !payload || payload.ok === false) {
        const requestError = new Error(
          (payload && (payload.message || payload.error)) ||
            "The request failed. Please try again."
        );
        requestError.hint = payload && payload.hint;
        throw requestError;
      }

      analysis.hidden = true;
      state.hasResult = true;
      errorBox.hidden = true;
      if (payload.prediction && payload.prediction.uncertain) {
        renderUncertainResult(payload);
      } else {
        renderConfidentResult(payload);
      }
      resultSection.hidden = false;
      updateActionState();
      window.requestAnimationFrame(function () {
        resultSection.focus({ preventScroll: true });
        resultSection.scrollIntoView({
          block: "start",
          behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
            ? "auto"
            : "smooth"
        });
      });
    } catch (error) {
      const message =
        error && error.name === "TypeError"
          ? "We couldn't reach the server. Check your connection and try again."
          : (error && error.message) ||
            "Something went wrong. Please try again.";
      showError("We couldn't classify that photo", message, error && error.hint);
      state.hasResult = false;
    } finally {
      setBusy(false);
    }
  }

  function cameraErrorMessage(error) {
    const name = error && error.name ? error.name : "";
    if (
      !window.isSecureContext &&
      window.location.hostname !== "localhost" &&
      window.location.hostname !== "127.0.0.1"
    ) {
      return {
        title: "Camera needs a secure connection",
        message:
          "Browsers only allow camera access on HTTPS or localhost. Upload a photo instead, or open the app over HTTPS."
      };
    }
    if (name === "NotAllowedError" || name === "PermissionDeniedError") {
      return {
        title: "Camera access was blocked",
        message:
          "Allow camera access in your browser's address-bar settings, then try again — or upload a photo instead."
      };
    }
    if (name === "NotFoundError" || name === "DevicesNotFoundError") {
      return {
        title: "No camera was found",
        message:
          "This device doesn't seem to have a camera available. You can upload a photo instead."
      };
    }
    if (name === "NotReadableError" || name === "TrackStartError") {
      return {
        title: "The camera is busy",
        message:
          "Another app may be using the camera. Close it and try again, or upload a photo instead."
      };
    }
    if (name === "OverconstrainedError") {
      return {
        title: "This camera isn't supported",
        message:
          "The camera couldn't provide a suitable video stream. Upload a photo instead."
      };
    }
    return {
      title: "Camera unavailable",
      message:
        "We couldn't start the camera. Check your browser permissions, or upload a photo instead."
    };
  }

  async function startCamera() {
    if (state.busy) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      const info = cameraErrorMessage({ name: "NotSupportedError" });
      showCameraError(info.title, info.message);
      return;
    }
    if (!window.isSecureContext && window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1") {
      const info = cameraErrorMessage({});
      showCameraError(info.title, info.message);
      return;
    }

    hideCameraStages();
    cameraStart.hidden = false;
    startCameraBtn.disabled = true;
    startCameraBtn.innerHTML =
      '<span class="btn-spinner" aria-hidden="true"></span> Starting camera...';
    cameraHint.textContent = "Waiting for camera permission...";

    try {
      stopStream();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1280 },
          height: { ideal: 960 }
        },
        audio: false
      });
      if (state.mode !== "camera") {
        stream.getTracks().forEach(function (track) {
          track.stop();
        });
        return;
      }
      state.stream = stream;
      cameraVideo.srcObject = stream;
      await cameraVideo.play().catch(function () {
        /* Autoplay may need a gesture; the user already clicked, usually fine. */
      });
      hideCameraStages();
      cameraLive.hidden = false;
    } catch (error) {
      stopStream();
      const info = cameraErrorMessage(error);
      showCameraError(info.title, info.message);
    } finally {
      startCameraBtn.disabled = false;
      startCameraBtn.innerHTML = icon("camera") + " Turn on camera";
      cameraHint.textContent = "Works on phones, tablets and laptops with a camera.";
    }
  }

  function showCameraError(title, message) {
    hideCameraStages();
    cameraPermission.hidden = false;
    cameraErrorTitle.textContent = title;
    cameraErrorText.textContent = message;
  }

  function capturePhoto() {
    if (state.busy) return;
    if (!cameraVideo.videoWidth || !cameraVideo.videoHeight) {
      showCameraError(
        "The camera isn't ready yet",
        "Wait a moment for the preview to appear, then tap Capture again."
      );
      return;
    }
    const maxSide = 1600;
    const scale = Math.min(
      1,
      maxSide / Math.max(cameraVideo.videoWidth, cameraVideo.videoHeight)
    );
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(cameraVideo.videoWidth * scale);
    canvas.height = Math.round(cameraVideo.videoHeight * scale);
    const context = canvas.getContext("2d");
    context.drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);
    state.capturedDataUrl = canvas.toDataURL("image/jpeg", 0.9);
    capturedImage.src = state.capturedDataUrl;
    capturedImage.alt = "Photo captured from your camera";
    stopStream();
    hideCameraStages();
    cameraReview.hidden = false;
    captureBtn.disabled = false;
  }

  function retakePhoto() {
    clearSelection({ keepTab: true });
    hideCameraStages();
    cameraStart.hidden = false;
    startCamera();
  }

  // Tabs -----------------------------------------------------------------
  tabs.forEach(function (tab) {
    tab.button.addEventListener("click", function () {
      activateTab(tab.mode);
    });
  });

  const tabList = document.querySelector(".source-tabs");
  tabList.addEventListener("keydown", function (event) {
    const currentIndex = tabs.findIndex(function (tab) {
      return tab.mode === state.mode;
    });
    let nextIndex = currentIndex;
    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      nextIndex = (currentIndex + 1) % tabs.length;
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      nextIndex = (currentIndex - 1 + tabs.length) % tabs.length;
    } else if (event.key === "Home") {
      nextIndex = 0;
    } else if (event.key === "End") {
      nextIndex = tabs.length - 1;
    } else {
      return;
    }
    event.preventDefault();
    activateTab(tabs[nextIndex].mode, true);
  });

  // Upload ---------------------------------------------------------------
  dropzone.addEventListener("click", function (event) {
    if (state.busy) return;
    if (event.target !== fileInput) fileInput.click();
  });
  dropzone.addEventListener("keydown", function (event) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      if (!state.busy) fileInput.click();
    }
  });
  fileInput.addEventListener("click", function (event) {
    event.stopPropagation();
  });
  fileInput.addEventListener("change", function () {
    const file = fileInput.files && fileInput.files[0];
    if (file) setSelectionFromFile(file);
    fileInput.value = "";
  });
  ["dragenter", "dragover"].forEach(function (eventName) {
    dropzone.addEventListener(eventName, function (event) {
      event.preventDefault();
      if (!state.busy) dropzone.classList.add("is-dragging");
    });
  });
  ["dragleave", "dragend"].forEach(function (eventName) {
    dropzone.addEventListener(eventName, function () {
      dropzone.classList.remove("is-dragging");
    });
  });
  dropzone.addEventListener("drop", function (event) {
    event.preventDefault();
    dropzone.classList.remove("is-dragging");
    if (state.busy) return;
    const file =
      event.dataTransfer &&
      event.dataTransfer.files &&
      event.dataTransfer.files[0];
    if (file) setSelectionFromFile(file);
  });
  window.addEventListener("dragover", function (event) {
    if (event.dataTransfer && Array.from(event.dataTransfer.types || []).includes("Files")) {
      event.preventDefault();
    }
  });
  window.addEventListener("drop", function (event) {
    if (event.dataTransfer && Array.from(event.dataTransfer.types || []).includes("Files")) {
      event.preventDefault();
    }
  });

  removeImageBtn.addEventListener("click", function () {
    clearSelection();
    dropzone.focus({ preventScroll: true });
  });

  // Camera ---------------------------------------------------------------
  startCameraBtn.addEventListener("click", startCamera);
  captureBtn.addEventListener("click", capturePhoto);
  retakeBtn.addEventListener("click", retakePhoto);
  retakeSelectedBtn.addEventListener("click", retakePhoto);
  useCaptureBtn.addEventListener("click", function () {
    if (state.busy || !state.capturedDataUrl) return;
    setSelectionFromCamera(state.capturedDataUrl);
    activateTab("camera");
  });
  cameraFallback.addEventListener("click", function () {
    stopStream();
    resetCameraView();
    activateTab("upload", true);
    dropzone.focus({ preventScroll: true });
  });

  classifyBtn.addEventListener("click", classify);
  retryButton.addEventListener("click", function () {
    if (!state.busy) classify();
  });

  // Privacy: stop the live camera when leaving the page or tab. -----------
  window.addEventListener("pagehide", stopStream);
  document.addEventListener("visibilitychange", function () {
    if (document.hidden && state.stream) {
      stopStream();
      hideCameraStages();
      cameraStart.hidden = false;
    }
  });

  updateActionState();
})();
