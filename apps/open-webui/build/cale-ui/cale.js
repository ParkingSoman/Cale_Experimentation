/* The "Get Cale" card for Open WebUI. Shows until Cale is installed and connected, then goes away.
   It only talks to this app's own backend (/api/cale/*), never to Cale directly. */
(function () {
  var shown = false, card, bar, title, text, btn, note, badge;

  function build() {
    var css = document.createElement("style");
    css.textContent =
      "#cale-ov{position:fixed;inset:0;z-index:99999;display:flex;align-items:center;justify-content:center;" +
      "background:rgba(10,10,12,.72);backdrop-filter:blur(6px);font-family:system-ui,-apple-system,sans-serif}" +
      "#cale-card{width:min(440px,calc(100vw - 32px));padding:28px;border-radius:16px;background:#1b1b1f;color:#ececf0;" +
      "box-shadow:0 20px 60px rgba(0,0,0,.5);border:1px solid #2e2e35}" +
      "#cale-card h2{margin:0 0 8px;font-size:20px;font-weight:600}" +
      "#cale-card p{margin:0 0 18px;font-size:14px;line-height:1.5;color:#b4b4bd}" +
      "#cale-btn{width:100%;padding:12px;border:0;border-radius:10px;background:#e8e8ee;color:#111;font-size:15px;" +
      "font-weight:600;cursor:pointer}#cale-btn:disabled{opacity:.5;cursor:default}" +
      "#cale-barwrap{height:8px;border-radius:4px;background:#2e2e35;overflow:hidden;margin:0 0 10px}" +
      "#cale-bar{height:100%;width:0;background:#7cc4a0;transition:width .3s}" +
      "#cale-note{margin:10px 0 0;font-size:12px;color:#8a8a94;min-height:16px}";
    document.head.appendChild(css);
    var ov = document.createElement("div");
    ov.id = "cale-ov";
    ov.innerHTML =
      '<div id="cale-card"><h2 id="cale-title"></h2><p id="cale-text"></p>' +
      '<div id="cale-barwrap" hidden><div id="cale-bar"></div></div>' +
      '<button id="cale-btn"></button><div id="cale-note"></div></div>';
    document.body.appendChild(ov);
    card = ov; bar = ov.querySelector("#cale-bar");
    title = ov.querySelector("#cale-title"); text = ov.querySelector("#cale-text");
    btn = ov.querySelector("#cale-btn"); note = ov.querySelector("#cale-note");
    shown = true;
  }

  function gb(n) { return (n / 1e9).toFixed(1); }

  /* A small badge that always says what Cale chose for this app's models, on this computer. */
  function showBadge(models) {
    if (!badge) {
      badge = document.createElement("div");
      badge.id = "cale-badge";
      badge.style.cssText = "position:fixed;left:12px;bottom:10px;z-index:9999;padding:6px 10px;border-radius:8px;" +
        "background:rgba(27,27,31,.88);color:#b4b4bd;font:12px system-ui,-apple-system,sans-serif;pointer-events:none";
      document.body.appendChild(badge);
    }
    var parts = (models || []).map(function (m) {
      var l = m.last;
      if (!l) return m.id + ": ready";
      var met = l.goal_met === null || l.goal_met === undefined ? "" : (l.goal_met ? ", goal met" : ", goal not met");
      return m.id + ": " + l.size_pct + "% size" + met + (l.seconds ? ", " + l.seconds.toFixed(1) + " s" : "");
    });
    badge.textContent = "Cale on this computer \u00b7 " + (parts.length ? parts.join(" \u00b7 ") : "starting");
  }

  function render(s) {
    if (s.state === "ready") {
      if (shown) { location.reload(); }
      showBadge(s.models);
      return;
    }
    if (!shown) build();
    var wrap = card.querySelector("#cale-barwrap");
    wrap.hidden = s.state !== "installing";
    btn.hidden = s.state === "installing";
    btn.disabled = false;
    note.textContent = "";
    if (s.state === "missing") {
      title.textContent = "Use AI that runs on this computer";
      text.textContent = "Cale keeps one AI model on your computer. You download it once. " +
        "Every app that uses Cale shares it, so the next app needs no download.";
      btn.textContent = "Get Cale (about " + (s.download_gb || 5) + " GB)";
      btn.onclick = function () { btn.disabled = true; fetch("/api/cale/install", { method: "POST" }); };
    } else if (s.state === "starting") {
      title.textContent = "Starting Cale";
      text.textContent = "Cale is installed. It is starting now. This takes a few seconds.";
      btn.hidden = true;
    } else if (s.state === "installing") {
      var i = s.install, label = "Setting up";
      if (i.stage === "python") label = "Preparing Cale";
      if (i.stage === "model") label = i.total ? "Downloading the model: " + gb(i.done) + " of " + gb(i.total) + " GB" : "Downloading the model";
      if (i.stage === "convert") label = "Preparing the model for this computer";
      if (i.stage === "service") label = "Starting Cale";
      title.textContent = "Installing Cale";
      text.textContent = label;
      bar.style.width = Math.max(3, i.pct) + "%";
    } else if (s.state === "needs_connect") {
      title.textContent = "Cale is on this computer";
      text.textContent = "Let Open WebUI use it. Open WebUI says what it needs: answers in a good time, titles and suggestions fast. " +
        "Cale picks the size that fits this computer.";
      btn.textContent = "Connect Open WebUI to Cale";
      btn.onclick = function () {
        btn.disabled = true;
        note.textContent = "Look for the Allow question on your screen.";
        fetch("/api/cale/connect", { method: "POST" }).then(function (r) { return r.json(); }).then(function (j) {
          if (!j.ok) { note.textContent = j.error; btn.disabled = false; }
        });
      };
    } else if (s.state === "failed") {
      title.textContent = "The install stopped";
      text.textContent = s.install.error || "Something went wrong.";
      btn.textContent = "Try again";
      btn.onclick = function () { btn.disabled = true; fetch("/api/cale/install", { method: "POST" }); };
    }
  }

  function poll() {
    fetch("/api/cale/status").then(function (r) { return r.json(); }).then(render)
      .catch(function () {}).then(function () { setTimeout(poll, 1000); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", poll); else poll();
})();
