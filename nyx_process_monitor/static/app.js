"use strict";

(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const cards = new Map();
  const statusNames = {
    running: "En cours", complete: "Terminé", failed: "Échec",
    absent: "Processus absent", unknown: "État incertain", queued: "En attente",
    pending: "En attente", waiting: "En attente", not_started: "En attente",
    completed: "Terminé", error: "Échec", preparing: "Préparation",
  };
  const dateTime = new Intl.DateTimeFormat("fr-FR", {
    day: "2-digit", month: "2-digit", year: "numeric", hour: "2-digit",
    minute: "2-digit", second: "2-digit",
  });
  const timeOnly = new Intl.DateTimeFormat("fr-FR", {
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
  const number = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 1 });
  let lastState = null;
  let lastError = null;
  let fetching = false;
  let refreshSeconds = 1;
  let frozenAt = null;

  function str(value, fallback = "") {
    return value === null || value === undefined ? fallback : String(value);
  }
  function numeric(value) {
    return typeof value === "number" && Number.isFinite(value);
  }
  function setText(element, value) {
    const next = str(value);
    if (element.textContent !== next) element.textContent = next;
  }
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = str(text);
    return node;
  }
  function date(value, short = false) {
    if (!value) return "non renseignée";
    const parsed = new Date(value);
    return Number.isNaN(parsed.getTime()) ? "non renseignée" : (short ? timeOnly : dateTime).format(parsed);
  }
  function duration(value, clock = false) {
    const seconds = Math.max(0, Math.floor(numeric(value) ? value : 0));
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const tail = clock ? ` ${String(seconds % 60).padStart(2, "0")} s` : "";
    if (days) return `${days} j ${hours} h ${minutes} min${tail}`;
    if (hours) return `${hours} h ${String(minutes).padStart(2, "0")} min${tail}`;
    return `${minutes} min${clock ? tail : ""}`;
  }
  function active(job) {
    return typeof job.active === "boolean" ? job.active : Array.isArray(job.processes) && job.processes.length > 0;
  }
  function phase(value) {
    return str(value, "Calcul actif").replace(/kalman_replay/g, "Recalcul Kalman").replace(/baseline_controls/g, "Contrôles de référence").replace(/unknown/g, "Phase non instrumentée");
  }
  function etaFor(job, card) {
    if (job.eta && numeric(job.eta.remaining_seconds)) return job.eta;
    // Defensive compatibility only: explicit weak heuristic, never disguised as a measured ETA.
    if (card.eta && card.eta.method === "display_fallback") return card.eta;
    const elapsed = numeric(job.elapsed_seconds) ? job.elapsed_seconds : 0;
    const remaining = Math.max(300, Math.max(3600, 2 * elapsed) - elapsed);
    return { remaining_seconds: remaining, low_seconds: remaining / 4, high_seconds: remaining * 4,
      confidence: "very_low", as_of: lastState.updated_at, method: "display_fallback",
      basis: "Hypothèse de secours : durée totale supposée égale à 1 h ou au double du temps écoulé (le plus grand), avec au moins 5 min restantes ; confiance très faible." };
  }
  function updateClocks() {
    const now = frozenAt === null ? Date.now() : frozenAt;
    cards.forEach(card => {
      const job = card.job;
      if (!job) return;
      const observedAt = Date.parse(lastState.updated_at);
      const sinceSnapshot = Number.isFinite(observedAt) ? Math.max(0, (now - observedAt) / 1000) : 0;
      setText($(".elapsed", card.node), `${duration((job.elapsed_seconds || 0) + sinceSnapshot, true)} écoulées`);
      const eta = card.eta;
      const asOf = Date.parse(eta.as_of || lastState.updated_at);
      const age = Number.isFinite(asOf) ? Math.max(0, (now - asOf) / 1000) : 0;
      const remaining = Math.max(0, eta.remaining_seconds - age);
      const low = Math.max(0, (numeric(eta.low_seconds) ? eta.low_seconds : eta.remaining_seconds / 4) - age);
      const high = Math.max(low, (numeric(eta.high_seconds) ? eta.high_seconds : eta.remaining_seconds * 4) - age);
      setText($(".eta-value", card.node), `≈ ${duration(remaining, true)}`);
      setText($(".eta-range", card.node), `Fourchette : ${duration(low)} – ${duration(high)}`);
      const estimatedEnd = eta.estimated_end_at || new Date((Number.isFinite(asOf) ? asOf : now) + eta.remaining_seconds * 1000).toISOString();
      setText($(".eta-end", card.node), frozenAt !== null ? "Dernière estimation · compte à rebours figé" : remaining <= 0 ? "Échéance estimée dépassée · calcul encore actif" : `Fin estimée vers ${date(estimatedEnd, true)}`);
    });
  }
  function cpu(value) { return numeric(value) ? `${number.format(value)} %` : "—"; }
  function memory(value) {
    if (!numeric(value)) return "—";
    return value >= 1024 ? `${number.format(value / 1024)} Go` : `${number.format(value)} Mo`;
  }
  function status(value) { return str(value, "unknown").toLowerCase(); }
  function statusLabel(value) { return statusNames[status(value)] || str(value, "État incertain"); }
  function warnings(target, values) {
    const items = Array.isArray(values) ? values.filter(Boolean).map(String) : [];
    const signature = JSON.stringify(items);
    if (target.dataset.signature !== signature) {
      target.replaceChildren(...items.map(item => el("p", "", item)));
      target.dataset.signature = signature;
    }
    target.hidden = items.length === 0;
  }
  function reportURL(value) {
    // Only backend-issued, same-origin artifact links may leave a result card.
    if (typeof value !== "string" || !value.startsWith("/artifact?")) return null;
    try {
      const url = new URL(value, location.origin);
      if (url.origin !== location.origin || url.pathname !== "/artifact" || !url.searchParams.get("id")) return null;
      return url.pathname + url.search;
    } catch (_) { return null; }
  }
  function renderZones(target, zones) {
    const data = Array.isArray(zones) ? zones : [];
    const signature = JSON.stringify(data);
    if (target.dataset.signature === signature) return;
    // Keep focused report links intact on a normal refresh with unchanged data.
    const focusedZone = target.contains(document.activeElement) ? document.activeElement.dataset.zone : null;
    if (focusedZone) return;
    target.replaceChildren(...data.map(zone => {
      const node = el("span", "zone");
      node.dataset.status = status(zone.status);
      node.title = [str(zone.phase), zone.updated_at ? `Mis à jour le ${date(zone.updated_at)}` : ""].filter(Boolean).join(" · ");
      node.append(el("b", "", zone.zone), el("span", "", statusLabel(zone.status)));
      const href = reportURL(zone.report_url);
      if (href) {
        const link = el("a", "zone-report", "↗");
        link.href = href; link.target = "_blank"; link.rel = "noopener noreferrer";
        link.dataset.zone = str(zone.zone); link.setAttribute("aria-label", `Ouvrir le rapport ${str(zone.zone)}`);
        node.append(link);
      }
      return node;
    }));
    target.dataset.signature = signature;
  }

  function processOrder(processes) {
    const data = Array.isArray(processes) ? processes : [];
    const ids = new Set(data.map(item => item.pid));
    const seen = new Set();
    const ordered = [];
    function visit(item, depth) {
      if (seen.has(item.pid)) return;
      seen.add(item.pid);
      ordered.push({ item, depth });
      data.filter(child => child.ppid === item.pid).forEach(child => visit(child, depth + 1));
    }
    data.filter(item => !ids.has(item.ppid)).forEach(item => visit(item, 0));
    data.forEach(item => visit(item, 0));
    return ordered;
  }
  function renderProcesses(card, processes) {
    const target = $(".process-tree", card.node);
    const data = processOrder(processes);
    const signature = JSON.stringify(data);
    if (target.dataset.signature === signature) return;
    // Preserve command selection while the user copies a PID or command.
    const selection = window.getSelection();
    if (selection && !selection.isCollapsed && selection.anchorNode && target.contains(selection.anchorNode)) return;
    target.replaceChildren();
    if (!data.length) target.append(el("p", "process-empty", "Aucun processus actif observé pour ce calcul."));
    data.forEach(({ item, depth }) => {
      const row = el("div", "process-row");
      row.dataset.depth = String(depth);
      row.style.paddingLeft = `${12 + Math.min(depth, 5) * 16}px`;
      const info = el("div", "process-info");
      info.append(el("span", "process-pid", `${depth ? "↳ " : ""}PID ${str(item.pid, "?")}`));
      info.append(el("span", "process-role", `${str(item.role, "Python")} · parent ${str(item.ppid, "?")}`));
      info.append(el("span", "process-created", `Créé le ${date(item.created_at)}`));
      info.append(el("span", "process-resources", `CPU ${cpu(item.cpu_percent)} · RAM ${memory(item.memory_mb)}`));
      row.append(info, el("code", "process-command", str(item.command, "Commande non disponible")));
      target.append(row);
    });
    target.dataset.signature = signature;
  }
  function renderLog(card, force = false) {
    if (!card.job) return;
    const output = $(".log-output", card.node);
    const raw = card.log === "stderr" ? card.job.stderr_tail : card.job.stdout_tail;
    const next = str(raw) || "Aucune ligne disponible dans ce journal.";
    if (output.textContent !== next) {
      const top = output.scrollTop;
      const left = output.scrollLeft;
      const selection = window.getSelection();
      const selected = selection && !selection.isCollapsed && selection.anchorNode && output.contains(selection.anchorNode);
      if (!selected || force) {
        output.textContent = next;
        output.scrollTop = $(".follow-log", card.node).checked ? output.scrollHeight : top;
        output.scrollLeft = left;
      }
    }
    $(".stderr-hint", card.node).hidden = card.log !== "stderr";
    setText($(".log-updated", card.node), card.job.logs_updated_at ? `Dernière écriture : ${date(card.job.logs_updated_at)}` : "Date d’écriture non disponible");
  }
  function newCard(id) {
    const node = $("#job-template").content.firstElementChild.cloneNode(true);
    const card = { id, node, log: "stdout", job: null, eta: null };
    node.dataset.jobId = id;
    $(".job-details", node).addEventListener("toggle", event => {
      if (event.target.open) {
        renderProcesses(card, card.job && card.job.processes);
        renderLog(card, true);
        const output = $(".log-output", node);
        if ($(".follow-log", node).checked) output.scrollTop = output.scrollHeight;
      }
    });
    node.querySelectorAll("[data-log]").forEach(button => {
      button.addEventListener("click", () => {
        card.log = button.dataset.log;
        node.querySelectorAll("[data-log]").forEach(tab => tab.setAttribute("aria-pressed", String(tab === button)));
        renderLog(card, true);
      });
    });
    $(".follow-log", node).addEventListener("change", () => {
      const output = $(".log-output", node);
      if ($(".follow-log", node).checked) output.scrollTop = output.scrollHeight;
    });
    $(".log-output", node).addEventListener("wheel", event => {
      if (event.deltaY < 0) $(".follow-log", node).checked = false;
    }, { passive: true });
    $(".log-output", node).addEventListener("scroll", event => {
      const output = event.currentTarget;
      if (output.scrollHeight - output.scrollTop - output.clientHeight > 30) {
        $(".follow-log", node).checked = false;
      }
    }, { passive: true });
    $(".log-output", node).addEventListener("keydown", event => {
      if (["ArrowUp", "PageUp", "Home"].includes(event.key)) $(".follow-log", node).checked = false;
    });
    $("#jobs").append(node);
    cards.set(id, card);
    return card;
  }
  function updateCard(card, job) {
    card.job = job;
    card.eta = etaFor(job, card);
    const node = card.node;
    node.dataset.status = status(job.status);
    setText($(".job-kind", node), job.kind === "detected" ? "PROCESSUS DÉTECTÉ" : "EXPÉRIENCE SUIVIE");
    setText($(".job-title", node), str(job.title, job.id));
    setText($(".job-subtitle", node), job.subtitle);
    setText($(".status-text", node), "Actif");
    setText($(".phase-label", node), phase(job.phase));
    $(".elapsed", node).title = `Démarrage : ${date(job.started_at)}. Temps écoulé depuis le lancement.`;
    const confidence = ["medium", "low", "very_low"].includes(card.eta.confidence) ? card.eta.confidence : "very_low";
    $(".eta-confidence", node).dataset.confidence = confidence;
    setText($(".eta-confidence", node), `Confiance ${ { medium: "moyenne", low: "faible", very_low: "très faible" }[confidence] }`);
    setText($(".eta-basis", node), str(card.eta.basis, "Estimation indicative recalculée à partir des observations disponibles."));
    $(".eta-panel", node).title = `Méthode : ${str(card.eta.method, "heuristique")} · Estimation du ${date(card.eta.as_of || lastState.updated_at)}`;
    const progress = job.progress || {};
    const hasPercent = numeric(progress.percent);
    const percent = hasPercent ? Math.max(0, Math.min(100, progress.percent)) : null;
    const track = $(".progress-track", node);
    track.classList.toggle("indeterminate", !hasPercent);
    if (hasPercent) {
      track.setAttribute("aria-valuemin", "0"); track.setAttribute("aria-valuemax", "100");
      track.setAttribute("aria-valuenow", String(percent));
      $(".progress-fill", node).style.width = `${percent}%`;
    } else {
      track.removeAttribute("aria-valuenow"); track.removeAttribute("aria-valuemin"); track.removeAttribute("aria-valuemax");
      $(".progress-fill", node).style.removeProperty("width");
    }
    const label = str(progress.label) || (numeric(progress.completed) && numeric(progress.total) && progress.total > 0 ? `${number.format(progress.completed)} / ${number.format(progress.total)} unités terminées` : "Progression détaillée non disponible");
    track.setAttribute("aria-valuetext", label);
    setText($(".progress-label", node), label);
    setText($(".progress-value", node), hasPercent ? `${number.format(percent)} %` : "Lot en cours");
    setText($(".cpu-value", node), cpu(job.cpu_percent));
    setText($(".memory-value", node), memory(job.memory_mb));
    const count = Array.isArray(job.processes) ? job.processes.length : 0;
    setText($(".process-count", node), `${count} processus`);
    renderZones($(".zones", node), job.zones);
    warnings($(".job-warnings", node), job.warnings);
    if ($(".job-details", node).open) { renderProcesses(card, job.processes); renderLog(card); }
  }
  function updateEmptyState() {
    $("#empty-state").hidden = cards.size > 0;
    if (lastState) {
      setText($("#empty-title"), "Aucun calcul actif");
      setText($("#empty-text"), "Le suivi reste ouvert. Les prochains calculs apparaîtront automatiquement ; les calculs terminés ne sont pas affichés.");
    }
  }
  function renderState(state) {
    lastState = state;
    const jobs = state.jobs.filter(job => job && typeof job === "object" && active(job));
    const present = new Set();
    jobs.forEach((job, index) => {
      const id = str(job.id, `job-${index}`);
      present.add(id); updateCard(cards.get(id) || newCard(id), job);
    });
    cards.forEach((card, id) => { if (!present.has(id)) { card.node.remove(); cards.delete(id); } });
    setText($("#count-running"), jobs.length);
    const processes = new Map();
    jobs.forEach(job => (job.processes || []).forEach(process => processes.set(`${process.pid}|${process.created_at}`, process)));
    setText($("#process-total"), `${processes.size} processus`);
    const values = [...processes.values()];
    const cpuKnown = values.length > 0 && values.every(process => numeric(process.cpu_percent));
    const ramKnown = values.length > 0 && values.every(process => numeric(process.memory_mb));
    setText($("#total-cpu"), cpu(cpuKnown || !jobs.length ? values.reduce((sum, process) => sum + (numeric(process.cpu_percent) ? process.cpu_percent : 0), 0) : null));
    setText($("#total-memory"), memory(ramKnown || !jobs.length ? values.reduce((sum, process) => sum + (numeric(process.memory_mb) ? process.memory_mb : 0), 0) : null));
    warnings($("#global-warnings"), state.warnings);
    refreshSeconds = 1;
    setText($("#refresh-label"), "Actualisation continue · 1 s");
    setText($("#last-update"), `Dernière lecture : ${date(state.updated_at, true)}`);
    updateEmptyState(); updateConnection(); updateClocks();
  }
  function updateConnection() {
    const serverTime = lastState && Date.parse(lastState.updated_at);
    const stale = lastState && (!Number.isFinite(serverTime) || Date.now() - serverTime > 12000);
    const fresh = Boolean(lastState && !lastError && !stale);
    if (!fresh && frozenAt === null) frozenAt = Date.now();
    if (fresh) frozenAt = null;
    document.body.dataset.fresh = String(fresh);
    setText($("#active-scope"), fresh ? "Sans historique · lecture seule" : "Dernier relevé · activité non confirmée");
    cards.forEach(card => setText($(".status-text", card.node), fresh ? "Actif" : "Non confirmé"));
    const connection = $("#connection");
    connection.dataset.state = lastError ? "offline" : stale ? "stale" : lastState ? "online" : "loading";
    setText($("#connection-label"), lastError ? "Connexion interrompue" : stale ? "Lecture ancienne — données à confirmer" : lastState ? "Suivi local connecté" : "Connexion au suivi local…");
    const error = $("#connection-error");
    error.hidden = !lastError && !stale;
    if (lastError) setText(error, `${lastError} ${lastState ? "Dernier relevé conservé : présence des processus non confirmée, estimations figées." : "Impossible de lire l’état des calculs pour le moment."} Nouvelle tentative automatique.`);
    else if (stale) setText(error, "La dernière lecture du serveur est ancienne. Présence des processus non confirmée ; estimations figées. Le suivi continue de vérifier automatiquement.");
    if (lastError && !lastState) {
      setText($("#empty-title"), "En attente du serveur local");
      setText($("#empty-text"), "Aucune action n’est effectuée sur les calculs. La connexion sera réessayée automatiquement.");
    }
  }
  async function refresh() {
    if (fetching) return;
    fetching = true;
    const requestStarted = Date.now();
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {
      const response = await fetch("/api/state", { cache: "no-store", signal: controller.signal, credentials: "same-origin" });
      if (!response.ok) throw new Error(`Le serveur répond HTTP ${response.status}.`);
      const state = await response.json();
      if (!state || state.app !== "nyx-process-monitor" || !Array.isArray(state.jobs)) throw new Error("La réponse du serveur est invalide.");
      lastError = null; renderState(state);
    } catch (error) {
      lastError = error.name === "AbortError" ? "Le serveur local ne répond pas à temps." : (error instanceof TypeError ? "Le serveur local est injoignable." : str(error.message, "Lecture impossible."));
      updateConnection();
    } finally {
      clearTimeout(timeout); fetching = false;
      setTimeout(refresh, Math.max(100, refreshSeconds * 1000 - (Date.now() - requestStarted)));
    }
  }
  window.addEventListener("online", () => { updateConnection(); });
  setInterval(() => { updateConnection(); updateClocks(); }, 1000);
  refresh();
})();
