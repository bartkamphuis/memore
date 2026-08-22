'''The whole front end: one HTML string, inline CSS and JS, no build step, no CDN.

A Python string rather than a file next to it, so the page ships with the package whatever
the installer does and cannot go missing from a wheel. It is long, and that is the cost of
being genuinely self-contained.

The layout carries the argument. Chat on the left is the ordinary thing; the **store on the
right is the point**, because a superseded fact staying visible next to the value that
replaced it is what "supersede, never delete" looks like when it is true.
'''

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>memore</title>
<style>
  :root {
    --bg:#0e1116; --panel:#151a21; --line:#232a34; --ink:#d7dee8; --dim:#8b97a8;
    --accent:#5ac8fa; --good:#5ad19a; --warn:#e8b657; --bad:#e2686a; --old:#5d6773;
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  }
  * { box-sizing:border-box; }
  body {
    margin:0; background:var(--bg); color:var(--ink);
    font:14px/1.55 var(--mono); height:100vh; display:flex; flex-direction:column;
  }
  header {
    padding:10px 16px; border-bottom:1px solid var(--line); background:var(--panel);
    display:flex; gap:18px; align-items:baseline; flex-wrap:wrap;
  }
  header b { color:var(--accent); font-weight:600; letter-spacing:.5px; }
  header span { color:var(--dim); font-size:12px; }
  header span i { color:var(--ink); font-style:normal; }
  header button {
    background:transparent; color:var(--dim); border:1px solid var(--line);
    border-radius:5px; padding:4px 10px; font:inherit; font-size:12px; cursor:pointer;
  }
  header #reset:hover { color:var(--bad); border-color:var(--bad); }
  header .right { margin-left:auto; display:flex; gap:8px; align-items:center; }
  header select {
    background:var(--bg); color:var(--ink); border:1px solid var(--line); border-radius:5px;
    padding:4px 8px; font:inherit; font-size:12px; cursor:pointer; max-width:280px;
  }
  header select:focus { outline:none; border-color:var(--accent); }

  /* The injected block, hoisted out of the per-turn trace into its own strip. It is the
     one piece of state that persists ACROSS turns now (demo/linger.py), so rendering it
     inside a turn would have shown a per-turn artefact where a standing one lives. */
  #context {
    border-bottom:1px solid var(--line); background:#0c1015;
    padding:10px 16px; max-height:34vh; overflow-y:auto; flex:0 0 auto;
  }
  #context .head { display:flex; gap:12px; align-items:baseline; }
  #context .head h2 { margin:0; text-transform:none; }
  #context .head .status { color:var(--dim); font-size:12px; }
  #context .head .status b { color:var(--good); font-weight:600; }
  #context .head button {
    margin-left:auto; background:transparent; color:var(--dim); border:1px solid var(--line);
    border-radius:5px; padding:2px 8px; font:inherit; font-size:11px; cursor:pointer;
  }
  /* Two independent reasons the block is not on screen, so two classes: there is no
     block (a reload, or a turn that injected nothing) and the reader collapsed it. One
     inline style toggled by the button would have fought the first. */
  #context.empty-ctx .ctx-rows { display:none; }
  #context.empty-ctx pre.block, #context.no-block pre.block,
  #context.hidden-block pre.block { display:none; }
  .ctx { display:flex; gap:10px; align-items:baseline; font-size:12px; padding:3px 0; }
  .ctx .w { color:var(--accent); width:52px; flex:0 0 auto; text-align:right; }
  .ctx.carried .w { color:var(--warn); }
  .ctx .bar { flex:0 0 64px; height:4px; background:#1b222c; border-radius:2px; overflow:hidden; }
  .ctx .bar i { display:block; height:100%; background:var(--accent); }
  .ctx.carried .bar i { background:var(--warn); }
  .ctx .txt { flex:1; color:#e6edf6; }
  .ctx.sup .txt { color:var(--old); text-decoration:line-through; }
  .ctx .why { color:#5f6b7a; font-size:11px; flex:0 0 auto; }
  main { flex:1; display:grid; grid-template-columns:1fr 400px; min-height:0; }
  @media (max-width:900px) { main { grid-template-columns:1fr; } #store { display:none; } }
  #chat { display:flex; flex-direction:column; min-height:0; border-right:1px solid var(--line); }
  #log { flex:1; overflow-y:auto; padding:16px; }
  #store { overflow-y:auto; padding:16px; background:#11151b; }
  form { display:flex; gap:8px; padding:12px 16px; border-top:1px solid var(--line); background:var(--panel); }
  input[type=text] {
    flex:1; background:var(--bg); border:1px solid var(--line); border-radius:6px;
    color:var(--ink); padding:9px 12px; font:inherit; outline:none;
  }
  input[type=text]:focus { border-color:var(--accent); }
  form button {
    background:var(--accent); color:#08131a; border:0; border-radius:6px;
    padding:9px 18px; font:inherit; font-weight:600; cursor:pointer;
  }
  form button:disabled { opacity:.4; cursor:default; }

  .msg { margin-bottom:14px; }
  .msg .who { color:var(--dim); font-size:11px; text-transform:uppercase; letter-spacing:.7px; }
  .msg.user .body { color:#eef3f9; }
  .msg.bot  .body { color:var(--ink); }
  .body { white-space:pre-wrap; word-break:break-word; }

  /* `.body` is pre-wrap so the model's own line breaks survive. The trace is generated
     HTML, and inside a pre-wrap parent every newline in that markup became a blank line --
     the trace rendered three screens tall. It opts back out. */
  .trace { margin:6px 0 10px; border-left:2px solid var(--line); padding-left:12px;
           white-space:normal; }
  .trace .row { color:var(--dim); font-size:12px; line-height:1.5; }
  .trace .row + .row { margin-top:4px; }
  .reply { margin-top:6px; white-space:pre-wrap; }
  .reply.streaming::after {
    content:"▋"; color:var(--accent); animation:blink 1s steps(2,start) infinite;
  }
  @keyframes blink { to { visibility:hidden; } }
  .t { color:#55606f; }            /* the elapsed-ms stamp on each stage */
  .waiting { color:var(--old); }
  .trace .row .k { color:#6f7d90; }
  .pill {
    display:inline-block; padding:0 6px; margin:0 4px; border-radius:4px; font-size:11px;
    border:1px solid currentColor;
  }
  /* The gap around a pill is CSS, not markup whitespace: `flat()` legitimately collapses
     newlines between tags, and a word space that only exists because of source indentation
     is a word space that disappears the first time the markup is reformatted. */
  .open { color:var(--good); } .shut { color:var(--old); }
  .NEW { color:var(--good); } .CONTRADICTION { color:var(--warn); }
  .DUPLICATE { color:var(--old); } .REFINEMENT { color:var(--accent); }
  pre.block {
    margin:6px 0; padding:8px 10px; background:#0a0d12; border:1px solid var(--line);
    border-radius:6px; color:#9fb3c8; font-size:12px; white-space:pre-wrap; overflow-x:auto;
  }
  .hit { font-size:12px; color:var(--dim); white-space:normal; }
  .hit .s { color:var(--accent); }
  .hit.sup { color:var(--old); text-decoration:line-through; }

  h2 { font-size:11px; text-transform:uppercase; letter-spacing:1px; color:var(--dim);
       margin:0 0 10px; font-weight:600; }
  .subject { margin-bottom:14px; }
  .subject .name { color:var(--accent); font-size:12px; }
  .fact { font-size:12px; padding:2px 0 2px 10px; border-left:2px solid var(--line);
          color:#e6edf6; }
  .fact.sup { color:var(--old); border-left-color:#2a323d; }
  .fact.sup .t { text-decoration:line-through; }
  .fact .meta { color:#5f6b7a; font-size:11px; }
  .empty { color:var(--old); font-size:12px; }
  .pf { font-size:12px; margin-bottom:6px; }
  .pf.bad { color:var(--bad); } .pf.ok { color:var(--dim); }
  #banner { padding:10px 16px; background:#2a1416; border-bottom:1px solid var(--bad); display:none; }
</style>
</head>
<body>
<header>
  <b>memore</b>
  <span>graph <i id="h-graph">…</i></span>
  <span>gate <i id="h-gate">…</i></span>
  <span>embed <i id="h-embed">…</i></span>
  <span>model <i id="h-model">…</i></span>
  <span>linger <i id="h-linger">…</i></span>
  <div class="right">
    <!-- Sessions, not graphs: see the app module docstring. `n live / n` is what
         distinguishes two of them -- the name alone does not say which one has the
         conversation you were demonstrating. -->
    <select id="sessions" title="the session recall is scoped to -- switching clears the conversation and the carried context"></select>
    <button id="reset" title="clear_session() -- drops every fact in this session">reset store</button>
  </div>
</header>
<div id="banner"></div>
<section id="context" class="empty-ctx">
  <div class="head">
    <h2>&lt;recalled_context&gt;</h2>
    <span class="status" id="ctx-status">nothing injected yet</span>
    <button id="ctx-toggle">hide block</button>
  </div>
  <div class="ctx-rows" id="ctx-rows"></div>
  <pre class="block" id="ctx-block"></pre>
</section>
<main>
  <section id="chat">
    <div id="log">
      <div class="msg bot"><div class="who">memore</div><div class="body">Tell me something about yourself, then ask about it a few turns later. Watch the right-hand pane: when you contradict yourself, the old fact is <b>superseded, not deleted</b> — it stays, struck through, next to the value that replaced it.</div></div>
    </div>
    <form id="f" autocomplete="off">
      <input type="text" id="m" placeholder="say something…" autofocus>
      <button id="send">send</button>
    </form>
  </section>
  <aside id="store"><h2>store</h2><div id="facts" class="empty">empty</div></aside>
</main>

<script>
const $ = (id) => document.getElementById(id);
const esc = (s) => (s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

function renderStore(groups) {
  const box = $("facts");
  if (!groups || !groups.length) { box.className = "empty"; box.textContent = "empty"; return; }
  box.className = "";
  box.innerHTML = groups.map(g => `
    <div class="subject">
      <div class="name">${esc(g.subject)}</div>
      ${g.facts.map(f => `
        <div class="fact${f.superseded ? " sup" : ""}">
          <span class="t">${esc(f.fact)}</span>
          <div class="meta">#${f.ordinal} · ${esc(f.attribute || "—")} · ${esc(f.type)}${
            f.superseded ? " · SUPERSEDED" : ""}${
            f.occurs_at ? " · " + esc(f.occurs_at) + (f.recurring ? " (recurring)" : "") : ""}</div>
        </div>`).join("")}
    </div>`).join("");
}

// Collapses the newlines this function's template literals introduce. Belt and braces
// next to `.trace { white-space: normal }` -- the trace is markup, not text, and it should
// not matter which parent it is dropped into.
//
// Newline-bearing gaps only. `/>\s+</` also ate the single space in `</span> <span>`,
// which is a real word gap: it rendered `recall[gate OPEN]` and `P2[NEW]`.
const flat = (s) => s.replace(/>[^\S\n]*\n\s*</g, "><").trim();
const ms = (v) => v >= 1000 ? (v / 1000).toFixed(1) + "s" : Math.round(v) + "ms";

// The block itself is NOT rendered per turn any more -- it lives in the #context strip,
// because it now persists across turns and a per-turn copy would suggest otherwise. The
// gate decision and the hits stay here: those are per-turn facts.
function recallRows(r) {
  const hits = r.hits.map(h =>
    `<div class="hit${h.superseded ? " sup" : ""}">· <span class="s">${h.similarity.toFixed(3)}</span> ${esc(h.fact)}${h.past ? " <i>[past]</i>" : ""}</div>`
  ).join("");
  return flat(`<div class="row"><span class="k">recall</span>
      <span class="pill ${r.gate_open ? "open" : "shut"}">gate ${r.gate_open ? "OPEN" : "SHUT"}</span>
      ${r.hits.length} fact(s) · ${r.latency_ms}ms <span class="t">@${ms(r.ms)}</span></div>`)
    + hits;
}

// `gate SHUT` above and a block still going to the model is not a contradiction -- it is
// this row. It says so in words rather than leaving the two events to be reconciled.
function lingerRows(l) {
  if (!l.carried.length) return "";
  const rows = l.carried.map(c =>
    `<div class="hit${c.superseded ? " sup" : ""}">· <span class="s">${c.weight.toFixed(3)}</span> ${esc(c.fact)} <i>(${c.age_turns} turn${c.age_turns === 1 ? "" : "s"} ago, seen ${c.seen}×)</i></div>`
  ).join("");
  return flat(`<div class="row"><span class="k">linger</span>
      <span class="pill ${l.rescued ? "open" : "shut"}">${l.rescued ? "CARRIED" : "carried"}</span>
      ${l.carried.length} fact(s) held over${l.rescued ? " — the gate shut and the context survived" : ""}
      <span class="t">@${ms(l.ms)}</span></div>`) + rows;
}

function writeRows(w) {
  if (!w.outcomes.length) {
    return flat(`<div class="row"><span class="k">write</span> nothing stored — transient turn
      (P1's salience gate returned nothing) <span class="t">@${ms(w.ms)}</span></div>`);
  }
  return w.outcomes.map(o => flat(`
    <div class="row">
      <span class="k">P2</span> <span class="pill ${o.case}">${o.case}</span>
      ordinal ${o.ordinal}${o.superseded_fact_id ? " · superseded the incumbent" : ""}
      <span class="t">@${ms(w.ms)}</span><br>
      <span class="k">P1</span> ${esc(o.fact)}<br>
      <span class="k">  </span> subject=<i>${esc(o.subject)}</i> attribute=<i>${esc(o.attribute || "—")}</i>
      single_valued=${o.single_valued} conf=${o.confidence}
    </div>`)).join("");
}

// ---- the #context strip -----------------------------------------------------------
// One live view of what is currently in `<recalled_context>`, standing apart from the
// chat because it is no longer a property of a single turn. Two kinds of row, and the
// distinction is the whole point: `fresh` is what the gate admitted THIS turn, scored
// against this question; `carried` is what the frecency cache is still holding from an
// earlier one, at its decayed weight and with its age in turns.
const CTX = {fresh: [], carried: [], block: null, gate_open: false, turn: 0};

function ctxRow(kind, weight, text, why, superseded, past) {
  const pct = Math.max(2, Math.min(100, Math.round(weight * 100)));
  return flat(`<div class="ctx ${kind}${superseded ? " sup" : ""}">
    <span class="w">${weight.toFixed(3)}</span>
    <span class="bar"><i style="width:${pct}%"></i></span>
    <span class="txt">${esc(text)}${past ? " [past]" : ""}${superseded ? " [superseded]" : ""}</span>
    <span class="why">${esc(why)}</span>
  </div>`);
}

function renderContext() {
  const box = $("context");
  const total = CTX.fresh.length + CTX.carried.length;
  box.classList.toggle("empty-ctx", total === 0);
  if (!total) {
    $("ctx-status").textContent = CTX.turn
      ? "empty — the gate shut and nothing was still being carried"
      : "nothing injected yet";
    box.classList.add("empty-ctx");
    $("ctx-rows").innerHTML = ""; $("ctx-block").textContent = "";
    return;
  }
  // `turn === 0` is a fresh page against a server mid-conversation: nothing was recalled
  // in THIS browser session, so claiming a count for it would be wrong.
  $("ctx-status").innerHTML = CTX.turn === 0
    ? `<b>${total} fact(s)</b> still carried, and headed for the next turn's prompt`
    : `<b>${total} fact(s)</b> in the prompt · ${CTX.fresh.length} recalled this turn · ` +
      `${CTX.carried.length} carried` +
      (CTX.carried.length && !CTX.gate_open ? " · <b>gate shut, context survived</b>" : "");
  box.classList.toggle("no-block", !CTX.block);
  $("ctx-rows").innerHTML =
    CTX.fresh.map(h => ctxRow("fresh", h.similarity, h.fact, "recalled now", h.superseded, h.past)).join("")
    + CTX.carried.map(c => ctxRow("carried", c.weight, c.fact,
        `${c.age_turns} turn${c.age_turns === 1 ? "" : "s"} ago · seen ${c.seen}× · from ${c.strength.toFixed(2)}`,
        c.superseded, c.past)).join("");
  $("ctx-block").textContent = CTX.block || "";
}

function renderSessions(list, current) {
  const sel = $("sessions");
  sel.innerHTML = list.map(s =>
    `<option value="${esc(s.session)}"${s.session === current ? " selected" : ""}>${esc(s.session)} — ${s.live}/${s.facts} live</option>`
  ).join("") + `<option value="__new__">+ new session…</option>`;
}

function add(who, cls, body) {
  const el = document.createElement("div");
  el.className = "msg " + cls;
  el.innerHTML = `<div class="who">${who}</div><div class="body">${body}</div>`;
  $("log").appendChild(el);
  $("log").scrollTop = $("log").scrollHeight;
  return el;
}

// `EventSource` is GET-only and the turn's body is a message, so the stream is read off a
// POST with a reader instead. The buffer matters: a `data:` line can be split across two
// reads, and appending straight through would corrupt or drop that frame.
async function* sse(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const {value, done} = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, {stream: true});
    let split;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      let event = "message", data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) yield {event, data: JSON.parse(data)};
    }
  }
}

async function boot() {
  const s = await (await fetch("/api/state")).json();
  $("h-graph").textContent = s.graph;
  $("h-gate").textContent = s.gate + " k=" + s.k;
  $("h-embed").textContent = s.embedder;
  $("h-model").textContent = s.model;
  $("h-linger").textContent = s.linger.enabled
    ? `half-life ${s.linger.half_life_turns} turns ≥ ${s.linger.floor}`
    : "off";
  renderSessions(s.sessions, s.session);
  renderStore(s.store);
  // A reload loses the turn log but NOT the carried set -- that lives on the server. A
  // strip saying "nothing injected yet" while the next turn would inject three facts is
  // the panel lying about the state it exists to show.
  CTX.fresh = []; CTX.carried = s.carried || []; CTX.block = null; CTX.turn = 0;
  renderContext();
  if (!s.ok) {
    // The three failures that otherwise look like an empty store. Naming them here is
    // most of the point of the preflight check.
    $("banner").style.display = "block";
    $("banner").innerHTML = s.preflight.map(c =>
      `<div class="pf ${c.ok ? "ok" : "bad"}">${c.ok ? "ok  " : "FAIL"} ${esc(c.name)} — ${esc(c.detail)}</div>`
    ).join("");
  }
}

$("f").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $("m").value.trim();
  if (!text) return;
  $("m").value = ""; $("send").disabled = true;
  add("you", "user", esc(text));

  // The three stages get their elements up front and fill in as events land. That is the
  // entire point of streaming this: the gate result is on screen in ~80ms, the reply
  // arrives over seconds, and the write path lands after the reply is finished.
  const bot = add("memore", "bot",
    `<div class="trace"><div class="stage-recall waiting">recall…</div>
     <div class="stage-linger"></div>
     <div class="stage-write"></div></div><div class="reply"></div>`);
  // Both lanes write into elements that already exist, so an event landing in either order
  // updates its own slot instead of appending and reflowing the other.
  const $recall = bot.querySelector(".stage-recall");
  const $linger = bot.querySelector(".stage-linger");
  const $write  = bot.querySelector(".stage-write");
  const $reply  = bot.querySelector(".reply");
  const follow = () => { $("log").scrollTop = $("log").scrollHeight; };

  try {
    const res = await fetch("/api/turn", {
      method: "POST", headers: {"content-type": "application/json"},
      body: JSON.stringify({message: text}),
    });
    if (!res.ok || !res.body) {
      $reply.textContent = (await res.json().catch(() => ({}))).error || ("HTTP " + res.status);
      return;
    }
    for await (const {event, data} of sse(res)) {
      if (event === "recall") {
        $recall.className = "stage-recall"; $recall.innerHTML = recallRows(data);
        // The strip is rebuilt from this turn's recall and then completed by `linger`,
        // rather than accumulated: the carried set is the server's to decide, and a
        // client that added to its own copy would drift from the block actually sent.
        CTX.turn += 1; CTX.fresh = data.hits; CTX.carried = []; CTX.gate_open = data.gate_open;
        CTX.block = data.block; renderContext();
      }
      else if (event === "linger") {
        CTX.carried = data.carried; CTX.block = data.block; renderContext();
        $linger.innerHTML = lingerRows(data);
      }
      else if (event === "reply_start") { $reply.className = "reply streaming"; }
      // The write lane is launched here, beside the reply rather than after it -- so the
      // placeholder appears while the first token is still being generated, and the result
      // usually lands mid-stream. That overlap is the thing worth seeing.
      else if (event === "write_start") {
        $write.innerHTML = `<div class="row waiting">write… <span class="t">@${ms(data.ms)} (in parallel)</span></div>`;
      }
      else if (event === "delta") { $reply.textContent += data.text; }
      else if (event === "reply_end") { $reply.className = "reply"; }
      else if (event === "write") { $write.innerHTML = writeRows(data); }
      else if (event === "write_error") {
        $write.innerHTML = `<div class="row"><span class="k">write</span> failed — ${esc(data.error)}</div>`;
      }
      else if (event === "store") { renderStore(data.store); }
      follow();
    }
  } catch (err) {
    $reply.textContent += " [stream failed: " + err + "]";
  } finally {
    $reply.className = "reply";
    $("send").disabled = false; $("m").focus();
  }
});

$("reset").addEventListener("click", async () => {
  if (!confirm("Drop every fact in this session?")) return;
  const r = await (await fetch("/api/reset", {method: "POST"})).json();
  renderStore([]);
  renderSessions(r.sessions, $("sessions").value);
  CTX.fresh = []; CTX.carried = []; CTX.block = null; CTX.turn = 0; renderContext();
  $("log").innerHTML = "";
  add("memore", "bot", "Store cleared. The conversation history and the carried context went with it.");
});

// Switching is a whole-app move: the store reads, the conversation and the carried context
// all belong to the session, and the server clears the last two. A name that is not in the
// list yet is simply a session with no facts -- there is nothing to create.
$("sessions").addEventListener("change", async (e) => {
  const sel = e.target;
  let name = sel.value;
  if (name === "__new__") {
    name = (prompt("Session name — an unused name starts an empty store:") || "").trim();
    if (!name) { await boot(); return; }
  }
  const r = await (await fetch("/api/session", {
    method: "POST", headers: {"content-type": "application/json"},
    body: JSON.stringify({session: name}),
  })).json();
  if (r.error) { await boot(); return; }
  renderSessions(r.sessions, r.session);
  renderStore(r.store);
  CTX.fresh = []; CTX.carried = []; CTX.block = null; CTX.turn = 0; renderContext();
  $("log").innerHTML = "";
  add("memore", "bot", `Now talking into session <b>${esc(r.session)}</b>. Recall is scoped to it, so nothing from the previous one is visible here.`);
});

$("ctx-toggle").addEventListener("click", () => {
  const hidden = $("context").classList.toggle("hidden-block");
  $("ctx-toggle").textContent = hidden ? "show block" : "hide block";
});

boot();
</script>
</body>
</html>
"""
