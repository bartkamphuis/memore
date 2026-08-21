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
    margin-left:auto; background:transparent; color:var(--dim); border:1px solid var(--line);
    border-radius:5px; padding:4px 10px; font:inherit; font-size:12px; cursor:pointer;
  }
  header button:hover { color:var(--bad); border-color:var(--bad); }
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
  <span>session <i id="h-session">…</i></span>
  <span>graph <i id="h-graph">…</i></span>
  <span>gate <i id="h-gate">…</i></span>
  <span>embed <i id="h-embed">…</i></span>
  <span>model <i id="h-model">…</i></span>
  <button id="reset" title="clear_session() -- drops every fact in this session">reset store</button>
</header>
<div id="banner"></div>
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

function recallRows(r) {
  const hits = r.hits.map(h =>
    `<div class="hit${h.superseded ? " sup" : ""}">· <span class="s">${h.similarity.toFixed(3)}</span> ${esc(h.fact)}${h.past ? " <i>[past]</i>" : ""}</div>`
  ).join("");
  return flat(`<div class="row"><span class="k">recall</span>
      <span class="pill ${r.gate_open ? "open" : "shut"}">gate ${r.gate_open ? "OPEN" : "SHUT"}</span>
      ${r.hits.length} fact(s) · ${r.latency_ms}ms <span class="t">@${ms(r.ms)}</span></div>`)
    + hits
    + (r.block ? `<pre class="block">${esc(r.block)}</pre>` : "");
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
  $("h-session").textContent = s.session;
  $("h-graph").textContent = s.graph;
  $("h-gate").textContent = s.gate + " k=" + s.k;
  $("h-embed").textContent = s.embedder;
  $("h-model").textContent = s.model;
  renderStore(s.store);
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
     <div class="stage-write"></div></div><div class="reply"></div>`);
  const $recall = bot.querySelector(".stage-recall");
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
      if (event === "recall") { $recall.className = "stage-recall"; $recall.innerHTML = recallRows(data); }
      else if (event === "reply_start") { $reply.className = "reply streaming"; }
      else if (event === "delta") { $reply.textContent += data.text; }
      else if (event === "reply_end") {
        $reply.className = "reply";
        $write.innerHTML = `<div class="row waiting">write… <span class="t">@${ms(data.ms)}</span></div>`;
      }
      else if (event === "write") { $write.innerHTML = writeRows(data); }
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
  await fetch("/api/reset", {method: "POST"});
  renderStore([]);
  $("log").innerHTML = "";
  add("memore", "bot", "Store cleared. The conversation history went with it.");
});

boot();
</script>
</body>
</html>
"""
