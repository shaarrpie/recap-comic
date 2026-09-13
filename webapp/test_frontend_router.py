# webapp/test_frontend_router.py — offline regression tests for the studio
# frontend core (webapp/static/index.html).
#
# Runs the REAL inline <script> in Node with a minimal DOM/localStorage/fetch
# shim: no browser, no server, fully deterministic. Covers the
# "Open Manual Crop does nothing" bug class reported from the Studio view:
#
#   1. store.queue round-trips through localStorage (save()/load() across a
#      simulated page reload)  — queue must NOT vanish on refresh.
#   2. store.settings.mode round-trips through localStorage.
#   3. nav()/hashchange semantics: identical hash -> NO event (the silent
#      no-op mode); same route + different session arg -> event fires.
#   4. render()'s catch converts a throwing view into the generic Error
#      panel (vs. vManualCrop's internal catches, which swallow errors into
#      a toast + early return and leave the previous view's DOM in place).
#   5. Static wiring contract for the manual-crop feature (skipped when the
#      feature is absent, e.g. on the QA branch baseline).
#
# Run: pytest webapp/test_frontend_router.py -q   (or pytest tests webapp -q)
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HTML_PATH = Path(__file__).resolve().parent / "static" / "index.html"


def _extract_script(html: str) -> str:
    # non-greedy: index.html has TWO script blocks (main + cinematic
    # addon); a greedy match would span the HTML between them
    m = re.search(r"<script>([\s\S]*?)</script>", html)
    assert m, "inline <script> block not found in index.html"
    return m.group(1)


HTML = HTML_PATH.read_text(encoding="utf-8")
SRC = _extract_script(HTML)
HAS_MANUAL_CROP = "vManualCrop" in SRC and "Open Manual Crop" in SRC
NODE = shutil.which("node")

requires_node = pytest.mark.skipif(NODE is None, reason="node runtime not available")
requires_manual = pytest.mark.skipif(
    not HAS_MANUAL_CROP, reason="manual-crop wiring absent in this checkout"
)

# ---------------------------------------------------------------- harness --
HARNESS = r"""
const fs = require("fs"), assert = require("assert");
const [, , htmlPath, scenario] = process.argv;
const html = fs.readFileSync(htmlPath, "utf8");
const m = html.match(/<script>([\s\S]*?)<\/script>/);
assert(m, "inline <script> not found");
const src = m[1];

/* ---- minimal browser shim (deterministic, offline) ---- */
const storage = {};            // localStorage backing, shared across boots
const listeners = {};         // event listeners (hashchange, ...)
const mkEl = () => ({
  innerHTML: "", textContent: "", value: "", checked: false,
  style: {}, dataset: {}, onclick: null, offsetWidth: 0, clientWidth: 100,
  classList: { add(){}, remove(){}, toggle(){} },
  append(){}, appendChild(){}, addEventListener(){},
  insertAdjacentHTML(){}, remove(){}, showModal(){}, close(){},
  focus(){}, closest(){ return null; }, querySelectorAll(){ return []; },
});
const elCache = new Map();
const documentShim = {
  querySelector(s){ if(!elCache.has(s)) elCache.set(s, mkEl()); return elCache.get(s); },
  querySelectorAll(){ return []; },
  createElement(){ return mkEl(); },
  body: mkEl(),
};
const loc = {
  _hash: "",
  get hash(){ return this._hash; },
  set hash(h){
    h = String(h);
    if(h === this._hash) return;         // real browsers: same hash -> no event
    this._hash = h;
    (listeners.hashchange || []).forEach(f => f({}));
  },
};
global.document = documentShim;
global.localStorage = {
  getItem: k => (k in storage ? storage[k] : null),
  setItem: (k, v) => { storage[k] = String(v); },
  removeItem: k => { delete storage[k]; },
};
global.location = loc;
global.addEventListener = (ev, f) => { (listeners[ev] = listeners[ev] || []).push(f); };
global.removeEventListener = () => {};
global.fetch = () => Promise.reject(new Error("offline test shim"));
global.Image = class { set src(_){ /* never loads */ } };
global.IntersectionObserver = class { observe(){} unobserve(){} disconnect(){} };
global.confirm = () => true;
global.navigator = { clipboard: { writeText: () => Promise.resolve() } };
global.Blob = class {};
global.URL = { createObjectURL: () => "blob:" };
global.requestAnimationFrame = f => setTimeout(f, 0);
global.cancelAnimationFrame = () => {};

const flush = () => new Promise(r => setTimeout(r, 20));

/* One "page load": fresh DOM cache, fresh hashchange listeners, then the
   real app script. The appended alias exposes the script's top-level
   bindings for assertions (they live inside the eval scope). */
function boot(){
  listeners.hashchange = [];
  elCache.clear();
  eval(src + "\n;globalThis.__APP={store,save,nav,render,parseHash,routes,viewEl};");
}

const out = o => { process.stdout.write("RESULT " + JSON.stringify(o)); };

(async () => {
  let res;
  try {
    if (scenario === "queue_roundtrip") {
      boot(); await flush();
      __APP.store.queue = [{session: "abc123def456", name: "strip.png", added: 1}];
      __APP.save();
      assert.ok("rc.studio.v1" in storage, "save() wrote the localStorage key");
      boot(); await flush();                     // simulated page refresh
      const q = __APP.store.queue;
      assert.strictEqual(q.length, 1, "queue survived reload");
      assert.strictEqual(q[0].session, "abc123def456");
      assert.strictEqual(q[0].name, "strip.png");
      res = { ok: true, queueLength: q.length };
    } else if (scenario === "mode_roundtrip") {
      boot(); await flush();
      __APP.store.settings.mode = "manual"; __APP.save();
      boot(); await flush();
      assert.strictEqual(__APP.store.settings.mode, "manual");
      __APP.store.settings.mode = "automation"; __APP.save();
      boot(); await flush();
      assert.strictEqual(__APP.store.settings.mode, "automation");
      res = { ok: true };
    } else if (scenario === "fresh_default") {
      boot(); await flush();
      res = { ok: true, mode: __APP.store.settings.mode ?? null };
    } else if (scenario === "legacy_settings") {
      /* payload written by a pre-manual-mode build: settings without `mode`,
         plus one queued strip */
      storage["rc.studio.v1"] = JSON.stringify({
        session: null, recent: [], names: {},
        queue: [{session: "abc123def456", name: "x.png", added: 1}],
        settings: { backend: "gemini", tts: "edge", voice: "en-US-AriaNeural",
                    style: "recap", api_key: "", endpoint: "", cf_account_id: "" },
      });
      boot(); await flush();
      res = { ok: true, mode: __APP.store.settings.mode ?? null,
              queueLength: __APP.store.queue.length };
    } else if (scenario === "hash_diff_arg") {
      boot(); await flush();
      loc.hash = "#/manual-crop/aaaaaaaaaaaa"; await flush();
      let events = 0;
      global.addEventListener("hashchange", () => events++);
      __APP.nav("manual-crop", "bbbbbbbbbbbb"); await flush();
      assert.strictEqual(events, 1, "hashchange fired for a different session arg");
      assert.strictEqual(loc.hash, "#/manual-crop/bbbbbbbbbbbb");
      res = { ok: true, events };
    } else if (scenario === "hash_same") {
      boot(); await flush();
      loc.hash = "#/manual-crop/aaaaaaaaaaaa"; await flush();
      let events = 0;
      global.addEventListener("hashchange", () => events++);
      __APP.nav("manual-crop", "aaaaaaaaaaaa"); await flush();
      assert.strictEqual(events, 0, "identical hash must not fire hashchange");
      res = { ok: true, events };
    } else if (scenario === "render_catch") {
      boot(); await flush();
      __APP.routes["manual-crop"] = async () => { throw new Error("boom no session"); };
      loc.hash = "#/manual-crop/cccccccccccc"; await flush();
      const htmlAfter = __APP.viewEl.innerHTML;
      assert.ok(htmlAfter.includes("boom no session"),
        "error panel missing; innerHTML: " + htmlAfter.slice(0, 200));
      assert.ok(htmlAfter.includes("Error"));
      res = { ok: true };
    } else {
      throw new Error("unknown scenario " + scenario);
    }
  } catch (e) {
    process.stdout.write("HARNESS_FAIL " + String((e && e.stack) || e));
    process.exit(1);
  }
  out(res);
  await new Promise(r => setImmediate(r));
  process.exit(0);
})();
"""


def _run_node(tmp_path: Path, scenario: str) -> dict:
    assert NODE, "node runtime not available"
    harness_file = tmp_path / "frontend_harness.js"
    harness_file.write_text(HARNESS, encoding="utf-8")
    proc = subprocess.run(
        [NODE, str(harness_file), str(HTML_PATH), scenario],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    combined = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        pytest.fail(f"node harness ({scenario}) failed:\n{combined}")
    lines = [ln for ln in (proc.stdout or "").splitlines()
             if ln.startswith("RESULT ")]
    assert lines, f"no RESULT line in harness output:\n{combined}"
    return json.loads(lines[0][len("RESULT "):])


# ------------------------------------------------------------------ tests --
@requires_node
def test_store_queue_roundtrip(tmp_path):
    """store.queue persists across save()/load() — a page refresh must not
    clear the queue (otherwise the Studio 'Open Manual Crop (N)' panel
    disappears and the button reports 'Queue is empty')."""
    res = _run_node(tmp_path, "queue_roundtrip")
    assert res["ok"] and res["queueLength"] == 1


@requires_node
def test_settings_mode_roundtrip(tmp_path):
    """store.settings.mode persists across save()/load() — otherwise manual
    mode silently resets to automation and the button label/branch changes."""
    assert _run_node(tmp_path, "mode_roundtrip")["ok"]


@requires_node
@requires_manual
def test_fresh_store_mode_default(tmp_path):
    """A fresh install defaults settings.mode to 'automation'."""
    res = _run_node(tmp_path, "fresh_default")
    assert res["mode"] == "automation"


@requires_node
def test_legacy_settings_payload_drops_mode_default(tmp_path):
    """CHARACTERIZATION (known defect, F3): a localStorage payload written
    before `mode` existed replaces the default settings object wholesale, so
    store.settings.mode is undefined after load — the Studio then renders
    the automation branch. Flip this pin when index.html merges nested
    defaults (proposed fix: default `mode` after the Object.assign load)."""
    res = _run_node(tmp_path, "legacy_settings")
    assert res["queueLength"] == 1
    assert res["mode"] is None, (
        "legacy payload now keeps a mode default — the fix landed; "
        "update this pin to assert 'automation'"
    )


@requires_node
def test_hashchange_fires_same_route_different_arg(tmp_path):
    """nav() to the same view with a DIFFERENT session arg changes the hash,
    so hashchange fires and render() runs — this is the path the
    'Open Manual Crop (N)' button depends on."""
    res = _run_node(tmp_path, "hash_diff_arg")
    assert res["events"] == 1


@requires_node
def test_hashchange_silent_on_identical_hash(tmp_path):
    """CHARACTERIZATION (latent defect, F2): assigning an identical hash
    fires NO hashchange, and nav() has no re-render fallback — clicking a
    button that navigates to the route you are already on does nothing.
    Flip this pin when nav() gains a same-hash render() fallback."""
    res = _run_node(tmp_path, "hash_same")
    assert res["events"] == 0, (
        "nav() now re-renders on identical hash — the fix landed; "
        "update this pin to expect 1 render"
    )


@requires_node
def test_render_error_panel_on_throwing_view(tmp_path):
    """render()'s catch surfaces a throwing view as the generic Error panel
    (hypothesis d). NOTE: vManualCrop's internal init catches never reach
    this catch — they toast and return, leaving the OLD view's DOM in place,
    which is the actual 'click does nothing' experience (see the static
    wiring contract below)."""
    assert _run_node(tmp_path, "render_catch")["ok"]


@requires_manual
def test_manual_crop_wiring_contract():
    """Static contract for the manual-crop feature. An unknown route in
    render() silently falls back to vStudio, so a missing route registration
    or handler binding is invisible at runtime — these pins fail loudly."""
    # route registered in the router map
    assert re.search(r'"manual-crop"\s*:\s*vManualCrop', SRC)
    # #runAll manual branch walks the queue into Manual Crop
    assert 'nav("manual-crop",snap[0].session)' in SRC
    assert 'toast("Queue is empty","warn");return' in SRC
    # button label template
    assert "Open Manual Crop (${store.queue.length})" in SRC
    # per-card "Crop ->" button wiring
    assert 'nav("manual-crop",b.dataset.qcrop)' in SRC
    # hashchange drives render
    assert 'addEventListener("hashchange",render)' in SRC
    # queue mutations persist (upload, remove-from-queue, dequeue-on-generate)
    assert SRC.count("store.queue=store.queue.filter(q=>q.session!==session);save();") >= 3
    # CHARACTERIZATION (known defect, F1): vManualCrop's init failures are
    # swallowed into toast + early return (old view's DOM stays visible).
    # Flip when the fix renders an explicit error state instead.
    body_start = SRC.index("async function vManualCrop")
    body_end = SRC.index("async function vNarrate")
    mc_body = SRC[body_start:body_end]
    assert mc_body.count('catch(e){toast(e.message,"err");return}') >= 2, (
        "vManualCrop init no longer swallows errors via toast+return — "
        "the fix landed; update this pin"
    )
