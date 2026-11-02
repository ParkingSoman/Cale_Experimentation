# serves: TY3
"""The zoom page (Proposition TY3 in use): one slider over the stored levels of the ResNet-18 merge rule.

Reads results_rz_zoom.json and cache/r18/zoom_builds.npz (the one-time computation). For each level k the page shows
the sealed-split numbers, and the three largest merged sets with the classes their re-fit readers favour most. The
chart plots KL against k for every build kind. Moving the slider only selects stored numbers (a lookup).
Writes zoom/zoom.html.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "..", "cache", "r18")
R = json.load(open(os.path.join(HERE, "results_rz_zoom.json")))
B = np.load(os.path.join(C, "zoom_builds.npz")); names = np.load(os.path.join(C, "head.npz"))["names"]
levels = []
for row in R["levels"]:
    k = row["k"]; M = B[f"k{k}_M"]; V = B[f"k{k}_V"]
    size = (M > 0).sum(0); big = np.argsort(size)[::-1][:3]
    sets = []
    for j in big:
        if size[j] < 2:
            continue
        v = V[:, j] - V[:, j].mean()
        sets.append({"size": int(size[j]), "classes": [str(names[c]) for c in np.argsort(v)[::-1][:3]]})
    lv = {"k": k, "bound": row["bound_train"], "merge": row["merge"], "refit": row["merge_refit"],
          "prune": row["prune_refit"], "random": row["random_refit"], "sets": sets}
    if "trained" in row:
        lv["trained"] = row["trained"]
    levels.append(lv)
data = json.dumps({"levels": levels, "n_sealed": R["n_sealed"], "top1_N": R["N_sealed_top1"]})

html = """<title>ResNet-18 Zoom</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,650&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono&display=swap">
<style>
/* Layout: a single instrument panel. Slider and readout on top, the curve below, the merged sets beside it. */
:root { --bg:#f6f7f4; --panel:#fcfcfb; --fg:#1d2420; --muted:#5d6a63; --rule:#d9dfda; --accent:#2f6f5e;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4;
  --display:"Source Serif 4", Georgia, serif; --body:"IBM Plex Sans", system-ui, sans-serif; --mono:"IBM Plex Mono", ui-monospace, monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#151a18; --panel:#1a1a19; --fg:#e4ebe6; --muted:#9aa8a0; --rule:#2e3833; --accent:#7cc2ad;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; color-scheme:dark; } }
:root[data-theme="dark"] { --bg:#151a18; --panel:#1a1a19; --fg:#e4ebe6; --muted:#9aa8a0; --rule:#2e3833; --accent:#7cc2ad;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; color-scheme:dark; }
body { background:var(--bg); color:var(--fg); font-family:var(--body); font-size:15px; line-height:1.55; }
main { max-width:1080px; margin:0 auto; padding-block:28px 56px; padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px)); display:flex; flex-direction:column; gap:22px; }
h1 { font-family:var(--display); font-weight:650; font-size:2rem; line-height:1.15; margin:0; text-wrap:balance; }
h2 { font-size:.78rem; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin:0 0 10px; font-weight:600; }
.lede { max-width:70ch; margin:8px 0 0; color:var(--muted); }
.panel { background:var(--panel); border:1px solid var(--rule); border-radius:6px; padding:18px; min-width:0; }
.ctl { display:flex; flex-wrap:wrap; align-items:center; gap:10px 18px; }
.ctl label { font-weight:600; }
#k { flex:1 1 260px; accent-color:var(--accent); }
#kout { font-family:var(--mono); font-size:1.6rem; color:var(--accent); font-variant-numeric:tabular-nums; min-width:9ch; }
.read { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:12px; margin-top:16px; }
.tile { border-top:2px solid var(--rule); padding-top:8px; }
.tile b { display:block; font-family:var(--mono); font-size:1.3rem; font-variant-numeric:tabular-nums; }
.tile span { color:var(--muted); font-size:.85rem; }
.cols { display:grid; grid-template-columns:minmax(0, 1.6fr) minmax(0, 1fr); gap:18px; }
@media (max-width:760px) { .cols { grid-template-columns:1fr; } }
.chartwrap { position:relative; overflow-x:auto; }
svg text { fill:var(--muted); font-family:var(--body); font-size:11px; }
.legend { display:flex; flex-wrap:wrap; gap:6px 16px; margin:0 0 8px; padding:0; list-style:none; font-size:.85rem; }
.legend i { display:inline-block; width:14px; height:3px; border-radius:2px; vertical-align:middle; margin-right:6px; }
.tip { position:absolute; pointer-events:none; background:var(--panel); border:1px solid var(--rule); border-radius:4px; padding:6px 8px; font-size:.8rem; color:var(--fg); white-space:nowrap; }
.sets { list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:10px; }
.sets li { border-left:2px solid var(--rule); padding-left:10px; }
.sets b { font-family:var(--mono); font-weight:400; color:var(--accent); }
.note { color:var(--muted); font-size:.88rem; max-width:75ch; margin:0; }
table { border-collapse:collapse; font-size:.82rem; font-variant-numeric:tabular-nums; width:100%; }
th, td { padding:4px 8px; text-align:right; border-bottom:1px solid var(--rule); white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
.tablewrap { overflow-x:auto; }
input:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
</style>
<main>
<header><h1>ResNet-18 Zoom</h1>
<p class="lede">The last layer of ResNet-18 has 512 channels. Slide to choose how many it keeps. Each size was built once, up front, by the merge rule: channels that the map says move together become one channel, and the output weights are re-fit. Moving the slider only picks a stored network. Numbers are on 10,000 photos the rule never saw.</p></header>
<section class="panel">
<div class="ctl"><label for="k">Channels kept</label><input id="k" type="range" min="0" max="0" step="1" value="0"><output id="kout" for="k"></output></div>
<div class="read">
<div class="tile"><b id="t-agree"></b><span>photos where the small network's top answer matches the full network's</span></div>
<div class="tile"><b id="t-top1"></b><span>top-1 accuracy (full network: <span id="t-n"></span>)</span></div>
<div class="tile"><b id="t-kl"></b><span>KL from the full network (0 = identical answers)</span></div>
<div class="tile"><b id="t-size"></b><span>of the last layer's channels kept</span></div>
</div>
</section>
<div class="cols">
<section class="panel"><h2>KL from the full network, by size</h2>
<ul class="legend" id="legend"></ul>
<div class="chartwrap" id="chartwrap"><svg id="chart" viewBox="0 0 640 300" width="100%" role="img" aria-label="KL against channels kept for five ways of building the small network"></svg><div class="tip" id="tip" hidden></div></div>
<p class="note">Lower is better. Trained is an estimate of the best possible at that size (boxes trained by gradient descent), shown at three sizes only. It is a check on the rule, not part of it.</p>
</section>
<section class="panel"><h2>Largest merged channels at this size</h2><ul class="sets" id="sets"></ul>
<p class="note" style="margin-top:12px">Each merged channel is listed by the classes its new output weights favour most.</p></section>
</div>
<section class="panel"><h2>All sizes (table view)</h2><div class="tablewrap"><table id="tbl"></table></div></section>
<p class="note">What this shows about ResNet-18: its channels are not near-copies of each other, so the proved bound is far too large to use here, and down to about 100 channels merging is no better than pruning or random groups. From about 48 down to 16 channels the merges beat both. At 4 channels or fewer nothing works. Trained boxes do better at every size checked, so the rule is not the best possible build here.</p>
</main>
<script>
const D = __DATA__;
const L = D.levels.slice().sort((a,b)=>a.k-b.k);
const S = [
 {key:"refit", name:"Merge rule, re-fit readers", c:"--s1"},
 {key:"merge", name:"Merge rule, written-down readers", c:"--s5", dash:"5 4"},
 {key:"prune", name:"Keep the strongest channels", c:"--s2"},
 {key:"random", name:"Random groups", c:"--s3"},
 {key:"trained", name:"Trained (check)", c:"--s4", dots:true}];
const slider = document.getElementById("k"); slider.max = L.length-1; slider.value = L.findIndex(l=>l.k===64);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const pct = x => (100*x).toFixed(1)+"%";
document.getElementById("t-n").textContent = pct(D.top1_N);
document.getElementById("legend").innerHTML = S.map(s=>`<li><i style="background:var(${s.c})"></i>${s.name}</li>`).join("");
const W=640,H=300,ml=46,mr=16,mt=12,mb=36;
const xs = k => ml + (Math.log2(k)/9)*(W-ml-mr);
const ymax = 6; const ys = v => mt + (1 - v/ymax)*(H-mt-mb);
function draw(sel){
  const svg = document.getElementById("chart"); let g = "";
  for (let v=0; v<=ymax; v+=1) g += `<line x1="${ml}" x2="${W-mr}" y1="${ys(v)}" y2="${ys(v)}" stroke="${css('--rule')}" stroke-width="1"/><text x="${ml-8}" y="${ys(v)+4}" text-anchor="end">${v}</text>`;
  [1,2,4,8,16,32,64,128,256,512].forEach(k=> g += `<text x="${xs(k)}" y="${H-mb+16}" text-anchor="middle">${k}</text>`);
  g += `<text x="${(ml+W-mr)/2}" y="${H-4}" text-anchor="middle">channels kept (log scale)</text>`;
  g += `<line x1="${xs(sel.k)}" x2="${xs(sel.k)}" y1="${mt}" y2="${H-mb}" stroke="${css('--muted')}" stroke-dasharray="2 3"/>`;
  S.forEach(s=>{
    const pts = L.filter(l=>l[s.key]).map(l=>[xs(l.k), ys(Math.min(l[s.key].kl, ymax)), l]);
    if (!s.dots) g += `<polyline fill="none" stroke="${css(s.c)}" stroke-width="2" ${s.dash?`stroke-dasharray="${s.dash}"`:""} points="${pts.map(p=>p[0]+","+p[1]).join(" ")}"/>`;
    pts.forEach(p=> g += `<circle cx="${p[0]}" cy="${p[1]}" r="${s.dots?5:3}" fill="${css(s.c)}" stroke="${css('--panel')}" stroke-width="2"/>`);
  });
  L.forEach(l=> g += `<rect x="${xs(l.k)-10}" y="${mt}" width="20" height="${H-mt-mb}" fill="transparent" data-k="${l.k}"/>`);
  svg.innerHTML = g;
}
const tip = document.getElementById("tip");
document.getElementById("chart").addEventListener("mousemove", e=>{
  const r = e.target.closest("rect[data-k]"); if(!r){tip.hidden=true;return;}
  const l = L.find(x=>x.k==r.dataset.k);
  tip.innerHTML = `<b>${l.k} channels</b><br>` + S.filter(s=>l[s.key]).map(s=>`${s.name}: ${l[s.key].kl.toFixed(3)}`).join("<br>");
  const wrap = document.getElementById("chartwrap").getBoundingClientRect();
  tip.style.left = Math.min(e.clientX-wrap.left+12, wrap.width-230)+"px"; tip.style.top = (e.clientY-wrap.top+12)+"px"; tip.hidden=false;
});
document.getElementById("chart").addEventListener("mouseleave", ()=> tip.hidden=true);
function show(){
  const l = L[+slider.value];
  document.getElementById("kout").textContent = l.k + " / 512";
  document.getElementById("t-agree").textContent = pct(l.refit.agree);
  document.getElementById("t-top1").textContent = pct(l.refit.top1);
  document.getElementById("t-kl").textContent = l.refit.kl.toFixed(3);
  document.getElementById("t-size").textContent = pct(l.k/512);
  document.getElementById("sets").innerHTML = l.sets.length ? l.sets.map(s=>`<li><b>${s.size} channels</b> into one: ${s.classes.join(", ")}</li>`).join("") : "<li>No merges at this size: every channel is kept.</li>";
  draw(l);
}
slider.addEventListener("input", show);
document.getElementById("tbl").innerHTML = "<tr><th>Channels</th><th>Agree</th><th>Top-1</th>" + S.map(s=>`<th>KL: ${s.name}</th>`).join("") + "</tr>" +
  L.slice().reverse().map(l=>`<tr><td>${l.k}</td><td>${pct(l.refit.agree)}</td><td>${pct(l.refit.top1)}</td>` + S.map(s=>`<td>${l[s.key]?l[s.key].kl.toFixed(3):"–"}</td>`).join("") + "</tr>").join("");
show();
</script>
""".replace("__DATA__", data)
open(os.path.join(HERE, "zoom.html"), "w").write(html)
print("wrote zoom.html", len(html) // 1024, "KB")
