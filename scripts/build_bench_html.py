#!/usr/bin/env python3
"""Dựng trang HTML so sánh bản dịch của các hệ — có search / filter / sort.

Đọc eval/bench_<sys>.jsonl (mỗi file 1 hệ) + điểm judge nếu có
(eval/bench_scores.json) rồi xuất eval/bench_compare.html — file tĩnh, mở bằng
trình duyệt, không cần server.

  python scripts/build_bench_html.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
EVAL = ROOT / "eval"
OUT = EVAL / "bench_compare.html"

SYS = [
    ("opus", "Claude-Opus5", "#7c3aed"),
    ("haiku", "Haiku 4.5", "#0891b2"),
    ("google", "Google Translate", "#ea580c"),
    ("v3", "BitNet v3 · 152M", "#16a34a"),
    ("v2", "BitNet v2 · 110M", "#65a30d"),
    ("100m", "BitNet 100M (avg5)", "#a16207"),
]


def main():
    rows = {}
    for r in (json.loads(l) for l in open(EVAL / "bench_new.jsonl", encoding="utf-8")):
        rows[r["id"]] = {"id": r["id"], "len": r["len"], "feat": r["feat"],
                         "src": r["src"], "hyp": {}, "score": {}, "mean": {}}
    for key, _, _ in SYS:
        p = EVAL / f"bench_{key}.jsonl"
        if not p.exists():
            continue
        for r in (json.loads(l) for l in open(p, encoding="utf-8")):
            if r["id"] in rows:
                rows[r["id"]]["hyp"][key] = r["hyp"]

    sp = EVAL / "bench_scores.json"
    if sp.exists():
        sc = json.load(open(sp, encoding="utf-8"))
        for sid, per in sc.items():
            if int(sid) in rows:
                rows[int(sid)]["score"] = per
    # Thang NGHIA (0=sai, 1=lech, 2=dung) — chi so CHINH: "cau co dung nghia khong".
    # Thang acc 0-5 cu tron do chinh xac voi do troi chay, khien ban dich dung y
    # nhung cung bi coi la "khong dung duoc".
    mp = EVAL / "bench_meaning.json"
    if mp.exists():
        for sid, per in json.load(open(mp, encoding="utf-8")).items():
            if int(sid) in rows:
                rows[int(sid)]["mean"] = per

    data = [rows[i] for i in sorted(rows)]
    syslist = [{"key": k, "name": n, "color": c} for k, n, c in SYS]
    html = TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False)) \
                   .replace("__SYS__", json.dumps(syslist, ensure_ascii=False))
    OUT.write_text(html, encoding="utf-8")
    n_scored = sum(1 for r in data if r["score"])
    print(f"-> {OUT}")
    print(f"   {len(data)} câu | {len([s for s in SYS if any(r['hyp'].get(s[0]) for r in data)])} hệ"
          f" | đã chấm điểm: {n_scored} câu")


TEMPLATE = r"""<meta charset="utf-8">
<title>So sánh bản dịch JA→VI</title>
<style>
:root{
  --bg:#ffffff; --fg:#18181b; --mut:#71717a; --line:#e4e4e7; --card:#fafafa;
  --ja:#1e293b; --jabg:#f1f5f9; --acc:#2563eb;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0b0b0e; --fg:#e7e7ea; --mut:#a1a1aa; --line:#27272a; --card:#131316;
  --ja:#dbeafe; --jabg:#17212f; --acc:#60a5fa;}}
:root[data-theme=dark]{--bg:#0b0b0e;--fg:#e7e7ea;--mut:#a1a1aa;--line:#27272a;
  --card:#131316;--ja:#dbeafe;--jabg:#17212f;--acc:#60a5fa}
:root[data-theme=light]{--bg:#fff;--fg:#18181b;--mut:#71717a;--line:#e4e4e7;
  --card:#fafafa;--ja:#1e293b;--jabg:#f1f5f9;--acc:#2563eb}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:15px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans JP",sans-serif}
header{position:sticky;top:0;z-index:9;background:var(--bg);
  border-bottom:1px solid var(--line);padding:14px 20px}
h1{margin:0 0 10px;font-size:17px;font-weight:650;letter-spacing:-.01em}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
input[type=search],select{background:var(--card);color:var(--fg);
  border:1px solid var(--line);border-radius:7px;padding:7px 10px;font:inherit;font-size:14px}
input[type=search]{flex:1;min-width:220px}
.chip{border:1px solid var(--line);background:var(--card);color:var(--mut);
  border-radius:20px;padding:5px 11px;font-size:13px;cursor:pointer;user-select:none}
.chip.on{background:var(--acc);border-color:var(--acc);color:#fff}
.count{color:var(--mut);font-size:13px;margin-left:auto}
main{padding:16px 20px 60px;max-width:1180px;margin:0 auto}
.item{border:1px solid var(--line);border-radius:11px;margin-bottom:14px;
  overflow:hidden;background:var(--bg)}
.src{background:var(--jabg);color:var(--ja);padding:12px 14px;
  font-size:15px;line-height:1.75;border-bottom:1px solid var(--line)}
.meta{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin-bottom:7px}
.tag{font-size:11px;font-weight:600;letter-spacing:.03em;text-transform:uppercase;
  color:var(--mut);border:1px solid var(--line);border-radius:5px;padding:1px 6px;background:var(--bg)}
.hyps{display:grid;grid-template-columns:150px 1fr 82px;gap:0}
.hyps>div{padding:9px 14px;border-bottom:1px solid var(--line)}
.hyps>div:nth-last-child(-n+3){border-bottom:none}
.sys{font-size:12.5px;font-weight:600;display:flex;align-items:center;gap:7px}
.dot{width:8px;height:8px;border-radius:50%;flex:none}
.txt{font-size:14.5px}
.sc{text-align:center;font-weight:700;font-size:12.5px;white-space:nowrap}
.s5{color:#16a34a}.s4{color:#65a30d}.s3{color:#ca8a04}.s2{color:#ea580c}.s1{color:#dc2626}
.none{color:var(--mut);font-style:italic}
mark{background:#fde68a;color:#000;border-radius:2px}
@media(max-width:720px){.hyps{grid-template-columns:1fr;gap:0}
  .hyps>div{border-bottom:none;padding:6px 14px}
  .sys{padding-top:10px}.sc{text-align:left}}
table.sum{border-collapse:collapse;width:100%;margin:6px 0 18px;font-size:14px}
table.sum th,table.sum td{border:1px solid var(--line);padding:7px 10px;text-align:right}
table.sum th:first-child,table.sum td:first-child{text-align:left}
table.sum thead th{background:var(--card);font-weight:600}
</style>

<header>
  <h1>So sánh bản dịch Nhật → Việt · bộ test câu thật chưa train</h1>
  <div class="bar">
    <input type="search" id="q" placeholder="Tìm trong câu gốc hoặc bản dịch…">
    <select id="flen"><option value="">Độ dài: tất cả</option>
      <option value="short">Câu ngắn</option><option value="long">Câu dài</option></select>
    <select id="ffeat"><option value="">Đặc điểm: tất cả</option></select>
    <select id="sort"><option value="id">Sắp: theo thứ tự</option>
      <option value="len">Sắp: câu dài trước</option>
      <option value="gap">Sắp: v3 hơn Google nhiều nhất</option>
      <option value="bad">Sắp: v3 kém nhất</option>
      <option value="wrong">Sắp: câu v3 SAI nghĩa</option></select>
    <span class="chip" id="theme">◐</span>
    <span class="count" id="cnt"></span>
  </div>
</header>
<main><div id="sum"></div><div id="list"></div></main>

<script>
const DATA=__DATA__, SYS=__SYS__;
const $=s=>document.querySelector(s);
const esc=s=>(s||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const feats=[...new Set(DATA.map(d=>d.feat))].sort();
$("#ffeat").innerHTML+=feats.map(f=>`<option value="${f}">${f}</option>`).join("");

function summary(rows){
  const any=rows.some(r=>Object.keys(r.score||{}).length);
  let h=`<table class="sum"><thead><tr><th>Hệ</th><th>Số câu</th>`;
  if(any)h+=`<th>% ĐÚNG nghĩa</th><th>% sai</th><th>Ngắn</th><th>Dài</th><th>acc TB</th>`;
  h+=`</tr></thead><tbody>`;
  for(const s of SYS){
    const has=rows.filter(r=>r.hyp[s.key]);
    if(!has.length)continue;
    h+=`<tr><td><span class="dot" style="background:${s.color};display:inline-block;margin-right:6px"></span>${s.name}</td><td>${has.length}</td>`;
    if(any){
      const sc=has.filter(r=>r.score&&r.score[s.key]!=null).map(r=>r.score[s.key]);
      const mn=has.filter(r=>r.mean&&r.mean[s.key]!=null).map(r=>r.mean[s.key]);
      const msh=has.filter(r=>r.len=="short"&&r.mean&&r.mean[s.key]!=null).map(r=>r.mean[s.key]);
      const mlo=has.filter(r=>r.len=="long"&&r.mean&&r.mean[s.key]!=null).map(r=>r.mean[s.key]);
      const avg=a=>a.length?(a.reduce((x,y)=>x+y,0)/a.length).toFixed(2):"–";
      const p=(a,v)=>a.length?Math.round(100*a.filter(x=>x===v).length/a.length)+"%":"–";
      h+=`<td><b>${p(mn,2)}</b></td><td>${p(mn,0)}</td><td>${p(msh,2)}</td><td>${p(mlo,2)}</td><td>${avg(sc)}</td>`;
    }
    h+=`</tr>`;
  }
  return h+`</tbody></table>`;
}

function render(){
  const q=$("#q").value.trim().toLowerCase(), fl=$("#flen").value, ff=$("#ffeat").value, so=$("#sort").value;
  let rows=DATA.filter(d=>{
    if(fl&&d.len!==fl)return false;
    if(ff&&d.feat!==ff)return false;
    if(q){const hay=(d.src+" "+Object.values(d.hyp).join(" ")).toLowerCase();
      if(!hay.includes(q))return false;}
    return true;});
  const g=(r,k)=>(r.score&&r.score[k]!=null)?r.score[k]:null;
  if(so=="len")rows=[...rows].sort((a,b)=>b.src.length-a.src.length);
  else if(so=="gap")rows=[...rows].sort((a,b)=>((g(b,"v3")??0)-(g(b,"google")??0))-((g(a,"v3")??0)-(g(a,"google")??0)));
  else if(so=="bad")rows=[...rows].sort((a,b)=>(g(a,"v3")??9)-(g(b,"v3")??9));
  else if(so=="wrong")rows=[...rows].sort((a,b)=>((a.mean&&a.mean.v3!=null)?a.mean.v3:9)-((b.mean&&b.mean.v3!=null)?b.mean.v3:9));
  $("#cnt").textContent=rows.length+" / "+DATA.length+" câu";
  $("#sum").innerHTML=summary(rows);
  const hl=t=>{t=esc(t); if(!q)return t;
    return t.replace(new RegExp("("+q.replace(/[.*+?^${}()|[\]\\]/g,"\\$&")+")","gi"),"<mark>$1</mark>");};
  $("#list").innerHTML=rows.map(d=>{
    let h=`<div class="item"><div class="src"><div class="meta">
      <span class="tag">#${d.id}</span><span class="tag">${d.len==="short"?"ngắn":"dài"}</span>
      <span class="tag">${d.feat}</span><span class="tag">${d.src.length} ký tự</span></div>${hl(d.src)}</div><div class="hyps">`;
    for(const s of SYS){
      if(!d.hyp[s.key])continue;
      const v=d.score?d.score[s.key]:null;
      const mv=(d.mean&&d.mean[s.key]!=null)?d.mean[s.key]:null;
      const MT={2:"✓ đúng",1:"~ lệch",0:"✗ SAI"}, MC={2:"s5",1:"s3",0:"s1"};
      h+=`<div class="sys"><span class="dot" style="background:${s.color}"></span>${s.name}</div>
          <div class="txt">${hl(d.hyp[s.key])}</div>
          <div class="sc ${mv!=null?MC[mv]:(v?"s"+v:"")}" title="acc ${v??"-"}/5">${mv!=null?MT[mv]:(v??"<span class='none'>–</span>")}</div>`;
    }
    return h+`</div></div>`;}).join("");
}
["q","flen","ffeat","sort"].forEach(i=>$("#"+i).addEventListener("input",render));
$("#theme").onclick=()=>{const r=document.documentElement;
  const cur=r.getAttribute("data-theme")||(matchMedia("(prefers-color-scheme:dark)").matches?"dark":"light");
  r.setAttribute("data-theme",cur==="dark"?"light":"dark");};
render();
</script>
"""


if __name__ == "__main__":
    main()
