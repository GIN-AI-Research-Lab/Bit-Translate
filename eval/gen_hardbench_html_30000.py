#!/usr/bin/env python3
"""Sinh HTML báo cáo 4-way hardbench step25000 (theme-aware, information-design).
Xuất: eval/hardbench_4way_25000.html (standalone) + eval/hardbench_4way_25000.frag.html
(fragment cho Artifact: style + content + script, không có doctype/html/head/body).

  python eval/gen_hardbench_html.py
"""
import html
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).parent
SYS = [("292M", "292M step30000"), ("google", "Google"), ("haiku", "Claude Haiku 4.5"),
       ("fable", "Claude Fable")]
FILES = {"292M": "hardbench_292m_step30000.jsonl", "google": "hardbench_google.jsonl",
         "haiku": "hardbench_haiku.jsonl", "fable": "hardbench_claude.jsonl"}
# mốc cũ để so tiến bộ (292M step25000, cùng panel judge trước Phase 1 vòng 3a)
PREV_110 = {"acc": {"vi2ja": 1.45, "ja2vi": 2.19}, "nat": {"vi2ja": 1.71, "ja2vi": 2.33}, "ok": 8.0}


def load(p):
    return {r["id"]: r for r in (json.loads(l) for l in open(ROOT / p, encoding="utf-8"))}


base = load("hardbench200.jsonl")
hyp = {k: load(v) for k, v in FILES.items()}
scores = {int(k): v for k, v in json.loads(
    (ROOT / "hardbench_4way_scores.json").read_text(encoding="utf-8"))["per_item"].items()}
ids = sorted(base)
dirs = ["vi2ja", "ja2vi"]
DIRLAB = {"vi2ja": "vi→ja", "ja2vi": "ja→vi"}


def cell(i, s, m):
    return scores.get(i, {}).get(s, {}).get(m, float("nan"))


def sysmean(s, m, d=None, dom=None):
    v = [cell(i, s, m) for i in ids
         if (d is None or base[i]["dir"] == d) and (dom is None or base[i]["domain"] == dom)]
    v = [x for x in v if x == x]
    return mean(v) if v else float("nan")


def pct_ok(s):
    v = [cell(i, s, "acc") for i in ids]
    v = [x for x in v if x == x]
    return 100 * sum(1 for x in v if x >= 4) / len(v) if v else 0


def sclass(x):
    if x != x:
        return "s0"
    return "s4" if x >= 4 else "s3" if x >= 3 else "s2" if x >= 2 else "s1"


doms = sorted({base[i]["domain"] for i in ids})
DOMLAB = {"caudai": "Câu dài", "hoithoai": "Hội thoại", "hop": "Họp/công sở",
          "it_deep": "IT chuyên sâu", "keigo": "Kính ngữ", "nguphap": "Ngữ pháp bẫy",
          "slang": "Tiếng lóng", "solieu": "Số liệu", "thanhngu": "Thành ngữ",
          "zeropronoun": "Chủ ngữ ẩn"}

# ---- scorecard tổng ----
sum_rows = []
for s, lab in SYS:
    is292 = s == "292M"
    tds = [f'<th scope=row class="nm{" me" if is292 else ""}">{html.escape(lab)}</th>']
    for m in ("acc", "nat"):
        for d in dirs:
            x = sysmean(s, m, d)
            tds.append(f'<td class="{sclass(x)}">{x:.2f}</td>')
    tds.append(f'<td class="ok">{pct_ok(s):.0f}%</td>')
    sum_rows.append(f'<tr{" class=me-row" if is292 else ""}>' + "".join(tds) + "</tr>")

# so với 110M (chỉ 292M có mốc trước)
delta = {}
for m in ("acc", "nat"):
    for d in dirs:
        delta[(m, d)] = sysmean("292M", m, d) - PREV_110[m][d]

# ---- heatmap domain (acc, 2 chiều) ----
dom_rows = []
for dom in doms:
    tds = [f'<th scope=row class=nm>{html.escape(DOMLAB.get(dom, dom))}</th>']
    for s, _ in SYS:
        x = sysmean(s, "acc", dom=dom)
        tds.append(f'<td class="{sclass(x)}">{x:.2f}</td>')
    dom_rows.append("<tr>" + "".join(tds) + "</tr>")

# ---- card từng câu ----
cards = []
for i in ids:
    b = base[i]
    d = b["dir"]
    dom = b["domain"]
    sl = "JA" if d == "ja2vi" else "VI"
    tl = "VI" if d == "ja2vi" else "JA"
    trs = []
    for s, lab in SYS:
        h = hyp[s].get(i, {}).get("hyp", "")
        acc, nat = cell(i, s, "acc"), cell(i, s, "nat")
        trs.append(
            f'<div class="tr{" me" if s == "292M" else ""}">'
            f'<div class=sh><span class=snm>{html.escape(lab)}</span>'
            f'<span class="pill {sclass(acc)}">acc {acc:.1f}</span>'
            f'<span class="pill {sclass(nat)}">nat {nat:.1f}</span></div>'
            f'<p class=tx>{html.escape(h) or "<em>—</em>"}</p></div>')
    a292 = cell(i, "292M", "acc")
    others = [cell(i, s, "acc") for s, _ in SYS[1:]]
    win = "win" if a292 == a292 and a292 >= max(others) else ""
    cards.append(
        f'<article class="card {win}" data-dir="{d}" data-dom="{dom}">'
        f'<header class=meta><span class=tag>{html.escape(DOMLAB.get(dom, dom))}</span>'
        f'<span class="tag dir">{sl}→{tl}</span><span class=cid>#{i}</span></header>'
        f'<p class=src><span class=lb>Gốc · {sl}</span>{html.escape(b["src"])}</p>'
        f'<p class=ref><span class=lb>Tham chiếu</span>{html.escape(b["ref"])}</p>'
        f'<div class=grid>{"".join(trs)}</div></article>')

dom_opts = "".join(f'<option value="{d}">{html.escape(DOMLAB.get(d, d))}</option>' for d in doms)

# ---- gate G1: 292M có thắng Google ở 3 domain ja2vi (thanhngu/solieu/hoithoai)? ----
GATE_DOMS = ["thanhngu", "solieu", "hoithoai"]
gate_rows = []
gate_pass = True
for dom in GATE_DOMS:
    m292 = sysmean("292M", "acc", d="ja2vi", dom=dom)
    mg = sysmean("google", "acc", d="ja2vi", dom=dom)
    ok = m292 >= mg
    gate_pass = gate_pass and ok
    gate_rows.append((dom, m292, mg, ok))


def dnum(x):
    return f'<span class="d {"up" if x > 0 else "dn" if x < 0 else ""}">{"+" if x >= 0 else ""}{x:.2f}</span>'


STYLE = """<style>
:root{
 --bg:#f4f6f9; --panel:#ffffff; --ink:#1b2430; --mut:#5c6b7f; --line:#dde3ec;
 --me:#3358d4; --me-bg:#eef2ff;
 --s4:#137a3a; --s3:#8a6100; --s2:#b4530f; --s1:#c22a34; --s0:#9aa5b3;
 --s4bg:#e3f5ea; --s3bg:#fbf1d6; --s2bg:#fbe9dc; --s1bg:#fbe0e2; --s0bg:#eef1f5;
 --shadow:0 1px 2px rgba(20,30,50,.06),0 8px 24px rgba(20,30,50,.05);
}
@media (prefers-color-scheme:dark){:root{
 --bg:#0e131a; --panel:#161d27; --ink:#e6edf5; --mut:#93a1b3; --line:#26313f;
 --me:#7aa2ff; --me-bg:#182238;
 --s4:#54c377; --s3:#dcb34a; --s2:#e8894e; --s1:#f0636e; --s0:#66748a;
 --s4bg:#12271a; --s3bg:#2a2410; --s2bg:#2c1f14; --s1bg:#2c1519; --s0bg:#1a222e;
 --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
}}
:root[data-theme=light]{
 --bg:#f4f6f9; --panel:#ffffff; --ink:#1b2430; --mut:#5c6b7f; --line:#dde3ec;
 --me:#3358d4; --me-bg:#eef2ff;
 --s4:#137a3a; --s3:#8a6100; --s2:#b4530f; --s1:#c22a34; --s0:#9aa5b3;
 --s4bg:#e3f5ea; --s3bg:#fbf1d6; --s2bg:#fbe9dc; --s1bg:#fbe0e2; --s0bg:#eef1f5;
}
:root[data-theme=dark]{
 --bg:#0e131a; --panel:#161d27; --ink:#e6edf5; --mut:#93a1b3; --line:#26313f;
 --me:#7aa2ff; --me-bg:#182238;
 --s4:#54c377; --s3:#dcb34a; --s2:#e8894e; --s1:#f0636e; --s0:#66748a;
 --s4bg:#12271a; --s3bg:#2a2410; --s2bg:#2c1f14; --s1bg:#2c1519; --s0bg:#1a222e;
}
*{box-sizing:border-box}
.hb{background:var(--bg);color:var(--ink);font:15px/1.55 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif;-webkit-font-smoothing:antialiased}
.hb .wrap{max-width:1120px;margin:0 auto;padding:32px 20px 80px}
.hb h1{font-size:clamp(22px,3.4vw,30px);font-weight:750;letter-spacing:-.02em;margin:0 0 6px;text-wrap:balance}
.hb h2{font-size:17px;font-weight:680;letter-spacing:-.01em;margin:38px 0 4px}
.hb .lede{color:var(--mut);max-width:66ch;margin:0 0 4px}
.hb .cap{color:var(--mut);font-size:12.5px;margin:6px 0 0}
.hb a{color:var(--me)}
.hb .eyebrow{font-size:12px;font-weight:640;letter-spacing:.09em;text-transform:uppercase;color:var(--me)}
.hb .tblwrap{overflow-x:auto;margin-top:14px;border:1px solid var(--line);border-radius:14px;background:var(--panel);box-shadow:var(--shadow)}
.hb table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
.hb th,.hb td{padding:11px 14px;text-align:center;border-bottom:1px solid var(--line);white-space:nowrap}
.hb tbody tr:last-child td,.hb tbody tr:last-child th{border-bottom:0}
.hb thead th{font-size:11.5px;font-weight:620;letter-spacing:.03em;text-transform:uppercase;color:var(--mut);background:transparent;text-align:center}
.hb thead th.grp{border-bottom:1px solid var(--line)}
.hb th.nm{text-align:left;font-weight:640;color:var(--ink)}
.hb th.nm.me{color:var(--me)}
.hb tr.me-row{background:var(--me-bg)}
.hb td{font-weight:600}
.hb td.s4{color:var(--s4)} .hb td.s3{color:var(--s3)} .hb td.s2{color:var(--s2)} .hb td.s1{color:var(--s1)} .hb td.s0{color:var(--s0)}
.hb td.ok{color:var(--ink);font-weight:680}
.hb .delta{display:flex;flex-wrap:wrap;gap:8px 22px;margin-top:12px;padding:14px 16px;border:1px solid var(--line);border-radius:12px;background:var(--panel)}
.hb .delta .t{font-size:13px;color:var(--mut)}
.hb .delta b{color:var(--ink)} .hb .d{font-weight:680;font-variant-numeric:tabular-nums}
.hb .d.up{color:var(--s4)} .hb .d.dn{color:var(--s1)}
.hb .ctrl{position:sticky;top:0;z-index:9;display:flex;flex-wrap:wrap;gap:8px;align-items:center;
 margin-top:8px;padding:12px 0;background:linear-gradient(var(--bg) 78%,transparent)}
.hb .seg{display:inline-flex;border:1px solid var(--line);border-radius:9px;overflow:hidden;background:var(--panel)}
.hb .seg button{border:0;background:transparent;color:var(--mut);padding:7px 13px;font:inherit;font-size:13px;cursor:pointer}
.hb .seg button.on{background:var(--me);color:#fff}
.hb select,.hb .tgl{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:9px;padding:7px 12px;font:inherit;font-size:13px;cursor:pointer}
.hb .tgl.on{background:var(--me);color:#fff;border-color:var(--me)}
.hb .cnt{margin-left:auto;color:var(--mut);font-size:13px;font-variant-numeric:tabular-nums}
.hb .cards{display:flex;flex-direction:column;gap:14px;margin-top:4px}
.hb .card{border:1px solid var(--line);border-radius:14px;background:var(--panel);padding:16px 17px;box-shadow:var(--shadow)}
.hb .card.win{border-color:color-mix(in srgb,var(--s4) 55%,var(--line))}
.hb .meta{display:flex;align-items:center;gap:8px;margin-bottom:10px}
.hb .tag{font-size:12px;color:var(--mut);border:1px solid var(--line);border-radius:20px;padding:2px 11px}
.hb .tag.dir{color:var(--me);border-color:color-mix(in srgb,var(--me) 40%,var(--line));font-variant-numeric:tabular-nums}
.hb .cid{margin-left:auto;color:var(--mut);font-size:12px;font-variant-numeric:tabular-nums}
.hb .src,.hb .ref{margin:0 0 8px;padding:9px 12px;border-radius:9px;font-size:14.5px}
.hb .src{background:var(--s0bg)} .hb .ref{background:var(--s4bg)}
.hb .lb{display:block;font-size:10.5px;font-weight:650;letter-spacing:.07em;text-transform:uppercase;color:var(--mut);margin-bottom:3px}
.hb .grid{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-top:2px}
@media (max-width:620px){.hb .grid{grid-template-columns:1fr}}
.hb .tr{border:1px solid var(--line);border-radius:10px;padding:9px 11px;background:var(--bg)}
.hb .tr.me{border-color:color-mix(in srgb,var(--me) 50%,var(--line));background:var(--me-bg)}
.hb .sh{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:5px}
.hb .snm{font-size:12.5px;font-weight:660;margin-right:auto}
.hb .pill{font-size:11px;font-weight:660;border-radius:5px;padding:1px 7px;font-variant-numeric:tabular-nums}
.hb .pill.s4{color:var(--s4);background:var(--s4bg)} .hb .pill.s3{color:var(--s3);background:var(--s3bg)}
.hb .pill.s2{color:var(--s2);background:var(--s2bg)} .hb .pill.s1{color:var(--s1);background:var(--s1bg)}
.hb .pill.s0{color:var(--s0);background:var(--s0bg)}
.hb .tx{margin:0;font-size:14.5px}
.hb .hidden{display:none}
.hb .legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12.5px;color:var(--mut);margin-top:10px}
.hb .legend i{font-style:normal;font-weight:660}
</style>"""

BODY = f"""<div class=hb><div class=wrap>
<p class=eyebrow>Benchmark chấm mù · hardbench200 · Phase 1 (vòng 3a) xong</p>
<h1>Bốn hệ dịch Việt↔Nhật trên 200 câu khó — model 292M bước 30000</h1>
<p class=lede>10 giám khảo Claude chấm mù (5 panel × 2), điểm <b>acc</b> (chính xác nghĩa) và
<b>nat</b> (tự nhiên) thang 0–5, độc lập từng bản, thứ tự A–D xáo trộn. <b>292M</b> là model
1.58-bit của dự án (chạy CPU); Google/Haiku/Fable là mốc so. Panel judge lần này khác lần
step25000 (giám khảo mới, điểm Google/Haiku/Fable xê dịch nhẹ theo — so sánh khoảng cách
tương đối với Google trong CÙNG một lần chấm mới có ý nghĩa, không so tuyệt đối qua các lần).</p>
<div class=legend>
<span><i style="color:var(--s4)">■</i> ≥4 dùng được</span>
<span><i style="color:var(--s3)">■</i> 3–4</span>
<span><i style="color:var(--s2)">■</i> 2–3</span>
<span><i style="color:var(--s1)">■</i> &lt;2 kém</span>
</div>

<h2>Điểm tổng — 200 câu</h2>
<div class=tblwrap><table>
<thead>
<tr><th></th><th class=grp colspan=2>acc (chính xác)</th><th class=grp colspan=2>nat (tự nhiên)</th><th rowspan=2 style="vertical-align:bottom">dùng được<br>acc≥4</th></tr>
<tr><th></th><th>vi→ja</th><th>ja→vi</th><th>vi→ja</th><th>ja→vi</th></tr>
</thead>
<tbody>{''.join(sum_rows)}</tbody>
</table></div>
<div class=delta><span class=t>Tiến bộ so <b>292M bước 25000</b> (trước Phase 1 vòng 3a):</span>
<span class=t>acc vi→ja {dnum(delta[('acc','vi2ja')])}</span>
<span class=t>acc ja→vi {dnum(delta[('acc','ja2vi')])}</span>
<span class=t>nat vi→ja {dnum(delta[('nat','vi2ja')])}</span>
<span class=t>nat ja→vi {dnum(delta[('nat','ja2vi')])}</span>
<span class=t>dùng được {dnum(pct_ok('292M')-PREV_110['ok'])}%</span></div>
<p class=cap><b>Gate G1 ({', '.join(DOMLAB.get(d,d) for d in GATE_DOMS)} ja→vi, thắng Google):
{'ĐẠT' if gate_pass else 'CHƯA ĐẠT'}</b> — """ + " · ".join(
    f'{DOMLAB.get(dom, dom)} 292M {m292:.2f} {"≥" if ok else "&lt;"} Google {mg:.2f}'
    for dom, m292, mg, ok in gate_rows
) + f"""<br>292M vẫn kém Google/Haiku/Fable rõ rệt trên toàn bộ 200 câu (dùng được acc≥4:
{pct_ok('292M'):.0f}% so Google {pct_ok('google'):.0f}%) — Phase 1 (data vòng 3a) mới thu hẹp
NHẸ khoảng cách so bước 25000, đúng như lộ trình 3–5 vòng lặp đã định, chưa phải điểm dừng.</p>

<h2>Độ chính xác theo chủ đề (acc, gộp 2 chiều)</h2>
<div class=tblwrap><table>
<thead><tr><th></th>{''.join(f'<th>{html.escape(lab)}</th>' for _,lab in SYS)}</tr></thead>
<tbody>{''.join(dom_rows)}</tbody>
</table></div>
<p class=cap>292M mạnh nhất ở số liệu/chủ-ngữ-ẩn ja→vi; vẫn yếu tiếng lóng &amp; câu dài vi→ja
(sinh tiếng Nhật) — điểm chết cấu trúc đã biết, chưa xử lý ở vòng data này.</p>

<h2>Từng câu — 4 bản dịch &amp; điểm</h2>
<div class=ctrl>
<div class=seg>
<button class=on data-v=all onclick="fd(this,'all')">Cả 2 chiều</button>
<button data-v=vi2ja onclick="fd(this,'vi2ja')">vi→ja</button>
<button data-v=ja2vi onclick="fd(this,'ja2vi')">ja→vi</button>
</div>
<select id=dom onchange=fdom()><option value=all>Mọi chủ đề</option>{dom_opts}</select>
<button class=tgl id=wb onclick=ftog()>Câu 292M ≥ mọi hệ khác</button>
<span class=cnt id=cnt></span>
</div>
<div class=cards id=cards>{''.join(cards)}</div>
</div></div>
<script>
(function(){{
 let D='all',M='all',W=false;
 const cards=[...document.querySelectorAll('.hb .card')],cnt=document.getElementById('cnt');
 function apply(){{let n=0;for(const c of cards){{
  const ok=(D=='all'||c.dataset.dir==D)&&(M=='all'||c.dataset.dom==M)&&(!W||c.classList.contains('win'));
  c.classList.toggle('hidden',!ok);if(ok)n++;}}cnt.textContent=n+' câu';}}
 window.fd=(b,v)=>{{D=v;b.parentElement.querySelectorAll('button').forEach(x=>x.classList.remove('on'));b.classList.add('on');apply();}};
 window.fdom=()=>{{M=document.getElementById('dom').value;apply();}};
 window.ftog=()=>{{W=!W;document.getElementById('wb').classList.toggle('on',W);apply();}};
 apply();
}})();
</script>"""

frag = STYLE + BODY
(ROOT / "hardbench_4way_30000.frag.html").write_text(frag, encoding="utf-8")
full = ('<!doctype html><html lang=vi><head><meta charset=utf-8>'
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        '<title>Hardbench 4-way — 292M step30000</title></head><body>'
        + frag + "</body></html>")
(ROOT / "hardbench_4way_30000.html").write_text(full, encoding="utf-8")
print("gate G1:", "DAT" if gate_pass else "CHUA DAT")
print("frag:", (ROOT / "hardbench_4way_30000.frag.html").stat().st_size // 1024, "KB;",
      "full:", (ROOT / "hardbench_4way_30000.html").stat().st_size // 1024, "KB;", len(cards), "câu")
