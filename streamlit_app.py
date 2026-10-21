"""Streamlit: Penetapan Kadar Fosfat dengan Spektrofotometer UV-Vis Shimadzu UV-1800.

Menjalankan:  streamlit run app.py
"""
import datetime as dt
import io
import math
import zipfile
from statistics import NormalDist

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


from dataclasses import dataclass

# ======================================================================
# BAGIAN 1 — PERHITUNGAN
# ======================================================================

# ---- Distribusi t-Student tanpa scipy ----------------------------------------------------------
def _betacf(a, b, x):
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-14:
            break
    return h


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1 - bt * _betacf(b, a, 1 - x) / b


def _t_cdf(t, df):
    p = 0.5 * _betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1 - p if t > 0 else p


def t_ppf(p, df):
    """Kuantil t-Student (p > 0,5) dengan metode bagi-dua."""
    df = max(float(df), 1.0)
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = (lo + hi) / 2
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def norm_ppf(p):
    return NormalDist().inv_cdf(p)



CONC_COL = "Konsentrasi (mg/L P)"
ABS_COLS = ["Abs 1", "Abs 2", "Abs 3", "Abs 4", "Abs 5"]

# Massa atom (g/mol) dan ketidakpastian (setengah lebar interval, distribusi persegi panjang)
ATOMIC = {
    "K": (39.0983, 0.0001),
    "H": (1.00794, 0.00007),
    "P": (30.973762, 0.000002),
    "O": (15.9994, 0.0003),
}
M_P = ATOMIC["P"][0]
M_PO4 = M_P + 4 * ATOMIC["O"][0]
P_TO_PO4 = M_PO4 / M_P  # 3.0661

# Toleransi gelas volumetrik kelas A (ISO 1042 labu ukur, ISO 648 pipet volume), dalam mL
FLASK_TOL = {5: 0.025, 10: 0.025, 25: 0.03, 50: 0.05, 100: 0.10, 200: 0.15,
             250: 0.15, 500: 0.25, 1000: 0.40, 2000: 0.60}
PIPET_TOL = {1: 0.008, 2: 0.010, 5: 0.015, 10: 0.020, 20: 0.030, 25: 0.030,
             50: 0.050, 100: 0.080}


def glass_tol(kind: str, volume: float, glass_class: str = "A") -> float:
    table = FLASK_TOL if kind == "flask" else PIPET_TOL
    keys = sorted(table)
    tol = float(np.interp(volume, keys, [table[k] for k in keys]))
    return tol * (2.0 if glass_class == "B" else 1.0)


def u_from_tol(tol: float, dist: str) -> float:
    """Ketidakpastian baku dari toleransi (setengah lebar)."""
    return tol / math.sqrt(6) if dist == "Segitiga" else tol / math.sqrt(3)


# --------------------------------------------------------------------------- kalibrasi
@dataclass
class CalFit:
    x: np.ndarray
    y: np.ndarray
    n: int
    xbar: float
    ybar: float
    sxx: float
    b: float
    a: float
    sy: float
    r: float
    r2: float
    se_b: float
    se_a: float
    dof: int
    lod: float
    loq: float
    resid: np.ndarray
    levels: pd.DataFrame


def long_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cols = [c for c in ABS_COLS if c in df.columns]
    d = df[[CONC_COL] + cols].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=[CONC_COL])
    m = d.melt(id_vars=CONC_COL, value_vars=cols).dropna()
    return m[CONC_COL].to_numpy(float), m["value"].to_numpy(float)


def fit_calibration(df: pd.DataFrame) -> CalFit:
    x, y = long_data(df)
    n = len(x)
    if n < 5 or len(np.unique(x)) < 3:
        raise ValueError("Data kalibrasi tidak cukup: minimal 3 level konsentrasi dan 5 pembacaan.")
    xbar, ybar = x.mean(), y.mean()
    sxx = float(((x - xbar) ** 2).sum())
    b = float(((x - xbar) * (y - ybar)).sum() / sxx)
    a = float(ybar - b * xbar)
    resid = y - (a + b * x)
    ssr = float((resid ** 2).sum())
    dof = n - 2
    sy = math.sqrt(ssr / dof)
    sst = float(((y - ybar) ** 2).sum())
    r2 = 1 - ssr / sst
    se_b = sy / math.sqrt(sxx)
    se_a = sy * math.sqrt(1 / n + xbar ** 2 / sxx)
    lv = (pd.DataFrame({"x": x, "y": y}).groupby("x")["y"]
          .agg(["mean", "std", "count"]).reset_index()
          .rename(columns={"x": CONC_COL, "mean": "A rata-rata", "std": "SD", "count": "n"}))
    return CalFit(x, y, n, xbar, ybar, sxx, b, a, sy, math.sqrt(max(r2, 0)), r2,
                  se_b, se_a, dof, 3.3 * sy / b, 10 * sy / b, resid, lv)


def conf_band(fit: CalFit, xg: np.ndarray, conf: float):
    t = t_ppf(1 - (1 - conf) / 2, fit.dof)
    yh = fit.a + fit.b * xg
    h = t * fit.sy * np.sqrt(1 / fit.n + (xg - fit.xbar) ** 2 / fit.sxx)
    return yh, yh - h, yh + h


# --------------------------------------------------------------------------- ketidakpastian
def stock_relative_u(m_g, u_m_lin, u_m_rep, purity, purity_hw, v_ml, v_tol, dist):
    """Ketidakpastian relatif larutan stok KH2PO4 (mg/L sebagai P)."""
    f_p = M_P / (ATOMIC["K"][0] + ATOMIC["H"][0] + M_P + 4 * ATOMIC["O"][0])
    # ketidakpastian massa molar (jumlah kuadrat kontribusi tiap unsur, persegi panjang)
    mm = ATOMIC["K"][0] + ATOMIC["H"][0] + M_P + 4 * ATOMIC["O"][0]
    u_mm = math.sqrt(sum((n * ATOMIC[s][1] / math.sqrt(3)) ** 2
                         for s, n in (("K", 1), ("H", 1), ("P", 1), ("O", 4))))
    u_mp = ATOMIC["P"][1] / math.sqrt(3)
    u_fp_rel = math.sqrt((u_mp / M_P) ** 2 + (u_mm / mm) ** 2)
    u_mass = math.sqrt(2 * (u_m_lin / math.sqrt(3)) ** 2 + u_m_rep ** 2)  # tara + bruto
    rows = [
        ("Penimbangan KH2PO4 (linearitas + repeatabilitas neraca)", u_mass / m_g, "B"),
        ("Kemurnian KH2PO4", (purity_hw / math.sqrt(3)) / purity, "B"),
        (f"Labu ukur stok {v_ml:g} mL", u_from_tol(v_tol, dist) / v_ml, "B"),
        ("Fraksi massa P dalam KH2PO4 (massa atom)", u_fp_rel, "B"),
    ]
    conc = m_g * 1000 * (purity / 100) * f_p / (v_ml / 1000)
    return rows, conc, f_p


def sf_round(U: float, x: float, sig: int = 2):
    """Bulatkan U ke `sig` angka penting dan x ke desimal yang sama."""
    if not np.isfinite(U) or U <= 0:
        return x, U, 4
    exp = math.floor(math.log10(U))
    dec = max(0, sig - 1 - exp)
    return round(x, dec), round(U, dec), dec


def sample_row_values(row: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(row[[c for c in ABS_COLS if c in row.index]], errors="coerce")
    return vals.dropna().to_numpy(float)


def evaluate_sample(row: pd.Series, fit: CalFit, cfg: dict) -> dict:
    """Hitung kadar dan anggaran ketidakpastian untuk satu sampel."""
    y = sample_row_values(row)
    m = len(y)
    if m == 0:
        raise ValueError("Tidak ada data absorbansi sampel.")
    y0 = float(y.mean())
    sd_rep = float(y.std(ddof=1)) if m > 1 else float("nan")
    x0 = (y0 - fit.a) / fit.b
    v1 = float(row.get("V aliquot (mL)", np.nan))
    v2 = float(row.get("V akhir (mL)", np.nan))
    if not (np.isfinite(v1) and np.isfinite(v2)) or v1 <= 0 or v2 <= 0:
        v1 = v2 = 1.0
    df_ = v2 / v1
    rec = cfg.get("rec_mean")  # fraksi, mis. 1.0
    corr = cfg.get("rec_correct", False) and rec
    c_p = x0 * df_ / (rec if corr else 1.0)

    # --- komponen
    comps = []
    u_x0 = fit.sy / fit.b * math.sqrt(1 / m + 1 / fit.n + (y0 - fit.ybar) ** 2 / (fit.b ** 2 * fit.sxx))
    comps.append(dict(Sumber="Kurva kalibrasi + repeatabilitas pembacaan sampel",
                      Tipe="A", Distribusi="Normal", Satuan="mg/L",
                      Nilai=u_x0, u_rel=abs(u_x0 / x0) if x0 else np.nan, dof=fit.dof))
    for name, ur, typ in cfg["std_rows"]:
        comps.append(dict(Sumber=name, Tipe=typ, Distribusi="-", Satuan="relatif",
                          Nilai=ur, u_rel=ur, dof=np.inf))
    if abs(df_ - 1) > 1e-12:
        cl = cfg["glass_class"]
        dist = cfg["dist"]
        up = u_from_tol(glass_tol("pipet", v1, cl), dist)
        uf = u_from_tol(glass_tol("flask", v2, cl), dist)
        comps.append(dict(Sumber=f"Pipet volume {v1:g} mL (pengenceran sampel)", Tipe="B", Distribusi=dist,
                          Satuan="relatif", Nilai=up / v1, u_rel=up / v1, dof=np.inf))
        comps.append(dict(Sumber=f"Labu ukur {v2:g} mL (pengenceran sampel)", Tipe="B", Distribusi=dist,
                          Satuan="relatif", Nilai=uf / v2, u_rel=uf / v2, dof=np.inf))
    if rec and cfg.get("rec_u_rel") is not None:
        comps.append(dict(Sumber="Recovery (uji spike)", Tipe="A", Distribusi="Normal", Satuan="relatif",
                          Nilai=cfg["rec_u_rel"], u_rel=cfg["rec_u_rel"], dof=cfg["rec_dof"]))
    if cfg.get("photometric"):
        acc = 0.002 if y0 <= 0.5 else 0.004
        ua = acc / math.sqrt(3)
        comps.append(dict(Sumber=f"Akurasi fotometrik UV-1800 (±{acc} Abs)", Tipe="B", Distribusi="Persegi panjang",
                          Satuan="mg/L", Nilai=ua / fit.b, u_rel=abs(ua / fit.b / x0) if x0 else np.nan,
                          dof=np.inf))

    bud = pd.DataFrame(comps)
    s2 = float((bud["u_rel"] ** 2).sum())
    u_c_rel = math.sqrt(s2)
    u_c = abs(c_p) * u_c_rel
    bud["u_i (mg/L P)"] = abs(c_p) * bud["u_rel"]
    bud["Kontribusi (%)"] = 100 * bud["u_rel"] ** 2 / s2

    terms = [(r.u_rel ** 4) / r.dof for r in bud.itertuples() if np.isfinite(r.dof)]
    veff = (u_c_rel ** 4) / sum(terms) if terms and sum(terms) > 0 else np.inf
    conf = cfg["conf"]
    if cfg["k_mode"] == "k = 2":
        k = 2.0
    else:
        k = float(t_ppf(1 - (1 - conf) / 2, veff)) if np.isfinite(veff) and veff < 1e4 else float(
            norm_ppf(1 - (1 - conf) / 2))
    U = k * u_c
    return dict(name=row.get("Nama sampel", ""), y=y, m=m, y0=y0, sd_rep=sd_rep,
                rsd_rep=(sd_rep / y0 * 100) if m > 1 and y0 else float("nan"),
                x0=x0, df=df_, v1=v1, v2=v2, c_p=c_p, c_po4=c_p * P_TO_PO4,
                u_c=u_c, u_c_rel=u_c_rel, veff=veff, k=k, U=U, U_rel=U / abs(c_p) * 100 if c_p else np.nan,
                budget=bud, in_range=bool(fit.x.min() <= x0 <= fit.x.max()),
                above_loq=bool(x0 >= fit.loq))


def mo_blue_shape(wl):
    """Bentuk spektrum ilustratif kompleks biru molibdenum (puncak ~880 nm, bahu ~710 nm)."""
    wl = np.asarray(wl, float)
    g = lambda m, s: np.exp(-((wl - m) / s) ** 2)
    norm = 1.0 + 0.55 * math.exp(-((880 - 710) / 45) ** 2)
    return (g(880, 60) + 0.55 * g(710, 45)) / norm


# ======================================================================
# BAGIAN 2 — SKEMA ALAT INTERAKTIF
# ======================================================================

_HTML = r"""
<!doctype html><html><head><meta charset="utf-8">
<style>
:root{--beam:#d62828;--bg:#0b1220;--card:#121c31;--ink:#e6edf7;--mut:#8fa3c4;--acc:#38bdf8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,Segoe UI,Roboto,sans-serif}
.wrap{padding:10px}
.ctl{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px;margin-bottom:8px}
.box{background:var(--card);border:1px solid #1f2d4a;border-radius:10px;padding:8px 12px}
.box label{font-size:12px;color:var(--mut);display:flex;justify-content:space-between}
.box b{color:var(--ink)}
input[type=range]{width:100%;accent-color:var(--acc)}
.btns{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px}
button{background:#1b2a4a;color:var(--ink);border:1px solid #2b3f68;border-radius:8px;padding:6px 11px;font-size:12px;cursor:pointer}
button:hover{background:#25396a}
svg{width:100%;height:auto;background:radial-gradient(ellipse at 30% 20%,#14213d,#0b1220);border:1px solid #1f2d4a;border-radius:12px}
.beam{stroke:var(--beam);stroke-width:3.2;fill:none;stroke-dasharray:10 8;animation:flow .7s linear infinite;stroke-linecap:round;filter:drop-shadow(0 0 3px var(--beam))}
.paused .beam{animation-play-state:paused}
@keyframes flow{to{stroke-dashoffset:-18}}
.comp{cursor:pointer}.comp .shape{stroke:#7f93b8;stroke-width:1.6;fill:#1a2744;transition:.2s}
.comp:hover .shape,.comp.hl .shape{stroke:var(--acc);stroke-width:3;filter:drop-shadow(0 0 8px var(--acc))}
.lbl{fill:#b8c7e3;font-size:12.5px}.lbl2{fill:#7f93b8;font-size:10.5px}
.read{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin:8px 0}
.read .box{text-align:center}.read .v{font:600 22px ui-monospace,Menlo,monospace;color:#7ee787}.read .k{font-size:11px;color:var(--mut)}
#info{background:var(--card);border:1px solid #1f2d4a;border-left:4px solid var(--acc);border-radius:8px;padding:9px 12px;font-size:13px;line-height:1.45;min-height:58px}
</style></head><body><div class="wrap" id="root">
<div class="ctl">
 <div class="box"><label><span>Panjang gelombang λ</span><b><span id="lv">880</span> nm</b></label><input id="lam" type="range" min="190" max="1100" step="1" value="__L0__"></div>
 <div class="box"><label><span>Konsentrasi fosfat (sebagai P)</span><b><span id="cv">0.50</span> mg/L</b></label><input id="conc" type="range" min="0" max="1.2" step="0.01" value="0.5"></div>
</div>
<div class="btns">
 <button id="bAnim">⏸ Jeda animasi</button><button id="bTour">▶ Tur jalur cahaya</button>
 <button id="bZero">Autozero</button><button id="bMeas">Ukur</button><button id="bOpt">Atur ke 880 nm</button><button id="bReset">Reset</button>
</div>
<svg id="sv" viewBox="0 0 1000 460" xmlns="http://www.w3.org/2000/svg">
 <!-- jalur cahaya -->
 <g id="beams">
  <polyline id="bWI" class="beam" points="84,105 150,175"/>
  <polyline id="bD2" class="beam" points="84,245 150,175"/>
  <polyline class="beam" points="150,175 250,175 345,290 440,175 560,175"/>
  <polyline class="beam" points="560,175 712,175"/>
  <polyline id="afterS" class="beam" points="768,175 888,175"/>
  <polyline class="beam" points="560,175 560,365 712,365"/>
  <polyline id="afterR" class="beam" points="768,365 888,365"/>
 </g>
 <!-- komponen -->
 <g class="comp" data-id="wi"><circle class="shape" cx="62" cy="105" r="22"/><text class="lbl" x="22" y="145">Lampu halogen (WI)</text><text class="lbl2" x="22" y="159">daerah tampak–NIR</text></g>
 <g class="comp" data-id="d2"><circle class="shape" cx="62" cy="245" r="22"/><text class="lbl" x="30" y="285">Lampu D₂</text><text class="lbl2" x="30" y="299">daerah UV</text></g>
 <g class="comp" data-id="m1"><rect class="shape" x="142" y="150" width="10" height="50" transform="rotate(-30 147 175)"/><text class="lbl" x="105" y="215">Cermin sumber</text><text class="lbl2" x="105" y="229">(pemilih lampu)</text></g>
 <g class="comp" data-id="s1"><rect class="shape" x="246" y="145" width="8" height="60"/><text class="lbl" x="207" y="135">Celah masuk</text></g>
 <g class="comp" data-id="gr"><path class="shape" d="M315,305 Q345,275 375,305 L375,315 L315,315 Z"/><text class="lbl" x="285" y="338">Kisi holografik</text><text class="lbl2" x="285" y="352">(monokromator)</text></g>
 <g class="comp" data-id="s2"><rect class="shape" x="436" y="145" width="8" height="60"/><text class="lbl" x="400" y="135">Celah keluar</text></g>
 <g class="comp" data-id="bs"><rect class="shape" x="545" y="160" width="30" height="30" transform="rotate(45 560 175)"/><text class="lbl" x="468" y="225">Pembelah berkas</text><text class="lbl2" x="468" y="239">(double beam)</text></g>
 <g class="comp" data-id="m2"><rect class="shape" x="545" y="350" width="30" height="30" transform="rotate(45 560 365)"/><text class="lbl" x="445" y="400">Cermin</text></g>
 <g class="comp" data-id="sc"><rect class="shape" x="712" y="135" width="56" height="80" rx="4"/><rect id="liq" x="716" y="158" width="48" height="52" fill="rgba(30,79,216,.1)"/><text class="lbl" x="690" y="125">Kuvet sampel</text></g>
 <g class="comp" data-id="rc"><rect class="shape" x="712" y="325" width="56" height="80" rx="4"/><rect x="716" y="348" width="48" height="52" fill="rgba(160,200,255,.07)"/><text class="lbl" x="684" y="318">Kuvet referensi (blanko)</text></g>
 <g class="comp" data-id="dt"><rect class="shape" x="888" y="150" width="50" height="50" rx="6"/><text class="lbl" x="895" y="222">Detektor</text><text class="lbl2" x="880" y="236">fotodioda Si</text></g>
 <g class="comp" data-id="dr"><rect class="shape" x="888" y="340" width="50" height="50" rx="6"/><text class="lbl" x="895" y="412">Detektor</text><text class="lbl2" x="882" y="426">referensi</text></g>
 <text class="lbl2" x="14" y="450">Model edukasi (tidak berskala) — verifikasi detail optik pada manual resmi Shimadzu UV-1800.</text>
</svg>
<div class="read">
 <div class="box"><div class="v" id="rA">0.000</div><div class="k">Absorbansi (A)</div></div>
 <div class="box"><div class="v" id="rT">100.0</div><div class="k">Transmitans (%T)</div></div>
 <div class="box"><div class="v" id="rL" style="font-size:16px">WI</div><div class="k">Lampu aktif</div></div>
 <div class="box"><div class="v" id="rB" style="font-size:16px">—</div><div class="k">Warna berkas</div></div>
</div>
<div id="info">Klik komponen mana pun untuk melihat fungsinya, atau tekan <b>Tur jalur cahaya</b>.</div>
</div>
<script>
const B=__B__;           // sensitivitas (A per mg/L P) pada 880 nm dari kurva kalibrasi
const LAMP_SW=340;       // pergantian lampu (nm)
function gs(l,m,s){return Math.exp(-Math.pow((l-m)/s,2));}
const NORM=gs(880,880,60)+0.55*gs(880,710,45);
function shape(l){return (gs(l,880,60)+0.55*gs(l,710,45))/NORM;}
function wl2rgb(w){
 if(w<380)return[138,111,209]; if(w>780)return[176,74,74];
 let r=0,g=0,b=0;
 if(w<440){r=-(w-440)/60;b=1}else if(w<490){g=(w-440)/50;b=1}else if(w<510){g=1;b=-(w-510)/20}
 else if(w<580){r=(w-510)/70;g=1}else if(w<645){r=1;g=-(w-645)/65}else{r=1}
 let f=w<420?0.3+0.7*(w-380)/40:(w>700?0.3+0.7*(780-w)/80:1);
 return[r,g,b].map(v=>Math.round(255*Math.pow(v*f,0.8)));
}
const $=id=>document.getElementById(id);
let zero=0, noise=0;
const INFO={
 wi:["Lampu halogen (WI)","Sumber kontinu untuk daerah tampak dan inframerah dekat. Analisis fosfat metode biru molibdenum diukur sekitar 880 nm sehingga memakai lampu ini."],
 d2:["Lampu deuterium (D₂)","Sumber kontinu untuk daerah UV (±190–340 nm). Tidak dipakai pada 880 nm, tetapi aktif saat λ di bawah titik pergantian lampu."],
 m1:["Cermin sumber","Memilih/mengarahkan berkas dari lampu aktif ke celah masuk monokromator. Lampu berganti otomatis pada panjang gelombang tertentu (default 340 nm)."],
 s1:["Celah masuk","Membatasi berkas yang masuk ke monokromator sehingga menghasilkan citra celah yang tajam pada kisi."],
 gr:["Kisi difraksi (monokromator)","Menguraikan cahaya polikromatik menjadi spektrum. Memutar kisi memilih λ yang keluar lewat celah keluar. Pada UV-1800 lebar celah spektral 1 nm."],
 s2:["Celah keluar","Meloloskan pita sempit (≈1 nm) di sekitar λ terpilih. Makin sempit celah, makin monokromatik tetapi intensitas turun."],
 bs:["Pembelah berkas","Membagi berkas monokromatik menjadi dua: ke kuvet sampel dan ke kuvet referensi (sistem double-beam), mengoreksi drift lampu."],
 m2:["Cermin","Membelokkan berkas referensi menuju kuvet referensi dan detektor referensi."],
 sc:["Kuvet sampel","Berisi larutan kompleks biru molibdenum (fosfat + molibdat + askorbat). Pada konsentrasi lebih tinggi larutan lebih biru dan menyerap lebih banyak cahaya 880 nm (hukum Lambert–Beer: A = ε·b·c)."],
 rc:["Kuvet referensi","Berisi blanko reagen (air + pereaksi yang diperlakukan sama). Dipakai pada Autozero sehingga serapan pelarut, pereaksi dan kuvet dikurangkan."],
 dt:["Detektor sampel","Fotodioda silikon mengubah intensitas cahaya yang lolos (I) menjadi sinyal listrik."],
 dr:["Detektor referensi","Mengukur intensitas acuan (I₀). Absorbansi A = −log(I/I₀)."]
};
function show(id){const i=INFO[id];$('info').innerHTML='<b>'+i[0]+'</b><br>'+i[1];
 document.querySelectorAll('.comp').forEach(e=>e.classList.toggle('hl',e.dataset.id===id));}
document.querySelectorAll('.comp').forEach(e=>e.addEventListener('click',()=>{stopTour();show(e.dataset.id)}));
function update(){
 const l=+$('lam').value,c=+$('conc').value;
 $('lv').textContent=l;$('cv').textContent=c.toFixed(2);
 const A=B*c*shape(l),T=Math.pow(10,-A);
 const rgb=wl2rgb(l);const col='rgb('+rgb.join(',')+')';
 document.documentElement.style.setProperty('--beam',col);
 $('afterS').style.opacity=Math.max(T,0.05);
 const d2=l<LAMP_SW;
 $('bWI').style.opacity=d2?0.08:1;$('bD2').style.opacity=d2?1:0.08;
 $('rL').textContent=d2?'D₂':'WI';
 $('rB').textContent=l<380?'UV (tak tampak)':(l>780?'NIR (tak tampak)':col);
 $('liq').setAttribute('fill','rgba(30,79,216,'+(Math.min(c/1.2,1)*0.85+0.05).toFixed(3)+')');
 const Ad=A-zero+noise;
 $('rA').textContent=Ad.toFixed(3);$('rT').textContent=(100*Math.pow(10,-(A-zero))).toFixed(1);
}
$('lam').oninput=()=>{noise=0;update()};$('conc').oninput=()=>{noise=0;update()};
$('bAnim').onclick=()=>{const p=$('root').classList.toggle('paused');$('bAnim').textContent=p?'▶ Lanjut animasi':'⏸ Jeda animasi'};
$('bZero').onclick=()=>{zero=B*(+$('conc').value)*shape(+$('lam').value);noise=0;update();
 $('info').innerHTML='<b>Autozero</b><br>Absorbansi saat ini dijadikan nol (tempatkan blanko di kedua kuvet: atur konsentrasi 0 lebih dulu).'};
$('bMeas').onclick=()=>{noise=(Math.random()-0.5)*0.0012;update()};
$('bOpt').onclick=()=>{$('lam').value=880;noise=0;update()};
$('bReset').onclick=()=>{$('lam').value=__L0__;$('conc').value=0.5;zero=0;noise=0;stopTour();update();
 $('info').innerHTML='Klik komponen mana pun untuk melihat fungsinya, atau tekan <b>Tur jalur cahaya</b>.'};
let tour=null,ti=0;const ORDER=['wi','d2','m1','s1','gr','s2','bs','sc','rc','m2','dt','dr'];
function stopTour(){if(tour){clearInterval(tour);tour=null;$('bTour').textContent='▶ Tur jalur cahaya'}}
$('bTour').onclick=()=>{if(tour){stopTour();return}ti=0;$('bTour').textContent='■ Hentikan tur';
 show(ORDER[0]);tour=setInterval(()=>{ti=(ti+1)%ORDER.length;show(ORDER[ti])},3200)};
update();
</script></body></html>
"""


def schematic_html(slope: float = 0.69, lam0: int = 880) -> str:
    return _HTML.replace("__B__", f"{slope:.5f}").replace("__L0__", str(int(lam0)))


# ======================================================================
# BAGIAN 3 — ANTARMUKA STREAMLIT
# ======================================================================
import inspect

# Kompatibel dengan Streamlit lama (use_container_width) dan baru (width="stretch")
W = ({"width": "stretch"} if "width" in inspect.signature(st.dataframe).parameters
     else {"use_container_width": True})


def embed_html(html: str, height: int):
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height, scrolling=True)



def vega(spec: dict, height: int = 380):
    full = {"$schema": "https://vega.github.io/schema/vega-lite/v5.json", "width": "container",
            "height": height, **spec}
    st.vega_lite_chart(full, **W)


def X(title, **kw):
    return {"field": "x", "type": "quantitative", "title": title, "scale": {"zero": False}, **kw}


st.set_page_config(page_title="UV-1800 | Penetapan Fosfat", page_icon="🧪", layout="wide")
st.markdown(
    """<style>
    .block-container{padding-top:1.4rem}
    div[data-testid="stMetricValue"]{font-size:1.5rem}
    .hasil{background:#0f2a1f;border:1px solid #1f6f4a;border-radius:12px;padding:14px 18px;font-size:1.15rem}
    </style>""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- data contoh
EX_STD = pd.DataFrame({
    CONC_COL: [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    "Abs 1": [0.002, 0.072, 0.141, 0.279, 0.418, 0.556, 0.695],
    "Abs 2": [0.003, 0.071, 0.140, 0.281, 0.416, 0.558, 0.692],
    "Abs 3": [0.002, 0.073, 0.142, 0.280, 0.419, 0.555, 0.696],
    "Abs 4": np.nan, "Abs 5": np.nan})
EX_SMP = pd.DataFrame({
    "Nama sampel": ["Air sungai (contoh)", "Air limbah (contoh)"],
    "V aliquot (mL)": [50.0, 25.0], "V akhir (mL)": [50.0, 50.0],
    "Abs 1": [0.182, 0.312], "Abs 2": [0.185, 0.315], "Abs 3": [0.180, 0.310],
    "Abs 4": np.nan, "Abs 5": np.nan})
EX_CHAIN = pd.DataFrame({
    "Tahap": ["Pengenceran antara: stok → 5 mg/L P"],
    "Pipet (mL)": [10.0], "Labu (mL)": [100.0],
    "Toleransi pipet (mL)": [0.020], "Toleransi labu (mL)": [0.10]})

for k, v in (("std", EX_STD), ("smp", EX_SMP), ("chain", EX_CHAIN)):
    if k not in st.session_state:
        st.session_state[k] = v.copy()

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🧪 UV-1800")
    st.caption("Penetapan kadar fosfat (PO₄-P) — metode biru molibdenum / asam askorbat")
    with st.expander("Identitas pengujian", expanded=False):
        analis = st.text_input("Analis", "")
        lab = st.text_input("Laboratorium / Institusi", "")
        tgl = st.date_input("Tanggal analisis", dt.date.today())
        sn = st.text_input("No. seri alat", "")
        sop = st.text_input("Acuan metode", "SNI 06-6989.31-2005 / APHA 4500-P E")
    st.subheader("Parameter alat")
    lam = st.number_input("λ analisis (nm)", 190, 1100, 880, 1)
    path = st.number_input("Panjang sel (cm)", 0.1, 10.0, 1.0, 0.1)
    st.subheader("Statistik")
    conf = st.select_slider("Tingkat kepercayaan", [0.90, 0.95, 0.99], 0.95, format_func=lambda v: f"{v:.0%}")
    k_mode = st.radio("Faktor cakupan k", ["k dari t-Student (v_eff)", "k = 2"], 0)
    dist = st.radio("Distribusi toleransi gelas", ["Segitiga", "Persegi panjang"], 0,
                    help="Segitiga (÷√6) umum untuk gelas volumetrik; persegi panjang (÷√3) lebih konservatif.")
    glass_class = st.radio("Kelas gelas volumetrik", ["A", "B"], 0)
    st.subheader("Kriteria penerimaan")
    r2_min = st.number_input("R² minimum", 0.90, 1.0, 0.995, 0.001, format="%.3f")
    rec_lo, rec_hi = st.slider("Rentang recovery (%)", 70, 130, (90, 110))
    st.divider()
    if st.button("Muat ulang data contoh"):
        for k, v in (("std", EX_STD), ("smp", EX_SMP), ("chain", EX_CHAIN)):
            st.session_state[k] = v.copy()
        st.rerun()
    st.caption("Data contoh bersifat ilustrasi, bukan hasil pengukuran nyata. Ganti dengan data Anda.")

st.title("Penetapan Kadar Fosfat — Spektrofotometer UV-Vis Shimadzu UV-1800")
st.caption("Skema alat • Prosedur • Spektrum • Kalibrasi • Perhitungan • Ketidakpastian • Pelaporan")

tabs = st.tabs(["1 · Skema alat", "2 · Prosedur", "3 · Spektrum & λmaks", "4 · Kalibrasi",
                "5 · Perhitungan sampel", "6 · Ketidakpastian", "7 · Pelaporan"])

# --------------------------------------------------------------------------- data editors (di tab 4 & 5) — hitung dulu
def get_fit():
    try:
        return fit_calibration(st.session_state["std"]), None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


fit, fit_err = get_fit()

# =========================================================================== 1. SKEMA
with tabs[0]:
    st.subheader("Skema jalur cahaya UV-1800 (interaktif)")
    st.caption("Geser λ dan konsentrasi: warna berkas, warna larutan, absorbansi dan %T ikut berubah. "
               "Sensitivitas diambil dari kurva kalibrasi Anda.")
    slope = fit.b if fit else 0.69
    embed_html(schematic_html(slope, int(lam)), 900)
    with st.expander("Spesifikasi ringkas UV-1800 (acuan, cek manual alat)"):
        st.markdown(
            "- Sistem optik: *double beam*, kisi holografik, fotodioda silikon\n"
            "- Sumber cahaya: lampu halogen (WI) dan deuterium (D₂)\n"
            "- Rentang λ: 190–1100 nm; lebar celah spektral 1 nm\n"
            "- Akurasi fotometrik (spesifikasi pabrik): ±0,002 A (0–0,5 A), ±0,004 A (0,5–1 A)\n"
            "- Perangkat lunak: UVProbe (mode Photometric / Spectrum / Quantitation)")

# =========================================================================== 2. PROSEDUR
with tabs[1]:
    st.subheader("Prosedur kerja (metode asam askorbat)")
    st.info("Ringkasan edukatif. Ikuti SOP laboratorium dan standar yang berlaku (mis. SNI 06-6989.31-2005 / APHA 4500-P E).")
    steps = [
        ("Persiapan pereaksi",
         "Larutan H₂SO₄ 5 N, kalium antimonil tartrat, amonium molibdat, asam askorbat 0,1 M, "
         "lalu **pereaksi kombinasi** (dibuat segar). Fenolftalein sebagai indikator."),
        ("Larutan induk fosfat",
         "Timbang KH₂PO₄ anhidrat (±219,5 mg) → 1000 mL ≈ 50 mg/L P. Buat larutan antara lalu deret standar."),
        ("Deret standar & blanko",
         "Contoh: 0; 0,1; 0,2; 0,4; 0,6; 0,8; 1,0 mg/L P. Blanko = air bebas fosfat + pereaksi."),
        ("Pengembangan warna",
         "Ke dalam volume tetap (mis. 50 mL) standar/sampel tambahkan pereaksi kombinasi (mis. 8 mL), "
         "kocok, diamkan ≥10 menit (maks. ±30 menit) — perlakuan sama untuk semua."),
        ("Setel UV-1800",
         "Nyalakan alat, tunggu pemanasan lampu, pilih mode Photometric, λ = 880 nm, sel 1 cm, **Autozero** dengan blanko."),
        ("Ukur absorbansi",
         "Ukur standar dari rendah ke tinggi, lalu sampel (replikat ≥3). Bilas kuvet antar-larutan."),
        ("Olah data",
         "Regresi linear A–C, cek R², residual, LOD/LOQ; hitung kadar sampel × faktor pengenceran."),
        ("Ketidakpastian & laporan",
         "Susun anggaran ketidakpastian, hitung U (k), laporkan hasil ± U.")]
    done = 0
    for i, (t, d) in enumerate(steps, 1):
        c1, c2 = st.columns([0.06, 0.94])
        ok = c1.checkbox(" ", key=f"step{i}", label_visibility="collapsed")
        done += ok
        c2.markdown(f"**{i}. {t}** — {d}")
    st.progress(done / len(steps), text=f"Progres: {done}/{len(steps)} langkah")

# =========================================================================== 3. SPEKTRUM
with tabs[2]:
    st.subheader("Spektrum serapan & penentuan λ maksimum")
    c1, c2 = st.columns([1, 2])
    with c1:
        mode = st.radio("Sumber spektrum", ["Simulasi (model ilustratif)", "Unggah CSV (λ, A)"])
        conc_scan = st.slider("Konsentrasi standar (mg/L P)", 0.1, 1.2, 1.0, 0.1)
        wl0, wl1 = st.slider("Rentang scan (nm)", 190, 1100, (500, 1000))
        seed = st.number_input("Seed derau", 0, 9999, 1)
    sens = fit.b if fit else 0.69
    if mode.startswith("Simulasi"):
        wl = np.arange(wl0, wl1 + 1, 1.0)
        rng = np.random.default_rng(int(seed))
        ab = conc_scan * sens * mo_blue_shape(wl) + rng.normal(0, 0.0005, len(wl))
        spec = pd.DataFrame({"λ (nm)": wl, "A": ab})
    else:
        up = c1.file_uploader("CSV dua kolom: panjang gelombang, absorbansi", type=["csv", "txt"])
        spec = None
        if up is not None:
            try:
                raw = pd.read_csv(up, sep=None, engine="python", header=None).apply(pd.to_numeric, errors="coerce").dropna()
                spec = raw.iloc[:, :2]
                spec.columns = ["λ (nm)", "A"]
            except Exception as e:  # noqa: BLE001
                st.error(f"Gagal membaca file: {e}")
    with c2:
        if spec is not None and len(spec):
            i = int(spec["A"].idxmax())
            lmax, amax = float(spec.loc[i, "λ (nm)"]), float(spec.loc[i, "A"])
            recs = [{"x": float(u), "y": float(v)} for u, v in zip(spec["λ (nm)"], spec["A"])]
            vega({"layer": [
                {"data": {"values": recs}, "mark": {"type": "line", "color": "#38bdf8"},
                 "encoding": {"x": X("Panjang gelombang (nm)"),
                              "y": {"field": "y", "type": "quantitative", "title": "Absorbansi (A)"}}},
                {"data": {"values": [{"x": lmax, "t": f"λmaks = {lmax:.0f} nm"}]},
                 "mark": {"type": "rule", "color": "#d62828", "strokeDash": [5, 4]},
                 "encoding": {"x": X("")}},
                {"data": {"values": [{"x": lmax, "t": f"λmaks = {lmax:.0f} nm"}]},
                 "mark": {"type": "text", "align": "left", "dx": 5, "dy": 8, "color": "#d62828"},
                 "encoding": {"x": X(""), "y": {"value": 8}, "text": {"field": "t"}}},
                {"data": {"values": [{"x": float(lam)}]},
                 "mark": {"type": "rule", "color": "#38bdf8", "strokeDash": [2, 3]},
                 "encoding": {"x": X("")}}]}, 420)
            st.metric("λ maksimum terdeteksi", f"{lmax:.0f} nm", f"A = {amax:.3f}")
            if abs(lmax - lam) > 10:
                st.warning("λ analisis berbeda >10 nm dari λmaks spektrum; pertimbangkan menyesuaikan.")
        else:
            st.info("Belum ada data spektrum.")
    st.caption("Spektrum simulasi hanya model ilustratif (puncak ±880 nm, bahu ±710 nm).")

# =========================================================================== 4. KALIBRASI
with tabs[3]:
    st.subheader("Data standar & kurva kalibrasi")
    st.caption("Isi konsentrasi (mg/L sebagai P) dan absorbansi tiap replikat. Baris dapat ditambah/hapus.")
    st.session_state["std"] = st.data_editor(
        st.session_state["std"], num_rows="dynamic", **W, key="std_ed",
        column_config={CONC_COL: st.column_config.NumberColumn(format="%.4f"),
                       **{c: st.column_config.NumberColumn(format="%.4f") for c in ABS_COLS}})
    fit, fit_err = get_fit()
    if fit_err:
        st.error(fit_err)
    else:
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Slope b (A per mg/L)", f"{fit.b:.5f}", f"± {fit.se_b:.5f}")
        k2.metric("Intersep a", f"{fit.a:.5f}", f"± {fit.se_a:.5f}")
        k3.metric("R²", f"{fit.r2:.5f}")
        k4.metric("s(y/x)", f"{fit.sy:.5f}")
        k5.metric("LOD (mg/L P)", f"{fit.lod:.4f}")
        k6.metric("LOQ (mg/L P)", f"{fit.loq:.4f}")
        if fit.r2 >= r2_min:
            st.success(f"R² = {fit.r2:.5f} ≥ {r2_min:.3f} — linearitas memenuhi kriteria.")
        else:
            st.error(f"R² = {fit.r2:.5f} < {r2_min:.3f} — linearitas tidak memenuhi kriteria.")
        t_a = t_ppf(1 - (1 - conf) / 2, fit.dof)
        if abs(fit.a) > t_a * fit.se_a:
            st.warning("Intersep berbeda nyata dari nol (uji-t): periksa kontaminasi blanko / koreksi blanko.")

        xg = np.linspace(min(0, fit.x.min()), fit.x.max() * 1.05, 100)
        yh, lo, hi = conf_band(fit, xg, conf)
        cA, cB = st.columns(2)
        sm = st.session_state["smp"]
        pts = []
        for _, r in sm.iterrows():
            try:
                ev = evaluate_sample(r, fit, dict(std_rows=[], glass_class=glass_class, dist=dist, conf=conf,
                                                  k_mode=k_mode))
                pts.append({"x": float(ev["x0"]), "y": float(ev["y0"]), "t": str(ev["name"])})
            except Exception:  # noqa: BLE001
                pass
        sgn = "+" if fit.a >= 0 else "−"
        yq = {"field": "y", "type": "quantitative", "title": "Absorbansi"}
        layers = [
            {"data": {"values": [{"x": float(u), "lo": float(l_), "hi": float(h_)} for u, l_, h_ in zip(xg, lo, hi)]},
             "mark": {"type": "area", "color": "#38bdf8", "opacity": 0.2},
             "encoding": {"x": X("C (mg/L P)", scale={"zero": True}),
                          "y": {"field": "lo", "type": "quantitative", "title": "Absorbansi"}, "y2": {"field": "hi"}}},
            {"data": {"values": [{"x": float(u), "y": float(v)} for u, v in zip(xg, yh)]},
             "mark": {"type": "line", "color": "#38bdf8"}, "encoding": {"x": X("C (mg/L P)"), "y": yq}},
            {"data": {"values": [{"x": float(u), "y": float(v)} for u, v in zip(fit.x, fit.y)]},
             "mark": {"type": "point", "filled": True, "color": "#f97316", "size": 60},
             "encoding": {"x": X("C (mg/L P)"), "y": yq}}]
        if pts:
            layers += [
                {"data": {"values": pts}, "mark": {"type": "point", "shape": "diamond", "filled": True,
                                                    "color": "#7ee787", "size": 140},
                 "encoding": {"x": X("C (mg/L P)"), "y": yq}},
                {"data": {"values": pts}, "mark": {"type": "text", "align": "left", "dx": 8, "dy": -8,
                                                    "color": "#7ee787"},
                 "encoding": {"x": X("C (mg/L P)"), "y": yq, "text": {"field": "t"}}}]
        with cA:
            st.markdown(f"**A = {fit.b:.5f}·C {sgn} {abs(fit.a):.5f}**  (R² = {fit.r2:.5f}) — "
                        f"oranye: standar, hijau: sampel, area biru: pita kepercayaan {conf:.0%}")
            vega({"layer": layers}, 380)
        rr = [{"x": float(u), "y": float(v)} for u, v in zip(fit.x, fit.resid)]
        rule = lambda v, d: {"data": {"values": [{"y": float(v)}]},
                             "mark": {"type": "rule", "strokeDash": d, "color": "#888"},
                             "encoding": {"y": {"field": "y", "type": "quantitative"}}}
        with cB:
            st.markdown("**Plot residual** (garis putus-putus = ±2·s(y/x))")
            vega({"layer": [
                {"data": {"values": rr}, "mark": {"type": "point", "filled": True, "color": "#f97316", "size": 60},
                 "encoding": {"x": X("C (mg/L P)"),
                              "y": {"field": "y", "type": "quantitative", "title": "Residual (A)"}}},
                rule(0, [1, 0]), rule(2 * fit.sy, [4, 4]), rule(-2 * fit.sy, [4, 4])]}, 380)
        with st.expander("Rekap per level & rumus"):
            st.dataframe(fit.levels, **W, hide_index=True)
            st.latex(r"b=\frac{\sum (x_i-\bar x)(y_i-\bar y)}{S_{xx}},\quad a=\bar y-b\bar x,\quad "
                     r"s_{y/x}=\sqrt{\frac{\sum (y_i-\hat y_i)^2}{n-2}}")
            st.latex(r"LOD=\frac{3{,}3\,s_{y/x}}{b},\qquad LOQ=\frac{10\,s_{y/x}}{b}")
            st.caption(f"n = {fit.n} pembacaan, derajat kebebasan = {fit.dof}, Sxx = {fit.sxx:.5f}")

# =========================================================================== 5. SAMPEL
results = {}
with tabs[4]:
    st.subheader("Data sampel & perhitungan kadar")
    st.caption("Faktor pengenceran DF = V akhir / V aliquot (pengenceran sebelum penambahan pereaksi; "
               "isi sama bila tanpa pengenceran). Penambahan pereaksi identik pada standar dan sampel, sehingga saling meniadakan.")
    st.session_state["smp"] = st.data_editor(
        st.session_state["smp"], num_rows="dynamic", **W, key="smp_ed",
        column_config={**{c: st.column_config.NumberColumn(format="%.4f") for c in ABS_COLS}})
    st.markdown("**Uji recovery (spike)** — opsional")
    rc1, rc2 = st.columns([2, 1])
    rec_txt = rc1.text_input("Recovery % (pisahkan koma)", "98.5, 101.2, 99.8, 97.9, 100.6")
    rec_corr = rc2.checkbox("Koreksi hasil dengan recovery", False)
    rec_vals = []
    for p in rec_txt.replace(";", ",").split(","):
        try:
            rec_vals.append(float(p))
        except ValueError:
            pass
    rec_cfg = {}
    if len(rec_vals) >= 2:
        rv = np.array(rec_vals) / 100
        rm, rs = rv.mean(), rv.std(ddof=1)
        u_r = rs / math.sqrt(len(rv)) / rm
        t_r = abs(1 - rm) / (rs / math.sqrt(len(rv))) if rs > 0 else np.inf
        t_c = t_ppf(0.975, len(rv) - 1)
        sig = t_r > t_c
        rec_cfg = dict(rec_mean=rm, rec_u_rel=u_r, rec_dof=len(rv) - 1, rec_correct=rec_corr)
        ok_rec = rec_lo <= rm * 100 <= rec_hi
        st.write(f"Recovery rata-rata = **{rm * 100:.1f}%**, SD = {rs * 100:.2f}%, n = {len(rv)}; "
                 f"{'memenuhi' if ok_rec else 'TIDAK memenuhi'} rentang {rec_lo}–{rec_hi}%. "
                 f"Uji-t terhadap 100%: {'berbeda nyata' if sig else 'tidak berbeda nyata'} "
                 f"(t = {t_r:.2f}, t kritis = {t_c:.2f}).")
    photometric = st.checkbox("Sertakan akurasi fotometrik UV-1800 sebagai komponen tambahan", False,
                              help="Sebagian sudah terwakili oleh sebaran residual regresi; centang hanya bila ingin konservatif.")

    st.markdown("**Larutan standar** — untuk anggaran ketidakpastian")
    s1, s2, s3 = st.columns(3)
    m_g = s1.number_input("Massa KH₂PO₄ (g)", 0.0001, 10.0, 0.2195, 0.0001, format="%.4f")
    purity = s2.number_input("Kemurnian (%)", 90.0, 100.0, 99.5, 0.1)
    purity_hw = s3.number_input("± ketidakpastian kemurnian (%)", 0.0, 5.0, 0.5, 0.1)
    s4, s5, s6 = st.columns(3)
    u_lin = s4.number_input("Linearitas neraca ± (g)", 0.00001, 0.01, 0.0002, 0.00005, format="%.5f")
    u_rep = s5.number_input("Repeatabilitas neraca, SD (g)", 0.00001, 0.01, 0.0001, 0.00005, format="%.5f")
    v_stock = s6.number_input("Labu stok (mL)", 10.0, 2000.0, 1000.0, 10.0)
    st.caption("Tahap pengenceran bersama (stok → larutan antara) yang berlaku untuk seluruh standar:")
    st.session_state["chain"] = st.data_editor(st.session_state["chain"], num_rows="dynamic",
                                               **W, key="chain_ed")

    if fit is None:
        st.error("Lengkapi data kalibrasi terlebih dahulu.")
    else:
        v_tol = glass_tol("flask", v_stock, glass_class)
        base_rows, c_stock, f_p = stock_relative_u(m_g, u_lin, u_rep, purity, purity_hw, v_stock, v_tol, dist)
        std_rows = [(n, u, t) for n, u, t in base_rows]
        for _, r in st.session_state["chain"].dropna(subset=["Pipet (mL)", "Labu (mL)"]).iterrows():
            tp = r.get("Toleransi pipet (mL)")
            tf = r.get("Toleransi labu (mL)")
            tp = glass_tol("pipet", r["Pipet (mL)"], glass_class) if pd.isna(tp) else tp
            tf = glass_tol("flask", r["Labu (mL)"], glass_class) if pd.isna(tf) else tf
            std_rows.append((f"{r['Tahap']} — pipet {r['Pipet (mL)']:g} mL",
                             u_from_tol(tp, dist) / r["Pipet (mL)"], "B"))
            std_rows.append((f"{r['Tahap']} — labu {r['Labu (mL)']:g} mL",
                             u_from_tol(tf, dist) / r["Labu (mL)"], "B"))
        cfg = dict(std_rows=std_rows, glass_class=glass_class, dist=dist, conf=conf, k_mode=k_mode,
                   photometric=photometric, **rec_cfg)

        rows, errs = [], []
        for idx, r in st.session_state["smp"].iterrows():
            if pd.isna(r.get("Nama sampel")) or str(r.get("Nama sampel")).strip() == "":
                continue
            try:
                ev = evaluate_sample(r, fit, cfg)
                base_name, n_dup = str(ev["name"]), 2
                while ev["name"] in results:  # nama sampel kembar -> beri nomor
                    ev["name"] = f"{base_name} ({n_dup})"
                    n_dup += 1
                results[ev["name"]] = ev
                xr, Ur, dec = sf_round(ev["U"], ev["c_p"])
                flag = []
                if not ev["in_range"]:
                    flag.append("di luar rentang kalibrasi")
                if not ev["above_loq"]:
                    flag.append("< LOQ")
                if ev["m"] > 1 and ev["rsd_rep"] > 5:
                    flag.append("RSD replikat >5%")
                rows.append({"Sampel": ev["name"], "n": ev["m"], "A rata-rata": round(ev["y0"], 4),
                             "SD": round(ev["sd_rep"], 4) if ev["m"] > 1 else np.nan,
                             "C larutan (mg/L P)": round(ev["x0"], 4), "DF": round(ev["df"], 3),
                             "Kadar P (mg/L)": f"{xr:.{dec}f}", "U (mg/L)": f"{Ur:.{dec}f}",
                             "Kadar PO₄ (mg/L)": f"{ev['c_po4']:.{dec}f}",
                             "Catatan": "; ".join(flag) or "OK"})
            except Exception as e:  # noqa: BLE001
                errs.append(f"{r.get('Nama sampel')}: {e}")
        for e in errs:
            st.warning(e)
        if rows:
            st.dataframe(pd.DataFrame(rows), **W, hide_index=True)
            st.latex(r"C_{P}=\frac{(\bar A_s-a)}{b}\times DF\ \ (\times 1/R),\qquad "
                     r"C_{PO_4}=C_P\times\frac{M_{PO_4}}{M_P}=C_P\times 3{,}066")
        st.caption(f"Larutan stok nominal terhitung = {c_stock:.2f} mg/L P (fraksi P dalam KH₂PO₄ = {f_p:.5f}). "
                   "Konsentrasi deret standar pada tab 4 diisi sesuai pembuatan sebenarnya.")

# =========================================================================== 6. KETIDAKPASTIAN
with tabs[5]:
    st.subheader("Estimasi ketidakpastian pengukuran")
    if not results:
        st.info("Belum ada hasil sampel yang valid.")
    else:
        pick = st.selectbox("Pilih sampel", list(results))
        ev = results[pick]
        b = ev["budget"].copy()
        xr, Ur, dec = sf_round(ev["U"], ev["c_p"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Kadar (mg/L P)", f"{xr:.{dec}f}")
        m2.metric("u_c (mg/L P)", f"{ev['u_c']:.{dec + 1}f}", f"{ev['u_c_rel'] * 100:.2f}% relatif")
        veff_txt = "∞" if not np.isfinite(ev["veff"]) else f"{ev['veff']:.1f}"
        m3.metric("k", f"{ev['k']:.2f}", f"v_eff = {veff_txt}")
        m4.metric("U (mg/L P)", f"{Ur:.{dec}f}", f"{ev['U_rel']:.2f}%")
        st.markdown(f"<div class='hasil'>Kadar fosfat (sebagai P) = <b>{xr:.{dec}f} ± {Ur:.{dec}f} mg/L</b> "
                    f"(k = {ev['k']:.2f}, tingkat kepercayaan {conf:.0%})</div>", unsafe_allow_html=True)
        show = b[["Sumber", "Tipe", "Distribusi", "Nilai", "u_rel", "u_i (mg/L P)", "Kontribusi (%)", "dof"]].copy()
        show["Nilai"] = show["Nilai"].map(lambda v: f"{v:.3e}")
        show["u_rel"] = show["u_rel"].map(lambda v: f"{v:.3e}")
        show["u_i (mg/L P)"] = show["u_i (mg/L P)"].map(lambda v: f"{v:.3e}")
        show["Kontribusi (%)"] = show["Kontribusi (%)"].map(lambda v: f"{v:.1f}")
        show["dof"] = show["dof"].map(lambda v: "∞" if not np.isfinite(v) else f"{v:.0f}")
        st.dataframe(show, **W, hide_index=True)
        bs = b.sort_values("Kontribusi (%)")
        st.markdown("**Kontribusi terhadap ketidakpastian gabungan (% dari u_c²)**")
        vega({"layer": [{
            "data": {"values": [{"s": str(r_["Sumber"]), "c": float(r_["Kontribusi (%)"])}
                                for _, r_ in bs.iterrows()]},
            "mark": {"type": "bar", "color": "#38bdf8"},
            "encoding": {"y": {"field": "s", "type": "nominal", "sort": "-x", "title": None,
                               "axis": {"labelLimit": 420}},
                         "x": {"field": "c", "type": "quantitative", "title": "%"}}}]},
             60 + 34 * len(bs))
        top = b.sort_values("Kontribusi (%)", ascending=False).iloc[0]
        st.info(f"Kontributor terbesar: **{top['Sumber']}** ({top['Kontribusi (%)']:.1f}%).")
        with st.expander("Model & rumus"):
            st.latex(r"C=\frac{\bar A_s-a}{b}\cdot\frac{V_2}{V_1}\cdot\frac{1}{R}")
            st.latex(r"u(x_0)=\frac{s_{y/x}}{b}\sqrt{\frac1m+\frac1n+\frac{(\bar A_s-\bar y)^2}{b^2S_{xx}}}")
            st.latex(r"\frac{u_c(C)}{C}=\sqrt{\left(\tfrac{u(x_0)}{x_0}\right)^2+\sum\left(\tfrac{u(q_i)}{q_i}\right)^2},"
                     r"\quad U=k\,u_c,\quad v_{eff}=\frac{u_c^4}{\sum u_i^4/v_i}")
            st.markdown(
                "- Repeatabilitas pembacaan sampel sudah termasuk pada suku 1/m di u(x₀) (tidak dihitung ganda).\n"
                "- Galat acak penyiapan tiap standar terwakili sebaran residual regresi; hanya bagian bersama "
                "(stok & pengenceran antara) ditambahkan terpisah.\n"
                "- Rumus pembulatan: U ke 2 angka penting, hasil mengikuti desimal U.")

# =========================================================================== 7. PELAPORAN
def build_report_html(fit, results, cfg_text):
    parts = [f"<h1>Laporan Penetapan Kadar Fosfat (PO₄-P)</h1>",
             f"<p><b>Instrumen:</b> Shimadzu UV-1800 (No. seri: {sn or '-'}) &nbsp; <b>λ:</b> {lam} nm &nbsp; "
             f"<b>Sel:</b> {path:g} cm<br><b>Analis:</b> {analis or '-'} &nbsp; <b>Laboratorium:</b> {lab or '-'} &nbsp; "
             f"<b>Tanggal:</b> {tgl}<br><b>Acuan metode:</b> {sop}</p>",
             "<h2>1. Kurva kalibrasi</h2>",
             f"<p>A = {fit.b:.5f}·C {'+' if fit.a >= 0 else '−'} {abs(fit.a):.5f}; R² = {fit.r2:.5f} "
             f"(kriteria ≥ {r2_min:.3f}: {'memenuhi' if fit.r2 >= r2_min else 'tidak memenuhi'}); "
             f"s(y/x) = {fit.sy:.5f}; n = {fit.n}; LOD = {fit.lod:.4f} mg/L; LOQ = {fit.loq:.4f} mg/L.</p>",
             fit.levels.round(5).to_html(index=False, border=0)]
    parts.append("<h2>2. Hasil sampel</h2>")
    for name, ev in results.items():
        xr, Ur, dec = sf_round(ev["U"], ev["c_p"])
        parts.append(f"<h3>{name}</h3><p>A rata-rata = {ev['y0']:.4f} (n = {ev['m']}); C larutan = {ev['x0']:.4f} mg/L; "
                     f"DF = {ev['df']:.3f}.<br><b>Kadar fosfat (sebagai P) = {xr:.{dec}f} ± {Ur:.{dec}f} mg/L</b> "
                     f"(k = {ev['k']:.2f}, tingkat kepercayaan {conf:.0%}); setara PO₄³⁻ = "
                     f"{ev['c_po4']:.{dec}f} ± {Ur * P_TO_PO4:.{dec}f} mg/L.</p>")
        b = ev["budget"][["Sumber", "Tipe", "u_rel", "u_i (mg/L P)", "Kontribusi (%)"]].copy()
        b["u_rel"] = b["u_rel"].map(lambda v: f"{v:.3e}")
        b["u_i (mg/L P)"] = b["u_i (mg/L P)"].map(lambda v: f"{v:.3e}")
        b["Kontribusi (%)"] = b["Kontribusi (%)"].map(lambda v: f"{v:.1f}")
        parts.append(b.to_html(index=False, border=0))
    parts.append(f"<h2>3. Catatan</h2><p>{cfg_text}</p>")
    css = ("<style>body{font-family:Arial,sans-serif;max-width:900px;margin:30px auto;color:#111}"
           "table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}"
           "th,td{border:1px solid #bbb;padding:4px 8px;text-align:left}th{background:#eef}"
           "h1{border-bottom:3px solid #1b4fd8}h2{color:#1b4fd8}</style>")
    return f"<!doctype html><html><head><meta charset='utf-8'><title>Laporan Fosfat</title>{css}</head><body>{''.join(parts)}</body></html>"


with tabs[6]:
    st.subheader("Pelaporan")
    if fit is None or not results:
        st.info("Lengkapi data kalibrasi dan sampel terlebih dahulu.")
    else:
        for name, ev in results.items():
            xr, Ur, dec = sf_round(ev["U"], ev["c_p"])
            st.markdown(
                f"**{name}** — kadar fosfat (sebagai P) = **{xr:.{dec}f} ± {Ur:.{dec}f} mg/L** "
                f"(setara PO₄³⁻ {ev['c_po4']:.{dec}f} ± {Ur * P_TO_PO4:.{dec}f} mg/L); "
                f"k = {ev['k']:.2f}, tingkat kepercayaan {conf:.0%}.")
        st.markdown("##### Pemeriksaan mutu")
        qc = [("Linearitas (R²)", f"{fit.r2:.5f}", fit.r2 >= r2_min, f"≥ {r2_min:.3f}")]
        if rec_cfg:
            qc.append(("Recovery rata-rata", f"{rec_cfg['rec_mean'] * 100:.1f}%",
                       rec_lo <= rec_cfg["rec_mean"] * 100 <= rec_hi, f"{rec_lo}–{rec_hi}%"))
        for nm, ev in results.items():
            qc.append((f"{nm}: di dalam rentang kalibrasi", "ya" if ev["in_range"] else "tidak", ev["in_range"],
                       f"{fit.x.min():g}–{fit.x.max():g} mg/L"))
            qc.append((f"{nm}: ≥ LOQ", f"{ev['x0']:.4f} vs {fit.loq:.4f}", ev["above_loq"], "C ≥ LOQ"))
        st.dataframe(pd.DataFrame([{"Parameter": a, "Hasil": b, "Status": "✅ Lulus" if c else "❌ Perlu ditinjau",
                                    "Kriteria": d} for a, b, c, d in qc]),
                     **W, hide_index=True)

        cfg_text = (f"Ketidakpastian diperluas U = k·u_c; k dari {k_mode}. Komponen: kurva kalibrasi (regresi), "
                    f"larutan standar (penimbangan, kemurnian, volumetri), pengenceran sampel"
                    f"{', recovery' if rec_cfg else ''}{', akurasi fotometrik' if photometric else ''}.")
        html = build_report_html(fit, results, cfg_text)
        meta = pd.DataFrame({"Item": ["Instrumen", "No. seri", "λ (nm)", "Sel (cm)", "Analis", "Laboratorium",
                                      "Tanggal", "Acuan metode", "Tingkat kepercayaan", "k"],
                             "Nilai": ["Shimadzu UV-1800", sn, lam, path, analis, lab, str(tgl), sop, conf, k_mode]})
        reg = pd.DataFrame({"Parameter": ["b", "a", "R2", "s(y/x)", "n", "LOD", "LOQ"],
                            "Nilai": [fit.b, fit.a, fit.r2, fit.sy, fit.n, fit.lod, fit.loq]})
        summ = pd.DataFrame([{"Sampel": n, "A rata-rata": e["y0"], "C larutan (mg/L P)": e["x0"], "DF": e["df"],
                              "Kadar P (mg/L)": e["c_p"], "u_c": e["u_c"], "k": e["k"], "U": e["U"],
                              "Kadar PO4 (mg/L)": e["c_po4"]} for n, e in results.items()])
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, d_ in (("metadata.csv", meta), ("data_standar.csv", st.session_state["std"]),
                              ("regresi.csv", reg), ("hasil.csv", summ)):
                zf.writestr(fname, d_.to_csv(index=False))
            for i_, (n, e) in enumerate(results.items(), 1):
                zf.writestr(f"anggaran_ketidakpastian_{i_}.csv", e["budget"].to_csv(index=False))
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇ Laporan (HTML, bisa dicetak ke PDF)", html, "laporan_fosfat_uv1800.html", "text/html")
        d2.download_button("⬇ Semua data & anggaran (ZIP berisi CSV)", zbuf.getvalue(), "fosfat_uv1800.zip",
                           "application/zip")
        d3.download_button("⬇ Ringkasan hasil (CSV)", summ.to_csv(index=False), "hasil_fosfat.csv", "text/csv")
        with st.expander("Pratinjau laporan"):
            embed_html(html, 700)
