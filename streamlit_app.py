"""Streamlit: Penetapan Kadar Fosfat (Air & Pupuk) dengan Spektrofotometer UV-Vis Shimadzu UV-1800.

Menjalankan:  streamlit run app.py
"""
import datetime as dt
import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ======================================================================
# BAGIAN 1 — PERHITUNGAN & STATISTIK
# ======================================================================

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
    df = max(float(df), 1.0)
    lo, hi = 0.0, 1e4
    for _ in range(200):
        mid = (lo + hi) / 2
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


CONC_COL = "Konsentrasi (mg/L P)"
ABS_COLS = ["Abs 1", "Abs 2", "Abs 3"]

ATOMIC = {
    "K": (39.0983, 0.0001),
    "H": (1.00794, 0.00007),
    "P": (30.973762, 0.000002),
    "O": (15.9994, 0.0003),
}
M_P = ATOMIC["P"][0]
M_PO4 = M_P + 4 * ATOMIC["O"][0]
P_TO_PO4 = M_PO4 / M_P  # ~3.0661


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
    r2: float
    dof: int
    loq: float


def long_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cols = [c for c in ABS_COLS if c in df.columns]
    d = df[[CONC_COL] + cols].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=[CONC_COL])
    m = d.melt(id_vars=CONC_COL, value_vars=cols).dropna()
    return m[CONC_COL].to_numpy(float), m["value"].to_numpy(float)


def fit_calibration(df: pd.DataFrame) -> CalFit:
    x, y = long_data(df)
    n = len(x)
    if n < 3 or len(np.unique(x)) < 2:
        raise ValueError("Data kalibrasi tidak cukup.")
    xbar, ybar = x.mean(), y.mean()
    sxx = float(((x - xbar) ** 2).sum())
    b = float(((x - xbar) * (y - ybar)).sum() / sxx)
    a = float(ybar - b * xbar)
    resid = y - (a + b * x)
    ssr = float((resid ** 2).sum())
    dof = n - 2
    sy = math.sqrt(ssr / dof) if dof > 0 else 0.001
    sst = float(((y - ybar) ** 2).sum())
    r2 = 1 - ssr / sst if sst > 0 else 1.0
    return CalFit(x, y, n, xbar, ybar, sxx, b, a, sy, max(r2, 0), dof, 10 * sy / b if b != 0 else 0)


# ======================================================================
# BAGIAN 2 — SKEMA 3D INTERAKTIF (CSS & HTML SIMULASI 3D UV-1800)
# ======================================================================

_HTML_3D = r"""
<!doctype html><html><head><meta charset="utf-8">
<style>
:root{--beam:#38bdf8;--bg:#0f172a;--card:#1e293b;--ink:#f8fafc;--mut:#94a3b8;--acc:#0ea5e9}
body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,sans-serif}
.wrap{padding:12px}
.box-3d{
  perspective: 1000px; background: var(--card); border: 1px solid #334155;
  border-radius: 12px; padding: 15px; box-shadow: 0 10px 25px -5px rgba(0,0,0,0.5);
  transform-style: preserve-3d; transition: transform 0.5s ease;
}
.spectro-model {
  display: flex; align-items: center; justify-content: space-between;
  background: linear-gradient(135deg, #1e293b, #0f172a);
  border: 2px solid #475569; border-radius: 10px; padding: 20px;
  box-shadow: inset 0 2px 10px rgba(0,0,0,0.6); position: relative;
}
.module {
  background: #334155; border: 1px solid #64748b; border-radius: 6px;
  padding: 10px; text-align: center; font-size: 11px; color: #e2e8f0;
  box-shadow: 0 4px 6px rgba(0,0,0,0.3); transform: translateZ(20px);
}
.light-beam {
  height: 4px; background: var(--beam); box-shadow: 0 0 12px var(--beam);
  flex-grow: 1; margin: 0 5px; animation: pulse 1.5s infinite alternate;
}
@keyframes pulse { from { opacity: 0.6; } to { opacity: 1; } }
.controls-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-top: 15px; }
.control-card { background: #1e293b; border: 1px solid #334155; padding: 10px; border-radius: 8px; }
.control-card label { font-size: 12px; color: var(--mut); display: block; margin-bottom: 5px; }
input[type=range] { width: 100%; accent-color: var(--acc); }
.readout { font-family: monospace; font-size: 20px; color: #4ade80; font-weight: bold; }
</style></head>
<body>
<div class="wrap">
  <div class="box-3d">
    <h3 style="margin-top:0; color:#38bdf8;">Simulasi Visual 3D Spektrofotometer UV-1800</h3>
    <div class="spectro-model">
      <div class="module">Sumber<br>(Lampu WI/D₂)</div>
      <div class="light-beam"></div>
      <div class="module">Monokromator<br>(Kisi Difraksi)</div>
      <div class="light-beam"></div>
      <div class="module">Kuvet Sampel<br>(Air / Pupuk)</div>
      <div class="light-beam"></div>
      <div class="module">Detektor<br>(Fotodioda Si)</div>
    </div>
    
    <div class="controls-grid">
      <div class="control-card">
        <label>Panjang Gelombang ($\lambda$): <span id="lval" style="color:#fff;">880</span> nm</label>
        <input id="lam" type="range" min="190" max="1100" value="880">
      </div>
      <div class="control-card">
        <label>Konsentrasi Simulasi: <span id="cval" style="color:#fff;">0.50</span> mg/L P</label>
        <input id="conc" type="range" min="0" max="2.0" step="0.01" value="0.5">
      </div>
      <div class="control-card" style="text-align:center;">
        <label>Nilai Absorbansi Terbaca (A)</label>
        <div class="readout" id="rA">0.345</div>
      </div>
    </div>
  </div>
</div>
<script>
const lam = document.getElementById('lam');
const conc = document.getElementById('conc');
const lval = document.getElementById('lval');
const cval = document.getElementById('cval');
const rA = document.getElementById('rA');

function updateSim() {
  lval.innerText = lam.value;
  cval.innerText = (+conc.value).toFixed(2);
  // Asumsi slope ~0.69 pada 880nm
  let slope = 0.69;
  let absorbance = slope * (+conc.value) * Math.exp(-Math.pow((+lam.value - 880)/80, 2));
  rA.innerText = absorbance.toFixed(3);
}

lam.oninput = updateSim;
conc.oninput = updateSim;
updateSim();
</script>
</body></html>
"""

# ======================================================================
# BAGIAN 3 — ANTARMUKA UTAMA STREAMLIT
# ======================================================================

st.set_page_config(page_title="UV-1800 | Analisis Fosfat Air & Pupuk", page_icon="🌾", layout="wide")

st.title("🧪 Analisis Kadar Fosfat ($PO_4-P$) Air & Pupuk — Shimadzu UV-1800")
st.caption("Modul Interaktif 3D • Input Data Mandiri • Kurva Kalibrasi • Perhitungan Sampel Air & Pupuk")

# Inisialisasi State Data Sampel & Standar Sendiri
if "std_data" not in st.session_state:
    st.session_state["std_data"] = pd.DataFrame({
        CONC_COL: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "Abs 1": [0.001, 0.140, 0.278, 0.415, 0.552, 0.690],
        "Abs 2": [0.002, 0.142, 0.280, 0.417, 0.554, 0.692],
        "Abs 3": [0.001, 0.141, 0.279, 0.416, 0.553, 0.691]
    })

if "water_data" not in st.session_state:
    st.session_state["water_data"] = pd.DataFrame({
        "Nama Sampel Air": ["Air Sumur Warga", "Air Sungai Irigasi"],
        "V aliquot (mL)": [25.0, 10.0],
        "V akhir (mL)": [50.0, 50.0],
        "Abs 1": [0.215, 0.380],
        "Abs 2": [0.217, 0.382],
        "Abs 3": [0.216, 0.381]
    })

if "fertilizer_data" not in st.session_state:
    st.session_state["fertilizer_data"] = pd.DataFrame({
        "Nama Sampel Pupuk": ["Pupuk NPK Mutiara", "Pupuk SP-36 Lokal"],
        "Massa Timbang (g)": [0.50, 0.25],
        "V Ekstraksi (mL)": [250.0, 250.0],
        "V Aliquot Pengenceran (mL)": [5.0, 10.0],
        "V Akhir Pengenceran (mL)": [50.0, 50.0],
        "Abs 1": [0.450, 0.520],
        "Abs 2": [0.452, 0.522],
        "Abs 3": [0.451, 0.521]
    })

tabs = st.tabs([
    "1 · Skema 3D Alat", 
    "2 · Data Standar Kalibrasi", 
    "3 · Input Sampel Air", 
    "4 · Input Sampel Pupuk", 
    "5 · Hasil & Perhitungan"
])

# --- TAB 1: SKEMA 3D ---
with tabs[0]:
    st.subheader("Visualisasi Model 3D Instrumen UV-1800")
    st.markdown("Gunakan kontrol interaktif di bawah untuk menguji respons optik instrumen terhadap perubahan panjang gelombang dan konsentrasi larutan analit.")
    components.html(_HTML_3D, height=360)

# --- TAB 2: DATA STANDAR ---
with tabs[1]:
    st.subheader("Pengaturan Data Deret Standar Kalibrasi")
    st.markdown("Silakan ubah atau masukkan sendiri nilai konsentrasi dan absorbansi pembacaan alat sesuai data riil Anda.")
    st.session_state["std_data"] = st.data_editor(st.session_state["std_data"], num_rows="dynamic", use_container_width=True)

    try:
        fit = fit_calibration(st.session_state["std_data"])
        st.success(f"Persamaan Kalibrasi Valid: $y = {fit.b:.4f}x + {fit.a:.4f}$ | $R^2 = {fit.r2:.5f}$")
    except Exception as e:
        st.warning(f"Belum dapat membentuk kurva: {e}")

# --- TAB 3: SAMPEL AIR ---
with tabs[2]:
    st.subheader("Input Data Sampel Air (Sungai / Limbah / Sumur)")
    st.markdown("Masukkan parameter sampel air, volume pengambilan (*aliquot*), volume labu ukur akhir, serta data absorbansi replikasinya.")
    st.session_state["water_data"] = st.data_editor(st.session_state["water_data"], num_rows="dynamic", use_container_width=True)

# --- TAB 4: SAMPEL PUPUK ---
with tabs[3]:
    st.subheader("Input Data Sampel Pupuk (NPK / SP-36 / Fosfat Alam)")
    st.markdown("Masukkan massa sampel pupuk yang ditimbang, volume total ekstraksi, dan volume pengenceran lanjutan sebelum pengukuran warna biru molibdenum.")
    st.session_state["fertilizer_data"] = st.data_editor(st.session_state["fertilizer_data"], num_rows="dynamic", use_container_width=True)

# --- TAB 5: PERHITUNGAN & LAPORAN AKHIR ---
with tabs[4]:
    st.subheader("Hasil Perhitungan Kadar Fosfat ($PO_4-P$ & $P_2O_5$ Total)")
    
    try:
        fit = fit_calibration(st.session_state["std_data"])
        
        # Hitung Air
        st.markdown("### 🌊 Hasil Analisis Sampel Air")
        water_df = st.session_state["water_data"]
        water_results = []
        for idx, row in water_df.iterrows():
            abs_cols_present = [c for c in ABS_COLS if c in row.index]
            vals = pd.to_numeric(row[abs_cols_present], errors="coerce").dropna().to_numpy()
            if len(vals) > 0:
                y_mean = vals.mean()
                # x0 = (y - a) / b (mg/L P)
                x0 = (y_mean - fit.a) / fit.b
                v_aliquot = float(row.get("V aliquot (mL)", 1.0))
                v_akhir = float(row.get("V akhir (mL)", 1.0))
                faktor_pengenceran = v_akhir / v_aliquot if v_aliquot > 0 else 1.0
                kadar_p = x0 * faktor_pengenceran
                kadar_po4 = kadar_p * 3.0661 # Konversi P ke PO4
                water_results.append({
                    "Nama Sampel": row.get("Nama Sampel Air", f"Sampel {idx+1}"),
                    "Abs Rata-rata": round(y_mean, 4),
                    "Kadar P (mg/L)": round(kadar_p, 3),
                    "Kadar PO4 (mg/L)": round(kadar_po4, 3)
                })
        if water_results:
            st.dataframe(pd.DataFrame(water_results), use_container_width=True)
        else:
            st.info("Belum ada data sampel air yang valid.")

        # Hitung Pupuk
        st.markdown("### 🌾 Hasil Analisis Sampel Pupuk (% P / % $P_2O_5$)")
        fert_df = st.session_state["fertilizer_data"]
        fert_results = []
        for idx, row in fert_df.iterrows():
            abs_cols_present = [c for c in ABS_COLS if c in row.index]
            vals = pd.to_numeric(row[abs_cols_present], errors="coerce").dropna().to_
