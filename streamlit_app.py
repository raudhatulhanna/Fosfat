import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Kalkulator Analisis & Validasi Metode", page_icon="🧪", layout="wide")

st.title("🧪 Kalkulator Analisis Kuantitatif & Validasi Metode")
st.markdown("Aplikasi web interaktif untuk perhitungan kurva kalibrasi, kadar sampel, $\% \text{Recovery}$, $\% \text{RSD}$, dan estimasi LOD/LOQ.")

# Sidebar Navigasi Menu
menu = st.sidebar.selectbox("Pilih Menu Perhitungan:", [
    "1. Kurva Kalibrasi & Regresi Linear",
    "2. Penentuan Kadar & Perhitungan Sampel",
    "3. Uji Akurasi (% Recovery)",
    "4. Uji Presisi (% RSD / % RPD)",
    "5. Estimasi LOD & LOQ"
])

# -------------------------------------------------------------------------
# MENU 1: KURVA KALIBRASI & REGRESI LINEAR
# -------------------------------------------------------------------------
if menu == "1. Kurva Kalibrasi & Regresi Linear":
    st.header("📈 Kurva Kalibrasi & Linearitas")
    st.write("Masukkan data konsentrasi standar ($X$) dan respon instrumen/absorbansi ($Y$).")

    # Input data standar dalam bentuk tabel interaktif
    num_points = st.number_input("Jumlah Titik Standar", min_value=3, max_value=10, value=5)
    
    default_data = pd.DataFrame({
        "Konsentrasi (ppm/mg/L)": [0.0, 2.0, 4.0, 6.0, 8.0][:num_points],
        "Absorbansi (Y)": [0.0, 0.150, 0.310, 0.455, 0.600][:num_points]
    })

    edited_df = st.data_editor(default_data, num_rows="dynamic")

    if st.button("Hitung Regresi Linear"):
        x = edited_df["Konsentrasi (ppm/mg/L)"].values
        y = edited_df["Absorbansi (Y)"].values

        # Perhitungan Regresi Linear (Least Squares)
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_xx = np.sum(x**2)
        sum_yy = np.sum(y**2)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x**2)
        intercept = (sum_y - slope * sum_x) / n
        r_value = (n * sum_xy - sum_x * sum_y) / np.sqrt((n * sum_xx - sum_x**2) * (n * sum_yy - sum_y**2))
        r_squared = r_value**2

        st.success("Perhitungan Berhasil!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Slope ($m$)", f"{slope:.4f}")
        col2.metric("Intercep ($c$)", f"{intercept:.4f}")
        col3.metric("Koefisien Korelasi ($r$)", f"{r_value:.4f}")
        st.info(f"Koefisien Determinasi ($R^2$): **{r_squared:.4f}**")
        st.latex(f"y = {slope:.4f}x + ({intercept:.4f})")

        # Plot Grafik Linearitas
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x, y, color='red', label='Data Pengamatan', zorder=5)
        
        # Garis regresi
        y_pred = slope * x + intercept
        ax.plot(x, y_pred, color='blue', label=f'Regresi: y = {slope:.4f}x + {intercept:.4f}')
        
        ax.set_xlabel("Konsentrasi")
        ax.set_xlabel("Konsentrasi")
        ax.set_ylabel("Absorbansi / Respon")
        ax.set_title("Kurva Kalibrasi Linearitas")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)

# -------------------------------------------------------------------------
# MENU 2: PENENTUAN KADAR SAMPEL
# -------------------------------------------------------------------------
elif menu == "2. Penentuan Kadar & Perhitungan Sampel":
    st.header("🧪 Perhitungan Konsentrasi Sampel")
    st.write("Menghitung kadar sampel berdasarkan persamaan kurva kalibrasi $y = mx + c$.")

    col1, col2 = st.columns(2)
    with col1:
        slope = st.number_input("Masukkan Slope ($m$)", value=0.075)
        intercept = st.number_input("Masukkan Intercep ($c$)", value=0.005)
    with col2:
        fp = st.number_input("Faktor Pengenceran (FP)", value=1.0)
        vol_smpl = st.number_input("Volume / Bobot Sampel", value=10.0)

    absorbansi_sampel = st.number_input("Masukkan Nilai Absorbansi/Respon Sampel ($Y$)", value=0.325)

    if st.button("Hitung Kadar"):
        # x = (y - c) / m
        kadar_baca = (absorbansi_sampel - intercept) / slope
        kadar_akhir = kadar_baca * fp

        st.markdown("---")
        st.subheader("Hasil Perhitungan:")
        st.write(f"Konsentrasi dari kurva kalibrasi ($x$): **{kadar_baca:.4f} ppm**")
        st.write(f"Konsentrasi Sampel Akhir (setelah dikali FP {fp}): **{kadar_akhir:.4f} ppm**")

# -------------------------------------------------------------------------
# MENU 3: UJI AKURASI (% RECOVERY)
# -------------------------------------------------------------------------
elif menu == "3. Uji Akurasi (% Recovery)":
    st.header("🎯 Perhitungan $\% \text{Recovery}$ (Akurasi)")
    st.write("Rumus: $\% \text{Recovery} = \frac{C_{\text{terukur}} - C_{\text{awal}}}{C_{\text{spike}}} \times 100\%$")

    c_awal = st.number_input("Konsentrasi Sampel Tanpa Penambahan ($C_{\text{awal}}$)", value=5.0)
    c_tambah = st.number_input("Konsentrasi Baku yang Ditambahkan ($C_{\text{spike}}$)", value=5.0)
    c_ukur = st.number_input("Konsentrasi Sampel Setelah Ditambah Baku ($C_{\text{terukur}}$)", value=9.8)

    if st.button("Hitung % Recovery"):
        recovery = ((c_ukur - c_awal) / c_tambah) * 100
        st.metric("% Recovery", f"{recovery:.2f}%")
        
        if 80 <= recovery <= 120:  # Contoh rentang penerimaan umum AOAC/AOCS
            st.success("✅ Nilai % Recovery berada dalam rentang keberterimaan umum (80% - 120%).")
        else:
            st.warning("⚠️ Nilai % Recovery di luar rentang standar umum.")

# -------------------------------------------------------------------------
# MENU 4: UJI PRESISI (% RSD atau % RPD)
# -------------------------------------------------------------------------
elif menu == "4. Uji Presisi (% RSD / % RPD)":
    st.header("📊 Perhitungan Presisi ($\% \text{RSD}$ atau $\% \text{RPD}$)")
    
    jenis_uji = st.radio("Pilih Jenis Uji Presisi:", ["% RSD (Replikasi Banyak)", "% RPD (Duplo / 2 Pengujian)"])

    if jenis_uji == "% RSD (Replikasi Banyak)":
        data_input = st.text_input("Masukkan data replikasi (pisahkan dengan koma)", "10.2, 10.4, 10.1, 10.3, 10.2")
        if st.button("Hitung % RSD"):
            try:
                vals = [float(i.strip()) for i in data_input.split(",")]
                mean_val = np.mean(vals)
                sd_val = np.std(vals, ddof=1) # Standar deviasi sampel
                rsd_val = (sd_val / mean_val) * 100

                st.write(o for o in []) # dummy spacing
                col1, col2, col3 = st.columns(3)
                col1.metric("Rata-rata (Mean)", f"{mean_val:.4f}")
                col2.metric("Standar Deviasi (SD)", f"{sd_val:.4f}")
                col3.metric("% RSD", f"{rsd_val:.2f}%")
            except ValueError:
                st.error("Format data salah. Pastikan angka dipisahkan dengan koma.")
    
    else:
        c1 = st.number_input("Hasil Pengujian 1 ($x_1$)", value=10.2)
        c2 = st.number_input("Hasil Pengujian 2 ($x_2$)", value=10.5)
        if st.button("Hitung % RPD"):
            rpd = (abs(c1 - c2) / ((c1 + c2) / 2)) * 100
            st.metric("% RPD", f"{rpd:.2f}%")

# -------------------------------------------------------------------------
# MENU 5: ESTIMASI LOD & LOQ
# -------------------------------------------------------------------------
elif menu == "5. Estimasi LOD & LOQ":
    st.header("📉 Estimasi Limit of Detection (LOD) & Limit of Quantification (LOQ)")
    st.write("Dihitung berdasarkan Standar Deviasi Blangko ($SD$) dan *Slope* ($m$) kurva kalibrasi.")
    st.latex(r"LOD = \frac{3.3 \times SD}{m}, \quad LOQ = \frac{10 \times SD}{m}")

    sd_val = st.number_input("Standar Deviasi ($SD$) Blangko / Simpangan Baku Residual", value=0.012)
    slope_val = st.number_input("Nilai Slope ($m$) dari Kurva Kalibrasi", value=0.075)

    if st.button("Hitung LOD & LOQ"):
        lod = (3.3 * sd_val) / slope_val
        loq = (10.0 * sd_val) / slope_val

        col1, col2 = st.columns(2)
        col1.metric("LOD (Batas Deteksi)", f"{lod:.4f} ppm")
        col2.metric("LOQ (Batas Kuantisasi)", f"{loq:.4f} ppm")
