import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Yogyakarta Air Quality Forecast", page_icon="🌫️", layout="centered")

DATA_PATH = "data_source/yogyakarta_air_quality_2024.csv"
MODEL_PATH = "model.joblib"


def pm25_category(value: float) -> tuple[str, str]:
    """Return (label, color) per Indonesian ISPU-style PM2.5 breakpoints (µg/m³, 24h avg)."""
    if value is None or np.isnan(value):
        return "Tidak diketahui", "#6b7280"
    if value <= 15.5:
        return "Baik", "#22c55e"
    if value <= 55.4:
        return "Sedang", "#facc15"
    if value <= 150.4:
        return "Tidak Sehat", "#f97316"
    if value <= 250.4:
        return "Sangat Tidak Sehat", "#ef4444"
    return "Berbahaya", "#7f1d1d"


st.markdown(
    """
    <style>
    .aq-hero {
        background: linear-gradient(135deg, #0c4a6e 0%, #0369a1 50%, #0ea5e9 100%);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .aq-hero h1 { color: white; font-size: 2rem; font-weight: 800; margin-bottom: 0.3rem; }
    .aq-hero p { color: #bae6fd; font-size: 1rem; margin: 0; }
    .aq-result {
        text-align: center;
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        margin: 1rem 0;
    }
    .aq-result .num { font-size: 2.8rem; font-weight: 800; }
    .aq-result .label { font-size: 1.1rem; font-weight: 600; margin-top: 0.2rem; }
    .aq-result .unit { font-size: 0.85rem; opacity: 0.9; }
    </style>

    <div class="aq-hero">
        <h1>🌫️ Yogyakarta Air Quality Forecast</h1>
        <p>Prediksi konsentrasi PM2.5 esok hari berdasarkan data historis 2024</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not (os.path.exists(DATA_PATH) and os.path.exists(MODEL_PATH)):
    st.error(
        "Data atau model belum tersedia. Jalankan `python extract_data.py` lalu "
        "`python train.py` terlebih dahulu di root proyek ini."
    )
    st.stop()


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
bundle = load_model()
model = bundle["model"]
feature_cols = bundle["feature_cols"]
medians = bundle["medians"]

tab1, tab2 = st.tabs(["📈 Tren & Prediksi", "🔮 Simulasi Manual"])

#historical trend + rolling forecast on real data
with tab1:
    st.markdown("##### Tren PM2.5 Tahun 2024 (Stasiun AQMS Gondokusuman)")
    chart_df = df[["date", "pm25"]].dropna()
    st.line_chart(chart_df.set_index("date"))

    st.markdown("##### Prediksi vs Aktual (30 hari terakhir dengan data lengkap)")
    recent = df.dropna(subset=["pm25"]).copy().reset_index(drop=True)
    recent["pm25_lag1"] = recent["pm25"].shift(1)
    recent["pm25_roll3"] = recent["pm25"].shift(1).rolling(3, min_periods=1).mean()
    recent["month"] = recent["date"].dt.month
    recent = recent.dropna(subset=["pm25_lag1", "pm25_roll3"]).tail(30)

    if len(recent) > 0:
        x = recent[feature_cols].fillna(pd.Series(medians))
        recent = recent.assign(prediksi_besok=model.predict(x))
        plot_df = recent[["date", "pm25", "prediksi_besok"]].set_index("date")
        plot_df.columns = ["PM2.5 aktual", "Prediksi (untuk H+1)"]
        st.line_chart(plot_df)
        st.caption(
            "'Prediksi (untuk H+1)' pada tanggal X adalah perkiraan model untuk PM2.5 "
            "keesokan harinya, dibuat menggunakan data yang tersedia hingga tanggal X."
        )
    else:
        st.info("Belum cukup data historis berurutan untuk menampilkan perbandingan ini.")

    latest_valid = df.dropna(subset=["pm25"]).iloc[-1]
    latest_date = latest_valid["date"].date()
    cat_label, cat_color = pm25_category(latest_valid["pm25"])
    st.markdown(
        f"""
        <div class="aq-result" style="background:{cat_color};">
            <div class="num">{latest_valid['pm25']:.1f}</div>
            <div class="unit">µg/m³ · PM2.5 pada {latest_date}</div>
            <div class="label">Kategori: {cat_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

#manual simulation
with tab2:
    st.markdown("##### Masukkan kondisi hari ini untuk memprediksi PM2.5 besok")
    col1, col2 = st.columns(2)
    with col1:
        pm25_today = st.number_input(
            "PM2.5 hari ini (µg/m³)", min_value=0.0, max_value=300.0,
            value=float(round(df["pm25"].dropna().median(), 1)), step=0.5,
        )
        pm25_yesterday = st.number_input(
            "PM2.5 kemarin (µg/m³)", min_value=0.0, max_value=300.0,
            value=float(round(df["pm25"].dropna().median(), 1)), step=0.5,
        )
    with col2:
        pm25_3day_avg = st.number_input(
            "Rata-rata PM2.5, 3 hari terakhir (µg/m³)", min_value=0.0, max_value=300.0,
            value=float(round(df["pm25"].dropna().median(), 1)), step=0.5,
        )
        month = st.selectbox(
            "Bulan", options=list(range(1, 13)),
            format_func=lambda m: pd.Timestamp(2024, m, 1).strftime("%B"),
            index=8,
        )

    if st.button("🔮 Prediksi PM2.5 Besok", use_container_width=True):
        x_input = pd.DataFrame(
            [{"pm25": pm25_today, "pm25_lag1": pm25_yesterday, "pm25_roll3": pm25_3day_avg, "month": month}]
        )[feature_cols]
        prediction = float(model.predict(x_input)[0])
        cat_label, cat_color = pm25_category(prediction)

        st.markdown(
            f"""
            <div class="aq-result" style="background:{cat_color};">
                <div class="num">{prediction:.1f}</div>
                <div class="unit">µg/m³ · perkiraan PM2.5 besok</div>
                <div class="label">Kategori: {cat_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
with st.expander("ℹ️ Tentang model & keterbatasan data"):
    st.markdown(
        """
        - Sumber data: Dinas Lingkungan Hidup Kota Yogyakarta, Tahun 2024
          (data konsentrasi harian AQMS Stasiun Gondokusuman + indeks ISPU harian).
        - Model: Random Forest Regressor, dilatih untuk memprediksi PM2.5 esok
          hari (`H+1`) dari PM2.5 hari ini.
        - Data historis hanya mencakup ~310 hari dengan data lengkap dalam satu tahun dari satu stasiun AQMS Gondokusuman. 
        - Model ini dibandingkan dengan baseline
          persistence ("besok = hari ini"). Pembanding ini cukup kuat karena
          tingginya autokorelasi harian PM2.5.
        """
    )
