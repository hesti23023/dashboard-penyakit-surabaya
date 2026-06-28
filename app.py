"""
Dashboard Analisis & Prediksi Kasus Penyakit Kota Surabaya
=============================================================
Dibangun dengan Streamlit. Menyajikan eksplorasi data, pemetaan
kerawanan kecamatan (clustering), serta evaluasi & hasil forecasting
4 model (Linear Regression, Random Forest, XGBoost, SARIMA).
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# -------------------------------------------------------------------
# KONFIGURASI HALAMAN
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Penyakit Surabaya",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path(__file__).parent / "data"

# -------------------------------------------------------------------
# PALET WARNA & STYLE
# -------------------------------------------------------------------
COLOR_PRIMARY = "#0F4C5C"      # teal gelap - identitas utama
COLOR_ACCENT = "#E36414"       # oranye terracotta - aksen/peringatan
COLOR_HIGH = "#C0392B"         # merah - rawan tinggi
COLOR_MED = "#E67E22"          # oranye - rawan sedang
COLOR_LOW = "#27AE60"          # hijau - rawan rendah
COLOR_BG_CARD = "#F7F5F0"

MODEL_COLORS = {
    "LR": "#9B59B6",
    "RF": "#C0392B",
    "XGB": "#E67E22",
    "SARIMA": "#27AE60",
}

CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {{
        font-family: 'Inter', sans-serif;
    }}
    h1, h2, h3 {{
        font-family: 'Lora', serif;
        color: {COLOR_PRIMARY};
    }}
    .main-header {{
        font-family: 'Lora', serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: {COLOR_PRIMARY};
        margin-bottom: 0.1rem;
    }}
    .sub-header {{
        font-size: 1rem;
        color: #5b6b6e;
        margin-bottom: 1.4rem;
    }}
    .metric-card {{
        background-color: {COLOR_BG_CARD};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 5px solid {COLOR_PRIMARY};
    }}
    .insight-box {{
        background-color: #EFF6F4;
        border-left: 5px solid {COLOR_PRIMARY};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
    }}
    .warning-box {{
        background-color: #FDF1E6;
        border-left: 5px solid {COLOR_ACCENT};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
    }}
    div[data-testid="stMetric"] {{
        background-color: {COLOR_BG_CARD};
        border-radius: 10px;
        padding: 0.8rem 1rem;
        border-left: 5px solid {COLOR_PRIMARY};
    }}
    section[data-testid="stSidebar"] {{
        background-color: #fbfaf8;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------------------------------------------------------
# LOAD DATA (cached)
# -------------------------------------------------------------------
@st.cache_data
def load_main_data():
    df = pd.read_csv(DATA_DIR / "data_bersih.csv.gz", compression="gzip", parse_dates=["Tanggal"])
    return df

@st.cache_data
def load_clustering():
    return pd.read_excel(DATA_DIR / "Hasil_Clustering_Kecamatan.xlsx")

@st.cache_data
def load_top5():
    return pd.read_excel(DATA_DIR / "Top5_Penyakit.xlsx")

@st.cache_data
def load_evaluasi():
    return pd.read_csv(DATA_DIR / "Tabel_Evaluasi_4_Model.csv")

@st.cache_data
def load_adf():
    return pd.read_excel(DATA_DIR / "Ringkasan_ADF.xlsx")

df_main = load_main_data()
df_cluster = load_clustering()
df_top5 = load_top5()
df_eval = load_evaluasi()
df_adf = load_adf()

RISK_COLOR_MAP = {
    "Rawan Tinggi": COLOR_HIGH,
    "Rawan Sedang": COLOR_MED,
    "Rawan Rendah": COLOR_LOW,
}

# -------------------------------------------------------------------
# SIDEBAR — NAVIGASI
# -------------------------------------------------------------------
st.sidebar.markdown("## 🏥 Navigasi Dashboard")
page = st.sidebar.radio(
    "Pilih halaman",
    [
        "🏠 Ringkasan Umum",
        "📊 Eksplorasi Data",
        "🗺️ Pemetaan Kerawanan",
        "📈 Forecasting & Evaluasi Model",
        "📝 Insight & Kesimpulan",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Sumber data: Data Kasus Penyakit Kota Surabaya per Kecamatan & Fasilitas "
    "Kesehatan, periode 2022–2026."
)
st.sidebar.caption("Dibuat dengan Streamlit · Analisis RAPIDS")

# =====================================================================
# HALAMAN 1 — RINGKASAN UMUM
# =====================================================================
if page == "🏠 Ringkasan Umum":
    st.markdown('<div class="main-header">Analisis & Prediksi Kasus Penyakit Kota Surabaya</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Dashboard interaktif hasil analisis data kasus penyakit '
        '31 kecamatan di Kota Surabaya, periode 2022–2026</div>',
        unsafe_allow_html=True,
    )

    total_kasus = df_main["jumlah_kasus"].sum()
    n_kecamatan = df_main["kecamatan"].nunique()
    n_penyakit = df_main["jenis_penyakit"].nunique()
    n_faskes = df_main["nama_faskes"].nunique()
    tahun_min, tahun_max = int(df_main["tahun"].min()), int(df_main["tahun"].max())

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Jumlah Kecamatan", f"{n_kecamatan}")
    col2.metric("Jenis Penyakit", f"{n_penyakit}")
    col3.metric("Fasilitas Kesehatan", f"{n_faskes}")
    col4.metric("Total Kasus", f"{total_kasus:,.0f}")
    col5.metric("Rentang Tahun", f"{tahun_min}–{tahun_max}")

    st.markdown("###")
    c1, c2 = st.columns([1.3, 1])

    with c1:
        st.subheader("Tren Total Kasus Bulanan — Kota Surabaya")
        tren = df_main.groupby("Tanggal", as_index=False)["jumlah_kasus"].sum()
        fig = px.area(
            tren, x="Tanggal", y="jumlah_kasus",
            color_discrete_sequence=[COLOR_PRIMARY],
        )
        fig.update_traces(line_width=2, fillcolor="rgba(15,76,92,0.15)")
        fig.update_layout(
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="", yaxis_title="Jumlah Kasus",
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Top 5 Penyakit Dominan")
        fig2 = px.bar(
            df_top5.sort_values("Total_Kasus"),
            x="Total_Kasus", y="Penyakit",
            orientation="h",
            color_discrete_sequence=[COLOR_ACCENT],
            text="Total_Kasus",
        )
        fig2.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig2.update_layout(
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Total Kasus", yaxis_title="",
            plot_bgcolor="white",
            yaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("###")
    col_a, col_b = st.columns(2)
    with col_a:
        kec_top = df_cluster.sort_values("total_kasus", ascending=False).iloc[0]
        st.markdown(
            f"""<div class="insight-box">
            <b>📍 Kecamatan paling rawan:</b> {kec_top['kecamatan']}<br>
            Total kasus tercatat <b>{kec_top['total_kasus']:,.0f}</b>, masuk kategori
            <b>{kec_top['risk_label']}</b> berdasarkan hasil clustering K-Means.
            </div>""",
            unsafe_allow_html=True,
        )
    with col_b:
        best_model = df_eval["Model_Terbaik"].value_counts().idxmax()
        win_count = df_eval["Model_Terbaik"].value_counts().max()
        st.markdown(
            f"""<div class="insight-box">
            <b>🏆 Model prediksi terbaik:</b> {best_model}<br>
            Memenangkan <b>{win_count} dari {len(df_eval)}</b> kategori penyakit utama
            dalam evaluasi RMSE/MAE/R².
            </div>""",
            unsafe_allow_html=True,
        )

# =====================================================================
# HALAMAN 2 — EKSPLORASI DATA
# =====================================================================
elif page == "📊 Eksplorasi Data":
    st.markdown('<div class="main-header">Eksplorasi Data Kasus Penyakit</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Telusuri data berdasarkan kecamatan, jenis penyakit, dan periode waktu</div>',
        unsafe_allow_html=True,
    )

    # --- Filter ---
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        kec_opts = ["Semua Kecamatan"] + sorted(df_main["kecamatan"].unique().tolist())
        sel_kec = st.selectbox("Kecamatan", kec_opts)
    with fc2:
        peny_opts = ["Semua Penyakit"] + sorted(df_main["jenis_penyakit"].unique().tolist())
        sel_peny = st.selectbox("Jenis Penyakit", peny_opts)
    with fc3:
        tahun_opts = ["Semua Tahun"] + sorted(df_main["tahun"].unique().tolist(), reverse=True)
        sel_tahun = st.selectbox("Tahun", tahun_opts)

    df_f = df_main.copy()
    if sel_kec != "Semua Kecamatan":
        df_f = df_f[df_f["kecamatan"] == sel_kec]
    if sel_peny != "Semua Penyakit":
        df_f = df_f[df_f["jenis_penyakit"] == sel_peny]
    if sel_tahun != "Semua Tahun":
        df_f = df_f[df_f["tahun"] == sel_tahun]

    st.markdown("###")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Kasus (filter aktif)", f"{df_f['jumlah_kasus'].sum():,.0f}")
    m2.metric("Jumlah Baris Data", f"{len(df_f):,}")
    m3.metric("Rata-rata Kasus / Periode", f"{df_f['jumlah_kasus'].mean():,.1f}" if len(df_f) else "0")

    st.markdown("###")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Tren Kasus dari Waktu ke Waktu")
        tren_f = df_f.groupby("Tanggal", as_index=False)["jumlah_kasus"].sum()
        if len(tren_f):
            fig = px.line(tren_f, x="Tanggal", y="jumlah_kasus", markers=True,
                          color_discrete_sequence=[COLOR_PRIMARY])
            fig.update_layout(height=380, plot_bgcolor="white",
                              margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="", yaxis_title="Jumlah Kasus")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tidak ada data untuk kombinasi filter ini.")

    with c2:
        st.subheader("Distribusi Kasus per Jenis Penyakit")
        dist = df_f.groupby("jenis_penyakit", as_index=False)["jumlah_kasus"].sum().sort_values("jumlah_kasus", ascending=True).tail(10)
        if len(dist):
            fig = px.bar(dist, x="jumlah_kasus", y="jenis_penyakit", orientation="h",
                        color_discrete_sequence=[COLOR_ACCENT])
            fig.update_layout(height=380, plot_bgcolor="white",
                              margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="Jumlah Kasus", yaxis_title="",
                              yaxis=dict(tickfont=dict(size=10)))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Tidak ada data untuk kombinasi filter ini.")

    st.markdown("###")
    st.subheader("Perbandingan Antar Kecamatan")
    kec_compare = df_f.groupby("kecamatan", as_index=False)["jumlah_kasus"].sum().sort_values("jumlah_kasus", ascending=False)
    if len(kec_compare):
        fig = px.bar(kec_compare, x="kecamatan", y="jumlah_kasus",
                    color="jumlah_kasus", color_continuous_scale=["#27AE60", "#E67E22", "#C0392B"])
        fig.update_layout(height=400, plot_bgcolor="white",
                          margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="", yaxis_title="Jumlah Kasus",
                          coloraxis_showscale=False)
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("###")
    with st.expander("🔍 Lihat Data Mentah (Tabel)"):
        st.dataframe(
            df_f[["kecamatan", "jenis_penyakit", "nama_faskes", "periode", "jumlah_kasus", "is_anomaly"]]
            .sort_values("periode", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
            height=350,
        )
        csv = df_f.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Unduh data terfilter (CSV)", csv, "data_filtered.csv", "text/csv")

    n_anomaly = int(df_f["is_anomaly"].sum())
    if n_anomaly > 0:
        st.markdown(
            f"""<div class="warning-box">⚠️ Terdeteksi <b>{n_anomaly}</b> titik data anomali
            (lonjakan/penurunan kasus tidak wajar) pada kombinasi filter saat ini.</div>""",
            unsafe_allow_html=True,
        )

# =====================================================================
# HALAMAN 3 — PEMETAAN KERAWANAN (CLUSTERING)
# =====================================================================
elif page == "🗺️ Pemetaan Kerawanan":
    st.markdown('<div class="main-header">Pemetaan Kerawanan Kecamatan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Hasil clustering K-Means (k=3) berdasarkan total kasus penyakit per kecamatan</div>',
        unsafe_allow_html=True,
    )

    silhouette = 0.4389  # dari Narasi_RAPIDS.txt
    n_tinggi = (df_cluster["risk_label"] == "Rawan Tinggi").sum()
    n_sedang = (df_cluster["risk_label"] == "Rawan Sedang").sum()
    n_rendah = (df_cluster["risk_label"] == "Rawan Rendah").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Silhouette Score", f"{silhouette:.4f}")
    c2.metric("🔴 Rawan Tinggi", f"{n_tinggi} kecamatan")
    c3.metric("🟠 Rawan Sedang", f"{n_sedang} kecamatan")
    c4.metric("🟢 Rawan Rendah", f"{n_rendah} kecamatan")

    st.markdown("###")
    df_plot = df_cluster.sort_values("total_kasus", ascending=True)
    fig = px.bar(
        df_plot, x="total_kasus", y="kecamatan", orientation="h",
        color="risk_label",
        color_discrete_map=RISK_COLOR_MAP,
        text="total_kasus",
        labels={"total_kasus": "Total Kasus", "kecamatan": "", "risk_label": "Kategori"},
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", textfont_size=10)
    fig.update_layout(
        height=820, plot_bgcolor="white",
        margin=dict(l=10, r=60, t=20, b=10),
        legend_title="Kategori Kerawanan",
        yaxis=dict(tickfont=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("###")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Tabel Detail Clustering")
        df_show = df_cluster.copy().sort_values("total_kasus", ascending=False)
        df_show["total_kasus"] = df_show["total_kasus"].map("{:,.0f}".format)
        df_show["rata_rata_bulanan"] = df_show["rata_rata_bulanan"].map("{:,.1f}".format)
        df_show["std_kasus"] = df_show["std_kasus"].map("{:,.1f}".format)
        st.dataframe(
            df_show[["kecamatan", "total_kasus", "rata_rata_bulanan", "std_kasus", "risk_label"]]
            .rename(columns={
                "kecamatan": "Kecamatan", "total_kasus": "Total Kasus",
                "rata_rata_bulanan": "Rata-rata Bulanan", "std_kasus": "Std. Deviasi",
                "risk_label": "Kategori",
            }),
            use_container_width=True, height=420, hide_index=True,
        )

    with col2:
        st.subheader("Proporsi Kecamatan per Kategori")
        risk_count = df_cluster["risk_label"].value_counts().reset_index()
        risk_count.columns = ["risk_label", "jumlah"]
        fig2 = px.pie(
            risk_count, names="risk_label", values="jumlah",
            color="risk_label", color_discrete_map=RISK_COLOR_MAP,
            hole=0.45,
        )
        fig2.update_traces(textinfo="label+percent")
        fig2.update_layout(height=420, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

        top3 = df_cluster.sort_values("total_kasus", ascending=False).head(3)["kecamatan"].tolist()
        st.markdown(
            f"""<div class="warning-box">⚠️ <b>3 kecamatan paling rawan:</b><br>
            {', '.join(top3)} — perlu prioritas intervensi kesehatan masyarakat.</div>""",
            unsafe_allow_html=True,
        )

# =====================================================================
# HALAMAN 4 — FORECASTING & EVALUASI MODEL
# =====================================================================
elif page == "📈 Forecasting & Evaluasi Model":
    st.markdown('<div class="main-header">Forecasting & Evaluasi Model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Perbandingan performa 4 model prediksi: Linear Regression, Random Forest, XGBoost, dan SARIMA</div>',
        unsafe_allow_html=True,
    )

    best_overall = df_eval["Model_Terbaik"].value_counts().idxmax()
    win_count = df_eval["Model_Terbaik"].value_counts().max()
    avg_rmse_best = df_eval["RF_RMSE"].mean()  # RF adalah model terbaik keseluruhan

    c1, c2, c3 = st.columns(3)
    c1.metric("Model Terbaik Keseluruhan", best_overall)
    c2.metric("Jumlah Kemenangan", f"{win_count} dari {len(df_eval)} penyakit")
    c3.metric("Rata-rata RMSE (RF)", f"{avg_rmse_best:,.2f}")

    st.markdown("###")
    st.subheader("Pilih Penyakit untuk Analisis Detail")
    peny_list = df_eval["Nama_Penyakit"].tolist()
    sel_peny_eval = st.selectbox("Jenis Penyakit", peny_list, label_visibility="collapsed")
    row = df_eval[df_eval["Nama_Penyakit"] == sel_peny_eval].iloc[0]

    metrics_df = pd.DataFrame({
        "Model": ["LR", "RF", "XGB", "SARIMA"],
        "RMSE": [row["LR_RMSE"], row["RF_RMSE"], row["XGB_RMSE"], row["SARIMA_RMSE"]],
        "MAE": [row["LR_MAE"], row["RF_MAE"], row["XGB_MAE"], row["SARIMA_MAE"]],
        "R2": [row["LR_R2"], row["RF_R2"], row["XGB_R2"], row["SARIMA_R2"]],
    })

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown(f"**RMSE & MAE — {sel_peny_eval}**")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=metrics_df["Model"], y=metrics_df["RMSE"], name="RMSE",
            marker_color=COLOR_PRIMARY,
        ))
        fig.add_trace(go.Bar(
            x=metrics_df["Model"], y=metrics_df["MAE"], name="MAE",
            marker_color=COLOR_ACCENT,
        ))
        fig.update_layout(
            barmode="group", height=360, plot_bgcolor="white",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    with cc2:
        st.markdown(f"**R² Score per Model — {sel_peny_eval}**")
        colors_r2 = [MODEL_COLORS[m] for m in metrics_df["Model"]]
        fig2 = go.Figure(go.Bar(
            x=metrics_df["Model"], y=metrics_df["R2"], marker_color=colors_r2,
            text=metrics_df["R2"].round(3), textposition="outside",
        ))
        fig2.add_hline(y=0, line_dash="dash", line_color="gray")
        fig2.update_layout(height=360, plot_bgcolor="white",
                          margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="R² Score")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(
        f"""<div class="insight-box">
        Model terbaik untuk <b>{sel_peny_eval}</b> adalah <b>{row['Model_Terbaik']}</b>,
        dengan RMSE terendah pada model tersebut dibanding 3 model lainnya.
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("###")
    st.subheader("Tabel Lengkap Evaluasi 4 Model — Top 5 Penyakit")
    df_eval_show = df_eval.copy()
    for c in df_eval_show.columns:
        if "RMSE" in c or "MAE" in c:
            df_eval_show[c] = df_eval_show[c].map("{:,.2f}".format)
        elif "R2" in c:
            df_eval_show[c] = df_eval_show[c].map("{:,.4f}".format)
    st.dataframe(
        df_eval_show.drop(columns=["Penyakit"]).rename(columns={"Nama_Penyakit": "Penyakit", "Model_Terbaik": "Model Terbaik"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("###")
    st.subheader("Uji Stasioneritas (Augmented Dickey-Fuller Test)")
    df_adf_show = df_adf.copy()
    df_adf_show["ADF_Statistic"] = df_adf_show["ADF_Statistic"].map("{:.4f}".format)
    df_adf_show["P_Value"] = df_adf_show["P_Value"].map("{:.2e}".format)
    df_adf_show["Stasioner"] = df_adf_show["Stasioner"].map({True: "✅ Ya", False: "❌ Tidak"})
    st.dataframe(
        df_adf_show.rename(columns={
            "Penyakit": "Penyakit", "ADF_Statistic": "Statistik ADF",
            "P_Value": "P-Value", "Stasioner": "Stasioner?",
            "Differencing_d": "Differencing (d)",
        }),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "Seluruh data deret waktu kasus penyakit dinyatakan **stasioner** (p-value < 0.05) "
        "tanpa perlu differencing tambahan (d=0), sehingga layak dimodelkan langsung dengan SARIMA."
    )

# =====================================================================
# HALAMAN 5 — INSIGHT & KESIMPULAN
# =====================================================================
elif page == "📝 Insight & Kesimpulan":
    st.markdown('<div class="main-header">Insight & Kesimpulan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Ringkasan naratif hasil analisis & prediksi kasus penyakit Kota Surabaya</div>',
        unsafe_allow_html=True,
    )

    total_kasus_pernafasan = df_top5.iloc[0]["Total_Kasus"]
    kec_top = df_cluster.sort_values("total_kasus", ascending=False).iloc[0]

    st.markdown(
        f"""
        ### 📌 Ringkasan Eksekutif

        Analisis ini mencakup **31 kecamatan** dan **22 jenis penyakit** di Kota Surabaya
        sepanjang periode **2022–2026**.

        - **Penyakit dominan**: *{df_top5.iloc[0]['Penyakit']}* dengan total
          **{total_kasus_pernafasan:,.0f}** kasus — jauh di atas penyakit lainnya.
        - **Kecamatan paling rawan**: **{kec_top['kecamatan']}**, dengan total kasus
          **{kec_top['total_kasus']:,.0f}**, dikategorikan sebagai *{kec_top['risk_label']}*.
        - **Clustering K-Means (k=3)** menghasilkan silhouette score **0.4389**, menunjukkan
          pemisahan kelompok yang cukup baik antara kecamatan rawan tinggi, sedang, dan rendah.
        - **Model prediksi terbaik secara keseluruhan**: **Random Forest (RF)**, memenangkan
          2 dari 5 kategori penyakit utama, dengan rata-rata RMSE terbaik **2.857,99**.
        """
    )

    st.markdown("###")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🦠 Top 5 Penyakit Dominan")
        for i, r in df_top5.iterrows():
            st.markdown(f"**{i+1}. {r['Penyakit']}** — {r['Total_Kasus']:,.0f} kasus")

    with col2:
        st.markdown("#### 🗺️ Top 5 Kecamatan Paling Rawan")
        top5_kec = df_cluster.sort_values("total_kasus", ascending=False).head(5)
        for i, r in top5_kec.reset_index(drop=True).iterrows():
            st.markdown(f"**{i+1}. {r['kecamatan']}** — {r['total_kasus']:,.0f} kasus ({r['risk_label']})")

    st.markdown("###")
    st.markdown("#### 🤖 Model Terbaik per Jenis Penyakit")
    for _, r in df_eval.iterrows():
        st.markdown(f"- **{r['Nama_Penyakit']}** → model terbaik: **{r['Model_Terbaik']}**")

    st.markdown("###")
    st.markdown(
        """
        ### 💡 Rekomendasi

        1. **Prioritaskan intervensi kesehatan** di kecamatan dengan kategori *Rawan Tinggi*
           (Kenjeran, Sawahan, Semampir, Wonokromo, Wonocolo), terutama untuk penyakit sistem
           pernafasan dan pencernaan yang mendominasi.
        2. **Gunakan model Random Forest** sebagai model utama untuk prediksi kasus penyakit
           pernafasan dan pencernaan, karena memberikan akurasi (RMSE) terbaik dibanding
           Linear Regression, XGBoost, maupun SARIMA.
        3. **Gunakan model SARIMA** untuk kategori penyakit dengan pola musiman yang lebih
           kuat, seperti faktor status kesehatan umum dan penyakit musculoskeletal.
        4. **Pantau anomali data secara berkala** — terdapat sejumlah titik anomali yang
           terdeteksi dan dapat menjadi sinyal dini lonjakan kasus di lapangan.
        """
    )

    st.markdown("---")
    st.caption(
        "Dashboard ini dibangun berdasarkan hasil analisis RAPIDS (Rangkaian Analisis "
        "Prediksi & Identifikasi Daerah Sensitif) terhadap data kasus penyakit Kota Surabaya."
    )
