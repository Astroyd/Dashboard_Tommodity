"""
GRRA - Yapay Zeka Destekli Satın Alma ve Envanter Karar Destek Sistemi
----------------------------------------------------------------------
Kurumsal Karar Destek ve Optimizasyon Platformu (Üst Sekme Navigasyonlu)
Veri kaynakları: data/envanter.db, data/dis_degiskenler.csv,
                 data/hibrit_tahmin_sonuc.csv, data/hibrit_simulasyon_sonuc.csv

Çalıştırmak için:
    streamlit run 08_dashboard.py
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ---------------------------------------------------------------------------
# 1. SAYFA YAPILANDIRMASI VE KURUMSAL CSS STİLİ
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GRRA | Satın Alma ve Envanter Karar Destek Sistemi",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Genel Sayfa Düzeni */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 97%;
    }
    h1, h2, h3, h4, h5 {
        font-family: 'Inter', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        color: #0f172a;
        font-weight: 600;
        letter-spacing: -0.015em;
    }

    /* Üst Kurumsal Başlık Bandı */
    .corp-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background-color: #0f172a;
        color: #f8fafc;
        padding: 14px 22px;
        border-radius: 4px;
        margin-bottom: 14px;
        border-bottom: 3px solid #2563eb;
    }
    .corp-title {
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        color: #ffffff;
        margin: 0;
    }
    .corp-subtitle {
        font-size: 0.80rem;
        color: #94a3b8;
        margin-top: 2px;
    }
    .corp-badges {
        display: flex;
        gap: 12px;
        font-size: 0.80rem;
    }
    .corp-badge {
        background-color: #1e293b;
        border: 1px solid #334155;
        color: #e2e8f0;
        padding: 6px 12px;
        border-radius: 3px;
        font-weight: 500;
    }

    /* En Üst Navigasyon Sekmeleri (Top-Level Tabs) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #f1f5f9;
        padding: 6px 8px;
        border-radius: 4px;
        border: 1px solid #cbd5e1;
        margin-bottom: 18px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        background-color: transparent;
        border-radius: 3px;
        padding: 0 18px;
        font-weight: 600;
        font-size: 0.86rem;
        color: #475569;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-bottom: 2px solid #2563eb !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    }

    /* KPI Metrik Kartları */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-top: 3px solid #0f172a;
        padding: 12px 16px;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.02);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.76rem;
        color: #475569;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.35rem;
        color: #0f172a;
        font-weight: 700;
    }

    /* Kurumsal ERP Tedarik Emri / Karar Fişi */
    .erp-order-card {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 4px;
        margin-top: 10px;
        margin-bottom: 22px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
        overflow: hidden;
    }
    .erp-header-buy {
        background-color: #065f46;
        color: #ffffff;
        padding: 10px 20px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .erp-header-hold {
        background-color: #1e3a8a;
        color: #ffffff;
        padding: 10px 20px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .erp-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        border-bottom: 1px solid #e2e8f0;
        background-color: #f8fafc;
    }
    .erp-cell {
        padding: 12px 20px;
        border-right: 1px solid #e2e8f0;
    }
    .erp-cell:last-child {
        border-right: none;
    }
    .erp-cell-label {
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }
    .erp-cell-value {
        font-size: 1.05rem;
        color: #0f172a;
        font-weight: 700;
    }
    .erp-rationale {
        padding: 12px 20px;
        font-size: 0.86rem;
        color: #334155;
        background-color: #ffffff;
        line-height: 1.5;
    }

    /* Yan Menü (Sidebar) Model Künyesi Kartı */
    .sidebar-model-box {
        background-color: #f8fafc;
        border: 1px solid #cbd5e1;
        border-left: 3px solid #2563eb;
        padding: 10px 12px;
        border-radius: 3px;
        font-size: 0.80rem;
        color: #1e293b;
        margin-top: 6px;
        margin-bottom: 12px;
        line-height: 1.45;
    }
    .sidebar-section-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #0f172a;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-bottom: 1px solid #cbd5e1;
        padding-bottom: 4px;
        margin-top: 14px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

DB = Path("data/envanter.db")
DIS_CSV = Path("data/dis_degiskenler.csv")
TAHMIN_CSV = Path("data/hibrit_tahmin_sonuc.csv")
SIM_CSV = Path("data/hibrit_simulasyon_sonuc.csv")

ADLAR = {"pamuk": "Pamuk", "aluminyum": "Alüminyum", "bugday": "Buğday"}

# Aktif Modellerin Detaylı Künyesi ve Amaçları
MODEL_KATALOGU = {
    "quant_yon_hibrit": {
        "ad": "Çoklu Frekans (MIDAS) + Olasılıksal Yön Hibriti (quant_yon_hibrit)",
        "kisa_ad": "MIDAS Olasılıksal Yön Hibriti (Şampiyon)",
        "mimari": "Huber Regresyon + L2 Lojistik Sınıflandırıcı + Makro Ridge Birleşimi",
        "girdiler": "CFTC COT Smart Money Spread, Dolar Endeksi (DXY), VIX, Maliyet Marjı, Hasat Anomalisi ve Volatilite MACD.",
        "amac": "Aykırı fiyat şoklarından etkilenmeden yön olasılığını tahmin etmek ve tüm sipariş senaryolarında sıfır zarar ile maksimum net ekonomik tasarruf sağlamak.",
        "rejim": "Hem kriz/şok dönemleri hem de standart piyasa rejimleri (En yüksek genel performans).",
    },
    "ridge_dis": {
        "ad": "Dış Değişkenli Ridge Regresyon (ridge_dis)",
        "kisa_ad": "Ridge + Makro Dışsal (ridge_dis)",
        "mimari": "L2 Cezalı Doğrusal Regresyon + TimeSeriesSplit Çapraz Doğrulama",
        "girdiler": "Maliyet Marjı (Cost Spread), Volatilite Ölçekli MACD ve Makro Emtia Sepeti (Brent Petrol, Doğal Gaz, Kömür, Üre, Mısır, Soya, Bakır) 1 ve 3 aylık değişimleri.",
        "amac": "Hammadde fiyatını üretim girdileri ve ikame ürün maliyetleriyle birlikte modelleyerek kısa-orta vadeli (1-3 ay) satın alma zamanlamasında maksimum maliyet tasarrufu sağlamak.",
        "rejim": "Enflasyonist dönemler, enerji krizleri ve arz-talep şokları.",
    },
    "hibrit_ensemble": {
        "ad": "Hibrit Kantitatif Formül + ML Topluluğu (hibrit_ensemble)",
        "kisa_ad": "Hibrit Formül + ML (hibrit_ensemble)",
        "mimari": "Ters-MSE Ağırlıklı Kantitatif Uzman Formüller + Ridge Regresyon Birleşimi",
        "girdiler": "MR60, MR120, Maliyet Marjı Z-Skoru, Volatilite MACD, Kısa Vade Sharpe Momentum ve Dış Piyasa İvmesi.",
        "amac": "Tek bir modele bağımlı kalmadan trend takipçisi ve ortalamaya dönüş formüllerini geçmiş hata performanslarına göre harmanlayarak tahmin varyansını düşürmek.",
        "rejim": "Yönün belirsiz olduğu geçiş dönemleri ve çoklu ufuk (1-6 ay) planlaması.",
    },
    "hibrit_ridge": {
        "ad": "Üstel Ağırlıklı Hibrit Ridge (hibrit_ridge)",
        "kisa_ad": "Üstel Ağırlıklı Ridge (hibrit_ridge)",
        "mimari": "Yakın Geçmiş Odaklı (Exponentially Weighted) L2 Regresyon + James-Stein Shrinkage",
        "girdiler": "6 Kantitatif Finans Sinyali + Mevsimsellik Bileşenleri (Sin/Cos).",
        "amac": "1990'lardaki eski piyasa verilerinin ağırlığını azaltıp son yıllardaki yapısal kırılmalara ve güncel piyasa rejimine hızla adapte olmak.",
        "rejim": "Yapısal kırılma sonrası yeni fiyat dengelerinin oluştuğu piyasalar.",
    },
    "mr_maliyet": {
        "ad": "Maliyet Düzeltilmiş Ortalamaya Dönüş (mr_maliyet)",
        "kisa_ad": "Maliyet Bazlı Mean-Reversion (mr_maliyet)",
        "mimari": "Sabit Terimli OLS + 10 Yıllık Denge ve Üretim Maliyeti Spread Formülü",
        "girdiler": "120 Aylık Ortalamadan Sapma (MR120) ve Üretim Maliyeti / İkame Marjı (Cost Spread).",
        "amac": "Fiyatın üretim maliyetine ve uzun vadeli tarihsel ortalamasına göre aşırı ucuz veya aşırı pahalı olduğu bölgeleri tespit etmek.",
        "rejim": "Yatay/dalgalı piyasalar ve uzun vadeli (6-12 ay) stratejik kontrat bağlama kararları.",
    },
    "ridge": {
        "ad": "Standart İçsel Ridge Regresyon (ridge)",
        "kisa_ad": "Standart İçsel Ridge (ridge)",
        "mimari": "Standartlaştırılmış L2 Doğrusal Regresyon (Alpha=10.0)",
        "girdiler": "Yalnızca hammaddenin kendi geçmiş getirileri (lag 1..6), 3-12 aylık momentum ve oynaklık.",
        "amac": "Dış veri gerektirmeden sadece içsel fiyat hareketleriyle temel referans (benchmark) tahmini üretmek.",
        "rejim": "Dışsal makro şokların olmadığı standart piyasa koşulları.",
    },
}

MALIYET_SEPETI = {
    "pamuk": {"petrol_brent": 0.35, "soya": 0.25, "misir": 0.20, "ure": 0.20},
    "aluminyum": {"dogalgaz_ab": 0.30, "komur": 0.30, "petrol_brent": 0.20, "bakir": 0.20},
    "bugday": {"misir": 0.35, "soya": 0.25, "ure": 0.25, "petrol_brent": 0.15},
}


# ---------------------------------------------------------------------------
# 2. VERİ YÜKLEME VE HESAPLAMA FONKSİYONLARI
# ---------------------------------------------------------------------------
@st.cache_data
def verileri_yukle():
    if not DB.exists():
        st.error("HATA: data/envanter.db veritabanı bulunamadı. Lütfen önce 03, 16 ve 17 numaralı scriptleri çalıştırın.")
        st.stop()

    with sqlite3.connect(DB) as con:
        ham = pd.read_sql("SELECT * FROM hammadde", con).set_index("id")
        fiyat = pd.read_sql("SELECT * FROM fiyat ORDER BY tarih", con, parse_dates=["tarih"])
        tuketim = pd.read_sql("SELECT * FROM tuketim ORDER BY tarih", con, parse_dates=["tarih"])
        tahmin = pd.read_sql("SELECT * FROM tahmin ORDER BY tarih", con, parse_dates=["tarih"])
        sim_stok = pd.read_sql("SELECT * FROM sim_stok ORDER BY tarih", con, parse_dates=["tarih"])
        sim_siparis = pd.read_sql("SELECT * FROM sim_siparis ORDER BY tarih", con, parse_dates=["tarih"])

    dis_df = pd.read_csv(DIS_CSV, parse_dates=["tarih"], index_col="tarih") if DIS_CSV.exists() else pd.DataFrame()
    tah_ozet = pd.read_csv(TAHMIN_CSV) if TAHMIN_CSV.exists() else pd.DataFrame()
    sim_ozet = pd.read_csv(SIM_CSV) if SIM_CSV.exists() else pd.DataFrame()
    return ham, fiyat, tuketim, tahmin, sim_stok, sim_siparis, dis_df, tah_ozet, sim_ozet


def envanter_tablosu_uret(ham, fiyat, tuketim, s_carpan=5.0, z_skor_hizmet=1.65):
    satirlar = []
    for hid in ham.index:
        p = ham.loc[hid]
        tuk = tuketim[tuketim["hammadde_id"] == hid]["miktar_ton"]
        fiy = fiyat[fiyat["hammadde_id"] == hid]["fiyat_usd_ton"]
        if tuk.empty or fiy.empty:
            continue

        d_ay, sigma_ay = float(tuk.mean()), float(tuk.std())
        d_yil = d_ay * 12.0
        f_son12 = float(fiy.iloc[-12:].mean())
        f_guncel = float(fiy.iloc[-1])
        S_eff = float(p["siparis_maliyeti"]) * s_carpan
        h_oran = float(p["tutma_orani"])
        h_usd = h_oran * f_son12
        L = int(p["tedarik_suresi_ay"])

        eoq = np.sqrt(2 * d_yil * S_eff / h_usd)
        ss = z_skor_hizmet * sigma_ay * np.sqrt(L)
        rop = d_ay * L + ss

        satirlar.append(dict(
            id=hid,
            hammadde=ADLAR.get(hid, p["ad"]),
            aylik_talep_ton=round(d_ay, 1),
            talep_std_ton=round(sigma_ay, 1),
            guncel_fiyat_usd=round(f_guncel, 1),
            ort_fiyat_usd=round(f_son12, 1),
            yillik_harcama_usd=round(d_yil * f_son12, 0),
            siparis_maliyeti_usd=round(S_eff, 0),
            tutma_orani_pct=round(h_oran * 100, 1),
            tedarik_suresi_ay=L,
            EOQ_ton=round(eoq, 1),
            guvenlik_stogu_ton=round(ss, 1),
            ROP_ton=round(rop, 1),
            yillik_siparis_sayisi=round(d_yil / eoq, 1),
        ))
    df = pd.DataFrame(satirlar).sort_values("yillik_harcama_usd", ascending=False).reset_index(drop=True)
    pay = df["yillik_harcama_usd"].cumsum() / df["yillik_harcama_usd"].sum()
    onceki = pay.shift(fill_value=0)
    df["ABC"] = np.where(onceki < 0.80, "A", np.where(onceki < 0.95, "B", "C"))
    df["butce_payi_pct"] = (df["yillik_harcama_usd"] / df["yillik_harcama_usd"].sum() * 100).round(1)
    return df.set_index("id")


def canli_faktoru_cozumle(fiyat_df: pd.DataFrame, dis_df: pd.DataFrame, hid: str) -> dict:
    s_fiy = fiyat_df[fiyat_df["hammadde_id"] == hid].set_index("tarih")["fiyat_usd_ton"].sort_index()
    lp = np.log(s_fiy)
    r1 = lp.diff()
    vol6 = r1.rolling(6, min_periods=3).std().clip(lower=0.01)
    vol12 = r1.rolling(12, min_periods=6).std().clip(lower=0.01)

    ma60 = lp.rolling(60, min_periods=24).mean()
    sd60 = lp.rolling(60, min_periods=24).std().clip(lower=0.02)
    ma120 = lp.rolling(120, min_periods=36).mean()
    sd120 = lp.rolling(120, min_periods=36).std().clip(lower=0.02)

    mr60_z = float(-((lp - ma60) / sd60).clip(-3.0, 3.0).iloc[-1])
    mr120_z = float(-((lp - ma120) / sd120).clip(-3.0, 3.0).iloc[-1])

    sepet = MALIYET_SEPETI.get(hid, {})
    maliyet_z_serisi = pd.Series(0.0, index=lp.index)
    if not dis_df.empty:
        mevcut = {k: w for k, w in sepet.items() if k in dis_df.columns}
        top_w = sum(mevcut.values()) if mevcut else 1.0
        for k, w in mevcut.items():
            x = np.log(dis_df[k].reindex(lp.index)).ffill().bfill()
            mu = x.rolling(60, min_periods=24).mean()
            sd = x.rolling(60, min_periods=24).std().replace(0, np.nan)
            maliyet_z_serisi += (w / top_w) * ((x - mu) / sd).fillna(0.0)

    fiyat_z = float(((lp - ma60) / sd60).fillna(0.0).iloc[-1])
    cost_spread_z = float(np.clip(maliyet_z_serisi.iloc[-1] - fiyat_z, -3.0, 3.0))
    vol_macd = float(((lp.rolling(3).mean() - lp.rolling(12).mean()) / (vol12 * np.sqrt(6))).clip(-3.0, 3.0).iloc[-1])
    mom3_sharpe = float(((lp - lp.shift(3)) / (vol6 * np.sqrt(3))).clip(-3.0, 3.0).iloc[-1])
    dis_ivme = float(maliyet_z_serisi.diff(3).clip(-3.0, 3.0).iloc[-1])

    return {
        "Maliyet / İkame Marjı (Cost Spread)": round(cost_spread_z, 2),
        "10 Yıllık Ortalamaya Dönüş (MR120)": round(mr120_z, 2),
        "5 Yıllık Ortalamaya Dönüş (MR60)": round(mr60_z, 2),
        "Volatilite Ölçekli Trend (Vol-MACD)": round(vol_macd, 2),
        "Kısa Vade Momentum (3A Sharpe)": round(mom3_sharpe, 2),
        "Makro Enerji ve Tahıl İvmesi (3A)": round(dis_ivme, 2),
    }


# ---------------------------------------------------------------------------
# 3. ANA EKRAN VE KONTROL PANELİ AKIŞI
# ---------------------------------------------------------------------------
def main():
    ham, fiyat, tuketim, tahmin, sim_stok, sim_siparis, dis_df, tah_ozet, sim_ozet = verileri_yukle()

    # =========================================================================
    # SOL KONTROL PANELİ (YAPILANDIRILMIŞ SİSTEM PARAMETRELERİ)
    # =========================================================================
    st.sidebar.markdown("## GRRA KONTROL PANELİ")
    st.sidebar.caption("Satın Alma ve Envanter Karar Destek Sistemi")

    # BÖLÜM 1: EMTİA VE ALGORİTMA SEÇİMİ
    st.sidebar.markdown('<div class="sidebar-section-title">1. Emtia ve Model Seçimi</div>', unsafe_allow_html=True)

    secili_hid = st.sidebar.selectbox(
        "Hedef Hammadde",
        options=list(ADLAR.keys()),
        format_func=lambda x: f"{ADLAR[x]} ({x.upper()})",
    )

    mevcut_modeller = [m for m in MODEL_KATALOGU.keys() if m in tahmin["model"].unique()]
    secili_model = st.sidebar.selectbox(
        "Aktif Tahmin Modeli",
        options=mevcut_modeller,
        format_func=lambda x: MODEL_KATALOGU[x]["kisa_ad"],
        index=0,
    )

    # Seçili Modelin Açıklama ve Amaç Kartı
    m_katalog = MODEL_KATALOGU[secili_model]
    st.sidebar.markdown(
        f"""<div class="sidebar-model-box">
        <b>Model Mimarisi:</b> {m_katalog['mimari']}<br><br>
        <b>Kullanım Amacı:</b> {m_katalog['amac']}<br><br>
        <b>Temel Girdiler:</b> {m_katalog['girdiler']}
        </div>""",
        unsafe_allow_html=True,
    )

    # BÖLÜM 2: OPERASYONEL ENVANTER PARAMETRELERİ
    st.sidebar.markdown('<div class="sidebar-section-title">2. Operasyonel Parametreler</div>', unsafe_allow_html=True)

    s_carpan = st.sidebar.select_slider(
        "Sabit Sipariş Maliyeti Çarpanı (S)",
        options=[1, 5, 10],
        value=5,
        help="1x: Yerel Tedarik ($300-$500), 5x: Bölgesel/Konteyner ($1500-$2500), 10x: Küresel İthalat ($3000-$5000)",
    )

    hizmet_secim = st.sidebar.select_slider(
        "Hedef Hizmet Düzeyi (Service Level)",
        options=["%90 (Z=1.28)", "%95 (Z=1.65)", "%99 (Z=2.33)"],
        value="%95 (Z=1.65)",
    )
    z_degeri = {"%90 (Z=1.28)": 1.28, "%95 (Z=1.65)": 1.65, "%99 (Z=2.33)": 2.33}[hizmet_secim]

    env_tablo = envanter_tablosu_uret(ham, fiyat, tuketim, s_carpan=float(s_carpan), z_skor_hizmet=z_degeri)
    m_bilgi = env_tablo.loc[secili_hid]

    # BÖLÜM 3: CANLI DEPO VE DUYARLILIK SİMÜLASYONU
    st.sidebar.markdown('<div class="sidebar-section-title">3. Depo ve Sinyal Kalibrasyonu</div>', unsafe_allow_html=True)

    mevcut_stok = st.sidebar.number_input(
        f"Mevcut {m_bilgi['hammadde']} Stoğu (Ton)",
        min_value=0.0,
        max_value=float(round(m_bilgi["ROP_ton"] * 3.0, 0)),
        value=float(round(m_bilgi["ROP_ton"] * 0.90, 1)),
        step=5.0,
        help="Depodaki anlık kullanılabilir fiziksel stok + yoldaki sipariş miktarı.",
    )

    beta_agresiflik = st.sidebar.slider(
        "Spekülatif Esneklik Katsayısı (Beta)",
        min_value=0.10,
        max_value=0.60,
        value=0.35,
        step=0.05,
        help="Modelin fiyat değişim sinyaline göre EOQ ve ROP değerlerini maksimum esnetme oranı.",
    )

    # =========================================================================
    # CANLI KARAR HESAPLAMALARI
    # =========================================================================
    tah_h1 = tahmin[(tahmin["hammadde_id"] == secili_hid) & (tahmin["model"] == secili_model) & (tahmin["ufuk"] == 1)]
    tah_h3 = tahmin[(tahmin["hammadde_id"] == secili_hid) & (tahmin["model"] == secili_model) & (tahmin["ufuk"] == 3)]

    son_dlog1 = float(tah_h1.iloc[-1]["tahmin_log_degisim"]) if not tah_h1.empty else 0.0
    son_dlog3 = float(tah_h3.iloc[-1]["tahmin_log_degisim"]) if not tah_h3.empty else 0.0
    std_dlog3 = float(tah_h3["tahmin_log_degisim"].std()) if len(tah_h3) > 12 else 0.03

    bilesik_beklenti = 0.35 * son_dlog1 + 0.65 * (son_dlog3 / 3.0)
    z_sinyal = float(np.clip(bilesik_beklenti / max(std_dlog3 / 3.0, 0.005), -3.0, 3.0))
    w_agirlik = float(np.tanh(1.25 * z_sinyal))

    klasik_eoq = float(m_bilgi["EOQ_ton"])
    klasik_rop = float(m_bilgi["ROP_ton"])
    d_aylik = float(m_bilgi["aylik_talep_ton"])

    if w_agirlik >= 0:
        ai_rop = klasik_rop + (beta_agresiflik * w_agirlik) * d_aylik
        ai_eoq = klasik_eoq * (1.0 + beta_agresiflik * w_agirlik)
    else:
        ai_rop = klasik_rop + (0.60 * beta_agresiflik * w_agirlik) * d_aylik
        ai_eoq = klasik_eoq * (1.0 + 0.70 * beta_agresiflik * w_agirlik)

    beklenen_1ay_pct = (np.exp(son_dlog1) - 1.0) * 100.0
    beklenen_3ay_pct = (np.exp(son_dlog3) - 1.0) * 100.0
    hedef_fiyat_3ay = m_bilgi["guncel_fiyat_usd"] * np.exp(son_dlog3)

    # =========================================================================
    # EN ÜST KURUMSAL BAŞLIK VE ANA NAVİGASYON SEKMELERİ
    # =========================================================================
    st.markdown(
        f"""<div class="corp-header">
            <div>
                <div class="corp-title">GRRA | YAPAY ZEKA DESTEKLİ SATIN ALMA VE ENVANTER KARAR DESTEK SİSTEMİ</div>
                <div class="corp-subtitle">Kantitatif Fiyat Tahmini, Dinamik EOQ/ROP Optimizasyonu ve Stokastik Politika Simülasyonu</div>
            </div>
            <div class="corp-badges">
                <div class="corp-badge">Emtia: <b>{m_bilgi['hammadde']} ({m_bilgi['ABC']} Sınıfı)</b></div>
                <div class="corp-badge">Model: <b>{secili_model.upper()}</b></div>
                <div class="corp-badge">Sipariş Maliyeti: <b>${m_bilgi['siparis_maliyeti_usd']:,.0f} ({s_carpan}x)</b></div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # EN ÜSTTEKİ 5 ANA MODÜL SEKMESİ
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Satın Alma Karar Merkezi ve Sinyal Analizi",
        "2. Fiyat Tahmini ve Şok Dönemi Analizi",
        "3. Politika Simülasyonu ve Backtest",
        "4. İstatistiksel Doğrulama ve Model Rehberi",
        "5. Envanter Parametreleri ve ABC Analizi",
    ])

    # =========================================================================
    # SEKME 1: SATIN ALMA KARAR MERKEZİ, TEDARİK EMRİ VE SİNYAL ANALİZİ
    # =========================================================================
    with tab1:
        # 5 KPI KARTI
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric(
            "Güncel Spot Fiyat",
            f"${m_bilgi['guncel_fiyat_usd']:,.1f} /ton",
            f"{m_bilgi['guncel_fiyat_usd'] - m_bilgi['ort_fiyat_usd']:+.1f} USD (12A Ort. Farkı)",
        )
        k2.metric(
            "3 Aylık Hedef Fiyat",
            f"${hedef_fiyat_3ay:,.1f} /ton",
            f"%{beklenen_3ay_pct:+.2f} ({'Artış Yönlü' if beklenen_3ay_pct >= 0 else 'Düşüş Yönlü'})",
        )
        k3.metric(
            "Dinamik Sipariş Noktası (ROP)",
            f"{ai_rop:.1f} Ton",
            f"{ai_rop - klasik_rop:+.1f} Ton (Klasik: {klasik_rop:.1f})",
        )
        k4.metric(
            "Spekülatif Parti Miktarı (EOQ)",
            f"{ai_eoq:.1f} Ton",
            f"%{((ai_eoq / klasik_eoq) - 1.0) * 100:+.1f} (Klasik: {klasik_eoq:.1f})",
        )
        k5.metric(
            f"Güvenlik Stoğu ({hizmet_secim[:3]})",
            f"{m_bilgi['guvenlik_stogu_ton']:.1f} Ton",
            f"Aylık Talep: {d_aylik:.0f} ± {m_bilgi['talep_std_ton']:.0f} Ton",
        )

        # KURUMSAL ERP TEDARİK EMRİ FİŞİ (PROCUREMENT REQUISITION SLIP)
        if mevcut_stok <= ai_rop:
            onerilen_miktar = max(ai_eoq, ai_rop - mevcut_stok + ai_eoq * 0.5)
            fatura_usd = onerilen_miktar * m_bilgi["guncel_fiyat_usd"]
            fark_ton = ai_eoq - klasik_eoq
            st.markdown(
                f"""<div class="erp-order-card">
                    <div class="erp-header-buy">
                        <span>KURUMSAL TEDARİK EMRİ FİŞİ — DURUM: SİPARİŞ TETİKLENDİ (ONAY BEKLİYOR)</span>
                        <span>BELGE KODU: GRRA-PR-{secili_hid.upper()}-{s_carpan}X</span>
                    </div>
                    <div class="erp-grid">
                        <div class="erp-cell">
                            <div class="erp-cell-label">Envanter Dengesi (Stok / Eşik)</div>
                            <div class="erp-cell-value">{mevcut_stok:.1f} Ton / {ai_rop:.1f} Ton</div>
                        </div>
                        <div class="erp-cell">
                            <div class="erp-cell-label">Önerilen Sipariş Miktarı</div>
                            <div class="erp-cell-value" style="color:#065f46;">{onerilen_miktar:.1f} Ton (EOQ Farkı: {fark_ton:+.1f} T)</div>
                        </div>
                        <div class="erp-cell">
                            <div class="erp-cell-label">Tahmini Sipariş Tutarı (FOB)</div>
                            <div class="erp-cell-value">${fatura_usd:,.0f} USD</div>
                        </div>
                        <div class="erp-cell">
                            <div class="erp-cell-label">Kantitatif Sinyal Gücü</div>
                            <div class="erp-cell-value">Z = {z_sinyal:+.2f} σ (3A: %{beklenen_3ay_pct:+.2f})</div>
                        </div>
                    </div>
                    <div class="erp-rationale">
                        <b>Karar Gerekçesi ve Operasyonel Talimat:</b> Depodaki kullanılabilir envanter seviyesi ({mevcut_stok:.1f} ton),
                        <b>{m_katalog['ad']}</b> tarafından hesaplanan dinamik yeniden sipariş noktasının ({ai_rop:.1f} ton) altına inmiştir.
                        Model 1 aylık ufukta <b>%{beklenen_1ay_pct:+.2f}</b>, 3 aylık ufukta <b>%{beklenen_3ay_pct:+.2f}</b> fiyat değişimi öngörmektedir.
                        Klasik EOQ politikasına ({klasik_eoq:.1f} ton) kıyasla parti büyüklüğü spekülatif formülle <b>{ai_eoq:.1f} ton</b> olarak optimize edilmiştir.
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            rezerv_fazlasi = mevcut_stok - ai_rop
            st.markdown(
                f"""<div class="erp-order-card">
                    <div class="erp-header-hold">
                        <span>ENVANTER KONTROL RAPORU — DURUM: STOK YETERLİ (SİPARİŞ BEKLEMEDE)</span>
                        <span>BELGE KODU: GRRA-MON-{secili_hid.upper()}-{s_carpan}X</span>
                    </div>
                    <div class="erp-grid">
                        <div class="erp-cell">
                            <div class="erp-cell-label">Envanter Dengesi (Stok / Eşik)</div>
                            <div class="erp-cell-value">{mevcut_stok:.1f} Ton / {ai_rop:.1f} Ton</div>
                        </div>
                        <div class="erp-cell">
                            <div class="erp-cell-label">Kullanılabilir Emniyet Tamponu</div>
                            <div class="erp-cell-value" style="color:#1e3a8a;">+{rezerv_fazlasi:.1f} Ton (Tetikleme Üstü)</div>
                        </div>
                        <div class="erp-cell">
                            <div class="erp-cell-label">Planlanan Sonraki Parti (AI EOQ)</div>
                            <div class="erp-cell-value">{ai_eoq:.1f} Ton (${ai_eoq * m_bilgi['guncel_fiyat_usd']:,.0f})</div>
                        </div>
                        <div class="erp-cell">
                            <div class="erp-cell-label">Kantitatif Sinyal Gücü</div>
                            <div class="erp-cell-value">Z = {z_sinyal:+.2f} σ (3A: %{beklenen_3ay_pct:+.2f})</div>
                        </div>
                    </div>
                    <div class="erp-rationale">
                        <b>Karar Gerekçesi ve Operasyonel Talimat:</b> Mevcut envanter pozisyonu ({mevcut_stok:.1f} ton),
                        <b>{m_katalog['ad']}</b> tarafından belirlenen dinamik sipariş eşiğinin ({ai_rop:.1f} ton) üzerindedir.
                        Stok tutma maliyetini (yıllık %{m_bilgi['tutma_orani_pct']}) minimize etmek amacıyla yeni alım emri oluşturulmamıştır.
                        Stok seviyesi {ai_rop:.1f} tona gerilediğinde {ai_eoq:.1f} tonluk sipariş partisi devreye alınacaktır.
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        # KANTİTATİF SİNYAL KADRANI VE FAKTÖR AYRIŞTIRMASI
        col_g1, col_g2 = st.columns([1, 1.45])

        with col_g1:
            st.markdown("#### Bileşik Fiyat Sinyali Göstergesi (Z-Skoru)")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=z_sinyal,
                number={"suffix": " σ", "font": {"size": 26, "color": "#0f172a"}},
                delta={"reference": 0.0, "increasing": {"color": "#b91c1c"}, "decreasing": {"color": "#047857"}},
                title={"text": "Sinyal Ölçeği (-3: Düşüş Beklentisi | +3: Artış Beklentisi)", "font": {"size": 13, "color": "#475569"}},
                gauge={
                    "axis": {"range": [-3.0, 3.0], "tickwidth": 1, "tickcolor": "#64748b"},
                    "bar": {"color": "#0f172a", "thickness": 0.28},
                    "bgcolor": "white",
                    "borderwidth": 1,
                    "bordercolor": "#cbd5e1",
                    "steps": [
                        {"range": [-3.0, -1.0], "color": "#d1fae5"},
                        {"range": [-1.0, 1.0], "color": "#f1f5f9"},
                        {"range": [1.0, 3.0], "color": "#fee2e2"},
                    ],
                    "threshold": {"line": {"color": "#0f172a", "width": 3}, "thickness": 0.75, "value": z_sinyal},
                },
            ))
            fig_gauge.update_layout(template="plotly_white", height=300, margin=dict(l=25, r=25, t=45, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.caption(
                "Metodoloji: 1 ve 3 aylık model tahminlerinin tarihsel standart sapmaya bölünmesiyle elde edilen sinyal/gürültü oranıdır. Tanh fonksiyonu ile EOQ ve ROP denklemlerine aktarılır."
            )

        with col_g2:
            st.markdown(f"#### Kantitatif Faktör Ayrıştırma Analizi ({m_bilgi['hammadde']})")
            faktorler = canli_faktoru_cozumle(fiyat, dis_df, secili_hid)
            f_df = pd.DataFrame({
                "Faktör": list(faktorler.keys()),
                "Z_Skoru": list(faktorler.values()),
            }).sort_values("Z_Skoru", ascending=True)
            f_df["Yön"] = np.where(f_df["Z_Skoru"] >= 0, "Yukarı Yönlü Fiyat Baskısı (+)", "Aşağı Yönlü Fiyat Baskısı (-)")

            fig_xai = px.bar(
                f_df, x="Z_Skoru", y="Faktör", orientation="h", color="Yön",
                text=f_df["Z_Skoru"].apply(lambda x: f"{x:+.2f}"),
                color_discrete_map={"Yukarı Yönlü Fiyat Baskısı (+)": "#b91c1c", "Aşağı Yönlü Fiyat Baskısı (-)": "#047857"},
                template="plotly_white",
            )
            fig_xai.add_vline(x=0.0, line_width=1.2, line_color="#334155")
            fig_xai.update_layout(
                height=330, xaxis_range=[-3.2, 3.2],
                xaxis_title="Standartlaştırılmış Faktör Değeri (Z-Skoru)",
                yaxis_title="", legend_title="Etki Yönü",
                margin=dict(l=10, r=20, t=15, b=20),
            )
            st.plotly_chart(fig_xai, use_container_width=True)

    # =========================================================================
    # SEKME 2: FİYAT TAHMİN PERFORMANSI VE ŞOK ANALİZİ
    # =========================================================================
    with tab2:
        ust_c1, ust_c2 = st.columns([3, 1])
        with ust_c1:
            st.markdown(f"#### {m_bilgi['hammadde']} — Walk-Forward Fiyat Tahmin Performansı ve Şok Analizi")
            st.caption(f"Aktif Model: {m_katalog['ad']} | Eğitim: İleriye Doğru Yürüyen Pencere (Veri Sızıntısız)")
        with ust_c2:
            secili_ufuk = st.radio("Tahmin Ufku (Ay)", options=[1, 3, 6], index=1, horizontal=True)

        t_df = tahmin[
            (tahmin["hammadde_id"] == secili_hid)
            & (tahmin["model"] == secili_model)
            & (tahmin["ufuk"] == secili_ufuk)
        ].copy()

        if not t_df.empty:
            fig_fiyat = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                row_heights=[0.68, 0.32],
                subplot_titles=(
                    f"{m_bilgi['hammadde']} Spot Fiyat Serisi ve {secili_ufuk} Aylık Walk-Forward Tahmin (USD/Ton)",
                    "Gerçekleşen ve Tahmin Edilen Logaritmik Fiyat Değişimi (%)",
                ),
            )
            fig_fiyat.add_trace(
                go.Scatter(x=t_df["hedef_tarih"], y=t_df["gercek_fiyat"], mode="lines",
                           name="Gerçekleşen Spot Fiyat", line=dict(color="#0f172a", width=2.0)),
                row=1, col=1,
            )
            fig_fiyat.add_trace(
                go.Scatter(x=t_df["hedef_tarih"], y=t_df["tahmin_fiyat"], mode="lines",
                           name=f"Model Tahmini ({secili_model})", line=dict(color="#2563eb", width=1.8, dash="dot")),
                row=1, col=1,
            )
            fig_fiyat.add_vrect(
                x0="2021-06-01", x1="2023-06-01", fillcolor="#cbd5e1", opacity=0.35, line_width=0,
                annotation_text="2021-2023 Küresel Arz ve Enerji Şoku Dönemi", annotation_position="top left",
                row=1, col=1,
            )

            fig_fiyat.add_trace(
                go.Bar(x=t_df["hedef_tarih"], y=t_df["gercek_log_degisim"] * 100,
                       name="Gerçek Değişim (%)", marker_color="#94a3b8"),
                row=2, col=1,
            )
            fig_fiyat.add_trace(
                go.Scatter(x=t_df["hedef_tarih"], y=t_df["tahmin_log_degisim"] * 100, mode="lines",
                           name="Tahmin Edilen Değişim (%)", line=dict(color="#b91c1c", width=1.6)),
                row=2, col=1,
            )
            fig_fiyat.update_layout(template="plotly_white", height=540, hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_fiyat, use_container_width=True)

    # =========================================================================
    # SEKME 3: MONTE CARLO SİMÜLASYONU VE MALİYET TASARRUFU (BACKTEST)
    # =========================================================================
    with tab3:
        st.markdown(f"#### 2020–2024 Test Dönemi Maliyet Analizi (200 Talep Senaryosu — {s_carpan}x Sipariş Maliyeti)")
        if not sim_ozet.empty:
            s_sub = sim_ozet[(sim_ozet["S_carpani"] == s_carpan) & (sim_ozet["hammadde"] == secili_hid)].copy()
            sc1, sc2 = st.columns([1.35, 1])

            with sc1:
                goster_kol = ["politika", "net_maliyet_usd", "tasarruf_usd", "rop_eoq_e_gore_pct", "ci95_alt", "ci95_ust", "hizmet_duzeyi", "yillik_siparis"]
                st.dataframe(
                    s_sub[goster_kol].rename(columns={
                        "politika": "Politika",
                        "net_maliyet_usd": "Net Maliyet (USD)",
                        "tasarruf_usd": "Net Tasarruf (USD)",
                        "rop_eoq_e_gore_pct": "ROP-EOQ Farkı (%)",
                        "ci95_alt": "%95 GA Alt",
                        "ci95_ust": "%95 GA Üst",
                        "hizmet_duzeyi": "Hizmet Düzeyi",
                        "yillik_siparis": "Yıllık Sipariş",
                    }),
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    "Muhasebe Denklemi: Net Ekonomik Maliyet = Toplam Nakit Çıkışı (Alım + Sipariş + Tutma + Acil Prim) - (Dönem Sonu Stok Değeri - Dönem Başı Stok Değeri). %95 Güven Aralığı (GA) sıfırı kesmiyorsa tasarruf istatistiksel olarak anlamlıdır."
                )

            with sc2:
                s_plot = s_sub[~s_sub["politika"].isin(["mukemmel", "rop_eoq"])].copy()
                s_plot["Durum"] = np.where(s_plot["tasarruf_usd"] >= 0, "Maliyet Tasarrufu (+)", "Ek Maliyet (-)")
                fig_tas = px.bar(
                    s_plot, x="politika", y="tasarruf_usd", color="Durum",
                    color_discrete_map={"Maliyet Tasarrufu (+)": "#047857", "Ek Maliyet (-)": "#b91c1c"},
                    text=s_plot["tasarruf_usd"].apply(lambda v: f"${v:+,.0f}"),
                    title=f"{m_bilgi['hammadde']} ({s_carpan}x) — Klasik ROP-EOQ'ya Göre Net Tasarruf (USD)",
                    labels={"politika": "Politika", "tasarruf_usd": "Net Tasarruf (USD)"},
                    template="plotly_white",
                )
                fig_tas.add_hline(y=0, line_color="#0f172a", line_width=1.2)
                fig_tas.update_layout(height=310, showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_tas, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Depo Stok Seviyesi ve Sipariş Zamanlaması (Klasik ROP-EOQ vs. Seçili Model Politikası)")
        st_rop = sim_stok[(sim_stok["hammadde_id"] == secili_hid) & (sim_stok["politika"] == "rop_eoq")]
        st_ai = sim_stok[(sim_stok["hammadde_id"] == secili_hid) & (sim_stok["politika"] == secili_model)]
        sp_ai = sim_siparis[(sim_siparis["hammadde_id"] == secili_hid) & (sim_siparis["politika"] == secili_model)]

        if not st_rop.empty and not st_ai.empty:
            fig_stok = go.Figure()
            fig_stok.add_trace(go.Scatter(
                x=st_rop["tarih"], y=st_rop["stok_ton"],
                mode="lines", name="Klasik ROP-EOQ Stoğu (Ton)",
                line=dict(color="#64748b", width=1.5, dash="dash"),
            ))
            fig_stok.add_trace(go.Scatter(
                x=st_ai["tarih"], y=st_ai["stok_ton"],
                mode="lines", name=f"Dinamik Politika Stoğu ({secili_model})",
                line=dict(color="#0f172a", width=2.0),
            ))
            if not sp_ai.empty:
                fig_stok.add_trace(
                    go.Scatter(
                        x=sp_ai["tarih"], y=sp_ai["miktar_ton"],
                        mode="markers", name="Sipariş Parti Miktarı (Ton)",
                        marker=dict(color="#2563eb", size=8, symbol="diamond"),
                    )
                )
            fig_stok.add_vline(x="2020-01-01", line_dash="dot", line_color="#475569")
            fig_stok.update_layout(
                template="plotly_white", height=380,
                xaxis_title="Tarih (2015-2019 Kalibrasyon Dönemi | 2020-2024 Test Dönemi)",
                yaxis_title="Miktar (Ton)", hovermode="x unified",
            )
            st.plotly_chart(fig_stok, use_container_width=True)

    # =========================================================================
    # SEKME 4: İSTATİSTİKSEL DOĞRULAMA (DM TESTİ) VE MODEL REHBERİ
    # =========================================================================
    with tab4:
        st.markdown("#### Sistemde Yer Alan Tahmin Modellerinin Metodolojik Rehberi ve Amaçları")
        rehber_satirlar = []
        for kod, d in MODEL_KATALOGU.items():
            rehber_satirlar.append({
                "Model Kodu": kod,
                "Model Adı": d["ad"],
                "Matematiksel Mimari": d["mimari"],
                "Kullanılan Değişkenler ve Formüller": d["girdiler"],
                "Temel Operasyonel Amacı": d["amac"],
                "En Uygun Piyasa Rejimi": d["rejim"],
            })
        st.dataframe(pd.DataFrame(rehber_satirlar), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Walk-Forward Tahmin Performansı ve Bartlett-HLN Diebold-Mariano İstatistiksel Test Sonuçları")
        if not tah_ozet.empty:
            col_d1, col_d2 = st.columns([1.4, 1])
            with col_d1:
                st.dataframe(tah_ozet, use_container_width=True, hide_index=True)
            with col_d2:
                t3_all = tah_ozet[tah_ozet["ufuk_ay"] == 3].copy()
                fig_yon = px.bar(
                    t3_all, x="hammadde", y="yon_dogrulugu", color="model", barmode="group",
                    title="3 Aylık Tahmin Ufkunda Yön Doğruluğu (Directional Accuracy)",
                    labels={"hammadde": "Hammadde", "yon_dogrulugu": "Yön Doğruluğu Oranı", "model": "Model"},
                    template="plotly_white",
                    color_discrete_sequence=["#0f172a", "#2563eb", "#047857", "#64748b", "#94a3b8", "#cbd5e1"],
                )
                fig_yon.add_hline(y=0.50, line_dash="dash", line_color="#b91c1c", annotation_text="%50 Referans Sınırı")
                fig_yon.update_layout(height=390, yaxis_range=[0.40, 0.62])
                st.plotly_chart(fig_yon, use_container_width=True)

    # =========================================================================
    # SEKME 5: ÇOKLU HAMMADDE, ABC ANALİZİ VE RAPOR İNDİRME
    # =========================================================================
    with tab5:
        st.markdown("#### Envanter Parametreleri Özeti ve ABC Bütçe Sınıflandırması")
        ca1, ca2 = st.columns([1.45, 1])
        with ca1:
            st.dataframe(env_tablo.reset_index(drop=True), use_container_width=True, hide_index=True)
            csv_data = env_tablo.reset_index().to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Envanter Parametreleri Tablosunu İndir (CSV)",
                data=csv_data,
                file_name=f"GRRA_Envanter_Parametreleri_{s_carpan}x.csv",
                mime="text/csv",
            )
        with ca2:
            fig_abc = px.pie(
                env_tablo.reset_index(), values="yillik_harcama_usd", names="hammadde",
                hole=0.45, color="ABC",
                color_discrete_map={"A": "#0f172a", "B": "#2563eb", "C": "#64748b"},
                title="Yıllık Hammadde Harcama Dağılımı ve ABC Sınıfları",
                template="plotly_white",
            )
            fig_abc.update_layout(height=340)
            st.plotly_chart(fig_abc, use_container_width=True)


if __name__ == "__main__":
    main()