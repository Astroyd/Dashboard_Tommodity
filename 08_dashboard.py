"""
GRRA - Yapay Zeka Destekli Satın Alma ve Envanter Karar Destek Sistemi
----------------------------------------------------------------------
Tasarım Dili: Material Soft UI / Bootstrap 5 SaaS Dashboard Stili
Veri Kaynakları: data/envanter.db, data/dis_degiskenler.csv,
                 data/hibrit_tahmin_sonuc.csv, data/hibrit_simulasyon_sonuc.csv

Çalıştırmak için:
    & c:\python315\python.exe -m streamlit run 08_dashboard.py
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
# 1. SAYFA YAPILANDIRMASI VE MATERIAL SOFT UI CSS STİLİ
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GRRA | Satın Alma ve Envanter KDS",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Ana Arka Plan ve Tipografi (Material Soft Gri Zemin) */
    .stApp {
        background-color: #f4f5f7;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1e293b;
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 96%;
    }

    /* Sol Menü (Temiz Beyaz Material Sidebar) */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e5e7eb;
        box-shadow: 4px 0 16px rgba(0, 0, 0, 0.02);
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 0 16px 0;
        border-bottom: 1px solid #f1f5f9;
        margin-bottom: 16px;
    }
    .sidebar-brand-box {
        background: linear-gradient(195deg, #323a54, #1a2035);
        color: #ffffff;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 8px 11px;
        border-radius: 8px;
        letter-spacing: 0.04em;
    }
    .sidebar-brand-text {
        font-size: 0.98rem;
        font-weight: 700;
        color: #1e293b;
        letter-spacing: -0.01em;
    }
    .sidebar-section {
        font-size: 0.72rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 18px;
        margin-bottom: 8px;
    }
    .sidebar-model-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 14px;
        font-size: 0.78rem;
        color: #475569;
        line-height: 1.5;
        margin-top: 6px;
        margin-bottom: 10px;
    }

    /* Üst Breadcrumb ve Başlık Çubuğu */
    .top-navbar {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 18px;
    }
    .breadcrumb {
        font-size: 0.80rem;
        color: #64748b;
        margin-bottom: 4px;
    }
    .breadcrumb span {
        color: #0f172a;
        font-weight: 600;
    }
    .page-title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #1e293b;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .page-subtitle {
        font-size: 0.88rem;
        color: #64748b;
        margin-top: 2px;
    }
    .status-pills {
        display: flex;
        gap: 10px;
        align-items: center;
    }
    .status-pill {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 8px 14px;
        font-size: 0.80rem;
        color: #475569;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
    }
    .status-pill b {
        color: #0f172a;
    }

    /* Üst Navigasyon Sekmeleri (Koyu Antrasit Hap Stili) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #ffffff;
        padding: 8px 10px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        margin-bottom: 22px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        background-color: transparent;
        border-radius: 8px;
        padding: 0 18px;
        font-weight: 600;
        font-size: 0.85rem;
        color: #64748b;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(195deg, #323a54, #1a2035) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 10px rgba(26, 32, 53, 0.25);
    }

    /* Görseldeki Özel KPI Kartları (Sağ Üstte Koyu Antrasit Rozetli) */
    .mat-kpi-card {
        background-color: #ffffff;
        border-radius: 14px;
        padding: 18px 20px 14px 20px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }
    .mat-kpi-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }
    .mat-kpi-label {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 500;
        margin-bottom: 4px;
    }
    .mat-kpi-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: #1e293b;
        letter-spacing: -0.02em;
    }
    .mat-kpi-icon {
        background: linear-gradient(195deg, #323a54, #1a2035);
        color: #ffffff;
        width: 44px;
        height: 44px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        box-shadow: 0 4px 10px rgba(26, 32, 53, 0.22);
    }
    .mat-kpi-divider {
        height: 1px;
        background-color: #f1f5f9;
        margin: 12px 0 10px 0;
    }
    .mat-kpi-footer {
        font-size: 0.80rem;
        color: #64748b;
    }
    .delta-pos {
        color: #16a34a;
        font-weight: 700;
    }
    .delta-neg {
        color: #dc2626;
        font-weight: 700;
    }
    .delta-neutral {
        color: #2563eb;
        font-weight: 700;
    }

    /* Grafik ve İçerik Kartları (White Surface Cards) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border-radius: 14px !important;
        border: 1px solid #e5e7eb !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03) !important;
        padding: 6px 10px !important;
    }
    .card-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 2px;
    }
    .card-subtitle {
        font-size: 0.82rem;
        color: #64748b;
        margin-bottom: 10px;
    }
    .card-footer {
        border-top: 1px solid #f1f5f9;
        padding-top: 8px;
        margin-top: 4px;
        font-size: 0.76rem;
        color: #94a3b8;
    }

    /* Tedarik Emri / Sipariş Özeti Kartı (Orders Overview Stili) */
    .order-banner {
        background-color: #ffffff;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        padding: 20px 24px;
        margin-bottom: 22px;
    }
    .order-banner-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 12px;
        margin-bottom: 14px;
    }
    .order-badge-buy {
        background-color: #dcfce7;
        color: #15803d;
        font-weight: 700;
        font-size: 0.76rem;
        padding: 5px 12px;
        border-radius: 6px;
        letter-spacing: 0.04em;
    }
    .order-badge-hold {
        background-color: #e0f2fe;
        color: #0369a1;
        font-weight: 700;
        font-size: 0.76rem;
        padding: 5px 12px;
        border-radius: 6px;
        letter-spacing: 0.04em;
    }
    .order-metrics-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 14px 18px;
        border: 1px solid #f1f5f9;
        margin-bottom: 12px;
    }
    .order-metric-item small {
        display: block;
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 3px;
    }
    .order-metric-item span {
        font-size: 1.02rem;
        font-weight: 700;
        color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

DB = Path("data/envanter.db")
DIS_CSV = Path("data/dis_degiskenler.csv")
TAHMIN_CSV = Path("data/hibrit_tahmin_sonuc.csv")
SIM_CSV = Path("data/hibrit_simulasyon_sonuc.csv")

ADLAR = {"pamuk": "Pamuk", "aluminyum": "Alüminyum", "bugday": "Buğday"}

MODEL_KATALOGU = {
    "quant_yon_hibrit": {
        "ad": "MIDAS Olasılıksal Yön ve Huber Hibriti (quant_yon_hibrit)",
        "kisa_ad": "MIDAS Olasılıksal Yön Hibriti (Şampiyon)",
        "mimari": "Huber Regresyon + L2 Lojistik Sınıflandırıcı + Makro Ridge",
        "girdiler": "CFTC COT Smart Money Spread, Dolar Endeksi (DXY), VIX, Maliyet Marjı, Hasat Anomalisi ve Vol-MACD.",
        "amac": "Aykırı fiyat şoklarından etkilenmeden yön olasılığını tahmin etmek ve tüm senaryolarda sıfır zarar ile maksimum tasarruf sağlamak.",
        "rejim": "Kriz/şok dönemleri ve standart piyasa rejimleri (En yüksek net tasarruf: $101,459).",
    },
    "ridge_dis": {
        "ad": "Dış Değişkenli Ridge Regresyon (ridge_dis)",
        "kisa_ad": "Ridge + Makro Dışsal (ridge_dis)",
        "mimari": "L2 Cezalı Doğrusal Regresyon + TimeSeriesSplit Çapraz Doğrulama",
        "girdiler": "Maliyet Marjı (Cost Spread), Volatilite MACD ve 7 Makro Emtia (Petrol, Doğal Gaz, Kömür, Üre, Mısır, Soya, Bakır).",
        "amac": "Hammadde fiyatını üretim girdileri ve ikame ürün maliyetleriyle birlikte modelleyerek kısa-orta vadeli alım zamanlamasını optimize etmek.",
        "rejim": "Enflasyonist dönemler, enerji krizleri ve arz-talep şokları.",
    },
    "hibrit_ensemble": {
        "ad": "Hibrit Kantitatif Formül + ML Topluluğu (hibrit_ensemble)",
        "kisa_ad": "Hibrit Formül + ML (hibrit_ensemble)",
        "mimari": "Ters-MSE Ağırlıklı Kantitatif Uzman Formüller + Ridge Birleşimi",
        "girdiler": "MR60, MR120, Maliyet Marjı Z-Skoru, Volatilite MACD, Kısa Vade Sharpe Momentum ve Dış Piyasa İvmesi.",
        "amac": "Trend takipçisi ve ortalamaya dönüş formüllerini geçmiş hata performanslarına göre harmanlayarak tahmin varyansını düşürmek.",
        "rejim": "Yönün belirsiz olduğu geçiş dönemleri ve çoklu ufuk (1-6 ay) planlaması.",
    },
    "hibrit_ridge": {
        "ad": "Üstel Ağırlıklı Hibrit Ridge (hibrit_ridge)",
        "kisa_ad": "Üstel Ağırlıklı Ridge (hibrit_ridge)",
        "mimari": "Yakın Geçmiş Odaklı (Exponentially Weighted) L2 Regresyon",
        "girdiler": "6 Kantitatif Finans Sinyali + Mevsimsellik Bileşenleri (Sin/Cos).",
        "amac": "Eski piyasa verilerinin ağırlığını azaltıp son yıllardaki yapısal kırılmalara ve güncel piyasa rejimine hızla adapte olmak.",
        "rejim": "Yapısal kırılma sonrası yeni fiyat dengelerinin oluştuğu piyasalar.",
    },
    "mr_maliyet": {
        "ad": "Maliyet Düzeltilmiş Ortalamaya Dönüş (mr_maliyet)",
        "kisa_ad": "Maliyet Bazlı Mean-Reversion (mr_maliyet)",
        "mimari": "Sabit Terimli OLS + 10 Yıllık Denge ve Üretim Maliyeti Spread Formülü",
        "girdiler": "120 Aylık Ortalamadan Sapma (MR120) ve Üretim Maliyeti / İkame Marjı (Cost Spread).",
        "amac": "Fiyatın üretim maliyetine ve uzun vadeli tarihsel ortalamasına göre aşırı ucuz veya pahalı olduğu bölgeleri tespit etmek.",
        "rejim": "Yatay/dalgalı piyasalar ve uzun vadeli (6-12 ay) kontrat kararları.",
    },
    "ridge": {
        "ad": "Standart İçsel Ridge Regresyon (ridge)",
        "kisa_ad": "Standart İçsel Ridge (ridge)",
        "mimari": "Standartlaştırılmış L2 Doğrusal Regresyon (Alpha=10.0)",
        "girdiler": "Yalnızca hammaddenin kendi geçmiş getirileri (lag 1..6), 3-12 aylık momentum ve oynaklık.",
        "amac": "Dış veri gerektirmeden sadece içsel fiyat hareketleriyle temel referans tahmini üretmek.",
        "rejim": "Dışsal makro şokların olmadığı standart piyasa koşulları.",
    },
}

MALIYET_SEPETI = {
    "pamuk": {"petrol_brent": 0.35, "soya": 0.25, "misir": 0.20, "ure": 0.20},
    "aluminyum": {"dogalgaz_ab": 0.30, "komur": 0.30, "petrol_brent": 0.20, "bakir": 0.20},
    "bugday": {"misir": 0.35, "soya": 0.25, "ure": 0.25, "petrol_brent": 0.15},
}


# ---------------------------------------------------------------------------
# 2. YARDIMCI GÖRSEL BİLEŞEN FONKSİYONLARI (MATERIAL KARTLAR)
# ---------------------------------------------------------------------------
def material_kpi_karti(baslik: str, deger: str, rozet: str, alt_vurgulu: str, alt_aciklama: str, durum: str = "pos"):
    renk_sinifi = {"pos": "delta-pos", "neg": "delta-neg", "neutral": "delta-neutral"}.get(durum, "delta-neutral")
    st.markdown(
        f"""<div class="mat-kpi-card">
            <div class="mat-kpi-top">
                <div>
                    <div class="mat-kpi-label">{baslik}</div>
                    <div class="mat-kpi-value">{deger}</div>
                </div>
                <div class="mat-kpi-icon">{rozet}</div>
            </div>
            <div class="mat-kpi-divider"></div>
            <div class="mat-kpi-footer">
                <span class="{renk_sinifi}">{alt_vurgulu}</span> {alt_aciklama}
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 3. VERİ YÜKLEME VE HESAPLAMA FONKSİYONLARI
# ---------------------------------------------------------------------------
@st.cache_data
def verileri_yukle():
    if not DB.exists():
        st.error("HATA: data/envanter.db veritabanı bulunamadı. Lütfen önce 03, 18 ve 17 numaralı scriptleri çalıştırın.")
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
# 4. ANA EKRAN VE MATERIAL DASHBOARD AKIŞI
# ---------------------------------------------------------------------------
def main():
    ham, fiyat, tuketim, tahmin, sim_stok, sim_siparis, dis_df, tah_ozet, sim_ozet = verileri_yukle()

    # =========================================================================
    # SOL MENÜ (MATERIAL SIDEBAR)
    # =========================================================================
    st.sidebar.markdown(
        """<div class="sidebar-brand">
            <div class="sidebar-brand-box">GRRA</div>
            <div class="sidebar-brand-text">Karar Destek Platformu</div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('<div class="sidebar-section">Emtia ve Model Yapılandırması</div>', unsafe_allow_html=True)

    secili_hid = st.sidebar.selectbox(
        "Hammadde Seçimi",
        options=list(ADLAR.keys()),
        format_func=lambda x: f"{ADLAR[x]} ({x.upper()})",
    )

    mevcut_modeller = [m for m in MODEL_KATALOGU.keys() if m in tahmin["model"].unique()]
    secili_model = st.sidebar.selectbox(
        "Aktif Tahmin Algoritması",
        options=mevcut_modeller,
        format_func=lambda x: MODEL_KATALOGU[x]["kisa_ad"],
        index=0,
    )

    m_katalog = MODEL_KATALOGU[secili_model]
    st.sidebar.markdown(
        f"""<div class="sidebar-model-card">
        <b style="color:#0f172a;">Mimari:</b> {m_katalog['mimari']}<br><br>
        <b style="color:#0f172a;">Operasyonel Amaç:</b> {m_katalog['amac']}<br><br>
        <b style="color:#0f172a;">Girdiler:</b> {m_katalog['girdiler']}
        </div>""",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('<div class="sidebar-section">Operasyonel Kısıtlar</div>', unsafe_allow_html=True)

    s_carpan = st.sidebar.select_slider(
        "Sipariş Maliyeti Çarpanı (S)",
        options=[1, 5, 10],
        value=5,
    )

    hizmet_secim = st.sidebar.select_slider(
        "Hedef Hizmet Düzeyi",
        options=["%90 (Z=1.28)", "%95 (Z=1.65)", "%99 (Z=2.33)"],
        value="%95 (Z=1.65)",
    )
    z_degeri = {"%90 (Z=1.28)": 1.28, "%95 (Z=1.65)": 1.65, "%99 (Z=2.33)": 2.33}[hizmet_secim]

    env_tablo = envanter_tablosu_uret(ham, fiyat, tuketim, s_carpan=float(s_carpan), z_skor_hizmet=z_degeri)
    m_bilgi = env_tablo.loc[secili_hid]

    st.sidebar.markdown('<div class="sidebar-section">Canlı Depo Kalibrasyonu</div>', unsafe_allow_html=True)

    mevcut_stok = st.sidebar.number_input(
        f"Mevcut {m_bilgi['hammadde']} Stoğu (Ton)",
        min_value=0.0,
        max_value=float(round(m_bilgi["ROP_ton"] * 3.0, 0)),
        value=float(round(m_bilgi["ROP_ton"] * 0.90, 1)),
        step=5.0,
    )

    beta_agresiflik = st.sidebar.slider(
        "Spekülatif Esneklik Katsayısı (Beta)",
        min_value=0.10,
        max_value=0.60,
        value=0.35,
        step=0.05,
    )

    # =========================================================================
    # CANLI HESAPLAMALAR
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
    fiyat_fark_12a_pct = ((m_bilgi["guncel_fiyat_usd"] / m_bilgi["ort_fiyat_usd"]) - 1.0) * 100.0
    eoq_degisim_pct = ((ai_eoq / klasik_eoq) - 1.0) * 100.0

    # =========================================================================
    # ÜST BREADCRUMB BAŞLIK VE HAP SEKMELER (MATERIAL TOPBAR)
    # =========================================================================
    st.markdown(
        f"""<div class="top-navbar">
            <div>
                <div class="breadcrumb">Modüller &nbsp;/&nbsp; <span>Satın Alma ve Envanter Kontrol Paneli</span></div>
                <div class="page-title">{m_bilgi['hammadde']} Tedarik ve Optimizasyon Kokpiti</div>
                <div class="page-subtitle">Kantitatif fiyat sinyalleri, dinamik EOQ/ROP karar motoru ve Monte Carlo maliyet doğrulaması.</div>
            </div>
            <div class="status-pills">
                <div class="status-pill">ABC Sınıfı: <b>{m_bilgi['ABC']} (%{m_bilgi['butce_payi_pct']} Bütçe)</b></div>
                <div class="status-pill">Aktif Model: <b>{secili_model}</b></div>
                <div class="status-pill">Sipariş Maliyeti: <b>${m_bilgi['siparis_maliyeti_usd']:,.0f} ({s_carpan}x)</b></div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Karar Kokpiti ve Sinyal Analizi",
        "Fiyat Tahmini ve Şok Analizi",
        "Politika Simülasyonu ve Backtest",
        "İstatistiksel Doğrulama (DM Testi)",
        "Envanter Parametreleri ve ABC",
    ])

    # =========================================================================
    # SEKME 1: KARAR KOKPİTİ (GÖRSELDEKİ 4'LÜ KPI + 3'LÜ GRAFİK KART DÜZENİ)
    # =========================================================================
    with tab1:
        # 1. SATIR: GÖRSELDEKİ SAĞ ÜSTÜ SİYAH ROZETLİ 4 KPI KARTI
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            material_kpi_karti(
                baslik="Güncel Spot Fiyat",
                deger=f"${m_bilgi['guncel_fiyat_usd']:,.1f}",
                rozet="USD",
                alt_vurgulu=f"%{fiyat_fark_12a_pct:+.1f}",
                alt_aciklama="son 12 aylık ortalamaya göre",
                durum="neg" if fiyat_fark_12a_pct > 0 else "pos",
            )
        with k2:
            material_kpi_karti(
                baslik="3 Aylık Hedef Fiyat",
                deger=f"${hedef_fiyat_3ay:,.1f}",
                rozet="3AY",
                alt_vurgulu=f"%{beklenen_3ay_pct:+.2f}",
                alt_aciklama=f"beklenen değişim ({secili_model})",
                durum="neg" if beklenen_3ay_pct > 0 else "pos",
            )
        with k3:
            material_kpi_karti(
                baslik="Dinamik Sipariş Eşiği (ROP)",
                deger=f"{ai_rop:.1f} T",
                rozet="ROP",
                alt_vurgulu=f"{ai_rop - klasik_rop:+.1f} Ton",
                alt_aciklama=f"klasik ROP'a göre ({klasik_rop:.1f} T)",
                durum="neutral",
            )
        with k4:
            material_kpi_karti(
                baslik="Spekülatif Parti (AI EOQ)",
                deger=f"{ai_eoq:.1f} T",
                rozet="EOQ",
                alt_vurgulu=f"%{eoq_degisim_pct:+.1f}",
                alt_aciklama=f"klasik EOQ'ya göre ({klasik_eoq:.1f} T)",
                durum="pos" if eoq_degisim_pct >= 0 else "neutral",
            )

        # 2. SATIR: GÖRSELDEKİ 3 YAN YANA BEYAZ GRAFİK KARTI (SİNYAL, XAI, BACKTEST ÖZETİ)
        c_card1, c_card2, c_card3 = st.columns(3)

        with c_card1:
            with st.container(border=True):
                st.markdown('<div class="card-title">Kantitatif Faktör Ayrıştırması</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card-subtitle">Bileşik Sinyal Gücü: <b>{z_sinyal:+.2f} σ</b> (Anlık Z-Skoru)</div>', unsafe_allow_html=True)

                faktorler = canli_faktoru_cozumle(fiyat, dis_df, secili_hid)
                f_df = pd.DataFrame({
                    "Faktör": [
                        "Maliyet Marjı", "MR120 (10Y)", "MR60 (5Y)",
                        "Vol-MACD", "Momentum (3A)", "Makro İvme"
                    ],
                    "Z_Skoru": list(faktorler.values()),
                })
                renkler = ["#16a34a" if v >= 0 else "#334155" for v in f_df["Z_Skoru"]]

                fig_f = go.Figure(go.Bar(
                    x=f_df["Faktör"], y=f_df["Z_Skoru"],
                    marker_color=renkler, marker_line_width=0, width=0.52,
                ))
                fig_f.add_hline(y=0, line_color="#cbd5e1", line_width=1)
                fig_f.update_layout(
                    template="plotly_white", height=230,
                    margin=dict(l=10, r=10, t=10, b=10),
                    yaxis=dict(range=[-3.0, 3.0], gridcolor="#f1f5f9", title=""),
                    xaxis=dict(title="", tickfont=dict(size=10)),
                )
                st.plotly_chart(fig_f, use_container_width=True)
                st.markdown('<div class="card-footer">Güncelleme: Aylık ve dışsal makro seriler aktif</div>', unsafe_allow_html=True)

        with c_card2:
            with st.container(border=True):
                st.markdown('<div class="card-title">Fiyat Trendi ve 3 Aylık Tahmin</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card-subtitle">Son 36 aylık gerçekleşen ve tahmin edilen yörünge</div>', unsafe_allow_html=True)

                t3_kisa = tahmin[
                    (tahmin["hammadde_id"] == secili_hid)
                    & (tahmin["model"] == secili_model)
                    & (tahmin["ufuk"] == 3)
                ].tail(36)

                fig_tr = go.Figure()
                if not t3_kisa.empty:
                    fig_tr.add_trace(go.Scatter(
                        x=t3_kisa["hedef_tarih"], y=t3_kisa["gercek_fiyat"],
                        mode="lines+markers", name="Gerçek",
                        line=dict(color="#16a34a", width=2.2), marker=dict(size=4),
                    ))
                    fig_tr.add_trace(go.Scatter(
                        x=t3_kisa["hedef_tarih"], y=t3_kisa["tahmin_fiyat"],
                        mode="lines", name="AI Tahmin",
                        line=dict(color="#1e293b", width=1.8, dash="dot"),
                    ))
                fig_tr.update_layout(
                    template="plotly_white", height=230, showlegend=False,
                    margin=dict(l=10, r=10, t=10, b=10),
                    yaxis=dict(gridcolor="#f1f5f9", title=""),
                    xaxis=dict(title="", tickfont=dict(size=10)),
                )
                st.plotly_chart(fig_tr, use_container_width=True)
                st.markdown(f'<div class="card-footer">Model: {m_katalog["kisa_ad"]}</div>', unsafe_allow_html=True)

        with c_card3:
            with st.container(border=True):
                st.markdown('<div class="card-title">Depo Stok Yolu (Son 36 Ay)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card-subtitle">Güvenlik Stoğu: <b>{m_bilgi["guvenlik_stogu_ton"]:.1f} T</b> | Hizmet: <b>{hizmet_secim[:3]}</b></div>', unsafe_allow_html=True)

                st_kisa = sim_stok[
                    (sim_stok["hammadde_id"] == secili_hid)
                    & (sim_stok["politika"] == secili_model)
                ].tail(36)
                if st_kisa.empty:
                    st_kisa = sim_stok[(sim_stok["hammadde_id"] == secili_hid) & (sim_stok["politika"] == "rop_eoq")].tail(36)

                fig_st_mini = go.Figure()
                if not st_kisa.empty:
                    fig_st_mini.add_trace(go.Scatter(
                        x=st_kisa["tarih"], y=st_kisa["stok_ton"],
                        mode="lines+markers", name="Stok (Ton)",
                        line=dict(color="#16a34a", width=2.2), marker=dict(size=4),
                    ))
                    fig_st_mini.add_hline(y=m_bilgi["guvenlik_stogu_ton"], line_dash="dash", line_color="#94a3b8")
                fig_st_mini.update_layout(
                    template="plotly_white", height=230, showlegend=False,
                    margin=dict(l=10, r=10, t=10, b=10),
                    yaxis=dict(gridcolor="#f1f5f9", title=""),
                    xaxis=dict(title="", tickfont=dict(size=10)),
                )
                st.plotly_chart(fig_st_mini, use_container_width=True)
                st.markdown('<div class="card-footer">Simülasyon: 200 stokastik talep senaryosu</div>', unsafe_allow_html=True)

        # 3. SATIR: KURUMSAL TEDARİK EMRİ KARTI (ALT BÖLÜM)
        if mevcut_stok <= ai_rop:
            onerilen_miktar = max(ai_eoq, ai_rop - mevcut_stok + ai_eoq * 0.5)
            fatura_usd = onerilen_miktar * m_bilgi["guncel_fiyat_usd"]
            fark_ton = ai_eoq - klasik_eoq
            st.markdown(
                f"""<div class="order-banner">
                    <div class="order-banner-head">
                        <div>
                            <div class="card-title">Kurumsal Tedarik Emri ve Operasyonel Reçete</div>
                            <div class="card-subtitle" style="margin-bottom:0;">Belge Kodu: GRRA-PR-{secili_hid.upper()}-{s_carpan}X &nbsp;|&nbsp; Aktif Politika: {m_katalog['ad']}</div>
                        </div>
                        <div class="order-badge-buy">SİPARİŞ TETİKLENDİ — ONAY BEKLİYOR</div>
                    </div>
                    <div class="order-metrics-row">
                        <div class="order-metric-item"><small>Mevcut Stok / Dinamik ROP</small><span>{mevcut_stok:.1f} T / {ai_rop:.1f} T</span></div>
                        <div class="order-metric-item"><small>Önerilen Sipariş Miktarı</small><span style="color:#15803d;">{onerilen_miktar:.1f} Ton ({fark_ton:+.1f} T EOQ Farkı)</span></div>
                        <div class="order-metric-item"><small>Tahmini Fatura Tutarı (FOB)</small><span>${fatura_usd:,.0f} USD</span></div>
                        <div class="order-metric-item"><small>Bileşik Sinyal Z-Skoru</small><span>{z_sinyal:+.2f} σ (1A: %{beklenen_1ay_pct:+.2f} | 3A: %{beklenen_3ay_pct:+.2f})</span></div>
                    </div>
                    <div style="font-size:0.86rem; color:#475569; line-height:1.55;">
                        <b>Karar Gerekçesi:</b> Depodaki kullanılabilir stok seviyesi ({mevcut_stok:.1f} ton), dinamik yeniden sipariş noktasının ({ai_rop:.1f} ton) altındadır.
                        Klasik EOQ politikasına ({klasik_eoq:.1f} ton) kıyasla sipariş partisi spekülatif formülle <b>{ai_eoq:.1f} ton</b> olarak optimize edilmiştir.
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            rezerv_fazlasi = mevcut_stok - ai_rop
            st.markdown(
                f"""<div class="order-banner">
                    <div class="order-banner-head">
                        <div>
                            <div class="card-title">Kurumsal Tedarik Emri ve Operasyonel Reçete</div>
                            <div class="card-subtitle" style="margin-bottom:0;">Belge Kodu: GRRA-MON-{secili_hid.upper()}-{s_carpan}X &nbsp;|&nbsp; Aktif Politika: {m_katalog['ad']}</div>
                        </div>
                        <div class="order-badge-hold">STOK YETERLİ — SİPARİŞ BEKLEMEDE</div>
                    </div>
                    <div class="order-metrics-row">
                        <div class="order-metric-item"><small>Mevcut Stok / Dinamik ROP</small><span>{mevcut_stok:.1f} T / {ai_rop:.1f} T</span></div>
                        <div class="order-metric-item"><small>Emniyet Tamponu Fazlası</small><span style="color:#0369a1;">+{rezerv_fazlasi:.1f} Ton (Eşik Üstü)</span></div>
                        <div class="order-metric-item"><small>Planlanan Sonraki Parti</small><span>{ai_eoq:.1f} Ton (${ai_eoq * m_bilgi['guncel_fiyat_usd']:,.0f})</span></div>
                        <div class="order-metric-item"><small>Bileşik Sinyal Z-Skoru</small><span>{z_sinyal:+.2f} σ (1A: %{beklenen_1ay_pct:+.2f} | 3A: %{beklenen_3ay_pct:+.2f})</span></div>
                    </div>
                    <div style="font-size:0.86rem; color:#475569; line-height:1.55;">
                        <b>Karar Gerekçesi:</b> Mevcut envanter pozisyonu ({mevcut_stok:.1f} ton), dinamik yeniden sipariş eşiğinin ({ai_rop:.1f} ton) üzerindedir.
                        Stok tutma maliyetini (yıllık %{m_bilgi['tutma_orani_pct']}) minimize etmek amacıyla yeni alım emri oluşturulmamıştır.
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

    # =========================================================================
    # SEKME 2: FİYAT TAHMİN PERFORMANSI VE ŞOK ANALİZİ
    # =========================================================================
    with tab2:
        with st.container(border=True):
            ust_c1, ust_c2 = st.columns([3, 1])
            with ust_c1:
                st.markdown(f'<div class="card-title">{m_bilgi["hammadde"]} — Walk-Forward Fiyat Tahmin Performansı (2012–2024)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="card-subtitle">Aktif Model: {m_katalog["ad"]} | Veri Sızıntısız İleriye Doğru Yürüyen Pencere</div>', unsafe_allow_html=True)
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
                               name="Gerçekleşen Spot Fiyat", line=dict(color="#1e293b", width=2.2)),
                    row=1, col=1,
                )
                fig_fiyat.add_trace(
                    go.Scatter(x=t_df["hedef_tarih"], y=t_df["tahmin_fiyat"], mode="lines",
                               name=f"Model Tahmini ({secili_model})", line=dict(color="#16a34a", width=2.0, dash="dot")),
                    row=1, col=1,
                )
                fig_fiyat.add_vrect(
                    x0="2021-06-01", x1="2023-06-01", fillcolor="#cbd5e1", opacity=0.30, line_width=0,
                    annotation_text="2021-2023 Küresel Arz ve Enerji Şoku", annotation_position="top left",
                    row=1, col=1,
                )
                fig_fiyat.add_trace(
                    go.Bar(x=t_df["hedef_tarih"], y=t_df["gercek_log_degisim"] * 100,
                           name="Gerçek Değişim (%)", marker_color="#cbd5e1"),
                    row=2, col=1,
                )
                fig_fiyat.add_trace(
                    go.Scatter(x=t_df["hedef_tarih"], y=t_df["tahmin_log_degisim"] * 100, mode="lines",
                               name="Tahmin Edilen Değişim (%)", line=dict(color="#16a34a", width=1.8)),
                    row=2, col=1,
                )
                fig_fiyat.update_layout(template="plotly_white", height=520, hovermode="x unified", margin=dict(l=15, r=15, t=40, b=15))
                st.plotly_chart(fig_fiyat, use_container_width=True)

    # =========================================================================
    # SEKME 3: MONTE CARLO SİMÜLASYONU VE MALİYET TASARRUFU (BACKTEST)
    # =========================================================================
    with tab3:
        sc1, sc2 = st.columns([1.35, 1])
        if not sim_ozet.empty:
            s_sub = sim_ozet[(sim_ozet["S_carpani"] == s_carpan) & (sim_ozet["hammadde"] == secili_hid)].copy()

            with sc1:
                with st.container(border=True):
                    st.markdown(f'<div class="card-title">2020–2024 Test Dönemi Maliyet Tablosu ({s_carpan}x Sipariş Maliyeti)</div>', unsafe_allow_html=True)
                    st.markdown('<div class="card-subtitle">200 eşleştirilmiş Monte Carlo talep senaryosu üzerinden net ekonomik maliyetler</div>', unsafe_allow_html=True)
                    goster_kol = ["politika", "net_maliyet_usd", "tasarruf_usd", "rop_eoq_e_gore_pct", "ci95_alt", "ci95_ust", "hizmet_duzeyi", "yillik_siparis"]
                    st.dataframe(
                        s_sub[goster_kol].rename(columns={
                            "politika": "Politika",
                            "net_maliyet_usd": "Net Maliyet ($)",
                            "tasarruf_usd": "Net Tasarruf ($)",
                            "rop_eoq_e_gore_pct": "ROP-EOQ Farkı (%)",
                            "ci95_alt": "%95 GA Alt",
                            "ci95_ust": "%95 GA Üst",
                            "hizmet_duzeyi": "Hizmet Düzeyi",
                            "yillik_siparis": "Yıllık Sipariş",
                        }),
                        use_container_width=True, hide_index=True,
                    )
                    st.markdown('<div class="card-footer">Net Maliyet = Nakit Çıkışı - Dönem Sonu/Başı Stok Değer Farkı</div>', unsafe_allow_html=True)

            with sc2:
                with st.container(border=True):
                    st.markdown(f'<div class="card-title">Klasik ROP-EOQ\'ya Göre Net Tasarruf (USD)</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="card-subtitle">{m_bilgi["hammadde"]} ({s_carpan}x) senaryosunda modellerin nakit katkısı</div>', unsafe_allow_html=True)
                    s_plot = s_sub[~s_sub["politika"].isin(["mukemmel", "rop_eoq"])].copy()
                    renk_bar = ["#16a34a" if v >= 0 else "#dc2626" for v in s_plot["tasarruf_usd"]]
                    fig_tas = go.Figure(go.Bar(
                        x=s_plot["politika"], y=s_plot["tasarruf_usd"],
                        marker_color=renk_bar, width=0.48,
                        text=s_plot["tasarruf_usd"].apply(lambda v: f"${v:+,.0f}"),
                        textposition="auto",
                    ))
                    fig_tas.add_hline(y=0, line_color="#1e293b", line_width=1.2)
                    fig_tas.update_layout(template="plotly_white", height=295, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig_tas, use_container_width=True)
                    st.markdown('<div class="card-footer">Pozitif değerler klasik EOQ politikasına göre net kârı gösterir</div>', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="card-title">Depo Stok Seviyesi ve Sipariş Zamanlaması Karşılaştırması</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card-subtitle">Klasik ROP-EOQ politikası ile {secili_model} politikasının 2015–2024 stok yörüngesi</div>', unsafe_allow_html=True)
            st_rop = sim_stok[(sim_stok["hammadde_id"] == secili_hid) & (sim_stok["politika"] == "rop_eoq")]
            st_ai = sim_stok[(sim_stok["hammadde_id"] == secili_hid) & (sim_stok["politika"] == secili_model)]
            sp_ai = sim_siparis[(sim_siparis["hammadde_id"] == secili_hid) & (sim_siparis["politika"] == secili_model)]

            if not st_rop.empty and not st_ai.empty:
                fig_stok = go.Figure()
                fig_stok.add_trace(go.Scatter(
                    x=st_rop["tarih"], y=st_rop["stok_ton"],
                    mode="lines", name="Klasik ROP-EOQ Stoğu (Ton)",
                    line=dict(color="#94a3b8", width=1.5, dash="dash"),
                ))
                fig_stok.add_trace(go.Scatter(
                    x=st_ai["tarih"], y=st_ai["stok_ton"],
                    mode="lines", name=f"Dinamik Politika Stoğu ({secili_model})",
                    line=dict(color="#16a34a", width=2.2),
                ))
                if not sp_ai.empty:
                    fig_stok.add_trace(go.Scatter(
                        x=sp_ai["tarih"], y=sp_ai["miktar_ton"],
                        mode="markers", name="Sipariş Parti Miktarı (Ton)",
                        marker=dict(color="#1e293b", size=7, symbol="diamond"),
                    ))
                fig_stok.add_vline(x="2020-01-01", line_dash="dot", line_color="#64748b")
                fig_stok.update_layout(template="plotly_white", height=350, hovermode="x unified", margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_stok, use_container_width=True)

    # =========================================================================
    # SEKME 4: İSTATİSTİKSEL DOĞRULAMA (DM TESTİ) VE MODEL REHBERİ
    # =========================================================================
    with tab4:
        with st.container(border=True):
            st.markdown('<div class="card-title">Tahmin Modellerinin Metodolojik Rehberi ve Operasyonel Amaçları</div>', unsafe_allow_html=True)
            st.markdown('<div class="card-subtitle">Sistemde yer alan algoritmaların matematiksel mimarisi ve uygun piyasa rejimleri</div>', unsafe_allow_html=True)
            rehber_satirlar = [
                {
                    "Model Kodu": kod,
                    "Model Adı": d["ad"],
                    "Matematiksel Mimari": d["mimari"],
                    "Kullanılan Değişkenler": d["girdiler"],
                    "Operasyonel Amacı": d["amac"],
                }
                for kod, d in MODEL_KATALOGU.items()
            ]
            st.dataframe(pd.DataFrame(rehber_satirlar), use_container_width=True, hide_index=True)

        if not tah_ozet.empty:
            col_d1, col_d2 = st.columns([1.4, 1])
            with col_d1:
                with st.container(border=True):
                    st.markdown('<div class="card-title">Walk-Forward ve Diebold-Mariano (Bartlett-HLN) Sonuçları</div>', unsafe_allow_html=True)
                    st.markdown('<div class="card-subtitle">1, 3 ve 6 aylık ufuklarda tüm modellerin hata ve yön doğruluğu metrikleri</div>', unsafe_allow_html=True)
                    st.dataframe(tah_ozet, use_container_width=True, hide_index=True)
            with col_d2:
                with st.container(border=True):
                    st.markdown('<div class="card-title">3 Aylık Ufukta Yön Doğruluğu Kıyaslaması</div>', unsafe_allow_html=True)
                    st.markdown('<div class="card-subtitle">%50 rastgele yürüyüş eşiğine göre yön başarısı</div>', unsafe_allow_html=True)
                    t3_all = tah_ozet[tah_ozet["ufuk_ay"] == 3].copy()
                    fig_yon = px.bar(
                        t3_all, x="hammadde", y="yon_dogrulugu", color="model", barmode="group",
                        template="plotly_white",
                        color_discrete_sequence=["#16a34a", "#1e293b", "#2563eb", "#64748b", "#94a3b8", "#cbd5e1"],
                    )
                    fig_yon.add_hline(y=0.50, line_dash="dash", line_color="#dc2626")
                    fig_yon.update_layout(height=360, yaxis_range=[0.40, 0.63], margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig_yon, use_container_width=True)

    # =========================================================================
    # SEKME 5: ENVANTER PARAMETRELERİ VE ABC ANALİZİ
    # =========================================================================
    with tab5:
        ca1, ca2 = st.columns([1.45, 1])
        with ca1:
            with st.container(border=True):
                st.markdown('<div class="card-title">Şirket Envanter Parametreleri ve Optimal EOQ/ROP Özeti</div>', unsafe_allow_html=True)
                st.markdown('<div class="card-subtitle">Yıllık tüketim, sabit sipariş maliyeti ve tutma oranlarına göre hesaplanan temel değerler</div>', unsafe_allow_html=True)
                st.dataframe(env_tablo.reset_index(drop=True), use_container_width=True, hide_index=True)
                csv_data = env_tablo.reset_index().to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Envanter Reçetesini İndir (.CSV)",
                    data=csv_data,
                    file_name=f"GRRA_Envanter_Parametreleri_{s_carpan}x.csv",
                    mime="text/csv",
                )
        with ca2:
            with st.container(border=True):
                st.markdown('<div class="card-title">Yıllık Hammadde Harcama Dağılımı (ABC Analizi)</div>', unsafe_allow_html=True)
                st.markdown('<div class="card-subtitle">Toplam satın alma bütçesinin hammaddelere göre dağılımı</div>', unsafe_allow_html=True)
                fig_abc = px.pie(
                    env_tablo.reset_index(), values="yillik_harcama_usd", names="hammadde",
                    hole=0.48, color="ABC",
                    color_discrete_map={"A": "#1e293b", "B": "#16a34a", "C": "#94a3b8"},
                    template="plotly_white",
                )
                fig_abc.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_abc, use_container_width=True)


if __name__ == "__main__":
    main()
