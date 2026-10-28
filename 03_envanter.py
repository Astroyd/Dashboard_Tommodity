"""
Adım 3 - Veri tabanı + envanter modülü
--------------------------------------
1) SQLite veri tabanını kurar (data/envanter.db)
2) Fiyat verisini (data/fiyatlar.csv) tabloya yükler
3) Tüketim verisini SİMÜLE eder (gerçek veri yok; parametreleri aşağıdan değiştir)
4) Her hammadde için EOQ, güvenlik stoğu, yeniden sipariş noktası ve ABC sınıfını hesaplar

Gerekli paketler: pandas, numpy
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB = Path("data/envanter.db")
FIYAT_CSV = Path("data/fiyatlar.csv")
SEED = 42
SIM_AY = 120  # Son kaç ay için tüketim simüle edilecek (10 yıl)

# ---------------------------------------------------------------------------
# VARSAYIMLAR - bunları raporda açıkça belirtmelisin, istediğin gibi değiştir
# birim_carpan: Pink Sheet fiyatını USD/ton'a çevirmek için
#   (Pamuk A Index USD/kg verilir -> x1000; alüminyum ve buğday zaten USD/ton)
# ---------------------------------------------------------------------------
HAMMADDELER = {
    "pamuk": dict(ad="Pamuk", birim_carpan=1000, aylik_tuketim=50,
                  tuketim_cv=0.15, mevsim_genlik=0.10, siparis_maliyeti=500,
                  tutma_orani=0.20, tedarik_suresi_ay=1),
    "aluminyum": dict(ad="Alüminyum", birim_carpan=1, aylik_tuketim=80,
                      tuketim_cv=0.12, mevsim_genlik=0.03, siparis_maliyeti=500,
                      tutma_orani=0.15, tedarik_suresi_ay=1),
    "bugday": dict(ad="Buğday", birim_carpan=1, aylik_tuketim=400,
                   tuketim_cv=0.15, mevsim_genlik=0.08, siparis_maliyeti=300,
                   tutma_orani=0.18, tedarik_suresi_ay=1),
}
Z_HIZMET = 1.65  # %95 hizmet düzeyi

SEMA = """
CREATE TABLE IF NOT EXISTS hammadde (
    id TEXT PRIMARY KEY,
    ad TEXT NOT NULL,
    birim TEXT DEFAULT 'ton',
    siparis_maliyeti REAL,        -- USD / sipariş
    tutma_orani REAL,             -- yıllık, fiyatın oranı
    tedarik_suresi_ay INTEGER
);
CREATE TABLE IF NOT EXISTS fiyat (
    tarih TEXT, hammadde_id TEXT, fiyat_usd_ton REAL,
    PRIMARY KEY (tarih, hammadde_id),
    FOREIGN KEY (hammadde_id) REFERENCES hammadde(id)
);
CREATE TABLE IF NOT EXISTS tuketim (
    tarih TEXT, hammadde_id TEXT, miktar_ton REAL,
    PRIMARY KEY (tarih, hammadde_id),
    FOREIGN KEY (hammadde_id) REFERENCES hammadde(id)
);
CREATE TABLE IF NOT EXISTS siparis (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hammadde_id TEXT, siparis_tarihi TEXT, teslim_tarihi TEXT,
    miktar_ton REAL, birim_fiyat_usd REAL,
    FOREIGN KEY (hammadde_id) REFERENCES hammadde(id)
);
CREATE TABLE IF NOT EXISTS stok_seviye (
    tarih TEXT, hammadde_id TEXT, stok_ton REAL,
    PRIMARY KEY (tarih, hammadde_id),
    FOREIGN KEY (hammadde_id) REFERENCES hammadde(id)
);
"""


def kur(con):
    con.executescript(SEMA)
    for hid, p in HAMMADDELER.items():
        con.execute(
            "INSERT OR REPLACE INTO hammadde VALUES (?,?,?,?,?,?)",
            (hid, p["ad"], "ton", p["siparis_maliyeti"], p["tutma_orani"],
             p["tedarik_suresi_ay"]),
        )


def fiyat_yukle(con):
    df = pd.read_csv(FIYAT_CSV, parse_dates=["tarih"])
    satirlar = []
    for hid, p in HAMMADDELER.items():
        if hid not in df.columns:
            continue
        s = df[["tarih", hid]].dropna()
        for t, v in zip(s["tarih"], s[hid]):
            satirlar.append((t.strftime("%Y-%m-%d"), hid, float(v) * p["birim_carpan"]))
    con.executemany("INSERT OR REPLACE INTO fiyat VALUES (?,?,?)", satirlar)


def tuketim_simule(con):
    """Aylık tüketim = ortalama x mevsimsellik x rastgele gürültü."""
    rng = np.random.default_rng(SEED)
    tarihler = pd.read_sql("SELECT DISTINCT tarih FROM fiyat ORDER BY tarih", con)["tarih"]
    tarihler = pd.to_datetime(tarihler).iloc[-SIM_AY:]
    satirlar = []
    for hid, p in HAMMADDELER.items():
        for t in tarihler:
            mevsim = 1 + p["mevsim_genlik"] * np.sin(2 * np.pi * (t.month - 1) / 12)
            gurultu = rng.normal(1, p["tuketim_cv"])
            miktar = max(0.0, p["aylik_tuketim"] * mevsim * gurultu)
            satirlar.append((t.strftime("%Y-%m-%d"), hid, round(miktar, 2)))
    con.executemany("INSERT OR REPLACE INTO tuketim VALUES (?,?,?)", satirlar)


def envanter_parametreleri(con):
    sonuc = []
    for hid, p in HAMMADDELER.items():
        tuk = pd.read_sql(
            "SELECT miktar_ton FROM tuketim WHERE hammadde_id=? ORDER BY tarih",
            con, params=(hid,))["miktar_ton"]
        fiyat = pd.read_sql(
            "SELECT fiyat_usd_ton FROM fiyat WHERE hammadde_id=? ORDER BY tarih",
            con, params=(hid,))["fiyat_usd_ton"]
        if tuk.empty or fiyat.empty:
            continue

        d_ay, sigma_ay = tuk.mean(), tuk.std()
        d_yil = d_ay * 12
        f = fiyat.iloc[-12:].mean()                 # son 12 ay ortalama fiyat
        h = p["tutma_orani"] * f                    # USD / ton / yıl
        eoq = np.sqrt(2 * d_yil * p["siparis_maliyeti"] / h)
        L = p["tedarik_suresi_ay"]
        ss = Z_HIZMET * sigma_ay * np.sqrt(L)
        rop = d_ay * L + ss

        sonuc.append(dict(
            hammadde=p["ad"], aylik_talep_ton=round(d_ay, 1),
            ort_fiyat_usd_ton=round(f, 0), yillik_harcama_usd=round(d_yil * f, 0),
            EOQ_ton=round(eoq, 1), guvenlik_stogu_ton=round(ss, 1),
            ROP_ton=round(rop, 1), yilda_siparis=round(d_yil / eoq, 1),
        ))

    tablo = pd.DataFrame(sonuc)
    # ABC: yıllık harcamaya göre kümülatif pay (A: ilk %80, B: sonraki %15, C: kalan)
    tablo = tablo.sort_values("yillik_harcama_usd", ascending=False).reset_index(drop=True)
    pay = tablo["yillik_harcama_usd"].cumsum() / tablo["yillik_harcama_usd"].sum()
    onceki = pay.shift(fill_value=0)
    tablo["ABC"] = np.where(onceki < 0.80, "A", np.where(onceki < 0.95, "B", "C"))
    return tablo


def main():
    if not FIYAT_CSV.exists():
        raise FileNotFoundError("Önce 01_veri_hazirla.py'yi çalıştır (data/fiyatlar.csv yok).")
    DB.parent.mkdir(exist_ok=True)
    with sqlite3.connect(DB) as con:
        kur(con)
        fiyat_yukle(con)
        tuketim_simule(con)
        tablo = envanter_parametreleri(con)
    pd.set_option("display.width", 200)
    print(tablo.to_string(index=False))
    print(f"\nVeri tabanı: {DB}")


if __name__ == "__main__":
    main()
