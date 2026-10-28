"""
Adım 2 - Keşifsel veri analizi (EDA)
------------------------------------
data/fiyatlar.csv dosyasını okur, grafikleri grafikler/ klasörüne kaydeder
ve ekrana özet istatistikleri yazdırır.

Gerekli paketler: pandas, matplotlib
    pip install pandas matplotlib
(İsteğe bağlı: statsmodels -> durağanlık testi için)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GIRDI = Path("data/fiyatlar.csv")
CIKTI = Path("grafikler")
CIKTI.mkdir(exist_ok=True)

ADLAR = {"pamuk": "Pamuk", "aluminyum": "Alüminyum", "bugday": "Buğday"}

df = pd.read_csv(GIRDI, parse_dates=["tarih"], index_col="tarih")
df = df[[c for c in ADLAR if c in df.columns]]

print("=== Genel bilgi ===")
print(df.describe().round(2))
print("\nEksik değerler:\n", df.isna().sum())

# 1) Zaman grafiği ---------------------------------------------------------
fig, eksenler = plt.subplots(len(df.columns), 1, figsize=(11, 8), sharex=True)
for ax, col in zip(eksenler, df.columns):
    ax.plot(df.index, df[col], lw=1.2)
    ax.axvspan(pd.Timestamp("2021-06-01"), pd.Timestamp("2023-06-01"),
               color="orange", alpha=0.2)
    ax.set_ylabel(ADLAR[col])
    ax.grid(alpha=0.3)
eksenler[0].set_title("Aylık fiyatlar (turuncu alan: 2021-2023 şok dönemi)")
plt.tight_layout()
plt.savefig(CIKTI / "01_fiyat_serileri.png", dpi=150)
plt.close()

# 2) Mevsimsellik ----------------------------------------------------------
# Her ayın fiyatını, o yılın ortalamasına oranlayarak trendi ve yıl etkisini ayıklıyoruz
oran = df.groupby(df.index.year).transform(lambda x: x / x.mean())
mevsim = oran.groupby(oran.index.month).mean()

fig, ax = plt.subplots(figsize=(10, 5))
for col in mevsim.columns:
    ax.plot(mevsim.index, mevsim[col], marker="o", label=ADLAR[col])
ax.axhline(1, color="gray", ls="--")
ax.set_xticks(range(1, 13))
ax.set_xlabel("Ay")
ax.set_ylabel("Yıllık ortalamaya oran")
ax.set_title("Aylık mevsimsellik profili")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(CIKTI / "02_mevsimsellik.png", dpi=150)
plt.close()

print("\n=== Mevsimsellik (en yüksek - en düşük ay farkı, %) ===")
for col in mevsim.columns:
    fark = (mevsim[col].max() - mevsim[col].min()) * 100
    print(f"{ADLAR[col]:10s}: {fark:.1f}  (en yüksek ay: {mevsim[col].idxmax()}, en düşük ay: {mevsim[col].idxmin()})")

# 3) Oynaklık (aylık log getiri) -------------------------------------------
getiri = np.log(df).diff().dropna()
print("\n=== Aylık log getiri standart sapması (oynaklık) ===")
print((getiri.std() * 100).round(2).rename(index=ADLAR).astype(str) + " %")

fig, ax = plt.subplots(figsize=(10, 5))
for col in getiri.columns:
    ax.plot(getiri.index, getiri[col].rolling(12).std() * 100, label=ADLAR[col])
ax.set_ylabel("12 aylık kayan oynaklık (%)")
ax.set_title("Oynaklık zaman içinde nasıl değişiyor?")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(CIKTI / "03_oynaklik.png", dpi=150)
plt.close()

# 4) Korelasyon ------------------------------------------------------------
print("\n=== Aylık getiriler arası korelasyon ===")
print(getiri.corr().round(2).rename(index=ADLAR, columns=ADLAR))

# 5) Durağanlık testi (varsa) ----------------------------------------------
try:
    from statsmodels.tsa.stattools import adfuller

    print("\n=== ADF testi (p < 0.05 -> durağan) ===")
    for col in df.columns:
        p_seviye = adfuller(df[col].dropna())[1]
        p_getiri = adfuller(getiri[col].dropna())[1]
        print(f"{ADLAR[col]:10s}: seviye p={p_seviye:.3f} | getiri p={p_getiri:.3f}")
except ImportError:
    print("\n(statsmodels kurulu değil, ADF testi atlandı: pip install statsmodels)")

print(f"\nGrafikler '{CIKTI}/' klasörüne kaydedildi.")
