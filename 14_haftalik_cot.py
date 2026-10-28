"""
Adım 14 - Haftalık tahmin + CFTC pozisyonlanma (COT) özellikleri (GELİŞTİRME aşaması)
------------------------------------------------------------------------------------
12 numaralı scriptin aynısı (aynı veri, aynı kilit: 2023-12-29 sonrası kullanılmaz,
aynı değerlendirme dönemi 2018-2023), ama modellere COT özellikleri eklenir:
  - managed money (fon) net pozisyonu / açık pozisyon ve 4 haftalık değişimi
  - 3 yıllık yüzdelik sırası (aşırı kalabalık pozisyon göstergesi)
  - producer/merchant (hedge eden üretici-tüccar) net pozisyonu
  - açık pozisyonun 4 haftalık log değişimi

Sızıntı önlemi: Rapor Salı'ya aittir, Cuma ~15:30'da yayımlanır. Bir Cuma kapanışında
yalnızca O CUMADAN ÖNCE yayımlanmış rapor kullanılır (allow_exact_matches=False).

Karşılaştırma: aynı tabloda "cotsuz" (12'deki özellikler) ve "cotlu" modeller.
Pamuk bu aşamada yok (temiz fiyat serisi yok; satın alınacak veriyle eklenecek).

Gerekli: data/cot.csv (13_cot_veri.py), 12_haftalik_gelistirme.py aynı klasörde.
"""

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

spec = importlib.util.spec_from_file_location("g12", "12_haftalik_gelistirme.py")
g12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g12)

COT = Path("data/cot.csv")
orijinal_ozellikler = g12.ozellikler


def cot_ozellikleri(w: pd.DataFrame, hedef: str) -> pd.DataFrame:
    cot = pd.read_csv(COT, parse_dates=["tarih", "kullanilabilir"])
    kolonlar = {f"{hedef}_mm_net_pct": "mm", f"{hedef}_prod_net_pct": "prod", f"{hedef}_oi": "oi"}
    c = cot[["kullanilabilir"] + list(kolonlar)].rename(columns=kolonlar).sort_values("kullanilabilir")
    # haftalık seri kendi yayım takvimine göre yeniden kurulur (rapor başına bir satır)
    c = c.set_index("kullanilabilir")
    f = pd.DataFrame(index=c.index)
    f["cot_mm"] = c["mm"]
    f["cot_mm_d4"] = c["mm"].diff(4)
    f["cot_mm_pct156"] = c["mm"].rolling(156, min_periods=52).apply(
        lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    f["cot_prod"] = c["prod"]
    f["cot_oi_d4"] = np.log(c["oi"]).diff(4)
    f = f.reset_index()
    # her Cuma için: o Cumadan ÖNCE yayımlanmış en son rapor
    hedef_tarihler = pd.DataFrame({"tarih": w.index})
    m = pd.merge_asof(hedef_tarihler.sort_values("tarih"), f.sort_values("kullanilabilir"),
                      left_on="tarih", right_on="kullanilabilir",
                      direction="backward", allow_exact_matches=False)
    return m.set_index("tarih").drop(columns="kullanilabilir")


def cotlu_ozellikler(w, hedef):
    return pd.concat([orijinal_ozellikler(w, hedef), cot_ozellikleri(w, hedef)], axis=1)


def main():
    if not COT.exists():
        raise FileNotFoundError("data/cot.csv yok. Önce 13_cot_veri.py'yi çalıştır.")
    w = g12.haftalik_veri()
    print(f"Haftalık veri: {w.index.min().date()} -> {w.index.max().date()} ({len(w)} hafta)")
    print(f"(Kilit: {g12.GELISTIRME_SON} sonrası veri okunmadı)\n")

    satirlar = []
    for hedef in g12.CAPRAZ:
        for h in g12.HORIZONLAR:
            sonuclar = {}
            for etiket, fonk in [("cotsuz", orijinal_ozellikler), ("cotlu", cotlu_ozellikler)]:
                g12.ozellikler = fonk
                sonuclar[etiket] = g12.walk_forward(w, hedef, h)
            g12.ozellikler = orijinal_ozellikler
            # ortak tarihlerde kıyasla
            p = {k: v.pivot(index="tarih", columns="model", values="tahmin") for k, v in sonuclar.items()}
            ortak = p["cotsuz"].index.intersection(p["cotlu"].index)
            gercek = (sonuclar["cotlu"].drop_duplicates("tarih").set_index("tarih")["gercek"]).reindex(ortak)
            e0 = (p["cotlu"].loc[ortak, "naif"] - gercek) ** 2
            for etiket in ["cotsuz", "cotlu"]:
                for m in ["ridge", "ridge_yari", "gbm"]:
                    e = (p[etiket].loc[ortak, m] - gercek) ** 2
                    satirlar.append(dict(
                        hammadde=hedef, ufuk_hafta=h, ozellik=etiket, model=m, n=len(ortak),
                        naife_oran=round(float(np.sqrt(e.mean() / e0.mean())), 3),
                        DM_p=round(g12.dm_p((e0 - e).values, h), 3),
                        yon_dogrulugu=round(float((np.sign(p[etiket].loc[ortak, m]) == np.sign(gercek)).mean()), 3)))
    s = pd.DataFrame(satirlar)
    s["DM_p_bonferroni"] = (s["DM_p"] * len(s)).clip(upper=1).round(3)
    pd.set_option("display.width", 200)
    print(s.to_string(index=False))
    s.to_csv("data/haftalik_cot_sonuc.csv", index=False)
    print("\nnaife_oran < 1 iyi. cotlu satırları cotsuz'dan iyiyse COT bilgi katıyor demektir.")


if __name__ == "__main__":
    main()
