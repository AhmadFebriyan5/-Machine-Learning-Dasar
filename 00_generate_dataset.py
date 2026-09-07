"""
Generator dataset simulasi 'mahasiswa_lulus.csv'
--------------------------------------------------
CATATAN PENTING:
Dataset asli dari soal tidak disertakan/diunggah ke sesi ini, sehingga
dataset berikut dibuat secara SINTETIS (disimulasikan) mengikuti skema
kolom yang diminta pada soal (500 baris, 7 fitur + 1 target), lengkap
dengan missing value, agar seluruh pipeline (preprocessing, modeling,
evaluasi, visualisasi) bisa didemonstrasikan end-to-end.

Jika kampus punya file mahasiswa_lulus.csv yang sesungguhnya, cukup
letakkan file itu di folder yang sama dengan nama yang sama -> seluruh
kode di 01_main_analysis.py akan tetap jalan tanpa perubahan, karena
nama & tipe kolom dibuat konsisten dengan soal.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 500

# --- Fitur dasar ---
ipk = np.clip(rng.normal(3.05, 0.45, N), 1.5, 4.0)
kehadiran = np.clip(rng.normal(82, 12, N), 40, 100)
jam_belajar = np.clip(rng.normal(14, 7, N), 0, 40)
organisasi = rng.choice(["Aktif", "Tidak Aktif"], size=N, p=[0.35, 0.65])
penghasilan_ortu = rng.choice(["Rendah", "Menengah", "Tinggi"], size=N, p=[0.3, 0.45, 0.25])
jenis_kelamin = rng.choice(["L", "P"], size=N, p=[0.48, 0.52])

# Beasiswa: sedikit lebih mungkin didapat mahasiswa IPK tinggi & penghasilan rendah
prob_beasiswa = 0.15 + 0.25 * (ipk > 3.3) + 0.15 * (penghasilan_ortu == "Rendah")
prob_beasiswa = np.clip(prob_beasiswa, 0, 0.9)
status_beasiswa = np.array(["Ya" if rng.random() < p else "Tidak" for p in prob_beasiswa])

# --- Target: Lulus_Tepat_Waktu ---
# Didorong terutama oleh IPK, Kehadiran, Jam_Belajar (faktor akademik)
# + sedikit efek Organisasi (positif kecil: soft skill, tapi bisa juga
#   menyita waktu -> efeknya dibuat kecil/netral agar realistis)
z = (
    0.35
    + 1.55 * (ipk - 3.0)
    + 0.045 * (kehadiran - 80)
    + 0.05 * (jam_belajar - 14)
    + 0.25 * (organisasi == "Aktif")
    + rng.normal(0, 0.6, N)
)
prob_lulus = 1 / (1 + np.exp(-z))
lulus_tepat_waktu = np.array(["Ya" if rng.random() < p else "Tidak" for p in prob_lulus])

df = pd.DataFrame({
    "IPK": np.round(ipk, 2),
    "Kehadiran": np.round(kehadiran, 1),
    "Jam_Belajar": np.round(jam_belajar, 1),
    "Organisasi": organisasi,
    "Penghasilan_Ortu": penghasilan_ortu,
    "Jenis_Kelamin": jenis_kelamin,
    "Status_Beasiswa": status_beasiswa,
    "Lulus_Tepat_Waktu": lulus_tepat_waktu,
})

# --- Suntikkan missing value secara acak (realistis untuk data mahasiswa) ---
for col, frac in [("IPK", 0.02), ("Kehadiran", 0.03), ("Jam_Belajar", 0.04),
                   ("Penghasilan_Ortu", 0.02)]:
    idx = rng.choice(N, size=int(N * frac), replace=False)
    df.loc[idx, col] = np.nan

df.to_csv("mahasiswa_lulus.csv", index=False)
print(df.shape)
print(df.isna().sum())
print(df["Lulus_Tepat_Waktu"].value_counts(normalize=True))
