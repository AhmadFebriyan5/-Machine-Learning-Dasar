# Prediksi Kelulusan Tepat Waktu Mahasiswa — Decision Tree vs Naive Bayes

Tugas Machine Learning Dasar — Studi kasus EduTech: memprediksi apakah mahasiswa
akan **lulus tepat waktu** menggunakan Decision Tree dan Naive Bayes.

> **Catatan dataset:** `mahasiswa_lulus.csv` di repo ini adalah dataset **simulasi**
> (500 baris, dibuat dengan seed tetap) mengikuti skema kolom pada soal, karena
> file dataset asli tidak disertakan ke dalam pengerjaan tugas. Ganti file ini
> dengan data asli kampus (nama kolom sama) untuk mendapatkan hasil real —
> seluruh kode akan tetap jalan tanpa perubahan.

## Isi Repo

| File | Deskripsi |
|---|---|
| `00_generate_dataset.py` | Generator dataset simulasi `mahasiswa_lulus.csv` (500 baris + missing value) |
| `01_main_analysis.py` | Pipeline lengkap: preprocessing → modeling (3 model) → evaluasi → visualisasi → analisis error → simulasi |
| `Analisis_Kelulusan_Mahasiswa.ipynb` | Versi notebook dari pipeline di atas, sudah dieksekusi (output & gambar tersimpan di notebook) |
| `Laporan_Kelulusan_Mahasiswa.docx` | Laporan lengkap (Bagian A, B, C) — max 8 halaman |
| `Slide_Presentasi_Kelulusan.pptx` | Slide presentasi 5 halaman (untuk presentasi ±7 menit) |
| `mahasiswa_lulus.csv` | Dataset yang digunakan (simulasi) |
| `hasil_metrik.csv`, `analisis_error.csv` | Output tabel evaluasi & contoh salah prediksi |
| `*.png` | Visualisasi (confusion matrix, pohon keputusan, bar chart metrik, feature importance) |

## Cara Menjalankan

```bash
pip install pandas numpy scikit-learn matplotlib

python 00_generate_dataset.py   # generate dataset (skip jika sudah punya data asli)
python 01_main_analysis.py      # jalankan seluruh pipeline analisis
```

Atau buka `Analisis_Kelulusan_Mahasiswa.ipynb` di Jupyter/Colab dan jalankan semua sel.

## Ringkasan Hasil

| Model | Akurasi | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Decision Tree (max_depth=3) | 0.670 | 0.671 | 0.919 | 0.776 |
| Decision Tree (max_depth=None) | 0.490 | 0.585 | 0.613 | 0.598 |
| Gaussian Naive Bayes | 0.670 | 0.693 | 0.839 | 0.759 |

Decision Tree tanpa batas kedalaman **overfit** (akurasi uji turun ke 0.49).
Rekomendasi: **Gaussian Naive Bayes** sebagai model utama prediksi risiko
(macro F1 & recall kelas minoritas lebih baik), didampingi **Decision Tree
(max_depth=3)** untuk interpretasi/komunikasi ke dosen wali.

Lihat `Laporan_Kelulusan_Mahasiswa.docx` untuk analisis lengkap Bagian A
(konsep), Bagian B (implementasi & evaluasi), dan Bagian C (rekomendasi & etika).

## Lisensi
Tugas kuliah — bebas digunakan untuk keperluan pembelajaran.
