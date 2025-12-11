# 📂 Dataset Information

Folder ini berisi dataset yang digunakan untuk melatih dan menguji model Deep Learning (BiLSTM + CNN) untuk deteksi emosi/sentimen pada teks.

## 📄 Deskripsi File

| Nama File | Deskripsi | Jumlah Baris (Estimasi) |
|-----------|-----------|-------------------------|
| `train.csv` | Data utama untuk proses training dan validasi model. | ~8,000+ baris |
| `test.csv` | Data terpisah (unseen) untuk evaluasi final dan batch prediction di aplikasi. | ~8,000+ baris |

## 📊 Struktur Data (Schema)

Setiap file CSV memiliki dua kolom utama:

1. **`tweets`**: Teks mentah (raw text) dari tweet pengguna. Berisi hashtag, mention, dan URL yang akan dibersihkan pada tahap preprocessing.
2. **`class`**: Label emosi/kategori sentimen.

## 🏷️ Kategori Kelas (Labels)

Dataset ini diklasifikasikan ke dalam 4 kategori utama:

* **Regular**: Teks normal tanpa majas atau sindiran.
* **Irony**: Kalimat yang mengandung ironi (kebalikan dari fakta).
* **Sarcasm**: Kalimat sarkasme yang tajam atau menyindir.
* **Figurative**: Kalimat kiasan atau metafora.

## ⚠️ Catatan Penggunaan
* Data ini memerlukan preprocessing (cleaning) sebelum dimasukkan ke dalam model (seperti menghapus URL, mention `@`, dan simbol `#`).
* Jangan mengubah nama kolom (`tweets`, `class`) agar script training dan aplikasi tetap berjalan lancar.
