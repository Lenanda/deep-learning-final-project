# 🚀 Emotion Detection App (Streamlit)

Folder ini berisi *source code* untuk antarmuka pengguna (User Interface) berbasis web menggunakan **Streamlit**. Aplikasi ini memungkinkan pengguna untuk melakukan prediksi sentimen secara *real-time* menggunakan model yang telah dilatih.

## 🛠️ Fitur Aplikasi

1. **Manual Prediction**:
   - Input teks bebas (kalimat/tweet).
   - Output berupa prediksi kelas (Irony/Sarcasm/Regular/Figurative), skor confidence, dan grafik probabilitas.
   
2. **Batch Prediction**:
   - Menganalisis data dari `data/test.csv`.
   - Menampilkan contoh teks dengan tingkat keyakinan (confidence) tertinggi untuk setiap kategori emosi.

## ⚙️ Cara Menjalankan Aplikasi

Pastikan Anda berada di **root directory** proyek (`deep-learning-final-project/`) saat menjalankan perintah ini, agar path ke folder `assets/` terbaca dengan benar.

### 1. Install Dependencies
Pastikan library yang dibutuhkan sudah terinstall:

```bash
pip install -r requirements.txt
