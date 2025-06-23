# 🗨️ Quotes Extraction (Ekstraksi Kutipan Berbahasa Indonesia)

Repositori ini berisi kode untuk **mengekstrak kutipan langsung dan tidak langsung** dari teks berita berbahasa Indonesia menggunakan pendekatan Named Entity Recognition (NER) berbasis RNN.

Dua model digunakan untuk melakukan ekstraksi kutipan secara lengkap dan kontekstual:

- `direct-quotes`: berfokus pada ekstraksi **kutipan langsung**
- `indirect-quotes`: berfokus pada ekstraksi **kutipan tidak langsung**

## 🔎 Entitas yang Diekstraksi

| Entitas       | Deskripsi                                                      | Diambil oleh `direct-quotes` | Diambil oleh `indirect-quotes` |
| ------------- | -------------------------------------------------------------- | ---------------------------- | ------------------------------ |
| `PERSON`      | Nama pembicara                                                 | ✅                           | ✅                             |
| `PERSONCOREF` | Rujukan kepada nama pembicara yang telah disebutkan sebelumnya | ✅                           | ✅                             |
| `ROLE`        | Peran atau jabatan pembicara                                   | ✅                           | ❌                             |
| `AFFILIATION` | Organisasi atau institusi pembicara                            | ✅                           | ❌                             |
| `CUE`         | Kata pengantar kutipan (misal: "ujar", "kata")                 | ✅                           | ✅                             |
| `CUECOREF`    | Penanda rujukan yang menunjukkan atribusi kutipan              | ✅                           | ✅                             |
| `STATEMENT`   | Isi kutipan                                                    | ✅                           | ✅                             |
| `DATETIME`    | Waktu kutipan/kejadian                                         | ✅                           | ❌                             |
| `LOCATION`    | Tempat kejadian                                                | ✅                           | ❌                             |
| `EVENT`       | Nama atau jenis acara                                          | ✅                           | ❌                             |
| `ISSUE`       | Masalah atau topik utama yang dibahas                          | ✅                           | ❌                             |

## 📊 F1 Score per Entitas

Model telah dievaluasi menggunakan metrik F1 Score pada data validasi untuk mengukur keseimbangan antara presisi dan recall dalam mengenali entitas. Hasilnya menunjukkan bahwa kedua model memiliki performa yang seimbang dalam tugas ekstraksi kutipan:
| Model | F1 Score |
| ------------- | ----------------- |
| `Direct Quotes` | 92.34 |
| `Indirect Quotes` | 89.73 |

Berikut hasil evaluasi model berdasarkan F1 Score pada data validasi:

| Entitas       | F1 Score (Direct) | F1 Score (Indirect) |
| ------------- | ----------------- | ------------------- |
| `PERSON`      | 79.59             | 43.48               |
| `PERSONCOREF` | 89.15             | 46.64               |
| `ROLE`        | 62.20             | 10.36               |
| `AFFILIATION` | 70.02             | 40.22               |
| `CUE`         | 95.3              | 81.13               |
| `CUECOREF`    | 90.75             | 0                   |
| `STATEMENT`   | 81.9              | 83.43               |
| `DATETIME`    | 86.35             | 76.77               |
| `LOCATION`    | 54.76             | 17.95               |
| `EVENT`       | 72.1              | -                   |

## 🧠 Arsitektur & Komponen

- **`cnn.py`**: Modul untuk proses karakter embedding menggunakan CNN.
- **`corpus.py`**: Melakukan preprocessing teks menjadi input yang siap diprediksi.
- **`idsentsegmenter/`**: Modul untuk memecah paragraf menjadi kalimat (kalimat tokenizer).
- **`quotes_extraction.py`**: Script utama untuk melakukan ekstraksi kutipan dan entitas.
- **`quotes_extraction_visualization.py`**: Script untuk menampilkan hasil ekstraksi dalam bentuk teks dengan highlight.
- **`requirements.txt`**: Daftar semua dependensi proyek.
- **`models/`**: Folder berisi file model dan metadata hasil pelatihan.
  📥 Catatan:
  Direktori models/ tidak disertakan langsung di dalam repositori karena ukurannya besar.
  Unduh terlebih dahulu melalui link Google Drive berikut:

🔗 Download models dari Google Drive
https://drive.google.com/drive/folders/15VqTcBYtoYZYk2uXxb7k0Lgr3DLxIQnZ?usp=sharing
Setelah diunduh, ekstrak isi arsip dan tempatkan ke dalam folder models/ di root proyek.

## 🚀 Cara Menjalankan

1. **Buat file `.env`**
2. **Install dependensi**
   ```bash
   pip install -r requirements.txt
   ```
3. **Jalankan ekstraksi kutipan**
   ```bash
   python quotes_extraction.py
   ```
