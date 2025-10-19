# Chatbot LLM berbasis RAG menggunakan Model Gemini Tentang Isi Kontrak

## 🎯Tujuan umum:
Membangun chatbot LLM berbasis RAG yang dapat memahami dan menjawab pertanyaan tentang isi kontrak tanpa harus dibaca manual oleh pengguna.

## 💼 Masalah bisnis yang ingin diselesaikan:
* Waktu analisis kontrak terlalu lama.
* Tingkat kesalahan manusia tinggi.
* Skalabilitas.

## 📚 Tentang Dataset
Dataset ini berisi 30 dokumen kontrak hukum independen yang ditulis dalam bahasa Inggris.
Setiap kontrak dirancang menyerupai dokumen dunia nyata dengan elemen yang realistis, mencakup:
1. Struktur dan format hukum yang autentik.
2. Tanggal, dan nilai kontrak yang sesuai konteks.
3. Beragam kategori, seperti pembiayaan properti, kendaraan, pendidikan, kesehatan, dan pinjaman umum.
Dataset ini menjadi dasar yang kuat untuk melatih dan menguji model kecerdasan buatan dalam tugas pemahaman dokumen hukum, analisis kontrak, serta ekstraksi klausul penting.

## 🧠 Pertanyaan bisnis yang bisa dijawab chatbot

| Kategori             | Contoh Pertanyaan                                      |
| -------------------- | ------------------------------------------------------ |
| Informasi Umum       | Siapa pihak peminjam dan pemberi pinjaman?             |
| Nilai Kontrak        | Berapa total pembiayaan?                               |
| Ketentuan Pembayaran | Berapa bunga per bulan dan jangka waktu pinjaman?      |
| Risiko & Penalti     | Apa konsekuensi jika terlambat membayar?               |
| Legalitas            | Pengadilan mana yang berwenang menyelesaikan sengketa? |
| Asuransi             | Apakah aset harus diasuransikan?                       |
| Terminasi            | Dalam kondisi apa kontrak dapat dibatalkan?            |

## 📊 Output bisnis yang diharapkan
* Chatbot mampu menjawab pertanyaan di atas berdasarkan isi dokumen PDF (bukan hasil halusinasi).
* Sistem dapat digunakan oleh berbagai pihak yang berinteraksi dengan dokumen hukum, antara lain:
 1. Firma hukum untuk mempercepat telaah kontrak dan due diligence.
 2. Departemen legal perusahaan untuk analisis risiko dan kepatuhan.
 3. Notaris & konsultan hukum dalam meninjau klausul penting.
 4. Pelaku bisnis & individu yang ingin memahami isi kontrak secara cepat dan jelas.

## 💰 Potensi penerapan bisnis
| Bidang                  | Implementasi                                                   |
| ----------------------- | -------------------------------------------------------------- |
| **Real Estate**         | Kontrak pembiayaan rumah dan apartemen.                        |
| **Vehicles**            | Kontrak pembiayaan kendaraan, termasuk mobil dan sepeda motor. |
| **Education**           | Perjanjian pinjaman pendidikan (student loan agreements).      |
| **Health**              | Kontrak terkait layanan medis dan asuransi kesehatan.          |
| **KGeneral Financing**  | Perjanjian pinjaman pribadi maupun bisnis.                     |

