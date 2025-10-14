# Chatbot LLM berbasis RAG menggunakan Model Gemini Tentang Isi Kontrak

## 🎯Tujuan umum:
Membangun chatbot LLM berbasis RAG yang dapat memahami dan menjawab pertanyaan tentang isi kontrak real estate financing (atau kontrak keuangan lain), tanpa harus dibaca manual oleh staf atau nasabah.

## 💼 Masalah bisnis yang ingin diselesaikan:
* Waktu analisis kontrak terlalu lama.
* Tingkat kesalahan manusia tinggi.
* Skalabilitas.

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
* Chatbot mampu menjawab pertanyaan di atas dengan akurasi tinggi, berdasarkan isi dokumen PDF (bukan hasil halusinasi).
* Dapat digunakan oleh:
    1. Customer service bank
    2. Legal officer
    3. Auditor internal
    4. Nasabah melalui portal web / chatbot
