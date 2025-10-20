import os
import re
import faiss
import numpy as np
import pandas as pd
import streamlit as st
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai import embed_content

# ==============================
# ⚙️ KONFIGURASI DASAR STREAMLIT
# ==============================
st.set_page_config(
    page_title="Legal Contract Analyzer",
    page_icon="📄",
    layout="wide"
)

# ==============================
# 🎨 CSS KUSTOM UNTUK STYLING
# ==============================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(145deg, #0d1117, #0a0f14);
    color: #e6edf3;
}
.main-title {
    text-align: center;
    font-size: 2.7rem;
    color: #58a6ff;
    font-weight: 700;
}
.subtitle {
    text-align: center;
    color: #a9b1ba;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}
.ai-box {
    background: rgba(30, 40, 55, 0.9);
    border-left: 4px solid #58a6ff;
    padding: 1rem 1.3rem;
    border-radius: 12px;
    margin-top: 1rem;
    font-size: 1.05rem;
    line-height: 1.6;
    box-shadow: 0 0 10px rgba(88,166,255,0.2);
}
.context-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 0.9rem;
    color: #a9b1ba;
    font-size: 0.95rem;
    line-height: 1.6;
}
mark {
    background-color: #fef08a;
    color: #000;
    border-radius: 4px;
    padding: 2px 4px;
}
.footer {
    text-align: center;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #30363d;
    color: #8b949e;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# 🔑 API KEY GEMINI
# ==============================
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
if not GEMINI_KEY:
    st.error("❌ API Key Gemini belum diset di Streamlit Secrets!")
else:
    genai.configure(api_key=GEMINI_KEY)

# ==============================
# 🤖 MODEL GEMINI
# ==============================
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.8,
    top_p=0.9,
    max_output_tokens=1000,
    convert_system_message_to_human=True,
    verbose=False,
)

# ==============================
# 🧹 CLEAN TEXT
# ==============================
def clean_text(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"[*_]+", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

# ==============================
# 📂 LOAD DATA ARTIFACT
# ==============================
artifact_folder = "artifacts"
index = faiss.read_index(os.path.join(artifact_folder, "faiss.index"))
chunks_df = pd.read_parquet(os.path.join(artifact_folder, "chunks.parquet"))
embeddings = np.load(os.path.join(artifact_folder, "embeddings.npy"))

# ==============================
# 🔍 RETRIEVAL
# ==============================
def retrieve_from_doc(query, file_name, top_k=5):
    doc_mask = chunks_df["filename"].str.lower() == file_name.lower()
    doc_chunks = chunks_df[doc_mask].reset_index(drop=True)
    doc_embeddings = embeddings[doc_mask]

    if len(doc_chunks) == 0:
        return []

    index_doc = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index_doc.add(doc_embeddings.astype("float32"))

    query_emb = embed_content(model="models/text-embedding-004", content=query)["embedding"]
    query_emb = np.array([query_emb]).astype("float32")

    distances, indices = index_doc.search(query_emb, top_k)
    return [doc_chunks.iloc[i]["text"] for i in indices[0] if 0 <= i < len(doc_chunks)]

# ==============================
# 🧠 ASK GEMINI (RAG)
# ==============================
def ask_gemini_rag(question, retrieved_chunks):
    joined_context = "\n\n".join(retrieved_chunks)
    prompt = f"""
Anda adalah asisten hukum profesional namun komunikatif.
Gunakan hanya konteks berikut untuk menjawab pertanyaan secara ringkas dan akurat.

KONTEKS:
{joined_context}

PERTANYAAN:
{question}

Aturan:
- Jawab langsung berdasarkan isi konteks.
- Jangan buat asumsi.
- Jika informasi tidak ada, katakan "Informasi tidak ditemukan dalam konteks."
"""
    response = llm.invoke(prompt)
    return clean_text(response.content.strip())

# ==============================
# ✨ HIGHLIGHT KONTEKS
# ==============================
def highlight_context(context_text, answer_text):
    keywords = re.findall(
        r"\$?\d[\d,\.]*|[A-Z]{2,}(?:\s[A-Z]{2,})*|\b[A-Z][a-z]+\b",
        answer_text
    )
    keywords = sorted(set(keywords), key=len, reverse=True)

    highlighted = context_text
    for kw in keywords:
        pattern = re.escape(kw)
        highlighted = re.sub(
            pattern,
            f"<mark>{kw}</mark>",
            highlighted,
            flags=re.IGNORECASE,
        )
    return highlighted

# ==============================
# 🏠 HEADER
# ==============================
st.markdown("<h1 class='main-title'>⚖️ Legal Contract Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Solusi Cerdas untuk Menganalisis Kontrak Hukum atau Perjanjian Pembiayaan Menggunakan Gemini & LangChain</p>", unsafe_allow_html=True)
with st.expander("ℹ️ Tentang Aplikasi"):
    st.markdown("""
    Aplikasi ini dirancang untuk membantu **analisis kontrak hukum atau perjanjian pembiayaan** 
    menggunakan pendekatan **Retrieval-Augmented Generation (RAG)**.

    🔍 **Cara kerja singkat:**
    1. Sistem mencari potongan teks paling relevan dari dokumen kontrak.
    2. Model **Gemini AI** kemudian menjawab pertanyaan Anda **berdasarkan konteks dokumen** — bukan asumsi.
    3. Hasil analisis disertai **sumber teks asli** agar transparan dan mudah diverifikasi.
    """)

st.markdown("---")

# ==============================
# 🧱 SIDEBAR (PENGATURAN)
# ==============================
with st.sidebar:
    st.header("📁 Pengaturan Dokumen")
    available_docs = sorted(chunks_df["filename"].unique().tolist())
    target_doc = st.selectbox("Pilih dokumen:", available_docs)

    st.markdown("---")
    st.subheader("⚙️ Metode Pertanyaan")
    mode = st.radio(
        "Pilih metode:",
        ["Pertanyaan Manual", "Pertanyaan Otomatis"],
        index=0
    )

# ==============================
# 📋 DATA PERTANYAAN OTOMATIS
# ==============================
auto_questions = {
        "pihak kontrak": [
            "Siapa pihak-pihak yang terlibat dalam perjanjian ini?",
            "Sebutkan siapa peminjam dan pemberi pinjaman dalam kontrak ini.",
            "Pihak mana saja yang disebutkan dalam kontrak pembiayaan ini?",
            "Siapa saja yang menandatangani perjanjian ini?",
            "Siapa pihak yang menerima pembiayaan dan pihak yang memberikan pembiayaan?"
        ],
        "pembayaran": [
            "Bagaimana sistem pembayaran diatur dalam kontrak ini?",
            "Berapa jangka waktu dan jumlah cicilan yang disepakati?",
            "Bagaimana ketentuan pembayaran dijelaskan oleh kontrak?",
            "Apakah pembayaran dilakukan setiap bulan atau sesuai kesepakatan tertentu?",
            "Ceritakan bagaimana proses pembayaran dijelaskan dalam perjanjian ini."
        ],
        "bunga": [
            "Berapa tingkat bunga yang ditetapkan dalam kontrak?",
            "Apakah terdapat bunga tambahan atau penyesuaian tahunan?",
            "Bagaimana suku bunga dihitung dalam perjanjian pembiayaan ini?",
            "Apakah tingkat bunga bersifat tetap atau berubah?",
            "Bagaimana cara penentuan bunga dijelaskan dalam kontrak?"
        ],
        "denda": [
            "Apa yang terjadi jika peminjam terlambat membayar cicilan?",
            "Apakah ada denda atau penalti atas keterlambatan pembayaran?",
            "Bagaimana kontrak mengatur konsekuensi keterlambatan pembayaran?",
            "Berapa besar denda yang dikenakan jika terjadi pelanggaran?",
            "Apakah kontrak menyebutkan sanksi terkait keterlambatan pembayaran?"
        ],
        "jaminan": [
            "Apa bentuk jaminan yang diberikan oleh peminjam?",
            "Bagaimana jaminan atau agunan diatur dalam kontrak ini?",
            "Siapa yang memegang hak atas jaminan sampai pembiayaan lunas?",
            "Apakah aset yang dibiayai dijadikan jaminan?",
            "Bagaimana proses eksekusi jaminan dijelaskan dalam perjanjian?"
        ],
        "hukum": [
            "Pengadilan mana yang berwenang menyelesaikan sengketa?",
            "Bagaimana kontrak mengatur yurisdiksi hukum?",
            "Apakah ada klausul yang menetapkan wilayah hukum tertentu?",
            "Jika terjadi sengketa, di mana perkara akan diselesaikan?",
            "Bagaimana ketentuan hukum dijelaskan dalam kontrak ini?"
        ]
}

# ==============================
# 📥 INPUT SESUAI MODE
# ==============================
if mode == "Pertanyaan Manual":
    st.markdown("#### 📝 Pertanyaan Manual")
    user_question = st.text_area(
        "Tulis pertanyaan Anda:",
        placeholder="Contoh: Apa sanksi jika peminjam terlambat membayar?",
        height=30
    )
    final_question = user_question.strip()
else:
    st.markdown("#### 💡 Pertanyaan Otomatis")
    category = st.selectbox("Kategori:", ["— Pilih kategori —"] + list(auto_questions.keys()))
    selected_auto_question = ""
    if category != "— Pilih kategori —":
        selected_auto_question = st.selectbox(
            "Pilih pertanyaan:",
            ["— Pilih pertanyaan —"] + auto_questions[category]
        )
    final_question = (
        selected_auto_question if selected_auto_question != "— Pilih pertanyaan —" else ""
    )

# ==============================
# 🚀 TOMBOL ANALISIS
# ==============================
if st.button("🚀 Analisis Kontrak", use_container_width=True):
    if not final_question:
        st.warning("⚠️ Harap isi atau pilih pertanyaan terlebih dahulu.")
    else:
        with st.spinner("🔎 Mencari konteks relevan..."):
            docs = retrieve_from_doc(final_question, target_doc)

        if not docs:
            st.error("❌ Tidak ada konteks ditemukan untuk dokumen ini.")
        else:
            with st.spinner("🧠 Menganalisis dengan Gemini..."):
                answer = ask_gemini_rag(final_question, docs)

            st.markdown("---")
            st.markdown("#### 🧩 Hasil Analisis Gemini")
            st.markdown(f"<div class='ai-box'>{answer}</div>", unsafe_allow_html=True)

            with st.expander("📚 Lihat Sumber Konteks dari Dokumen"):
                st.markdown(f"**📄 Dokumen:** *{target_doc}*")
                combined_text = " ".join(docs)
                highlighted_md = highlight_context(clean_text(combined_text), answer)
                st.markdown(f"<div class='context-box'>{highlighted_md}</div>", unsafe_allow_html=True)

# ==============================
# 🦶 FOOTER
# ==============================
st.markdown("<div class='footer'>💼 Dibangun oleh <b>Imam Bari Setiawan</b> | Powered by Gemini & LangChain 🚀</div>", unsafe_allow_html=True)
