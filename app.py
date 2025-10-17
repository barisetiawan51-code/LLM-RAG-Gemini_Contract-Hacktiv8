import os
import re
import faiss
import numpy as np
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai import embed_content

# ==============================
# ⚙️ KONFIGURASI DASAR STREAMLIT
# ==============================
st.set_page_config(
    page_title="⚖️ Legal Contract Analyzer",
    page_icon="📄",
    layout="wide"
)

# ==============================
# 🎨 CSS KUSTOM UNTUK STYLING
# ==============================
st.markdown("""
<style>
/* 🌈 Background animasi */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #0d1117, #0a0f14);
    animation: gradient 10s ease infinite;
    background-size: 400% 400%;
}
@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ✨ Heading */
.main-title {
    text-align: center;
    font-size: 2.5rem;
    color: #58a6ff;
    font-weight: 700;
    margin-top: 0.5rem;
}
.subtitle {
    text-align: center;
    color: #a9b1ba;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* 💬 Card Style */
.card {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    margin-bottom: 1.5rem;
}

/* 💡 AI Answer Box */
.ai-box {
    background: linear-gradient(145deg, #1e2530, #0f141a);
    border-left: 4px solid #58a6ff;
    padding: 1rem 1.3rem;
    border-radius: 12px;
    margin-top: 1rem;
    font-size: 1.05rem;
    line-height: 1.6;
    animation: fadeIn 0.8s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

/* 📄 Konteks Card */
.context-line {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 0.8rem;
    margin: 0.4rem 0;
    color: #a9b1ba;
    font-size: 0.95rem;
}
.highlight {
    color: #d2a8ff;
    font-weight: 600;
}

/* 🦶 Footer */
.footer {
    text-align: center;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #30363d;
    color: #8b949e;
    font-size: 0.85rem;
}

/* ✍️ Typing animation effect */
.typing {
    overflow: hidden;
    white-space: nowrap;
    border-right: 3px solid #58a6ff;
    animation: typing 3s steps(40, end), blink 0.75s step-end infinite;
}
@keyframes typing {
    from { width: 0 }
    to { width: 100% }
}
@keyframes blink {
    50% { border-color: transparent }
}
</style>
""", unsafe_allow_html=True)

# ==============================
# 🔑 API KEY GEMINI
# ==============================
os.environ["GOOGLE_API_KEY"] = st.secrets.get("GEMINI_API_KEY", "")

# ==============================
# 🤖 MODEL GEMINI
# ==============================
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.9,
    top_p=0.9,
    max_output_tokens=800,
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
# 📂 LOAD DATA
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
Kamu adalah asisten hukum profesional.
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
# 🏠 HEADER & INTRO
# ==============================
st.markdown("<h1 class='main-title'>⚖️ Legal Contract Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Menganalisis isi kontrak hukum menggunakan kecerdasan buatan Gemini + LangChain RAG</p>", unsafe_allow_html=True)

with st.expander("ℹ️ Tentang Aplikasi", expanded=False):
    st.markdown("""
    Aplikasi ini dirancang untuk membantu **analisis kontrak hukum atau perjanjian pembiayaan**
    menggunakan pendekatan **Retrieval-Augmented Generation (RAG)**.

    **Langkah kerja sistem:**
    1. Menemukan bagian teks paling relevan dari dokumen kontrak.
    2. Menggunakan **Gemini AI** untuk menjawab pertanyaan berdasarkan isi dokumen.
    3. Menampilkan hasil **dengan sumber teks asli** agar transparan.

    💬 *Contoh pertanyaan:*
    - Siapa pihak yang terlibat dalam kontrak?
    - Apa ketentuan bunga atau denda keterlambatan?
    - Bagaimana jadwal pembayaran dilakukan?
    """)

st.markdown("---")

# ==============================
# 🧱 SIDEBAR
# ==============================
with st.sidebar:
    st.header("📁 Pengaturan Dokumen")
    available_docs = sorted(chunks_df["filename"].unique().tolist())
    target_doc = st.selectbox("Pilih dokumen:", available_docs)
    top_k = st.slider("🔎 Jumlah konteks teratas", 3, 10, 5)

    st.markdown("---")
    st.markdown("""
    ### 💡 Tips Bertanya
    - Gunakan kalimat **jelas & spesifik**
    - Contoh:
        - 🏦 Berapa jumlah pinjaman?
        - 👥 Siapa pihak yang terlibat?
        - ⏰ Apa sanksi keterlambatan pembayaran?
    """)

# ==============================
# 💬 INPUT
# ==============================
user_question = st.text_area(
    "Masukkan pertanyaan Anda:",
    placeholder="Contoh: Siapa pihak peminjam dalam kontrak ini?",
    height=120
)
st.markdown("</div>", unsafe_allow_html=True)

# ==============================
# 🚀 TOMBOL ANALISIS
# ==============================
if st.button("🚀 Analisis Kontrak", use_container_width=True):
    if not user_question.strip():
        st.warning("⚠️ Harap isi pertanyaan terlebih dahulu.")
    else:
        with st.spinner("🔎 Mencari konteks relevan..."):
            docs = retrieve_from_doc(user_question, target_doc, top_k=top_k)

        if not docs:
            st.error("❌ Tidak ada konteks ditemukan untuk dokumen ini.")
        else:
            with st.spinner("🧠 Menganalisis dengan Gemini..."):
                answer = ask_gemini_rag(user_question, docs)

            # === Jawaban ===
            st.markdown("### 🧩 Hasil Analisis Gemini")
            st.markdown(f"<div class='ai-box'>{answer}</div>", unsafe_allow_html=True)

            # === Konteks ===
            with st.expander("📜 Lihat Konteks dari Dokumen"):
                for i, chunk in enumerate(docs, start=1):
                    st.markdown(
                        f"<div class='context-line'><span class='highlight'>Konteks {i}:</span> {clean_text(chunk)}</div>",
                        unsafe_allow_html=True,
                    )

# ==============================
# 🦶 FOOTER
# ==============================
st.markdown("<div class='footer'>💼 Dibangun oleh <b>Imam Bari Setiawan</b> | Powered by Gemini & LangChain 🚀</div>", unsafe_allow_html=True)
