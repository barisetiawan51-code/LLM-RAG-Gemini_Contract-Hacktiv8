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
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #0d1117, #0a0f14);
    background-size: 400% 400%;
    animation: gradient 12s ease infinite;
}
@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.main-title {
    text-align: center;
    font-size: 2.5rem;
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
    background: linear-gradient(145deg, #1e2530, #0f141a);
    border-left: 4px solid #58a6ff;
    padding: 1rem 1.3rem;
    border-radius: 12px;
    margin-top: 1rem;
    font-size: 1.05rem;
    line-height: 1.6;
}
.context-line {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 0.8rem;
    margin: 0.6rem 0;
    color: #a9b1ba;
    font-size: 0.95rem;
    line-height: 1.6;
}
mark {
    background-color: #58a6ff33;
    color: #fff;
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
# ✨ HIGHLIGHT KONTEKS RELEVAN
# ==============================
def highlight_relevant_context(text, answer):
    keywords = re.findall(r"\b[A-Za-z]{4,}\b", answer)
    for kw in set(keywords):
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        text = pattern.sub(
            lambda m: f"<mark>{m.group(0)}</mark>",
            text
        )
    return text

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
# 🏠 HEADER
# ==============================
st.markdown("<h1 class='main-title'>⚖️ Legal Contract Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Menganalisis isi kontrak hukum menggunakan Gemini + LangChain RAG</p>", unsafe_allow_html=True)

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
    placeholder="Contoh: Apa sanksi jika peminjam terlambat membayar?",
    height=120
)

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
            st.markdown("---")
            st.markdown("### 🧩 Hasil Analisis Gemini")
            st.markdown(f"<div class='ai-box'>{answer}</div>", unsafe_allow_html=True)

            # === Sumber konteks ===
            st.markdown("### 📚 Sumber Konteks dari Dokumen")
            st.markdown(f"**📄 Dokumen:** *{target_doc}*")

            combined_text = " ... ".join(docs)
            highlighted = highlight_relevant_context(clean_text(combined_text), answer)
            st.markdown(f"<div class='context-line'>{highlighted}</div>", unsafe_allow_html=True)

# ==============================
# 🦶 FOOTER
# ==============================
st.markdown("<div class='footer'>💼 Dibangun oleh <b>Imam Bari Setiawan</b> | Powered by Gemini & LangChain 🚀</div>", unsafe_allow_html=True)
