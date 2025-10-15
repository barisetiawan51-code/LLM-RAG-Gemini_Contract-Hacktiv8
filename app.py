import os
import re
import faiss
import numpy as np
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai import embed_content

# ============================
# ⚙️ KONFIGURASI STREAMLIT
# ============================
st.set_page_config(
    page_title="⚖️ Legal Contract RAG (Gemini + LangChain)",
    page_icon="📄",
    layout="wide"
)

# 🎨 CSS kustom
st.markdown("""
<style>
    body {
        background-color: #0d1117;
        color: #e6edf3;
    }
    .ai-box {
        background-color: #161b22;
        border-left: 4px solid #58a6ff;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .context-line {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 0.6rem;
        margin: 0.4rem 0;
        color: #a9b1ba;
        font-size: 0.9rem;
    }
    .highlight {
        color: #d2a8ff;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# 🔐 API KEY GEMINI
# ============================
os.environ["GOOGLE_API_KEY"] = st.secrets.get("GEMINI_API_KEY", "")

# Model Gemini
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.9,
    top_p=0.9,
    max_output_tokens=800,
    convert_system_message_to_human=True,
    verbose=False,
)

# ============================
# 🧹 CLEAN TEXT
# ============================
def clean_text(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"[*_]+", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

# ============================
# 📂 LOAD DATA
# ============================
artifact_folder = "artifacts"
index = faiss.read_index(os.path.join(artifact_folder, "faiss.index"))
chunks_df = pd.read_parquet(os.path.join(artifact_folder, "chunks.parquet"))
embeddings = np.load(os.path.join(artifact_folder, "embeddings.npy"))

# ============================
# 🔍 RETRIEVAL
# ============================
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

# ============================
# 🧠 ASK GEMINI RAG
# ============================
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

# ============================
# 🖥️ STREAMLIT UI
# ============================
st.title("⚖️ Legal Contract Analyzer")
st.caption("Analisis isi kontrak pembiayaan menggunakan Gemini + LangChain RAG")

# Sidebar
with st.sidebar:
    st.header("📁 Pengaturan Dokumen")

    available_docs = sorted(chunks_df["filename"].unique().tolist())
    target_doc = st.selectbox("Pilih dokumen:", available_docs)
    top_k = st.slider("🔎 Jumlah konteks teratas", 3, 10, 5)

    st.markdown("---")

    st.markdown("""
    ### 💡 Tips Bertanya
    - Ajukan pertanyaan **spesifik dan langsung ke isi kontrak.**
    - Contoh pertanyaan:
        - 🏦 *Berapa jumlah pinjaman?*  
        - ⏰ *Apa sanksi keterlambatan pembayaran?*  
        - 👥 *Siapa pihak yang terlibat dalam kontrak?*  
        - 💰 *Bagaimana ketentuan bunga atau cicilan?*
    """)

# Input pengguna
user_question = st.text_area(
    "💬 Masukkan pertanyaan Anda:",
    placeholder="Contoh: Siapa pihak peminjam dalam kontrak ini?",
)

# ============================
# 🚀 TOMBOL ANALISIS
# ============================
if st.button("🚀 Analisis", use_container_width=True):
    if not user_question.strip():
        st.warning("Harap isi pertanyaan terlebih dahulu.")
    else:
        with st.spinner("🔎 Mencari konteks relevan..."):
            docs = retrieve_from_doc(user_question, target_doc, top_k=top_k)

        if not docs:
            st.error("Tidak ada konteks ditemukan untuk dokumen ini.")
        else:
            with st.spinner("🧠 Meminta analisis Gemini..."):
                answer = ask_gemini_rag(user_question, docs)

            # === Jawaban Gemini ===
            st.markdown("### 🧠 Jawaban Gemini")
            st.markdown(f"<div class='ai-box'>{answer}</div>", unsafe_allow_html=True)

            # === Kalimat Sumber (dalam expander dropdown) ===
            with st.expander("📜 Lihat Kalimat Sumber dari Dokumen", expanded=False):
                for i, chunk in enumerate(docs, start=1):
                    st.markdown(
                        f"<div class='context-line'><span class='highlight'>Konteks {i}:</span> {clean_text(chunk)}</div>",
                        unsafe_allow_html=True,
                    )

# Footer
st.markdown("---")
st.caption("💼 Built by **Imam Bari Setiawan** | Powered by Gemini & LangChain 🚀")
