import os
import re
import faiss
import numpy as np
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai import embed_content

# ============================
# 🎯 KONFIGURASI STREAMLIT
# ============================
st.set_page_config(
    page_title="⚖️ Legal Contract Analyzer (RAG + Gemini)",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === Gaya CSS Kustom ===
st.markdown("""
<style>
    body {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: 'Inter', sans-serif;
    }
    .main {
        background-color: #0d1117;
        color: #e6edf3;
    }
    h1, h2, h3, h4 {
        color: #58a6ff;
    }
    .stTextArea textarea, .stTextInput input {
        background-color: #161b22;
        color: #f0f6fc;
        border-radius: 8px;
        border: 1px solid #30363d;
    }
    .stButton>button {
        background: linear-gradient(90deg, #238636, #2ea043);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: bold;
        padding: 0.6rem 1.5rem;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2ea043, #3fb950);
    }
    .ai-box {
        background-color: #161b22;
        border-left: 5px solid #58a6ff;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .context-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 0.7rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
        color: #a9b1ba;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# 🔐 KONFIGURASI API GEMINI
# ============================
os.environ["GOOGLE_API_KEY"] = st.secrets.get("GEMINI_API_KEY", "")

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    temperature=0.8,           # lebih natural
    top_p=0.9,                 # variasi gaya bahasa
    max_output_tokens=800,     # agar bisa panjang
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
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

# ============================
# 📂 LOAD ARTIFACTS
# ============================
artifact_folder = "artifacts"
index = faiss.read_index(os.path.join(artifact_folder, "faiss.index"))
chunks_df = pd.read_parquet(os.path.join(artifact_folder, "chunks.parquet"))
embeddings = np.load(os.path.join(artifact_folder, "embeddings.npy"))

# ============================
# 🔍 RETRIEVAL FUNCTION
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
# 💬 RAG FUNCTION
# ============================
def ask_gemini_rag(question, context):
    prompt = f"""
Kamu adalah asisten hukum profesional yang menganalisis kontrak pembiayaan.
Gunakan konteks di bawah untuk menjawab dengan akurat, ringkas, dan sopan.

KONTEKS:
{context}

PERTANYAAN:
{question}

Aturan:
- Jawab berdasarkan teks, jangan berasumsi.
- Jika informasi tidak tersedia, katakan dengan jelas.
- Gunakan bahasa Indonesia formal yang mudah dipahami.
"""
    response = llm.invoke(prompt)
    return clean_text(response.content.strip())

# ============================
# 🖥️ STREAMLIT UI
# ============================
st.title("⚖️ Legal Contract Analyzer")
st.caption("AI Assistant untuk menganalisis isi kontrak pembiayaan menggunakan **LangChain + Gemini RAG**")

with st.sidebar:
    st.header("📁 Pengaturan Dokumen")
    available_docs = sorted(chunks_df["filename"].unique().tolist())
    target_doc = st.selectbox("Pilih dokumen kontrak:", available_docs, index=0)
    top_k = st.slider("🔎 Jumlah konteks teratas", 3, 10, 5)
    st.markdown("---")
    st.info("💡 Tips:\nGunakan pertanyaan seperti:\n- 'Siapa pihak peminjam?'\n- 'Berapa jumlah pembiayaan?'")

# ============================
# 🧠 USER INPUT
# ============================
user_question = st.text_area("💬 Masukkan pertanyaan Anda:", placeholder="Contoh: Apa denda jika peminjam terlambat membayar?")

if st.button("🚀 Analisis Kontrak", use_container_width=True):
    if not user_question.strip():
        st.warning("❗ Harap isi pertanyaan terlebih dahulu.")
    else:
        with st.spinner("Menganalisis dokumen dan memanggil Gemini..."):
            try:
                docs = retrieve_from_doc(user_question, target_doc, top_k=top_k)
                if not docs:
                    st.error("Tidak ada konteks ditemukan untuk dokumen ini.")
                else:
                    context_text = "\n\n".join(clean_text(d) for d in docs)
                    answer = ask_gemini_rag(user_question, context_text)

                    st.markdown("### 🧠 Jawaban Gemini")
                    st.markdown(f"<div class='ai-box'>{answer}</div>", unsafe_allow_html=True)

                    with st.expander("📜 Konteks yang digunakan"):
                        st.markdown(f"<div class='context-box'>{context_text}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Terjadi kesalahan: {e}")

# ============================
# ⚙️ FOOTER
# ============================
st.markdown("---")
st.caption("💼 Built by **Imam Bari Setiawan** | Powered by Gemini & LangChain 🚀")
