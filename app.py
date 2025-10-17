import os
import re
import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
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
# 🎨 CSS KUSTOM UNTUK TAMPILAN
# ==============================
st.markdown("""
<style>
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

.main-title {
    text-align: center;
    font-size: 2.6rem;
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

.card {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    margin-bottom: 1.5rem;
}

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
    background-color: #58a6ff33;
    color: #ffffff;
    font-weight: 600;
    padding: 2px 4px;
    border-radius: 4px;
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
    temperature=0.7,
    top_p=0.9,
    max_output_tokens=800,
    convert_system_message_to_human=True,
)

# ==============================
# 🧹 CLEAN TEXT
# ==============================
def clean_text(text):
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
Jawab pertanyaan di bawah ini secara ringkas, akurat, dan sopan, berdasarkan isi dokumen kontrak berikut.

KONTEKS:
{joined_context}

PERTANYAAN:
{question}

Aturan:
- Jawab hanya berdasarkan isi dokumen.
- Jangan buat asumsi.
- Jika tidak ada informasi yang relevan, tulis "Informasi tidak ditemukan dalam konteks."
"""
    response = llm.invoke(prompt)
    return clean_text(response.content.strip())

# ==============================
# 🪄 HIGHLIGHT OTOMATIS KONTEKS RELEVAN
# ==============================
def highlight_relevant_parts(answer, chunks):
    """
    Mencari potongan dokumen yang paling mirip dengan jawaban Gemini
    dan memberikan highlight pada bagian itu.
    """
    if not answer or not chunks:
        return []

    # Dapatkan embedding jawaban & setiap chunk
    ans_emb = np.array(embed_content(model="models/text-embedding-004", content=answer)["embedding"]).reshape(1, -1)
    chunk_embs = np.array([embed_content(model="models/text-embedding-004", content=c)["embedding"] for c in chunks])
    sims = cosine_similarity(ans_emb, chunk_embs)[0]

    # Pilih top-3 bagian paling relevan
    top_idx = np.argsort(sims)[-3:][::-1]
    highlighted_chunks = []
    for i, c in enumerate(chunks):
        if i in top_idx:
            highlighted_chunks.append(f"<div class='context-line'><span class='highlight'>{clean_text(c)}</span></div>")
        else:
            highlighted_chunks.append(f"<div class='context-line'>{clean_text(c)}</div>")
    return highlighted_chunks

# ==============================
# 🏠 HEADER
# ==============================
st.markdown("<h1 class='main-title'>⚖️ Legal Contract Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Menganalisis isi kontrak hukum menggunakan Gemini + LangChain RAG</p>", unsafe_allow_html=True)

st.markdown("---")

# ==============================
# 🧱 SIDEBAR
# ==============================
with st.sidebar:
    st.header("📁 Pengaturan Dokumen")
    available_docs = sorted(chunks_df["filename"].unique().tolist())
    target_doc = st.selectbox("Pilih dokumen:", available_docs)
    top_k = st.slider("🔎 Jumlah konteks teratas", 3, 10, 5)

# ==============================
# 💬 INPUT
# ==============================
user_question = st.text_area(
    "Masukkan pertanyaan Anda:",
    placeholder="Contoh: Siapa pihak peminjam dalam kontrak ini?",
    height=120
)

# ==============================
# 🚀 ANALISIS
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

            st.markdown("---")
            st.markdown("### 🧩 Hasil Analisis Gemini")
            st.markdown(f"<div class='ai-box'>{answer}</div>", unsafe_allow_html=True)

            # Highlight otomatis bagian konteks yang mirip dengan jawaban
            st.markdown("### 📜 Sumber Konteks dari Dokumen")
            for chunk_html in highlight_relevant_parts(answer, docs):
                st.markdown(chunk_html, unsafe_allow_html=True)

# ==============================
# 🦶 FOOTER
# ==============================
st.markdown("<div class='footer'>💼 Dibangun oleh <b>Imam Bari Setiawan</b> | Powered by Gemini & LangChain 🚀</div>", unsafe_allow_html=True)
