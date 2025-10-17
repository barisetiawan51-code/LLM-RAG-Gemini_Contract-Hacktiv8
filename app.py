import os
import re
import faiss
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from google.generativeai import embed_content

# ==============================
# ⚙️ KONFIGURASI STREAMLIT
# ==============================
st.set_page_config(
    page_title="⚖️ Legal Contract Analyzer",
    page_icon="📄",
    layout="wide"
)

# ==============================
# 🎨 CSS TAMBAHAN UNTUK UI
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
# 🔑 API KEY
# ==============================
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
if not GEMINI_KEY:
    st.error("❌ API Key Gemini belum diset di Streamlit Secrets!")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GEMINI_KEY

# ==============================
# 📦 LOAD ARTIFACTS
# ==============================
artifact_folder = "/content/drive/MyDrive/Project Portofolio/LLM-RAG-Gemini-Hacktiv8/LLM_RAG_artifacts/"
index = faiss.read_index(os.path.join(artifact_folder, "faiss.index"))
chunks_df = pd.read_parquet(os.path.join(artifact_folder, "chunks.parquet"))
embeddings = np.load(os.path.join(artifact_folder, "embeddings.npy"))

# ==============================
# 🤖 MODEL GEMINI
# ==============================
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    temperature=0.7,
    top_p=0.9,
    max_output_tokens=700,
)

embedding_fn = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# ==============================
# 🔍 RETRIEVAL FUNCTION
# ==============================
def retrieve_from_doc(query, file_name, top_k=5):
    doc_mask = chunks_df["filename"].str.lower() == file_name.lower()
    doc_chunks = chunks_df[doc_mask].reset_index(drop=True)
    doc_embeddings = embeddings[doc_mask]

    if len(doc_chunks) == 0:
        return []

    index_doc = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index_doc.add(doc_embeddings.astype("float32"))

    query_emb = embedding_fn.embed_query(query)
    query_emb = np.array([query_emb]).astype("float32")

    distances, indices = index_doc.search(query_emb, top_k)
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(doc_chunks):
            row = doc_chunks.iloc[idx]
            results.append({
                "text": row["text"],
                "filename": row["filename"]
            })
    return results

# ==============================
# 🧠 ASK GEMINI (RAG)
# ==============================
def ask_gemini_rag(question, retrieved_chunks):
    joined_context = "\n\n".join([r["text"] for r in retrieved_chunks])
    prompt = f"""
Kamu adalah asisten hukum profesional.
Gunakan hanya konteks berikut untuk menjawab pertanyaan secara ringkas dan akurat.

KONTEKS:
{joined_context}

PERTANYAAN:
{question}

Aturan:
- Jawab langsung berdasarkan isi konteks.
- Jangan tulis 'Berdasarkan konteks'.
- Jika informasi tidak ditemukan, jawab: "Informasi tidak ditemukan dalam konteks."
"""
    response = llm.invoke(prompt)
    return response.content.strip()

# ==============================
# 💡 HIGHLIGHT SEMANTIK
# ==============================
def highlight_by_semantic_similarity(context_text, answer, threshold=0.65):
    """Menyorot bagian konteks yang paling semantik mirip dengan jawaban."""
    sentences = re.split(r'(?<=[.!?]) +', context_text)
    if not sentences:
        return context_text

    try:
        context_embs = [
            embed_content(model="models/text-embedding-004", content=sent)["embedding"]
            for sent in sentences
        ]
        answer_emb = embed_content(model="models/text-embedding-004", content=answer)["embedding"]

        sims = cosine_similarity([answer_emb], context_embs)[0]

        highlighted = []
        for sent, sim in zip(sentences, sims):
            if sim >= threshold:
                highlighted.append(f"<mark>{sent.strip()}</mark>")
            else:
                highlighted.append(sent.strip())
        return " ".join(highlighted)
    except Exception:
        return context_text

# ==============================
# 🏠 HEADER
# ==============================
st.markdown("<h1 class='main-title'>⚖️ Legal Contract Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analisis isi kontrak hukum dengan Gemini RAG dan Highlight Kontekstual</p>", unsafe_allow_html=True)

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
    ### 💡 Contoh Pertanyaan
    - Siapa pihak peminjam?
    - Apa sanksi jika terjadi keterlambatan?
    - Berapa jumlah pinjaman?
    """)

# ==============================
# 💬 INPUT USER
# ==============================
user_question = st.text_area(
    "Masukkan pertanyaan Anda:",
    placeholder="Contoh: Siapa pihak peminjam?",
    height=120
)

# ==============================
# 🚀 PROSES ANALISIS
# ==============================
if st.button("🚀 Analisis Kontrak", use_container_width=True):
    if not user_question.strip():
        st.warning("⚠️ Harap isi pertanyaan terlebih dahulu.")
    else:
        with st.spinner("🔎 Mengambil konteks relevan..."):
            retrieved = retrieve_from_doc(user_question, target_doc, top_k=top_k)

        if not retrieved:
            st.error("❌ Tidak ada konteks relevan ditemukan.")
        else:
            with st.spinner("🤖 Menganalisis dengan Gemini..."):
                answer = ask_gemini_rag(user_question, retrieved)

            # === Jawaban ===
            st.markdown("---")
            st.markdown("### 🧩 Hasil Analisis Gemini")
            st.markdown(f"<div class='ai-box'>{answer}</div>", unsafe_allow_html=True)

            # === Konteks ===
            st.markdown("### 📚 Sumber Konteks dari Dokumen")
            st.markdown(f"**📄 Dokumen:** *{target_doc}*")

            combined_text = " ".join([r["text"] for r in retrieved])
            highlighted_md = highlight_by_semantic_similarity(combined_text, answer)
            st.markdown(f"<div class='context-line'>{highlighted_md}</div>", unsafe_allow_html=True)

# ==============================
# 🦶 FOOTER
# ==============================
st.markdown("<div class='footer'>💼 Dibangun oleh <b>Imam Bari Setiawan</b> | Powered by Gemini & LangChain 🚀</div>", unsafe_allow_html=True)
