import os
import re
import faiss
import random
import numpy as np
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from google.generativeai import embed_content

# ==============================
# ⚙️ CONFIG STREAMLIT
# ==============================
st.set_page_config(page_title="Legal Contract Chatbot (RAG)", page_icon="📜", layout="wide")

# ==============================
# 🎨 CSS Styling
# ==============================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(145deg, #0d1117, #0a0f14);
    color: #e6edf3;
}
.main > div { padding-top: 0rem !important; }           
.main-title { text-align: center; font-size: 2.7rem; color: #58a6ff; font-weight: 700; }
.subtitle { text-align: center; color: #a9b1ba; font-size: 1.1rem; margin-bottom: 2rem; }
.ai-box { background: rgba(30, 40, 55, 0.9); border-left: 4px solid #58a6ff;
    padding: 1rem 1.3rem; border-radius: 12px; margin-top: 1rem; font-size: 1.05rem;
    line-height: 1.6; box-shadow: 0 0 10px rgba(88,166,255,0.2);
}
.context-box { background: #161b22; border: 1px solid #30363d; border-radius: 10px;
    padding: 0.9rem; color: #a9b1ba; font-size: 0.95rem; line-height: 1.6;
}
mark { background-color: #fef08a; color: #000; border-radius: 4px; padding: 2px 4px; }
.footer { text-align: center; margin-top: 3rem; padding-top: 1rem;
    border-top: 1px solid #30363d; color: #8b949e; font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# 🔑 API KEY
# ==============================
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "")
if not GEMINI_KEY:
    st.error("❌ API Key Gemini belum diset di Streamlit Secrets!")
else:
    os.environ["GOOGLE_API_KEY"] = GEMINI_KEY

# ==============================
# 🤖 MODEL
# ==============================
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0.8,
    top_p=0.9,
    max_output_tokens=1000,
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
# 📂 LOAD ARTIFACTS
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
# 💬 STYLE PROMPTS
# ==============================
style_prompts = {
    "informative": "Berikan jawaban yang jelas, padat, dan disertai konteks tambahan bila relevan.",
    "formal": "Gunakan gaya bahasa hukum yang formal dan profesional.",
    "natural": "Gunakan gaya bahasa alami seperti percakapan sehari-hari.",
    "ringkas": "Berikan jawaban singkat dan langsung ke poin utama."
}

# ==============================
# 🧠 ASK GEMINI
# ==============================
def ask_gemini_rag(question, retrieved_chunks, style="informative"):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
    Kamu adalah asisten hukum profesional.
    {style_prompts.get(style, style_prompts['informative'])}

    Berdasarkan konteks berikut:
    {context}

    Pertanyaan:
    {question}

    Jawaban dalam Bahasa Indonesia:
    """
    response = llm.invoke(prompt)
    return clean_text(response.content.strip())

# ==============================
# 🎯 AUTO QUESTION GENERATOR
# ==============================
def generate_questions(topic, n=3):
    variations = {
        "pihak kontrak": [
            "Siapa pihak-pihak yang terlibat dalam perjanjian ini?",
            "Sebutkan siapa peminjam dan pemberi pinjaman dalam kontrak ini.",
            "Pihak mana saja yang disebutkan dalam kontrak pembiayaan ini?"
        ],
        "pembayaran": [
            "Bagaimana sistem pembayaran diatur dalam kontrak ini?",
            "Berapa jangka waktu dan jumlah cicilan yang disepakati?",
            "Bagaimana ketentuan pembayaran dijelaskan oleh kontrak?"
        ],
        "denda": [
            "Apa yang terjadi jika peminjam terlambat membayar cicilan?",
            "Apakah ada denda atau penalti atas keterlambatan pembayaran?",
            "Bagaimana kontrak mengatur konsekuensi keterlambatan pembayaran?"
        ],
        "jaminan": [
            "Apa bentuk jaminan yang diberikan oleh peminjam?",
            "Bagaimana jaminan diatur dalam kontrak ini?",
            "Apakah aset yang dibiayai dijadikan jaminan?"
        ],
        "hukum": [
            "Pengadilan mana yang berwenang menyelesaikan sengketa?",
            "Bagaimana kontrak mengatur yurisdiksi hukum?",
            "Apakah ada klausul yang menetapkan wilayah hukum tertentu?"
        ]
    }
    return random.sample(variations.get(topic, ["Topik tidak dikenal."]), n)

# ==============================
# 🏠 HEADER
# ==============================
st.markdown("<h1 class='main-title'>⚖️ Legal Contract Chatbot (RAG)</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Chatbot cerdas untuk analisis kontrak hukum menggunakan Gemini + RAG</p>", unsafe_allow_html=True)
st.markdown("---")

# ==============================
# 🧱 SIDEBAR
# ==============================
with st.sidebar:
    st.header("⚙️ Pengaturan")
    available_docs = sorted(chunks_df["filename"].unique().tolist())
    target_doc = st.selectbox("📄 Pilih dokumen:", available_docs)
    style = st.selectbox("🗣️ Gaya jawaban:", list(style_prompts.keys()))
    top_k = st.slider("🔎 Jumlah konteks teratas", 3, 10, 5)
    mode = st.radio("💬 Mode pertanyaan:", ["Manual", "Otomatis"])

# ==============================
# 🚀 MODE MANUAL
# ==============================
if mode == "Manual":
    question = st.text_area("Masukkan pertanyaan Anda:", placeholder="Contoh: Apa sanksi jika peminjam terlambat membayar?", height=100)
    if st.button("🚀 Analisis Kontrak", use_container_width=True):
        if not question.strip():
            st.warning("⚠️ Harap isi pertanyaan terlebih dahulu.")
        else:
            with st.spinner("🔎 Mencari konteks relevan..."):
                docs = retrieve_from_doc(question, target_doc, top_k)
            if not docs:
                st.error("❌ Tidak ada konteks ditemukan.")
            else:
                with st.spinner("🧠 Menganalisis dengan Gemini..."):
                    answer = ask_gemini_rag(question, docs, style)
                st.markdown("### 🧩 Hasil Analisis Gemini")
                st.markdown(f"<div class='ai-box'>{answer}</div>", unsafe_allow_html=True)

# ==============================
# 🤖 MODE OTOMATIS
# ==============================
else:
    topic = st.selectbox("Pilih topik pertanyaan:", ["pihak kontrak", "pembayaran", "denda", "jaminan", "hukum"])
    if st.button("✨ Generate Pertanyaan Otomatis", use_container_width=True):
        auto_qs = generate_questions(topic)
        for q in auto_qs:
            st.markdown(f"**❓ {q}**")
            docs = retrieve_from_doc(q, target_doc, top_k)
            if not docs:
                st.warning("Tidak ditemukan konteks relevan.")
                continue
            answer = ask_gemini_rag(q, docs, style)
            st.markdown(f"🧠 **Jawaban:** {answer}")
            st.divider()

# ==============================
# 🦶 FOOTER
# ==============================
st.markdown("<div class='footer'>💼 Dibangun oleh <b>Imam Bari Setiawan</b> | Powered by Gemini & LangChain 🚀</div>", unsafe_allow_html=True)
