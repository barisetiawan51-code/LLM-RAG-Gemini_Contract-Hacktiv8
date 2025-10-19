import os
import re
import faiss
import numpy as np
import pandas as pd
import streamlit as st
import random
import google.generativeai as genai

# ==========================
# ⚙️ Konfigurasi Dasar
# ==========================
st.set_page_config(page_title="📜 Legal Contract Chatbot (RAG)", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
body {
    background: #0d1117;
    color: #e6edf3;
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
    font-size: 1rem;
    margin-bottom: 1.5rem;
}
.answer-box {
    background: rgba(30, 40, 55, 0.9);
    border-left: 4px solid #58a6ff;
    padding: 1rem 1.3rem;
    border-radius: 12px;
    margin-top: 1rem;
    font-size: 1.05rem;
    line-height: 1.6;
    box-shadow: 0 0 10px rgba(88,166,255,0.2);
}
.footer {
    text-align: center;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #30363d;
    color: #8b949e;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>⚖️ Legal Contract Chatbot (RAG)</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analisis kontrak hukum otomatis menggunakan Gemini AI dengan mode pertanyaan manual dan otomatis</p>", unsafe_allow_html=True)

# ==========================
# 🔐 Konfigurasi API Gemini
# ==========================
if "GEMINI_API_KEY" in st.secrets:
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
else:
    GEMINI_KEY = os.getenv("api_token_gemini_LLM")

if not GEMINI_KEY:
    st.error("❌ API key Gemini belum diset di secrets atau environment variable!")
else:
    genai.configure(api_key=GEMINI_KEY)

embedding_model = "models/text-embedding-004"
model = genai.GenerativeModel("models/gemini-2.5-pro")

# ==========================
# 📂 Load Artifact
# ==========================
artifact_folder = "artifacts"
index = faiss.read_index(os.path.join(artifact_folder, "faiss.index"))
chunks_df = pd.read_parquet(os.path.join(artifact_folder, "chunks.parquet"))

# ==========================
# 🔍 Retrieval Function
# ==========================
def retrieve(query, target_doc=None, top_k=5):
    query_emb = genai.embed_content(model=embedding_model, content=query)["embedding"]
    query_emb = np.array([query_emb]).astype("float32")
    distances, indices = index.search(query_emb, top_k)

    if target_doc:
        filtered = chunks_df[chunks_df["filename"] == target_doc]
        return filtered.iloc[indices[0]]["text"].tolist()
    else:
        return [chunks_df.iloc[i]["text"] for i in indices[0]]

# ==========================
# 🧠 RAG Response Function
# ==========================
def ask_gemini_rag(question, context, style="informative"):
    style_prompts = {
        "formal": "Gunakan gaya bahasa hukum yang formal dan profesional.",
        "natural": "Gunakan gaya bahasa alami seperti percakapan sehari-hari.",
        "ringkas": "Berikan jawaban singkat dan langsung ke poin utama.",
        "informative": "Berikan jawaban yang jelas, padat, dan disertai konteks tambahan bila relevan."
    }

    prompt = f"""
    Kamu adalah asisten hukum cerdas yang menjelaskan isi kontrak pembiayaan.
    {style_prompts.get(style, style_prompts["informative"])}

    Berdasarkan konteks berikut:
    {context}

    Pertanyaan: {question}

    Jawaban dalam Bahasa Indonesia secara informatif.
    """
    response = model.generate_content(prompt)
    return response.text.strip()

# ==========================
# 🎯 Generator Pertanyaan
# ==========================
def generate_questions(topic, n=3):
    variations = {
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
    if topic not in variations:
        return ["Topik tidak dikenal."]
    return random.sample(variations[topic], n)

# ==========================
# 🧱 Streamlit Interface
# ==========================
contracts = sorted(chunks_df["filename"].unique())
target_doc = st.selectbox("📄 Pilih kontrak yang ingin dianalisis:", contracts)
style = st.selectbox("🗣️ Pilih gaya jawaban:", ["informative", "formal", "natural", "ringkas"])
mode = st.radio("💡 Pilih mode tanya:", ["Ketik pertanyaan sendiri", "Gunakan pertanyaan otomatis"])

if mode == "Ketik pertanyaan sendiri":
    user_question = st.text_input("Masukkan pertanyaan Anda:")
    if st.button("🚀 Tanyakan"):
        docs = retrieve(user_question, target_doc)
        context = "\n\n".join(docs)
        answer = ask_gemini_rag(user_question, context, style)
        st.markdown("### 💬 Jawaban:")
        st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

else:
    topic = st.selectbox("📚 Pilih topik pertanyaan otomatis:", 
                         ["pihak kontrak", "pembayaran", "bunga", "denda", "jaminan", "hukum"])
    if st.button("🎯 Generate pertanyaan otomatis"):
        auto_qs = generate_questions(topic)
        for q in auto_qs:
            st.markdown(f"**❓ {q}**")
            docs = retrieve(q, target_doc)
            context = "\n\n".join(docs)
            answer = ask_gemini_rag(q, context, style)
            st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)
            st.divider()

# ==========================
# 🦶 Footer
# ==========================
st.markdown("<div class='footer'>💼 Dibangun oleh <b>Imam Bari Setiawan</b> | Powered by Gemini & LangChain 🚀</div>", unsafe_allow_html=True)
