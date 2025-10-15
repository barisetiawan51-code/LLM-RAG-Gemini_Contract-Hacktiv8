import os
import re
import faiss
import numpy as np
import pandas as pd
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

# === Konfigurasi API ===
os.environ["GOOGLE_API_KEY"] = st.secrets.get("GEMINI_API_KEY", "")
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",     # atau "gemini-1.5-pro" jika ingin akurasi tinggi
    temperature=0.8,              # lebih tinggi = lebih ekspresif
    top_p=0.9,                    # menjaga variasi respons
    max_output_tokens=800,        # supaya jawabannya bisa agak panjang
    convert_system_message_to_human=True,  # biar gaya lebih natural
    verbose=True,                 # tampilkan log interaksi LangChain
)

# === Fungsi Bersih Teks ===
def clean_text(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"[*_]+", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

# === Load Index dan Metadata ===
artifact_folder = "artifacts"
index = faiss.read_index(os.path.join(artifact_folder, "faiss.index"))
chunks_df = pd.read_parquet(os.path.join(artifact_folder, "chunks.parquet"))
embeddings = np.load(os.path.join(artifact_folder, "embeddings.npy"))

# === Fungsi Retrieve ===
def retrieve_from_doc(query, file_name, top_k=5):
    doc_mask = chunks_df["filename"].str.lower() == file_name.lower()
    doc_chunks = chunks_df[doc_mask].reset_index(drop=True)
    doc_embeddings = embeddings[doc_mask]

    if len(doc_chunks) == 0:
        return []

    # FAISS index lokal untuk dokumen
    index_doc = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index_doc.add(doc_embeddings.astype("float32"))

    # Gunakan embedding dari Gemini
    from google.generativeai import embed_content
    query_emb = embed_content(model="models/embedding-001", content=query)["embedding"]
    query_emb = np.array([query_emb]).astype("float32")

    # Pencarian FAISS
    distances, indices = index_doc.search(query_emb, top_k)
    return [doc_chunks.iloc[i]["text"] for i in indices[0] if 0 <= i < len(doc_chunks)]

# === Fungsi RAG ===
def ask_gemini_rag(question, context):
    prompt = f"""
Kamu adalah asisten hukum yang ahli dalam menganalisis isi kontrak pembiayaan.
Gunakan konteks berikut untuk menjawab pertanyaan secara spesifik dan akurat.

KONTEKS:
{context}

PERTANYAAN:
{question}

Petunjuk:
- Ambil informasi langsung dari teks (jangan buat asumsi).
- Jika informasi tidak ditemukan, tulis bahwa data tidak tersedia dalam konteks.
- Gunakan bahasa Indonesia yang profesional, ringkas, dan informatif.
"""
    response = llm.invoke(prompt)
    return response.content.strip()

# === Streamlit UI ===
st.set_page_config(page_title="📄 Legal Contract RAG", page_icon="⚖️")
st.title("📄 Legal Contract Analyzer (Gemini + LangChain RAG)")

st.sidebar.header("⚙️ Pengaturan")
target_doc = st.sidebar.text_input("Nama Dokumen:", "Contract_4.pdf")
top_k = st.sidebar.slider("Jumlah konteks teratas (Top K)", 3, 10, 5)

user_question = st.text_area("💬 Masukkan pertanyaan Anda:")

if st.button("🔍 Analisis Kontrak"):
    with st.spinner("Menganalisis dokumen..."):
        docs = retrieve_from_doc(user_question, target_doc, top_k=top_k)
        if not docs:
            st.warning("Tidak ada konteks ditemukan untuk dokumen ini.")
        else:
            context_text = "\n\n".join(clean_text(d) for d in docs)
            answer = ask_gemini_rag(user_question, context_text)
            st.subheader("🧠 Jawaban Gemini:")
            st.write(clean_text(answer))

# === Footer ===
st.markdown("---")
st.caption("RAG GEMINI with LANGCHAIN from IMAM BARI SETIAWAN")
