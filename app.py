import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests

# --- CONFIGURAÇÕES ---
HF_API_TOKEN = st.secrets["hf_token"]
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}
EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# --- FUNÇÕES AUXILIARES ---

def extract_text_from_pdf(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text, max_chunk_size=300):
    sentences = text.split(". ")
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_embeddings(chunks):
    return EMBEDDING_MODEL.encode(chunks).astype("float32")

def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def query_huggingface(question, context):
    prompt = f"""
Leia o conteúdo abaixo e responda de forma direta à pergunta.

[Conteúdo]
{context}

[Pergunta]
{question}

[Resposta]
"""
    response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt})
    if response.status_code == 200:
        return response.json()[0]["generated_text"].split("[Resposta]")[-1].strip()
    else:
        return f"Erro: {response.text}"

# --- APP STREAMLIT ---

st.set_page_config(page_title="Assistente PDF", layout="centered")
st.title("📚 Assistente PDF com LLM")
st.markdown("Faça upload de um ou mais PDFs e faça perguntas sobre o conteúdo.")

# Sessão
if "messages" not in st.session_state:
    st.session_state.messages = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# Exibir mensagens anteriores
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Upload e processamento
with st.sidebar:
    st.header("📂 Upload de PDF")
    pdf_files = st.file_uploader("Escolha seus arquivos", accept_multiple_files=True, type="pdf")
    if st.button("🔍 Processar PDFs"):
        raw_text = extract_text_from_pdf(pdf_files)
        chunks = get_text_chunks(raw_text)
        embeddings = generate_embeddings(chunks)
        index = create_faiss_index(embeddings)

        st.session_state.chunks = chunks
        st.session_state.faiss_index = index
        st.success("PDFs processados com sucesso!")

# Perguntas
query = st.chat_input("Digite sua pergunta")
if query and st.session_state.faiss_index:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Buscar contexto relevante
    question_embedding = EMBEDDING_MODEL.encode([query]).astype("float32")
    D, I = st.session_state.faiss_index.search(question_embedding, k=3)
    context = "\n".join([st.session_state.chunks[i] for i in I[0]])

    with st.spinner("Pensando..."):
        answer = query_huggingface(query, context)

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

elif query:
    st.warning("Por favor, carregue e processe os PDFs primeiro.")

# Rodapé
st.markdown("---")
st.caption("Desenvolvido com ❤️ usando Streamlit e Hugging Face")
